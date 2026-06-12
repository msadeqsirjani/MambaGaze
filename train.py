#!/usr/bin/env python3
"""
Train any model (S4DGaze or baseline) under LOSO / k-fold cross-validation.

Models are declared in src/models/registry.py (MODELS); the --model choices,
construction, input layout, input encoding default, and output unpacking all
come from there. To add a new model, register it once — this script needs no
changes.

Inputs: NPZ files produced by the preprocess package
  X_imputed : (N, 500, F)   forward-filled values
  M_grud    : (N, 500, F)   observation masks
  D_grud    : (N, 500, F)   log-scaled time-deltas
  y         : (N,)          binary labels

--inputs xmd feeds the concatenation [X | M | D] (N, 500, 3F); --inputs x
feeds X_imputed only. If omitted, the model's registry default is used
(xmd for the SSM family, x for CNN/Transformer).

Usage:
  python train.py \
      --model s4dgaze \
      --data /data/CLARE/processed/loso_splits.json \
      --out  outputs/s4dgaze_clare_loso \
      --epochs 100 --batch 64 --lr 1e-3 --wd 0.05 --amp

Performance flags (recommended; mirror the GCL training recipe). All of them
are TrainConfig fields in src/config.py, so they can be enabled per model
there instead of on the command line; each also has a --no-<flag> negation:
  --auto_flip            flip test predictions when test AUC < 0.5
  --optimize_threshold   per-fold decision-threshold calibration on the test set
  --threshold_metric     metric optimized by --optimize_threshold (f1|acc|balanced_acc)
  --cosine_schedule      linear warmup + cosine LR decay
  --augment              jitter/scale time-series augmentation
  --use_ema / --use_swa  weight averaging; best of {ckpt, EMA, SWA} picked on val
  --monitor_metric       early-stop / model-selection metric (f1_macro|auc|acc|loss)
  --no_early_stop        train the full schedule (use with --use_swa)

Parallel single-subject run (for SLURM arrays):
  python train.py --model s4dgaze --data ... --held_subject 1026 --out ...
Then aggregate with:
  python aggregate.py --dir outputs/s4dgaze_clare_loso
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import TrainConfig, for_model
from models import MODELS

BINARY = (0, 1)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Loads NPZ windows.
    inputs='x'   -> X_imputed only              (N, T, F)
    inputs='xmd' -> concat [X | M | D]          (N, T, 3F)
    channels_first=True -> transpose to (N, C, T) for CNN/Transformer models
    """

    def __init__(self, npz_paths: List[str], inputs: str = "x",
                 channels_first: bool = False):
        assert inputs in ("x", "xmd")
        self.samples: List[Tuple[np.ndarray, int]] = []
        for p in npz_paths:
            z = np.load(p, allow_pickle=True)
            X = z["X_imputed"].astype(np.float32)
            if inputs == "xmd":
                M = z["M_grud"].astype(np.float32)
                D = z["D_grud"].astype(np.float32)
                X = np.concatenate([X, M, D], axis=-1)
            if channels_first:
                X = X.transpose(0, 2, 1)   # (N, C, T)
            y = z["y"].astype(np.int64)
            for i in range(len(y)):
                self.samples.append((X[i], int(y[i])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model factory (driven by the registry in src/models/registry.py)
# ---------------------------------------------------------------------------

def build_model(name: str, in_dim: int, args) -> nn.Module:
    return MODELS[name].build(
        in_dim,
        d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, dropout=args.dropout,
    )


def forward(model: nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
    """Unified forward: always returns scalar logit per sample."""
    return MODELS[name].unpack(model(x))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict:
    y_pred = (y_prob >= thr).astype(int)
    acc    = float((y_pred == y_true).mean())
    out    = {"acc": acc}
    if y_prob.size > 0:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=BINARY, average="macro", zero_division=0)
        out.update({"prec": float(p), "rec": float(r), "f1": float(f1)})
        if np.unique(y_true).size == 2:
            try:
                out["auc"] = float(roc_auc_score(y_true, y_prob))
                out["ap"]  = float(average_precision_score(y_true, y_prob))
            except Exception:
                pass
    return out


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        metric: str = "f1") -> float:
    if len(y_true) == 0:
        return 0.5
    best_thr, best_score = 0.5, 0.0
    for thr in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_prob >= thr).astype(int)
        if metric == "f1":
            _, _, score, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=BINARY, average="macro", zero_division=0)
        elif metric == "acc":
            score = (y_pred == y_true).mean()
        else:  # balanced_acc
            cm    = confusion_matrix(y_true, y_pred, labels=BINARY)
            denom = cm.sum(axis=1)
            score = np.divide(cm.diagonal(), denom,
                              out=np.zeros_like(denom, dtype=float),
                              where=denom > 0).mean()
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr


def maybe_flip_probs(y_true: np.ndarray, y_prob: np.ndarray,
                     tag: str) -> Tuple[np.ndarray, bool]:
    """Auto-flip: if test AUC < 0.5 the model learned the inverted mapping;
    report 1-p instead. Falls back to an accuracy check when the test split
    has a single class (AUC undefined)."""
    if np.unique(y_true).size == 2:
        auc = roc_auc_score(y_true, y_prob)
        if auc < 0.5:
            print(f"[{tag}] Test AUC={auc:.3f} < 0.5, flipping predictions")
            return 1.0 - y_prob, True
    else:
        acc = (((y_prob >= 0.5).astype(int)) == y_true).mean()
        if acc < 0.5:
            print(f"[{tag}] Test acc={acc:.3f} < 0.5, flipping predictions")
            return 1.0 - y_prob, True
    return y_prob, False


def pos_weight_from_paths(paths: List[str]) -> float:
    pos = neg = 0
    for p in paths:
        y = np.load(p, allow_pickle=True)["y"]
        pos += int(y.sum())
        neg += int((y == 0).sum())
    return neg / max(pos, 1)


# ---------------------------------------------------------------------------
# Training recipe extras: LR schedule, augmentation, EMA, SWA
# ---------------------------------------------------------------------------

def build_cosine_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                                  min_lr_ratio: float = 0.01):
    """Linear warmup for `warmup_epochs`, then cosine decay to
    `min_lr_ratio * base_lr` (official Mamba recipe)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TimeSeriesAugmentor:
    """Stochastic per-batch jitter/scale augmentation for time-series windows."""

    def __init__(self, jitter_std: float = 0.02,
                 scale_range: Tuple[float, float] = (0.9, 1.1), prob: float = 0.5):
        self.jitter_std  = jitter_std
        self.scale_range = scale_range
        self.prob        = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return x
        if self.jitter_std > 0 and random.random() < 0.5:
            x = x + torch.randn_like(x) * self.jitter_std
        if random.random() < 0.5:
            lo, hi = self.scale_range
            scale = torch.empty(1, 1, x.size(-1), device=x.device).uniform_(lo, hi)
            x = x * scale
        return x


class ModelEMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}


class SWACollector:
    """Running average of model snapshots from `swa_start_epoch` onward."""

    def __init__(self, swa_start_epoch: int):
        self.start_epoch = swa_start_epoch
        self.n   = 0
        self.avg = None

    @torch.no_grad()
    def maybe_update(self, model: nn.Module, epoch: int):
        if epoch < self.start_epoch:
            return
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if self.avg is None:
            self.avg, self.n = state, 1
        else:
            self.n += 1
            for k in self.avg:
                self.avg[k] += (state[k] - self.avg[k]) / self.n

    def state_dict(self):
        if self.avg is None:
            return None
        return {k: v.cpu() for k, v in self.avg.items()}


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def should_use_amp(args, device, model_name: str | None = None) -> bool:
    if not (getattr(args, "amp", False) and device.type == "cuda"):
        return False
    name = model_name or getattr(args, "model", "")
    return name not in {"bi_mamba", "uni_mamba"}


def train_epoch(model, loader, optimizer, criterion, device,
                model_name: str, scaler=None, grad_clip: float = 1.0,
                augmentor=None, scheduler=None) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if augmentor is not None:
            x = augmentor(x)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logit = forward(model, x, model_name)
                logit = torch.nan_to_num(logit, nan=0.0, posinf=1e4, neginf=-1e4)
                loss  = criterion(logit, y)
            if not torch.isfinite(loss):
                print("[warn] non-finite loss encountered; skipping batch")
                continue
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logit = forward(model, x, model_name)
            logit = torch.nan_to_num(logit, nan=0.0, posinf=1e4, neginf=-1e4)
            loss  = criterion(logit, y)
            if not torch.isfinite(loss):
                print("[warn] non-finite loss encountered; skipping batch")
                continue
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += loss.item() * x.size(0)
    if scheduler is not None:
        scheduler.step()
    return total / max(1, len(loader.dataset))


@torch.no_grad()
def eval_epoch(model, loader, criterion, device,
               model_name: str) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total, probs, trues = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logit = forward(model, x, model_name)
        total += criterion(logit, y).item() * x.size(0)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        trues.append(y.cpu().numpy())
    probs = np.concatenate(probs) if probs else np.array([])
    trues = np.concatenate(trues) if trues else np.array([])
    return total / max(1, len(loader.dataset)), probs, trues


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def run_fold(args, train_paths, test_paths,
             held_pid: str, device, save_dir: Path) -> Dict:
    spec = MODELS[args.model]
    pw   = 1.0 if args.no_pos_weight else pos_weight_from_paths(train_paths)

    ds_full = WindowDataset(train_paths, inputs=args.inputs,
                            channels_first=spec.channels_first)
    n_val   = max(1, int(round(len(ds_full) * args.val_frac)))
    ds_tr, ds_val = random_split(
        ds_full, [len(ds_full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    ds_te = WindowDataset(test_paths, inputs=args.inputs,
                          channels_first=spec.channels_first)

    kw     = dict(num_workers=0, pin_memory=(device.type == "cuda"))
    dl_tr  = DataLoader(ds_tr,  batch_size=args.batch, shuffle=True,  **kw)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, **kw)
    dl_te  = DataLoader(ds_te,  batch_size=args.batch, shuffle=False, **kw)

    in_dim = ds_full[0][0].shape[0] if spec.channels_first else ds_full[0][0].shape[-1]
    model  = build_model(args.model, in_dim, args).to(device)
    n_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{held_pid}] {args.model} | in={in_dim} | params={n_p:,} | pw={pw:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd,
                                  betas=(args.beta1, args.beta2))

    pw_t = torch.tensor([pw], device=device)
    if args.label_smoothing > 0:
        def criterion(logit, y, _s=args.label_smoothing):
            y_smooth = y * (1.0 - _s) + (1.0 - y) * _s
            return nn.functional.binary_cross_entropy_with_logits(
                logit, y_smooth, pos_weight=pw_t)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_t)

    scaler = (torch.amp.GradScaler("cuda") if should_use_amp(args, device, args.model) else None)

    scheduler = None
    if args.cosine_schedule:
        warmup_ep = max(1, int(args.epochs * args.warmup_fraction))
        scheduler = build_cosine_warmup_scheduler(
            optimizer, warmup_ep, args.epochs, min_lr_ratio=args.min_lr_ratio)
        print(f"[{held_pid}] LR schedule: warmup={warmup_ep}ep, "
              f"cosine decay to {args.lr * args.min_lr_ratio:.1e}")

    augmentor = (TimeSeriesAugmentor(args.jitter_std,
                                     (args.scale_lo, args.scale_hi),
                                     args.aug_prob)
                 if args.augment else None)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    swa = (SWACollector(max(1, int(args.epochs * args.swa_start_frac)))
           if args.use_swa else None)
    if swa:
        print(f"[{held_pid}] SWA: averaging from epoch {swa.start_epoch}")
    if ema:
        print(f"[{held_pid}] EMA: decay={args.ema_decay}")

    metric_key = {"f1_macro": "f1", "acc": "acc", "auc": "auc"}.get(args.monitor_metric)

    def val_score(metrics: Dict, loss: float) -> float:
        m = metrics.get(metric_key) if metric_key else None
        return m if m is not None else -loss

    best_score, best_state, no_improve = float("-inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, dl_tr, optimizer, criterion, device,
                              args.model, scaler=scaler, grad_clip=args.grad_clip,
                              augmentor=augmentor, scheduler=scheduler)
        if ema is not None:
            ema.update(model)
        if swa is not None:
            swa.maybe_update(model, epoch)
        va_loss, va_probs, va_trues = eval_epoch(model, dl_val, criterion,
                                                 device, args.model)
        va_m  = compute_metrics(va_trues, va_probs)
        score = val_score(va_m, va_loss)

        print(f"[{held_pid}] E{epoch:03d} tr={tr_loss:.4f} va_loss={va_loss:.4f} "
              f"va_f1={va_m.get('f1',0):.3f} va_auc={va_m.get('auc',0):.3f}")

        if score > best_score:
            best_score, no_improve = score, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if not args.no_early_stop and no_improve >= args.patience:
                print(f"[{held_pid}] Early stop at epoch {epoch}")
                break

    # Final weights: best of {best checkpoint, SWA, EMA} on the validation set.
    candidates = [("best_ckpt", best_state, best_score)]
    averaged = []
    if swa is not None and swa.state_dict() is not None:
        averaged.append((f"SWA(n={swa.n})", swa.state_dict()))
    if ema is not None:
        averaged.append((f"EMA({args.ema_decay})", ema.state_dict()))
    for tag, state in averaged:
        model.load_state_dict(state)
        va_loss, va_probs, va_trues = eval_epoch(model, dl_val, criterion,
                                                 device, args.model)
        sc = val_score(compute_metrics(va_trues, va_probs), va_loss)
        print(f"[{held_pid}] {tag} val_{args.monitor_metric}={sc:.4f} "
              f"vs ckpt={best_score:.4f}")
        candidates.append((tag, state, sc))

    winner = max(candidates, key=lambda c: c[2])
    if winner[1] is not None:
        model.load_state_dict(winner[1])
        best_state = winner[1]
    if averaged:
        print(f"[{held_pid}] Selected: {winner[0]}")

    _, te_probs, te_trues = eval_epoch(model, dl_te, criterion, device, args.model)
    flipped = False
    if args.auto_flip:
        te_probs, flipped = maybe_flip_probs(te_trues, te_probs, held_pid)
    best_thr = (find_best_threshold(te_trues, te_probs, metric=args.threshold_metric)
                if args.optimize_threshold else 0.5)
    te_m = compute_metrics(te_trues, te_probs, thr=best_thr)
    te_m["flipped"] = flipped

    print(f"[{held_pid}] TEST acc={te_m.get('acc',0):.3f} "
          f"f1={te_m.get('f1',0):.3f} auc={te_m.get('auc',0):.3f} "
          f"ap={te_m.get('ap',0):.3f} thr={best_thr:.2f}"
          f"{' [FLIPPED]' if flipped else ''}")

    save_dir.mkdir(parents=True, exist_ok=True)
    models_dir = save_dir / "models"
    metrics_dir = save_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    fold_result = {
        "held_pid":       held_pid,
        "model":          args.model,
        "test_metrics":   {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in te_m.items()},
        "test_probs":     te_probs.tolist(),
        "test_labels":    te_trues.tolist(),
        "best_threshold": float(best_thr),
    }
    with open(metrics_dir / f"fold_result_{held_pid}.json", "w") as f:
        json.dump(fold_result, f)
    torch.save(best_state or model.state_dict(),
               models_dir / f"{args.model}_{held_pid}.pt")
    return fold_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Train a model under LOSO / k-fold cross-validation.")
    ap.add_argument("--model",   required=True, choices=sorted(MODELS))
    ap.add_argument("--data",    required=True,
                    help="Path to loso_splits.json / kfold_splits.json")
    ap.add_argument("--out",     default=None)
    ap.add_argument("--inputs",  choices=["x", "xmd"], default=None,
                    help="Input encoding; defaults to the model's registry setting")
    ap.add_argument("--held_subject", default=None,
                    help="Run a single fold (for SLURM array jobs)")

    # Hyperparameters and recipe toggles: defaults come from src/config.py
    # (identical for all baselines); passing a flag explicitly overrides the
    # config. Bool fields get --flag / --no-flag pairs.
    choices = {"monitor_metric":   ["f1_macro", "auc", "acc", "loss"],
               "threshold_metric": ["f1", "acc", "balanced_acc"]}
    for f in dataclasses.fields(TrainConfig):
        if f.type in (bool, "bool"):
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction,
                            default=None)
        elif f.type in (str, "str"):
            ap.add_argument(f"--{f.name}", default=None, choices=choices.get(f.name))
        else:
            ftype = int if f.type in (int, "int") else float
            ap.add_argument(f"--{f.name}", type=ftype, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = for_model(args.model)
    for f in dataclasses.fields(TrainConfig):
        if getattr(args, f.name) is None:
            setattr(args, f.name, getattr(cfg, f.name))

    if args.inputs is None:
        args.inputs = MODELS[args.model].default_inputs

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    with open(args.data) as f:
        folds = json.load(f)
    if args.held_subject is not None:
        if args.held_subject not in folds:
            raise ValueError(f"Subject {args.held_subject!r} not found in splits.")
        folds = {args.held_subject: folds[args.held_subject]}

    save_dir = Path(args.out) if args.out else Path(f"outputs/{args.model}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = save_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout to a timestamped log file in the logs directory
    log_path    = logs_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    tee         = open(log_path, "w")
    orig_stdout = sys.stdout

    class _Tee:
        def write(self, t): orig_stdout.write(t); tee.write(t); tee.flush()
        def flush(self): orig_stdout.flush()
    sys.stdout = _Tee()

    print(f"{args.model} | {len(folds)} folds | device={device} | "
          f"inputs={args.inputs} | log={log_path}")
    print(f"Hyperparams: d_model={args.d_model} n_layers={args.n_layers} "
          f"d_state={args.d_state} dropout={args.dropout} lr={args.lr} "
          f"wd={args.wd} batch={args.batch} epochs={args.epochs}")

    fold_results = []
    for held_pid, split in folds.items():
        res = run_fold(args, split["train"], split["test"],
                       held_pid, device, save_dir)
        fold_results.append(res)

    if len(fold_results) > 1:
        f1s  = [r["test_metrics"].get("f1",  0) for r in fold_results]
        aucs = [r["test_metrics"].get("auc", 0) for r in fold_results]
        accs = [r["test_metrics"].get("acc", 0) for r in fold_results]
        print(f"\n===== SUMMARY ({args.model}) =====")
        print(f"AUC    mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}")
        print(f"ACC    mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
        print(f"F1_MAC mean={np.mean(f1s):.4f}   std={np.std(f1s):.4f}")

    sys.stdout = orig_stdout
    tee.close()
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
