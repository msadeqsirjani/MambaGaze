"""
Power monitor for NVIDIA Jetson platforms.

Supports three sysfs interfaces, tried in priority order:

  1. iio_device/in_power*_input  — older JetPack 5.x kernels (AGX Orin, Orin NX)
     Driver: ina3221 / ina3221x via IIO subsystem.
     Files report power directly in mW.

  2. hwmon power*_input          — Jetson Nano / Xavier NX (hwmon, µW)
     /sys/class/hwmon/hwmon*/power*_input — values are in µW, converted to mW.

  3. hwmon in*_input + curr*_input  — JetPack 6.x kernels (Orin Nano, Orin NX,
     AGX Orin).  The ina3221 driver exposes bus voltage (mV) and shunt current
     (mA) separately; power is computed as P = V × I / 1000 per channel.

     Channel selection per chip (to avoid double-counting sub-rails):
       • If any labeled channel matches a "total board input" pattern
         (VDD_IN, VIN_SYS*, …) → use only those channels from that chip.
       • Otherwise → sum all labeled channels on that chip.
     Results are summed across all ina3221 chips found.

Falls back silently on non-Jetson hardware (returns empty stats).

Known label mapping across Orin family:
  Orin Nano / Orin NX  : VDD_IN (total), VDD_CPU_GPU_CV, VDD_SOC
  AGX Orin             : VIN_SYS_5V0 (total), VDD_GPU_SOC, VDD_CPU_CV
"""

from __future__ import annotations

import glob
import os
import threading
import time
from typing import Dict, List, Tuple

# Labels that indicate a total board-input rail.
# A channel whose label matches ANY of these (and NONE of _SUBSYS_MARKERS)
# is treated as the primary supply measurement for its chip.
_TOTAL_MARKERS: Tuple[str, ...] = (
    "VDD_IN",       # Orin Nano, Orin NX
    "VIN_SYS",      # AGX Orin DevKit  (e.g. VIN_SYS_5V0)
    "VIN_",         # generic VIN_ prefix
    "VDD_IN_SYS",   # alternative naming
)

# Sub-system keywords — a channel whose label contains any of these is NOT
# a total rail even if it also matches _TOTAL_MARKERS.
_SUBSYS_MARKERS: Tuple[str, ...] = (
    "CPU", "GPU", "SOC", "MEM", "CV",
    "DRAM", "DDR", "VDDQ",
)


def _is_total_rail(label: str) -> bool:
    u = label.upper()
    return (any(m in u for m in _TOTAL_MARKERS) and
            not any(m in u for m in _SUBSYS_MARKERS))


class PowerMonitor:
    """
    Background thread that samples Jetson power rails every `interval_ms` ms.

    Usage:
        pm = PowerMonitor(interval_ms=50)
        pm.start()
        # ... run inference ...
        stats = pm.stop()
        print(stats["total_power_mw"])
    """

    def __init__(self, interval_ms: int = 50):
        self.interval_s  = interval_ms / 1000.0
        self._samples: List[float] = []
        self._running    = False
        self._thread: threading.Thread | None = None

        self._power_paths = self._find_power_paths()
        # V×I pairs are only needed when no direct power files exist
        self._vi_pairs: List[Tuple[str, str]] = (
            [] if self._power_paths else self._find_vi_pairs()
        )

    # ------------------------------------------------------------------
    # Path discovery — interface 1 & 2 (direct mW / µW files)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_power_paths() -> List[str]:
        """Paths that report power directly (iio in_power*_input or hwmon power*_input)."""
        candidates: List[str] = []

        # Interface 1: IIO subsystem (JetPack 5.x, AGX Orin / Orin NX)
        for pat in (
            "/sys/bus/i2c/drivers/ina3221x/*/iio_device/in_power*_input",
            "/sys/bus/i2c/drivers/ina3221/*/iio_device/in_power*_input",
        ):
            candidates.extend(glob.glob(pat))

        # Interface 2: hwmon power*_input in µW (Jetson Nano / Xavier NX)
        candidates.extend(glob.glob("/sys/class/hwmon/hwmon*/power*_input"))

        return sorted(set(candidates))

    # ------------------------------------------------------------------
    # Path discovery — interface 3 (V × I via ina3221 hwmon, JetPack 6.x)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_vi_pairs() -> List[Tuple[str, str]]:
        """
        Return (voltage_path, current_path) pairs for ina3221 hwmon channels.

        Scans /sys/class/hwmon/hwmon*/ entries whose `name` file reads "ina3221"
        or "ina3221x".  Per chip:
          • Only labeled channels are considered (unlabeled indices are driver
            internal / virtual channels such as shunt-sum).
          • Channels whose label matches a total-input pattern (VDD_IN,
            VIN_SYS_5V0, …) are preferred to avoid double-counting sub-rails
            that are electrically downstream of the main supply.
          • If no labeled channels exist (rare), falls back to the first three
            physical channels (INA3221 has exactly 3 measurement inputs).
        Results from all chips are summed — chips on a given board measure
        non-overlapping power domains.
        """
        result: List[Tuple[str, str]] = []

        for hwmon_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*/")):
            try:
                name = open(os.path.join(hwmon_dir, "name")).read().strip()
            except OSError:
                continue
            if name not in ("ina3221", "ina3221x"):
                continue

            chip_all: List[Tuple[str, str]] = []    # all labeled channels
            chip_total: List[Tuple[str, str]] = []  # total-input channels

            for i in range(1, 10):
                v_path = os.path.join(hwmon_dir, f"in{i}_input")
                c_path = os.path.join(hwmon_dir, f"curr{i}_input")
                if not (os.path.exists(v_path) and os.path.exists(c_path)):
                    continue

                label_path = os.path.join(hwmon_dir, f"in{i}_label")
                try:
                    label = open(label_path).read().strip()
                except OSError:
                    # No label → virtual/derived channel (e.g. shunt-sum); skip.
                    continue

                pair = (v_path, c_path)
                chip_all.append(pair)
                if _is_total_rail(label):
                    chip_total.append(pair)

            if not chip_all:
                # Edge case: driver exposes no labels; use first 3 physical channels.
                for i in range(1, 4):
                    v = os.path.join(hwmon_dir, f"in{i}_input")
                    c = os.path.join(hwmon_dir, f"curr{i}_input")
                    if os.path.exists(v) and os.path.exists(c):
                        chip_all.append((v, c))

            result.extend(chip_total if chip_total else chip_all)

        return result

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_mw(path: str) -> float:
        try:
            val = int(open(path).read().strip())
            # hwmon power*_input reports in µW; iio in_power*_input in mW
            return val / 1000.0 if "hwmon" in path and "power" in os.path.basename(path) else float(val)
        except Exception:
            return 0.0

    @staticmethod
    def _read_vi_mw(v_path: str, c_path: str) -> float:
        """Read voltage (mV) and current (mA), return power in mW."""
        try:
            v_mv = int(open(v_path).read().strip())
            c_ma = int(open(c_path).read().strip())
            return (v_mv * c_ma) / 1000.0
        except Exception:
            return 0.0

    def _sample(self) -> float:
        total  = sum(self._read_mw(p) for p in self._power_paths)
        total += sum(self._read_vi_mw(v, c) for v, c in self._vi_pairs)
        return total

    def _run(self):
        while self._running:
            self._samples.append(self._sample())
            time.sleep(self.interval_s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._samples  = []
        self._running  = True
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return {}
        import numpy as np
        arr = np.array(self._samples)
        return {
            "total_power_mw": float(arr.mean()),
            "peak_power_mw":  float(arr.max()),
            "min_power_mw":   float(arr.min()),
            "n_samples":      len(arr),
            "interval_ms":    int(self.interval_s * 1000),
        }
