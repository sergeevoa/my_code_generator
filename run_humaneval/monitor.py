"""
Resource monitoring: VRAM, GPU utilization, and process RAM.

ResourceMonitor polls hardware counters in a background thread and exposes
peak / average readings after the monitored interval ends.

Requires optional dependencies:
  - pynvml  (pip install pynvml)  — GPU metrics
  - psutil  (pip install psutil)  — RAM metrics
Both degrade gracefully to zeroes when unavailable.
"""

import threading
from typing import Any, Dict, List, Optional

# ── Optional monitoring libraries ─────────────────────────────────────────────
# Fallback variables are typed as Any so that Pylance does not narrow their
# type to None and complain about every subsequent attribute access.

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    _pynvml: Any = None
    NVML_AVAILABLE = False

try:
    import psutil as _psutil
    PSUTIL_AVAILABLE = True
except Exception:
    _psutil: Any = None
    PSUTIL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────

def get_vram_used_mb() -> float:
    """Return current GPU 0 VRAM usage in MB, or 0.0 if pynvml unavailable."""
    if not NVML_AVAILABLE:
        return 0.0
    try:
        handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
        info = _pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used) / (1024 * 1024)
    except Exception:
        return 0.0


class ResourceMonitor:
    """
    Background-thread monitor that samples GPU VRAM, GPU utilization %, and
    process RSS every `interval` seconds.

    Usage:
        monitor = ResourceMonitor()
        monitor.start()
        ... do work ...
        readings = monitor.stop()   # {"vram_peak_mb": ..., "gpu_util_avg_pct": ..., "ram_peak_mb": ...}
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._vram_readings: List[float] = []
        self._gpu_util_readings: List[float] = []
        self._ram_readings: List[float] = []

        self._nvml_handle = None
        if NVML_AVAILABLE:
            try:
                self._nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

        self._proc = _psutil.Process() if PSUTIL_AVAILABLE else None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread."""
        self._stop.clear()
        self._vram_readings.clear()
        self._gpu_util_readings.clear()
        self._ram_readings.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        """
        Stop polling and return aggregated readings.

        Returns a dict suitable for the ``resources`` field in a task record:
            vram_peak_mb      — peak VRAM used during the interval
            gpu_util_avg_pct  — average GPU compute utilization %
            ram_peak_mb       — peak process RSS during the interval
        """
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        return {
            "vram_peak_mb":     round(max(self._vram_readings,     default=0.0), 1),
            "gpu_util_avg_pct": round(
                sum(self._gpu_util_readings) / len(self._gpu_util_readings)
                if self._gpu_util_readings else 0.0,
                1,
            ),
            "ram_peak_mb":      round(max(self._ram_readings,      default=0.0), 1),
        }

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._nvml_handle is not None:
                try:
                    mem = _pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self._vram_readings.append(int(mem.used) / (1024 * 1024))
                    util = _pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    self._gpu_util_readings.append(float(util.gpu))
                except Exception:
                    pass

            if self._proc is not None:
                try:
                    self._ram_readings.append(
                        self._proc.memory_info().rss / (1024 * 1024)
                    )
                except Exception:
                    pass

            self._stop.wait(self.interval)
