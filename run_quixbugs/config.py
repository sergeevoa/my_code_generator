"""
Run-time constants for the QuixBugs benchmark.

All tunable values are read from environment variables (loaded from .env by
__main__.py before this module is imported).  Defaults match the local dev
machine; override them in .env when running on a different host.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

# Project root (run_quixbugs/ → project root)
ROOT: Path = Path(__file__).parent.parent

# Directory where all benchmark output files are written
RESULTS_DIR: Path = ROOT / "quixbugs_results"

# Local cache for downloaded QuixBugs programs and test cases
CACHE_DIR: Path = ROOT / "quixbugs_cache"

# ── LLM server ────────────────────────────────────────────────────────────────
MODEL    = os.getenv("LLAMA_MODEL",    "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8080/v1")

# ── Agent behaviour ───────────────────────────────────────────────────────────
# Total ReACT steps per task (one step = one LLM API call)
MAX_REACT_STEPS = int(os.getenv("QB_MAX_REACT_STEPS", "10"))
# Max tokens the model may generate per step
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
# Max debug iterations (execute_code calls) per task — separate limits per mode
MAX_ITER_FULL  = int(os.getenv("QB_MAX_ITER_FULL",  "5"))
MAX_ITER_DEBUG = int(os.getenv("QB_MAX_ITER_DEBUG", "2"))
# Wall-clock time limit per task (seconds); exceeded → FAIL, next task
TASK_TIMEOUT_S = int(os.getenv("QB_TASK_TIMEOUT_S", "600"))

# ── Seeds ─────────────────────────────────────────────────────────────────────
# 4 fixed seeds chosen for meaningfully different LLM sampling:
#   7    — small prime, shifts token probabilities distinctly
#   42   — the canonical baseline seed
#   137  — fine-structure constant ≈ 1/137; prime, numerically distant from 42
#   2718 — truncation of e ≈ 2.71828; large, different modular properties
SEEDS: List[int] = [7, 42, 137, 2718]

# ── Hardware metadata (written into run_info once per run) ────────────────────
HARDWARE: Dict[str, Any] = {
    "gpu":           os.getenv("HW_GPU",            "NVIDIA RTX 3090"),
    "gpu_count":     int(os.getenv("HW_GPU_COUNT",  "1")),
    "vram_total_mb": int(os.getenv("HW_VRAM_TOTAL_MB", "24576")),
    "cpu":           os.getenv("HW_CPU",            "Unknown"),
    "cpu_cores":     int(os.getenv("HW_CPU_CORES",  "16")),
    "ram_total_mb":  int(os.getenv("HW_RAM_TOTAL_MB", "32768")),
}

# ── run_info template (filled at benchmark start) ─────────────────────────────
RUN_INFO_TEMPLATE: Dict[str, Any] = {
    "type":     "run_info",
    "mode":     None,   # set to "FULL" or "DEBUG" at runtime
    "model":    MODEL,
    "seeds":    SEEDS,
    "max_iter_full":  MAX_ITER_FULL,
    "max_iter_debug": MAX_ITER_DEBUG,
    "hardware": HARDWARE,
}
