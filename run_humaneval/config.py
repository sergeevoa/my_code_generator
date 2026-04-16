"""
Run-time constants for the HumanEval benchmark.

All tunable values are read from environment variables (loaded from .env by
__main__.py before this module is imported).  Defaults match the local dev
machine; override them in .env when running on a different host.
"""

import os
from pathlib import Path
from typing import Any, Dict

# Project root (run_humaneval/ → project root)
ROOT: Path = Path(__file__).parent.parent

# Directory where all benchmark output files are written
RESULTS_DIR: Path = ROOT / "humaneval_results"

# ── LLM server ────────────────────────────────────────────────────────────────
MODEL    = os.getenv("LLAMA_MODEL",    "Qwen2.5-Coder-7B-Instruct-AWQ")
BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8080/v1")

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_REACT_STEPS = int(os.getenv("MAX_REACT_STEPS", "8"))

# ── Hardware metadata (written into run_info once per run) ────────────────────
HARDWARE: Dict[str, Any] = {
    "gpu":           os.getenv("HW_GPU",            "NVIDIA RTX 4090"),
    "gpu_count":     int(os.getenv("HW_GPU_COUNT",  "1")),
    "vram_total_mb": int(os.getenv("HW_VRAM_TOTAL_MB", "24564")),
    "cpu":           os.getenv("HW_CPU",            "AMD Ryzen 9 7950X"),
    "cpu_cores":     int(os.getenv("HW_CPU_CORES",  "16")),
    "ram_total_mb":  int(os.getenv("HW_RAM_TOTAL_MB", "65536")),
}

# ── vLLM / llama-server settings (written into run_info once per run) ─────────
VLLM_CONFIG: Dict[str, Any] = {
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEM_UTIL",   "0.9")),
    "max_model_len":          int(os.getenv("VLLM_MAX_MODEL_LEN",     "8192")),
    "dtype":                  os.getenv("VLLM_DTYPE",                 "float16"),
    "quantization":           os.getenv("VLLM_QUANTIZATION",          "awq"),
}

# ── run_info template (filled at benchmark start) ─────────────────────────────
RUN_INFO_TEMPLATE: Dict[str, Any] = {
    "type":                     "run_info",
    "mode":                     None,   # set to "FULL" or "DEBUG" at runtime
    "model":                    MODEL,
    "hardware":                 HARDWARE,
    "vllm":                     VLLM_CONFIG,
    "vram_after_model_load_mb": None,   # measured at runtime
    "seed":                     42,
}
