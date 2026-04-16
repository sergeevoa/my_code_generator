"""
HumanEval dataset loader.

Tries the `datasets` library first (cleanest, works offline after first download).
Falls back to downloading the gzip'd JSONL directly from GitHub.
"""

import gzip
import io
import json
import sys
import urllib.request
from typing import Any, Dict, List


def load_humaneval() -> List[Dict[str, Any]]:
    """
    Return the full HumanEval test set as a list of task dicts.

    Each dict has at minimum:
        task_id        : str  — e.g. "HumanEval/0"
        prompt         : str  — function signature + docstring
        test           : str  — ``def check(candidate): ...`` test suite
        entry_point    : str  — function name to call in check()
        canonical_solution: str  — reference implementation (not used by agent)
    """
    # ── Attempt 1: HuggingFace datasets library ───────────────────────────────
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("openai/human-eval", split="test", trust_remote_code=True)
        tasks: List[Dict[str, Any]] = [dict(row) for row in ds]
        print(f"[dataset] Loaded {len(tasks)} tasks via `datasets`.", flush=True)
        return tasks
    except Exception as exc:
        print(
            f"[dataset] `datasets` unavailable ({exc}), trying GitHub fallback ...",
            file=sys.stderr,
            flush=True,
        )

    # ── Attempt 2: download gzip'd JSONL from GitHub ──────────────────────────
    url = (
        "https://raw.githubusercontent.com/openai/human-eval/"
        "master/data/HumanEval.jsonl.gz"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
        with gzip.open(io.BytesIO(raw)) as gz:
            lines = gz.read().decode("utf-8").strip().split("\n")
        tasks = [json.loads(line) for line in lines if line.strip()]
        print(f"[dataset] Loaded {len(tasks)} tasks from GitHub.", flush=True)
        return tasks
    except Exception as exc2:
        raise RuntimeError(
            "Cannot load the HumanEval dataset.\n"
            "  Option A — install datasets:  pip install datasets\n"
            "  Option B — check internet connectivity for the GitHub fallback.\n"
            f"  Last error: {exc2}"
        ) from exc2
