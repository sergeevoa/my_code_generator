"""
Core benchmark loop and result aggregation for QuixBugs.

run_benchmark(mode)   — async entry point; drives the full or debug run across
                        all seeds and both agent versions (B0, B1), writes raw
                        JSONL + summary JSON, handles checkpoint-based resume.

compute_summary()     — reads a completed raw JSONL and produces all aggregated
                        metrics including McNemar test and convergence curve data.

Record schema (one line per completed task/version/seed):
    {
      "type":                "task",
      "task_id":             str,
      "version":             "B0" | "B1",
      "seed":                int,
      "passed":              bool,          # official verifier verdict on final_completion
      "iterations":          int,           # agent execute_code calls (1 … MAX_ITER)
      "passed_at_iter_k":    list[bool|null],
      "error_type_at_iter_k":list[str|null],
      "prompt_tokens":       int,
      "completion_tokens":   int,
      "wall_time_s":         float,
      "final_completion":    str,
    }
"""

import asyncio
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sandbox.executor import SandboxContainer  # available via sys.path set in __init__

from .agent import run_agent_for_debug
from .client import SeededTrackingClient
from .config import (
    BASE_URL,
    HARDWARE,
    MAX_ITER_DEBUG,
    MAX_ITER_FULL,
    MAX_TOKENS,
    MODEL,
    RESULTS_DIR,
    RUN_INFO_TEMPLATE,
    SEEDS,
    TASK_TIMEOUT_S,
)
from .dataset import load_quixbugs
from .verifier import verify_solution

# Convergence curve is reported at these iteration checkpoints
_CURVE_POINTS = [1, 2, 3, 5]


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers (no NumPy/SciPy required for basic metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


def _bootstrap_ci(
    b0_passes: List[int],
    b1_passes: List[int],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap 95 % CI for Δpass@1 = mean(B1) − mean(B0)."""
    rng = random.Random(rng_seed)
    n = len(b0_passes)
    deltas = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        d = sum(b1_passes[i] for i in idxs) / n - sum(b0_passes[i] for i in idxs) / n
        deltas.append(d)
    deltas.sort()
    lo = deltas[int(alpha / 2 * n_bootstrap)]
    hi = deltas[int((1 - alpha / 2) * n_bootstrap)]
    return round(lo, 4), round(hi, 4)


def _mcnemar_pvalue(n01: int, n10: int) -> Optional[float]:
    """
    McNemar's test p-value (exact binomial, two-tailed) for the discordant pairs.
    n01 = only B1 passed, n10 = only B0 passed.
    Uses scipy if available, otherwise returns None.
    """
    try:
        from scipy.stats import binom_test  # type: ignore
        p = binom_test(n01, n01 + n10, 0.5)
        return round(float(p), 6)
    except ImportError:
        pass
    try:
        from scipy.stats import binomtest  # type: ignore
        res = binomtest(n01, n01 + n10, 0.5)
        return round(float(res.pvalue), 6)
    except (ImportError, Exception):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-run metrics (one seed × one version)
# ─────────────────────────────────────────────────────────────────────────────

def _per_run_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute metrics for one (version, seed) group."""
    n = len(records)
    if n == 0:
        return {}

    passed_mask    = [r["passed"] for r in records]
    iter_counts    = [r["iterations"] for r in records]
    prompt_toks    = [r["prompt_tokens"] for r in records]
    compl_toks     = [r["completion_tokens"] for r in records]
    wall_times     = [r["wall_time_s"] for r in records]

    pass_at_1 = sum(passed_mask) / n

    # pass@1 at iteration k — fraction solved within k iterations
    pass_at_k: Dict[str, float] = {}
    for k in _CURVE_POINTS:
        solved_by_k = sum(
            1 for r in records
            if r["passed"] and r["iterations"] <= k
        )
        pass_at_k[f"pass@1_at_iter_{k}"] = round(solved_by_k / n, 4)

    # Mean iterations to success (only tasks that passed)
    success_iters = [r["iterations"] for r in records if r["passed"]]
    mean_iters_success  = round(_mean(success_iters),   2) if success_iters else None
    median_iters_success = round(_median(success_iters), 2) if success_iters else None

    # Error type distribution on final failed tasks
    failed_final_errors: Dict[str, int] = {}
    for r in records:
        if not r["passed"] and r.get("error_type_at_iter_k"):
            etype = r["error_type_at_iter_k"][-1]
            key = etype or "unknown"
            failed_final_errors[key] = failed_final_errors.get(key, 0) + 1

    total_prompt    = sum(prompt_toks)
    total_compl     = sum(compl_toks)
    total_wall      = sum(wall_times)
    n_passed        = sum(passed_mask)
    tok_per_solved  = round(total_compl / n_passed, 1) if n_passed else None
    sec_per_solved  = round(total_wall  / n_passed, 2) if n_passed else None

    return {
        "n_tasks":               n,
        "n_passed":              n_passed,
        "pass@1":                round(pass_at_1, 4),
        **pass_at_k,
        "mean_iters_to_success":   mean_iters_success,
        "median_iters_to_success": median_iters_success,
        "error_dist_final_failed": failed_final_errors,
        "total_prompt_tokens":     total_prompt,
        "total_completion_tokens": total_compl,
        "total_wall_time_s":       round(total_wall, 2),
        "completion_tokens_per_solved_task": tok_per_solved,
        "wall_time_s_per_solved_task":       sec_per_solved,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated metrics (across 4 seeds for each version + B0 vs B1 comparison)
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_metrics(
    b0_records: List[Dict[str, Any]],
    b1_records: List[Dict[str, Any]],
    seeds: List[int],
) -> Dict[str, Any]:
    """
    Compute aggregate metrics across all seeds for both versions.
    b0_records / b1_records are the complete lists of task records for each version.
    """

    # ── 1. pass@1 per version per seed (for mean ± std) ──────────────────────
    def _pass_at_1_per_seed(records: List[Dict[str, Any]], version: str) -> List[float]:
        vals = []
        for seed in seeds:
            seed_recs = [r for r in records if r["seed"] == seed and r["version"] == version]
            if seed_recs:
                vals.append(sum(r["passed"] for r in seed_recs) / len(seed_recs))
        return vals

    b0_pass1_by_seed = _pass_at_1_per_seed(b0_records, "B0")
    b1_pass1_by_seed = _pass_at_1_per_seed(b1_records, "B1")

    b0_pass1_mean = round(_mean(b0_pass1_by_seed) or 0.0, 4)
    b0_pass1_std  = round(_std(b0_pass1_by_seed)  or 0.0, 4)
    b1_pass1_mean = round(_mean(b1_pass1_by_seed) or 0.0, 4)
    b1_pass1_std  = round(_std(b1_pass1_by_seed)  or 0.0, 4)

    # ── 2. Δpass@1 with bootstrap CI ─────────────────────────────────────────
    # For bootstrap, build aligned lists of (b0_passed, b1_passed) per (task, seed) pair.
    # Index both record sets by (task_id, seed).
    b0_idx = {(r["task_id"], r["seed"]): r for r in b0_records}
    b1_idx = {(r["task_id"], r["seed"]): r for r in b1_records}
    aligned_keys = sorted(set(b0_idx.keys()) & set(b1_idx.keys()))

    b0_aligned = [int(b0_idx[k]["passed"]) for k in aligned_keys]
    b1_aligned = [int(b1_idx[k]["passed"]) for k in aligned_keys]

    delta_pass1 = round(
        (sum(b1_aligned) - sum(b0_aligned)) / len(aligned_keys)
        if aligned_keys else 0.0,
        4,
    )
    bootstrap_ci = _bootstrap_ci(b0_aligned, b1_aligned) if aligned_keys else (None, None)

    # ── 3. McNemar 2×2 table ─────────────────────────────────────────────────
    both_pass  = sum(1 for k in aligned_keys if b0_idx[k]["passed"] and b1_idx[k]["passed"])
    only_b0    = sum(1 for k in aligned_keys if b0_idx[k]["passed"] and not b1_idx[k]["passed"])
    only_b1    = sum(1 for k in aligned_keys if not b0_idx[k]["passed"] and b1_idx[k]["passed"])
    both_fail  = sum(1 for k in aligned_keys if not b0_idx[k]["passed"] and not b1_idx[k]["passed"])
    mcnemar_p  = _mcnemar_pvalue(only_b1, only_b0)

    # ── 4. Trace contribution rate ────────────────────────────────────────────
    # Among tasks B0 failed (after all iterations), what fraction does B1 fix?
    b0_failed_keys = [k for k in aligned_keys if not b0_idx[k]["passed"]]
    trace_contrib  = (
        sum(1 for k in b0_failed_keys if b1_idx[k]["passed"]) / len(b0_failed_keys)
        if b0_failed_keys else None
    )

    # ── 5. Convergence speedup ────────────────────────────────────────────────
    # Among tasks solved by BOTH versions, mean(iterations_B0 − iterations_B1).
    both_solved = [
        k for k in aligned_keys
        if b0_idx[k]["passed"] and b1_idx[k]["passed"]
    ]
    conv_speedup = (
        round(
            sum(b0_idx[k]["iterations"] - b1_idx[k]["iterations"] for k in both_solved)
            / len(both_solved),
            3,
        )
        if both_solved else None
    )

    # ── 6. Mean / median iterations to success per version ───────────────────
    b0_success_iters = [r["iterations"] for r in b0_records if r["passed"]]
    b1_success_iters = [r["iterations"] for r in b1_records if r["passed"]]

    # ── 7. Convergence curve (main graph data) ────────────────────────────────
    # For each version and each k, compute mean ± std of pass@1_at_k across seeds.
    def _curve_stats(records: List[Dict[str, Any]], version: str) -> Dict[str, Any]:
        curve: Dict[str, Any] = {}
        for k in _CURVE_POINTS:
            per_seed = []
            for seed in seeds:
                seed_recs = [r for r in records if r["seed"] == seed and r["version"] == version]
                if not seed_recs:
                    continue
                solved = sum(1 for r in seed_recs if r["passed"] and r["iterations"] <= k)
                per_seed.append(solved / len(seed_recs))
            curve[f"iter_{k}"] = {
                "mean": round(_mean(per_seed) or 0.0, 4),
                "std":  round(_std(per_seed)  or 0.0, 4),
            }
        return curve

    # ── 8. Cost-efficiency ────────────────────────────────────────────────────
    def _cost(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_solved  = sum(1 for r in records if r["passed"])
        tot_compl = sum(r["completion_tokens"] for r in records)
        tot_wall  = sum(r["wall_time_s"] for r in records)
        return {
            "total_completion_tokens":  tot_compl,
            "total_wall_time_s":        round(tot_wall, 2),
            "completion_tokens_per_solved": round(tot_compl / n_solved, 1) if n_solved else None,
            "wall_time_s_per_solved":       round(tot_wall  / n_solved, 2) if n_solved else None,
        }

    return {
        "n_aligned_pairs": len(aligned_keys),
        "B0": {
            "pass@1_mean":             b0_pass1_mean,
            "pass@1_std":              b0_pass1_std,
            "mean_iters_to_success":   round(_mean(b0_success_iters) or 0.0, 3),
            "median_iters_to_success": round(_median(b0_success_iters) or 0.0, 3),
            "convergence_curve":       _curve_stats(b0_records, "B0"),
            "cost":                    _cost(b0_records),
        },
        "B1": {
            "pass@1_mean":             b1_pass1_mean,
            "pass@1_std":              b1_pass1_std,
            "mean_iters_to_success":   round(_mean(b1_success_iters) or 0.0, 3),
            "median_iters_to_success": round(_median(b1_success_iters) or 0.0, 3),
            "convergence_curve":       _curve_stats(b1_records, "B1"),
            "cost":                    _cost(b1_records),
        },
        "comparison": {
            "delta_pass@1":              delta_pass1,
            "bootstrap_ci_95pct":        list(bootstrap_ci),
            "mcnemar_table": {
                "both_pass":  both_pass,
                "only_B0":    only_b0,
                "only_B1":    only_b1,
                "both_fail":  both_fail,
            },
            "mcnemar_pvalue":            mcnemar_p,
            "trace_contribution_rate":   round(trace_contrib, 4) if trace_contrib is not None else None,
            "convergence_speedup_mean":  conv_speedup,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary builder
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(raw_file: Path, mode: str, seeds: List[int]) -> Dict[str, Any]:
    """Read *raw_file* (JSONL) and return the full aggregated summary dict."""
    all_records: List[Dict[str, Any]] = []
    with open(raw_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "task":
                all_records.append(rec)

    if not all_records:
        return {"mode": mode, "total_records": 0}

    b0_records = [r for r in all_records if r.get("version") == "B0"]
    b1_records = [r for r in all_records if r.get("version") == "B1"]

    # Per-run breakdown
    per_run: Dict[str, Any] = {}
    for version, recs in [("B0", b0_records), ("B1", b1_records)]:
        for seed in seeds:
            seed_recs = [r for r in recs if r["seed"] == seed]
            key = f"{version}_seed{seed}"
            per_run[key] = _per_run_metrics(seed_recs)

    aggregate = _aggregate_metrics(b0_records, b1_records, seeds)

    return {
        "mode":          mode,
        "total_records": len(all_records),
        "seeds":         seeds,
        "per_run":       per_run,
        "aggregate":     aggregate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed summary (single seed × both versions)
# ─────────────────────────────────────────────────────────────────────────────

def compute_seed_summary(raw_file: Path, seed: int) -> Dict[str, Any]:
    """Read *raw_file* (JSONL) and return aggregated summary for a single seed."""
    all_records: List[Dict[str, Any]] = []
    with open(raw_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "task" and rec.get("seed") == seed:
                all_records.append(rec)

    if not all_records:
        return {"mode": "SEED", "seed": seed, "total_records": 0}

    b0_records = [r for r in all_records if r.get("version") == "B0"]
    b1_records = [r for r in all_records if r.get("version") == "B1"]

    b0_metrics = _per_run_metrics(b0_records)
    b1_metrics = _per_run_metrics(b1_records)

    def _curve(records: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(records)
        return (
            {
                f"iter_{k}": round(
                    sum(1 for r in records if r["passed"] and r["iterations"] <= k) / n, 4
                )
                for k in _CURVE_POINTS
            }
            if n else {}
        )

    # Comparison aligned by task_id (single seed → no multi-seed averaging)
    b0_idx = {r["task_id"]: r for r in b0_records}
    b1_idx = {r["task_id"]: r for r in b1_records}
    aligned_keys = sorted(set(b0_idx.keys()) & set(b1_idx.keys()))

    b0_aligned = [int(b0_idx[k]["passed"]) for k in aligned_keys]
    b1_aligned = [int(b1_idx[k]["passed"]) for k in aligned_keys]

    delta_pass1 = round(
        (sum(b1_aligned) - sum(b0_aligned)) / len(aligned_keys) if aligned_keys else 0.0,
        4,
    )
    bootstrap_ci = _bootstrap_ci(b0_aligned, b1_aligned) if aligned_keys else (None, None)

    both_pass = sum(1 for k in aligned_keys if b0_idx[k]["passed"] and b1_idx[k]["passed"])
    only_b0   = sum(1 for k in aligned_keys if b0_idx[k]["passed"] and not b1_idx[k]["passed"])
    only_b1   = sum(1 for k in aligned_keys if not b0_idx[k]["passed"] and b1_idx[k]["passed"])
    both_fail = sum(1 for k in aligned_keys if not b0_idx[k]["passed"] and not b1_idx[k]["passed"])
    mcnemar_p = _mcnemar_pvalue(only_b1, only_b0)

    b0_failed_keys = [k for k in aligned_keys if not b0_idx[k]["passed"]]
    trace_contrib = (
        sum(1 for k in b0_failed_keys if b1_idx[k]["passed"]) / len(b0_failed_keys)
        if b0_failed_keys else None
    )

    both_solved = [k for k in aligned_keys if b0_idx[k]["passed"] and b1_idx[k]["passed"]]
    conv_speedup = (
        round(
            sum(b0_idx[k]["iterations"] - b1_idx[k]["iterations"] for k in both_solved)
            / len(both_solved),
            3,
        )
        if both_solved else None
    )

    return {
        "mode":          "SEED",
        "seed":          seed,
        "total_records": len(all_records),
        "B0": {**b0_metrics, "convergence_curve": _curve(b0_records)},
        "B1": {**b1_metrics, "convergence_curve": _curve(b1_records)},
        "comparison": {
            "delta_pass@1":             delta_pass1,
            "bootstrap_ci_95pct":       list(bootstrap_ci),
            "mcnemar_table": {
                "both_pass": both_pass,
                "only_B0":   only_b0,
                "only_B1":   only_b1,
                "both_fail": both_fail,
            },
            "mcnemar_pvalue":           mcnemar_p,
            "trace_contribution_rate":  round(trace_contrib, 4) if trace_contrib is not None else None,
            "convergence_speedup_mean": conv_speedup,
        },
    }


def _write_seed_summaries(raw_file: Path, seeds: List[int]) -> None:
    """Write quixbugs_seed{N}_summary.json for every seed that has data in raw_file."""
    for seed in seeds:
        summary = compute_seed_summary(raw_file, seed)
        if summary.get("total_records", 0) == 0:
            continue
        out = RESULTS_DIR / f"quixbugs_seed{seed}_summary.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"  Seed {seed} summary → {out.name}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_benchmark(mode: str, target_seed: Optional[int] = None) -> None:
    """
    Run the QuixBugs benchmark in the chosen mode.

    For each (seed, task) pair, both B0 (no trace) and B1 (trace-augmented)
    agents are run so that their outputs can be directly compared.

    mode="debug"
        Runs 2 tasks × 4 seeds × 2 versions = 16 agent invocations.
        Always starts fresh; ignores any existing checkpoint.

    mode="full"
        Runs all tasks × 4 seeds × 2 versions.  On restart, skips already-
        completed (task_id, version, seed) triples using a checkpoint file.
        After completion also writes per-seed summary files.

    mode="seed"
        Runs all tasks for target_seed only.  Uses the same raw file and
        checkpoint as "full" mode so a subsequent "full" run skips
        already-done seeds.  Writes quixbugs_seed{N}_summary.json.
    """

    # ── Load dataset ──────────────────────────────────────────────────────────
    all_tasks = load_quixbugs()
    if not all_tasks:
        print("[ERROR] No tasks loaded — aborting.", file=sys.stderr)
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Select tasks and files ────────────────────────────────────────────────
    if mode == "debug":
        selected_tasks  = all_tasks[:2]
        seeds_to_run:   List[int]       = SEEDS
        checkpoint_file: Optional[Path] = None
        completed_ids:   Set[str]       = set()
        fresh_start     = True
        max_iter        = MAX_ITER_DEBUG
        raw_file        = RESULTS_DIR / "quixbugs_debug_raw.jsonl"
        summary_file: Optional[Path]    = RESULTS_DIR / "quixbugs_debug_summary.json"
        print(
            f"[DEBUG] Running {len(selected_tasks)} tasks × {len(seeds_to_run)} seeds "
            f"× 2 versions = {len(selected_tasks) * len(seeds_to_run) * 2} agent invocations "
            f"(max_iter={max_iter}).",
            flush=True,
        )
    else:
        # "full" or "seed" — both share the same raw file and checkpoint
        selected_tasks  = all_tasks
        max_iter        = MAX_ITER_FULL
        checkpoint_file = RESULTS_DIR / ".quixbugs_full_checkpoint.json"
        raw_file        = RESULTS_DIR / "quixbugs_full_raw.jsonl"
        completed_ids   = set()
        fresh_start     = True

        if mode == "seed":
            if target_seed is None:
                print("[ERROR] --seed is required with --mode seed.", file=sys.stderr)
                return
            seeds_to_run = [target_seed]
            summary_file = None
        else:
            seeds_to_run = SEEDS
            summary_file = RESULTS_DIR / "quixbugs_full_summary.json"

        if checkpoint_file.exists():
            try:
                cp = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                completed_ids = set(cp.get("completed_ids", []))
                if completed_ids:
                    fresh_start = False
                    label = f"SEED={target_seed}" if mode == "seed" else "FULL"
                    print(
                        f"[{label}] Resuming — {len(completed_ids)} total runs already done. "
                        f"Checkpoint: seed={cp.get('current_seed')}, "
                        f"version={cp.get('current_version')}, "
                        f"task_idx={cp.get('current_task_idx')}",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[WARN] Cannot read checkpoint ({exc}). Starting fresh.",
                    file=sys.stderr,
                )

        total_runs = len(selected_tasks) * len(seeds_to_run) * 2
        if fresh_start:
            label = f"SEED={target_seed}" if mode == "seed" else "FULL"
            print(f"[{label}] Starting fresh — {total_runs} total agent invocations.", flush=True)

    # ── Create client and sandbox ─────────────────────────────────────────────
    client = SeededTrackingClient(model=MODEL, base_url=BASE_URL)
    container = SandboxContainer()
    try:
        container.start()
    except RuntimeError as exc:
        print(f"[INFRASTRUCTURE ERROR] Cannot start sandbox: {exc}", file=sys.stderr)
        return

    # ── Write run_info header (fresh start only) ──────────────────────────────
    if fresh_start:
        run_info: Dict[str, Any] = dict(RUN_INFO_TEMPLATE)
        run_info["mode"]     = mode.upper()
        run_info["model"]    = MODEL
        run_info["n_tasks"]  = len(selected_tasks)
        run_info["max_iter"] = max_iter
        with open(raw_file, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(run_info, ensure_ascii=False) + "\n")

    # ── Benchmark loop: seed → task → version ────────────────────────────────
    total_runs_done     = 0
    total_runs_passed   = 0
    # Snapshot the count of runs already completed from previous sessions.
    # Must not change as the current session adds to completed_ids.
    runs_from_checkpoint = len(completed_ids)

    try:
        for seed_idx, seed in enumerate(seeds_to_run):
            for task_idx, task in enumerate(selected_tasks):
                for version in ("B0", "B1"):
                    trace_debug = (version == "B1")
                    run_id = f"{task['name']}::{version}::{seed}"

                    if run_id in completed_ids:
                        print(f"[SKIP] {run_id}", flush=True)
                        continue

                    done_count = runs_from_checkpoint + total_runs_done + 1
                    total_count = len(selected_tasks) * len(seeds_to_run) * 2
                    print(
                        f"\n[RUN {done_count}/{total_count}] "
                        f"{task['name']}  seed={seed}  {version}",
                        flush=True,
                    )

                    t_start = time.monotonic()

                    # ── Run agent (with per-task wall-clock timeout) ───────────
                    try:
                        agent_result = await asyncio.wait_for(
                            run_agent_for_debug(
                                client,
                                task,
                                container,
                                max_tokens=MAX_TOKENS,
                                trace_debug=trace_debug,
                                seed=seed,
                                max_iter=max_iter,
                            ),
                            timeout=TASK_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        wall_time_s = round(time.monotonic() - t_start, 2)
                        print(
                            f"  [TIMEOUT] Exceeded {TASK_TIMEOUT_S}s — marking as FAIL.",
                            flush=True,
                        )
                        record = {
                            "type":                  "task",
                            "task_id":               task["name"],
                            "version":               version,
                            "seed":                  seed,
                            "passed":                False,
                            "iterations":            0,
                            "passed_at_iter_k":      [],
                            "error_type_at_iter_k":  ["task_timeout"],
                            "prompt_tokens":         0,
                            "completion_tokens":     0,
                            "wall_time_s":           wall_time_s,
                            "final_completion":      "",
                        }
                        if version == "B1":
                            record["trace_iterations"] = 0
                        with open(raw_file, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if checkpoint_file is not None:
                            completed_ids.add(run_id)
                            checkpoint_file.write_text(
                                json.dumps(
                                    {
                                        "completed_ids":    sorted(completed_ids),
                                        "current_seed":     seed,
                                        "current_version":  version,
                                        "current_task_idx": task_idx,
                                        "last_updated":     datetime.now().isoformat(),
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                        total_runs_done += 1
                        print(
                            f"  [FAIL]  iters=0  tokens=0  time={wall_time_s}s"
                            f"  error=task_timeout",
                            flush=True,
                        )
                        continue

                    # ── Official final verification ───────────────────────────
                    final_code = agent_result["final_completion"] or ""
                    if final_code:
                        passed_final, error_type = verify_solution(task, final_code, container)
                        # verify_solution returns (None, ...) when no test cases
                        if passed_final is None:
                            # Fall back: agent passed its own tests?
                            last_iter_passed = (
                                agent_result["passed_at_iter_k"][-1]
                                if agent_result["passed_at_iter_k"] else False
                            )
                            passed_final = bool(last_iter_passed)
                            error_type = agent_result.get("agent_error_type")
                    else:
                        passed_final = False
                        error_type   = agent_result.get("agent_error_type") or "no_solution"

                    wall_time_s = round(time.monotonic() - t_start, 2)

                    # ── Build record ──────────────────────────────────────────
                    record: Dict[str, Any] = {
                        "type":                  "task",
                        "task_id":               task["name"],
                        "version":               version,
                        "seed":                  seed,
                        "passed":                bool(passed_final),
                        "iterations":            agent_result["iterations"],
                        "passed_at_iter_k":      agent_result["passed_at_iter_k"],
                        "error_type_at_iter_k":  agent_result["error_type_at_iter_k"],
                        "prompt_tokens":         agent_result["prompt_tokens"],
                        "completion_tokens":     agent_result["completion_tokens"],
                        "wall_time_s":           wall_time_s,
                        "final_completion":      final_code,
                    }
                    if version == "B1":
                        record["trace_iterations"] = agent_result["trace_iterations"]

                    with open(raw_file, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # ── Update checkpoint (full mode) ─────────────────────────
                    if checkpoint_file is not None:
                        completed_ids.add(run_id)
                        checkpoint_file.write_text(
                            json.dumps(
                                {
                                    "completed_ids":    sorted(completed_ids),
                                    "current_seed":     seed,
                                    "current_version":  version,
                                    "current_task_idx": task_idx,
                                    "last_updated":     datetime.now().isoformat(),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )

                    total_runs_done += 1
                    if passed_final:
                        total_runs_passed += 1

                    status = "PASS" if passed_final else "FAIL"
                    trace_info = (
                        f"  trace_iters={record.get('trace_iterations', 0)}"
                        if version == "B1" else ""
                    )
                    print(
                        f"  [{status}]  iters={record['iterations']}"
                        + trace_info
                        + f"  tokens={record['completion_tokens']}"
                        + f"  time={record['wall_time_s']}s"
                        + (f"  error={error_type}" if not passed_final and error_type else ""),
                        flush=True,
                    )
    finally:
        container.stop()

    # ── Compute and persist summary ───────────────────────────────────────────
    sep = "═" * 64

    if mode == "seed":
        seed_summary = compute_seed_summary(raw_file, target_seed)
        seed_summary_file = RESULTS_DIR / f"quixbugs_seed{target_seed}_summary.json"
        with open(seed_summary_file, "w", encoding="utf-8") as fh:
            json.dump(seed_summary, fh, ensure_ascii=False, indent=2)

        b0  = seed_summary.get("B0", {})
        b1  = seed_summary.get("B1", {})
        cmp = seed_summary.get("comparison", {})

        print(f"\n{sep}")
        print(f"  QuixBugs SEED={target_seed} — {total_runs_done} runs  ({total_runs_passed} passed)")
        print(f"  {'─' * 56}")
        print(f"  {'Metric':<42}  {'B0':>8}  {'B1':>8}")
        print(f"  {'─' * 56}")
        rows = [
            ("pass@1",
             str(b0.get("pass@1", "?")),
             str(b1.get("pass@1", "?"))),
            ("Mean iters to success",
             str(b0.get("mean_iters_to_success", "?")),
             str(b1.get("mean_iters_to_success", "?"))),
            ("Median iters to success",
             str(b0.get("median_iters_to_success", "?")),
             str(b1.get("median_iters_to_success", "?"))),
        ]
        for label, v0, v1 in rows:
            print(f"  {label:<42}  {v0:>8}  {v1:>8}")
        print(f"  {'─' * 56}")
        print(f"  Δpass@1 (B1−B0):        {cmp.get('delta_pass@1', '?')}"
              f"   CI={cmp.get('bootstrap_ci_95pct', '?')}")
        print(f"  Trace contribution rate: {cmp.get('trace_contribution_rate', '?')}")
        print(f"  Convergence speedup:     {cmp.get('convergence_speedup_mean', '?')} iter")
        print(f"  McNemar p-value:         {cmp.get('mcnemar_pvalue', 'N/A (scipy missing)')}")
        print(sep)
        print(f"\n  Raw data  → {raw_file.name}")
        print(f"  Summary   → {seed_summary_file.name}")
        print(f"  Checkpoint→ {checkpoint_file.name}")
        print()
    else:
        summary = compute_summary(raw_file, mode.upper(), SEEDS)
        with open(summary_file, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        if mode == "full":
            print("\n  Per-seed summaries:")
            _write_seed_summaries(raw_file, SEEDS)

        agg = summary.get("aggregate", {})
        b0  = agg.get("B0", {})
        b1  = agg.get("B1", {})
        cmp = agg.get("comparison", {})

        print(f"\n{sep}")
        print(f"  QuixBugs {'DEBUG' if mode == 'debug' else 'FULL'} — "
              f"{total_runs_done} runs  ({total_runs_passed} passed)")
        print(f"  {'─' * 56}")
        print(f"  {'Metric':<42}  {'B0':>8}  {'B1':>8}")
        print(f"  {'─' * 56}")

        rows = [
            ("pass@1 (mean ± std)",
             f"{b0.get('pass@1_mean','?')} ±{b0.get('pass@1_std','?')}",
             f"{b1.get('pass@1_mean','?')} ±{b1.get('pass@1_std','?')}"),
            ("Mean iters to success",
             str(b0.get("mean_iters_to_success", "?")),
             str(b1.get("mean_iters_to_success", "?"))),
            ("Median iters to success",
             str(b0.get("median_iters_to_success", "?")),
             str(b1.get("median_iters_to_success", "?"))),
        ]
        for label, v0, v1 in rows:
            print(f"  {label:<42}  {v0:>8}  {v1:>8}")

        print(f"  {'─' * 56}")
        print(f"  Δpass@1 (B1−B0):        {cmp.get('delta_pass@1', '?')}"
              f"   CI={cmp.get('bootstrap_ci_95pct', '?')}")
        print(f"  Trace contribution rate: {cmp.get('trace_contribution_rate', '?')}")
        print(f"  Convergence speedup:     {cmp.get('convergence_speedup_mean', '?')} iter")
        print(f"  McNemar p-value:         {cmp.get('mcnemar_pvalue', 'N/A (scipy missing)')}")
        print(sep)
        print(f"\n  Raw data  → {raw_file.name}")
        print(f"  Summary   → {summary_file.name}")
        if checkpoint_file:
            print(f"  Checkpoint→ {checkpoint_file.name}")
        print()
