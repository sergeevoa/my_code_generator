"""
Core benchmark loop and result aggregation.

run_benchmark(mode)  — async entry point; drives the full or debug run,
                        writes raw JSONL + summary JSON, handles resume.

compute_summary()    — reads a completed raw JSONL and produces aggregated metrics.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent import run_agent_for_eval
from .client import TrackingLlamaClient
from .config import MODEL, BASE_URL, ROOT, RUN_INFO_TEMPLATE
from .dataset import load_humaneval
from .monitor import ResourceMonitor, get_vram_used_mb
from .verifier import verify_solution


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(raw_file: Path, mode: str) -> Dict[str, Any]:
    """
    Read ``raw_file`` (a JSONL with ``run_info`` + ``task`` records) and
    return a dict containing aggregated metrics for the entire run.
    """
    task_records: List[Dict[str, Any]] = []
    with open(raw_file, encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            rec: Dict[str, Any] = json.loads(raw_line)
            if rec.get("type") == "task":
                task_records.append(rec)

    if not task_records:
        return {"mode": mode, "total_tasks": 0, "passed_tasks": 0, "metrics": {}}

    n = len(task_records)
    passed = sum(1 for r in task_records if r["passed"])

    def _avg(dotted_key: str) -> float:
        keys = dotted_key.split(".")
        total = 0.0
        for rec in task_records:
            val: Any = rec
            for k in keys:
                val = val[k]
            total += float(val)
        return total / n

    return {
        "mode":         mode,
        "total_tasks":  n,
        "passed_tasks": passed,
        "metrics": {
            "pass@1":                round(passed / n,                    4),
            "avg_iterations":        round(_avg("iterations"),            2),
            "avg_completion_tokens": round(_avg("completion_tokens"),     1),
            "avg_wall_time_s":       round(_avg("wall_time_s"),           2),
            "avg_vram_peak_mb":      round(_avg("resources.vram_peak_mb"), 1),
            "avg_vram_delta_mb":     round(_avg("resources.vram_delta_mb"), 1),
            "avg_gpu_util_pct":      round(_avg("resources.gpu_util_avg_pct"), 1),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_benchmark(mode: str) -> None:
    """
    Run the HumanEval benchmark in the chosen mode and persist results.

    mode="debug"
        Runs exactly 3 tasks — index 0, middle, last — for a quick sanity check.
        Always starts fresh; ignores any existing checkpoint.

    mode="full"
        Runs all 164 tasks.  On restart, automatically skips tasks that have
        already been recorded in ``.humaneval_full_checkpoint.json``.
    """

    # ── Load dataset ──────────────────────────────────────────────────────────
    all_tasks: List[Dict[str, Any]] = load_humaneval()
    total = len(all_tasks)

    # ── Select tasks and resolve file paths ───────────────────────────────────
    checkpoint_file: Optional[Path]
    completed_ids: set

    if mode == "debug":
        indices = [0, total // 2, total - 1]
        selected_tasks = [all_tasks[i] for i in indices]
        checkpoint_file = None
        completed_ids   = set()
        fresh_start     = True
        raw_file     = ROOT / "humaneval_debug_raw.jsonl"
        summary_file = ROOT / "humaneval_debug_summary.json"
        print(f"[DEBUG] Running 3 tasks — indices {indices}", flush=True)

    else:  # full
        selected_tasks  = all_tasks
        checkpoint_file = ROOT / ".humaneval_full_checkpoint.json"
        raw_file        = ROOT / "humaneval_full_raw.jsonl"
        summary_file    = ROOT / "humaneval_full_summary.json"
        completed_ids   = set()
        fresh_start     = True

        if checkpoint_file.exists():
            try:
                cp = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                completed_ids = set(cp.get("completed_ids", []))
                if completed_ids:
                    fresh_start = False
                    print(
                        f"[FULL] Resuming — {len(completed_ids)}/{total} tasks "
                        f"already done.",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[WARN] Cannot read checkpoint ({exc}). Starting fresh.", file=sys.stderr)

        if fresh_start:
            print(f"[FULL] Starting fresh — {total} tasks total.", flush=True)

    # ── Create client ─────────────────────────────────────────────────────────
    client = TrackingLlamaClient(model=MODEL, base_url=BASE_URL)

    # ── VRAM baseline (model assumed already loaded in llama-server) ──────────
    vram_baseline_mb = get_vram_used_mb()

    # ── Write run_info header (only on fresh start) ───────────────────────────
    if fresh_start:
        run_info: Dict[str, Any] = dict(RUN_INFO_TEMPLATE)
        run_info["mode"]                    = mode.upper()
        run_info["model"]                   = MODEL
        run_info["vram_after_model_load_mb"] = round(vram_baseline_mb, 1)
        with open(raw_file, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(run_info, ensure_ascii=False) + "\n")

    # ── Task loop ─────────────────────────────────────────────────────────────
    tasks_run = 0
    tasks_passed = 0
    n_selected = len(selected_tasks)

    for task in selected_tasks:
        task_id: str = str(task["task_id"])

        if task_id in completed_ids:
            print(f"[SKIP] {task_id}", flush=True)
            continue

        tasks_run += 1
        done_so_far = len(completed_ids) + tasks_run
        print(f"\n[TASK] {task_id}  ({done_so_far}/{n_selected})", flush=True)

        # ── Resource monitoring ───────────────────────────────────────────────
        monitor = ResourceMonitor()
        monitor.start()
        t_start = time.monotonic()

        # ── Run agent ─────────────────────────────────────────────────────────
        agent_result = await run_agent_for_eval(client, task)

        # ── Independent HumanEval verification ───────────────────────────────
        passed, error_type = verify_solution(
            agent_result["final_completion"] or "", task
        )
        # Prefer agent-level error if verification found no solution
        if not passed and not error_type and agent_result["agent_error_type"]:
            error_type = agent_result["agent_error_type"]

        wall_time_s = round(time.monotonic() - t_start, 2)

        # ── Collect resource readings ─────────────────────────────────────────
        resources = monitor.stop()
        vram_after = get_vram_used_mb()
        resources["vram_delta_mb"]      = round(max(0.0, vram_after - vram_baseline_mb), 1)
        resources["sandbox_peak_ram_mb"] = 0  # Docker container RAM not tracked here

        # ── Build and persist task record ─────────────────────────────────────
        record: Dict[str, Any] = {
            "type":              "task",
            "task_id":           task_id,
            "passed":            passed,
            "iterations":        agent_result["iterations"],
            "error_type":        error_type,
            "prompt_tokens":     agent_result["prompt_tokens"],
            "completion_tokens": agent_result["completion_tokens"],
            "wall_time_s":       wall_time_s,
            "resources":         resources,
            "final_completion":  agent_result["final_completion"] or "",
        }

        with open(raw_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ── Update checkpoint (full mode) ─────────────────────────────────────
        if checkpoint_file is not None:
            completed_ids.add(task_id)
            checkpoint_file.write_text(
                json.dumps(
                    {
                        "completed_ids": sorted(completed_ids),
                        "last_updated":  datetime.now().isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if passed:
            tasks_passed += 1

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}]  "
            f"iters={record['iterations']}  "
            f"tokens={record['completion_tokens']}  "
            f"time={record['wall_time_s']}s  "
            f"vram_peak={resources['vram_peak_mb']:.0f} MB"
            + (f"  error={error_type}" if error_type else ""),
            flush=True,
        )

    # ── Print and persist summary ─────────────────────────────────────────────
    summary = compute_summary(raw_file, mode.upper())
    with open(summary_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    m = summary.get("metrics", {})
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  Tasks run: {tasks_run}   Passed: {tasks_passed}")
    print(f"  {'─' * 50}")
    print(f"  {'Metric':<36}  {'Value':>10}")
    print(f"  {'─' * 50}")
    for label, key in [
        ("pass@1",                    "pass@1"),
        ("Avg. iterations",           "avg_iterations"),
        ("Avg. completion tokens",    "avg_completion_tokens"),
        ("Avg. wall time (s)",        "avg_wall_time_s"),
        ("Avg. VRAM peak (MB)",       "avg_vram_peak_mb"),
        ("Avg. VRAM Δ over baseline", "avg_vram_delta_mb"),
        ("Avg. GPU utilization (%)",  "avg_gpu_util_pct"),
    ]:
        print(f"  {label:<36}  {m.get(key, 'N/A'):>10}")
    print(sep)
    print(f"\n  Raw data  → {raw_file.name}")
    print(f"  Summary   → {summary_file.name}")
    if checkpoint_file:
        print(f"  Checkpoint→ {checkpoint_file.name}")
    print()
