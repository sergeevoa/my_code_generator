"""
CLI entry point for the QuixBugs benchmark package.

    python -m run_quixbugs --mode debug              # 2 tasks x 4 seeds x 2 versions
    python -m run_quixbugs --mode full               # all tasks x 4 seeds x 2 versions
    python -m run_quixbugs --mode seed --seed 7      # all tasks for seed 7 only

Both versions (B0 = plain self-debug, B1 = trace-augmented self-debug) are
always run together so their results can be directly compared.

Environment variables (override in .env):
    LLAMA_MODEL        — model name/path on the llama server
    LLAMA_BASE_URL     — server base URL (default: http://localhost:8080/v1)
    MAX_TOKENS         — max tokens per LLM step (default: 4096)
    QB_MAX_REACT_STEPS — max ReACT steps per agent call (default: 10)
    QB_MAX_ITER_FULL   — max debug iterations in full/seed mode (default: 5)
    QB_MAX_ITER_DEBUG  — max debug iterations in debug mode (default: 2)
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load project .env (two levels up: run_quixbugs/ → project root)
load_dotenv(Path(__file__).parent.parent / ".env")

from .benchmark import run_benchmark  # noqa: E402  (import after dotenv)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m run_quixbugs",
        description="Run QuixBugs benchmark: compare plain self-debug (B0) vs "
                    "trace-augmented self-debug (B1) across 4 seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m run_quixbugs --mode debug              # quick sanity check (2 tasks)\n"
            "  python -m run_quixbugs --mode full               # full benchmark (~40 tasks, resumable)\n"
            "  python -m run_quixbugs --mode seed --seed 7      # single seed, integrates with full\n"
            "  python -m run_quixbugs --mode seed --seed 42\n"
            "\n"
            "  # or via the root-level launcher:\n"
            "  python run_quixbugs.py --mode debug\n"
            "  python run_quixbugs.py --mode seed --seed 137\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["full", "debug", "seed"],
        required=True,
        help=(
            "'debug' — runs 2 tasks x 4 seeds x 2 versions = 16 agent invocations "
            "for a quick sanity check. Always starts fresh. "
            "'full'  — runs all ~40 tasks x 4 seeds x 2 versions ~ 320 agent invocations; "
            "auto-resumes from checkpoint if interrupted; writes per-seed summaries at end. "
            "'seed'  — runs all tasks for one seed (--seed N); shares the same raw file and "
            "checkpoint as 'full', so a subsequent 'full' run skips already-done seeds. "
            "Writes quixbugs_seed{N}_summary.json on completion."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Seed to run (required when --mode seed). Supported values: 7, 42, 137, 2718.",
    )
    args = parser.parse_args()

    if args.mode == "seed" and args.seed is None:
        parser.error("--seed N is required when --mode seed")

    asyncio.run(run_benchmark(args.mode, target_seed=args.seed))


if __name__ == "__main__":
    main()
