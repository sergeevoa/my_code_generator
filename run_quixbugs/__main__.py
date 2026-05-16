"""
CLI entry point for the QuixBugs benchmark package.

    python -m run_quixbugs --mode debug   # 2 tasks × 4 seeds × 2 versions (quick check)
    python -m run_quixbugs --mode full    # all tasks × 4 seeds × 2 versions (resumable)

Both versions (B0 = plain self-debug, B1 = trace-augmented self-debug) are
always run together so their results can be directly compared.

Environment variables (override in .env):
    LLAMA_MODEL       — model name/path on the llama server
    LLAMA_BASE_URL    — server base URL (default: http://localhost:8080/v1)
    MAX_TOKENS        — max tokens per LLM step (default: 4096)
    QB_MAX_REACT_STEPS— max ReACT steps per agent call (default: 10)
    QB_MAX_ITER       — max debug iterations per task (default: 5)
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
            "  python -m run_quixbugs --mode debug   # quick sanity check (2 tasks)\n"
            "  python -m run_quixbugs --mode full    # full benchmark (all ~39 tasks, resumable)\n"
            "\n"
            "  # or via the root-level launcher:\n"
            "  python run_quixbugs.py --mode debug\n"
            "  python run_quixbugs.py --mode full\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["full", "debug"],
        required=True,
        help=(
            "'debug' — runs 2 tasks × 4 seeds × 2 versions = 16 agent invocations "
            "for a quick sanity check. Always starts fresh. "
            "'full'  — runs all ~39 tasks × 4 seeds × 2 versions ≈ 312 agent invocations; "
            "auto-resumes from checkpoint if interrupted."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.mode))


if __name__ == "__main__":
    main()
