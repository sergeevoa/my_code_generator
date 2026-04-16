"""
CLI entry point for the HumanEval benchmark package.

    python -m run_humaneval --mode debug
    python -m run_humaneval --mode full
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load project .env (two levels up: run_humaneval/ → project root)
load_dotenv(Path(__file__).parent.parent / ".env")

from .benchmark import run_benchmark  # noqa: E402  (import after dotenv)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m run_humaneval",
        description="Run HumanEval benchmark against the code-generation agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m run_humaneval --mode debug   # quick sanity check (3 tasks)\n"
            "  python -m run_humaneval --mode full    # full benchmark (164 tasks, resumable)\n"
            "\n"
            "  # or via the root-level launcher:\n"
            "  python run_humaneval.py --mode debug\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["full", "debug"],
        required=True,
        help=(
            "'full'  — run all 164 tasks; auto-resumes if interrupted. "
            "'debug' — run 3 tasks (first, middle, last) for a quick sanity check."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.mode))


if __name__ == "__main__":
    main()
