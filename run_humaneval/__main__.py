"""
CLI entry point for the HumanEval benchmark package.

    python -m run_humaneval --mode debug
    python -m run_humaneval --mode full
    python -m run_humaneval --mode full --trace-debug   # B1+trace ablation
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load project .env (two levels up: run_humaneval/ → project root)
load_dotenv(Path(__file__).parent.parent / ".env")

from .benchmark import run_benchmark  # noqa: E402  (import after dotenv)
from .config import TRACE_DEBUG        # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m run_humaneval",
        description="Run HumanEval benchmark against the code-generation agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m run_humaneval --mode debug             # quick sanity check (3 tasks)\n"
            "  python -m run_humaneval --mode full              # full benchmark (164 tasks, resumable)\n"
            "  python -m run_humaneval --mode full --trace-debug # B1+trace ablation\n"
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
    parser.add_argument(
        "--trace-debug",
        action="store_true",
        default=False,
        help=(
            "Enable trace-augmented self-debugging (B1+trace ablation). "
            "On each failed execute_code, the code is re-run with AST instrumentation "
            "and the execution trace is fed back to the model. "
            "Results are written to separate files (_trace suffix). "
            "Can also be enabled via TRACE_DEBUG=1 in .env."
        ),
    )
    args = parser.parse_args()

    # CLI flag takes precedence; fall back to env var value from config
    trace_debug: bool = args.trace_debug or TRACE_DEBUG

    asyncio.run(run_benchmark(args.mode, trace_debug=trace_debug))


if __name__ == "__main__":
    main()
