#!/usr/bin/env python3
"""
Root-level launcher — delegates to the run_quixbugs package.

Runs both agent versions (B0 = plain self-debug, B1 = trace-augmented self-debug)
across 4 fixed seeds so that the effect of execution trace augmentation can be
measured cleanly.

Usage:
    python run_quixbugs.py --mode debug   # 2 tasks × 4 seeds × 2 versions (quick check)
    python run_quixbugs.py --mode full    # all tasks × 4 seeds × 2 versions (resumable)

Alternatively, invoke the package directly:
    python -m run_quixbugs --mode debug
    python -m run_quixbugs --mode full
"""

from run_quixbugs.__main__ import main

if __name__ == "__main__":
    main()
