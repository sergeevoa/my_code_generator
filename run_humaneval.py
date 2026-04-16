#!/usr/bin/env python3
"""
Root-level launcher — delegates to the run_humaneval package.

Usage:
    python run_humaneval.py --mode debug   # 3 tasks: first, middle, last
    python run_humaneval.py --mode full    # all 164 tasks (auto-resumable)

Alternatively, invoke the package directly:
    python -m run_humaneval --mode debug
"""

from run_humaneval.__main__ import main

if __name__ == "__main__":
    main()
