"""HumanEval benchmark runner package."""

import sys
from pathlib import Path

# Make project's src/ importable from all sub-modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
