"""
QuixBugs dataset loader.

Downloads buggy Python programs and JSONL test cases from the QuixBugs GitHub
repository and caches them locally under quixbugs_cache/.

Test case file format: JSONL — one JSON value per line, each being
    [inputs_list, expected_output]
where inputs_list is the list of positional arguments to the function.

31 out of 40 programs have JSON test cases. The remaining 9 (graph/Node
programs) have no JSON test cases and fall back to the agent's own verdict.

Each task dict contains:
    name        : str  — program name, e.g. "bitcount"
    func_name   : str  — name of the function under test (extracted from source)
    buggy_code  : str  — the buggy Python source (with QuixBugs-specific imports cleaned)
    testcases   : list — list of ([arg1, arg2, ...], expected_output) tuples
    uses_node   : bool — True if the program works with linked-list Node objects
    raw_json    : str  — raw JSON string of test cases (for debugging)
"""

import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import CACHE_DIR

# ── Complete program list (40 programs from the QuixBugs GitHub repo) ────────
# Verified against: https://github.com/jkoppel/QuixBugs/tree/master/python_programs
# Excluded: node.py (helper class), *_test.py files (test runners)
PROGRAM_NAMES: List[str] = [
    "bitcount",
    "breadth_first_search",
    "bucketsort",
    "depth_first_search",
    "detect_cycle",
    "find_first_in_sorted",
    "find_in_sorted",
    "flatten",
    "gcd",
    "get_factors",
    "hanoi",
    "is_valid_parenthesization",
    "kheapsort",
    "knapsack",
    "kth",
    "lcs_length",
    "levenshtein",
    "lis",
    "longest_common_subsequence",
    "max_sublist_sum",
    "mergesort",
    "minimum_spanning_tree",
    "next_palindrome",
    "next_permutation",
    "pascal",
    "possible_change",
    "powerset",
    "quicksort",
    "reverse_linked_list",
    "rpn_eval",
    "shortest_path_length",
    "shortest_path_lengths",
    "shortest_paths",
    "shunting_yard",
    "sieve",
    "sqrt",
    "subsequences",
    "to_base",
    "topological_ordering",
    "wrap",
]

# Programs that use linked-list Node objects in their logic
_NODE_PROGRAMS = {
    "breadth_first_search",
    "depth_first_search",
    "detect_cycle",
    "reverse_linked_list",
    "topological_ordering",
}

# Programs confirmed to have NO JSON test cases in the QuixBugs repo.
# Verified via GitHub API — fetching these URLs always returns 404.
# Verification for these programs falls back to the agent's own test output.
_NO_JSON_TESTCASES = {
    "breadth_first_search",
    "depth_first_search",
    "detect_cycle",
    "minimum_spanning_tree",
    "reverse_linked_list",
    "shortest_path_length",
    "shortest_path_lengths",
    "shortest_paths",
    "topological_ordering",
}

_GITHUB_BASE = (
    "https://raw.githubusercontent.com/jkoppel/QuixBugs/master"
)
_PROGRAM_URL = _GITHUB_BASE + "/python_programs/{name}.py"
_TESTCASE_URL = _GITHUB_BASE + "/json_testcases/{name}.json"

# Imports that exist only in QuixBugs repo structure and must be stripped
_QUIXBUGS_IMPORTS = re.compile(
    r"^\s*from\s+node\s+import\s+.*$|^\s*import\s+node\s*$",
    re.MULTILINE,
)


def _fetch(url: str, timeout: int = 30) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except Exception as exc:
        print(f"[dataset] fetch failed for {url}: {exc}", file=sys.stderr)
        return None


def _extract_func_name(source: str) -> Optional[str]:
    """Return the name of the first top-level function defined in source."""
    m = re.search(r"^def\s+(\w+)\s*\(", source, re.MULTILINE)
    return m.group(1) if m else None


def _clean_source(source: str) -> str:
    """Strip QuixBugs-specific imports that can't be resolved in the sandbox."""
    return _QUIXBUGS_IMPORTS.sub("", source).strip()


def _parse_testcases(raw_json: str) -> Optional[List[Tuple[List[Any], Any]]]:
    """
    Parse QuixBugs JSONL test cases.

    The format is one JSON value per line (JSONL), where each line is:
        [inputs_list, expected_output]

    For example, gcd.json:
        [[17, 0], 17]
        [[13, 13], 13]

    And bucketsort.json:
        [[[], 14], []]
        [[[3, 11, 2, 9, 1, 5], 12], [1, 2, 3, 5, 9, 11]]

    Returns list of (inputs, expected) tuples, or None on failure.
    """
    result = []
    for line in raw_json.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        inputs, expected = entry
        if not isinstance(inputs, list):
            inputs = [inputs]
        result.append((inputs, expected))

    return result if result else None


def _load_or_fetch(cache_path: Path, url: str) -> Optional[str]:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    content = _fetch(url)
    if content is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
    return content


def load_quixbugs(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Load all QuixBugs tasks, downloading and caching files as needed.

    Returns a list of task dicts sorted by program name.
    Programs whose source or test cases cannot be retrieved are silently skipped
    with a warning to stderr.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tasks: List[Dict[str, Any]] = []

    for name in PROGRAM_NAMES:
        prog_cache = CACHE_DIR / f"{name}.py"
        tc_cache   = CACHE_DIR / f"{name}.json"

        if force_refresh:
            prog_cache.unlink(missing_ok=True)
            tc_cache.unlink(missing_ok=True)

        source = _load_or_fetch(prog_cache, _PROGRAM_URL.format(name=name))
        if source is None:
            print(f"[dataset] SKIP {name} — cannot fetch program source.", file=sys.stderr)
            continue

        if name in _NO_JSON_TESTCASES:
            raw_json = None
        else:
            raw_json = _load_or_fetch(tc_cache, _TESTCASE_URL.format(name=name))

        func_name = _extract_func_name(source)
        if func_name is None:
            print(f"[dataset] SKIP {name} — no function definition found.", file=sys.stderr)
            continue

        testcases = _parse_testcases(raw_json) if raw_json else None
        if not testcases:
            reason = "no JSON testcases in repo" if name in _NO_JSON_TESTCASES else "fetch failed"
            print(
                f"[dataset] INFO {name} — {reason}; "
                "verification will use agent's own test output.",
                file=sys.stderr,
            )

        tasks.append({
            "name":       name,
            "func_name":  func_name,
            "buggy_code": _clean_source(source),
            "testcases":  testcases or [],
            "uses_node":  name in _NODE_PROGRAMS,
            "raw_json":   raw_json or "",
        })

    print(f"[dataset] Loaded {len(tasks)}/{len(PROGRAM_NAMES)} QuixBugs tasks.", flush=True)
    return tasks
