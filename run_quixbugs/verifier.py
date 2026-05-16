"""
QuixBugs solution verifier.

build_test_harness()  — builds an executable Python script that tests a given
                        solution against the official QuixBugs test cases.

verify_solution()     — runs the harness in the sandbox and returns (passed, error_type).

extract_func_from_code() — extracts a named function definition from agent code
                           (used to isolate just the fixed function from code that
                           also contains test assertions).

extract_error_type()  — classifies the error type from sandbox output.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from sandbox.executor import SandboxContainer  # available via sys.path set in __init__

# ── Node / helper classes injected into every test harness ───────────────────
# QuixBugs Node programs use `from node import Node` which cannot be resolved
# in the sandbox.  We inline the class definition instead.

_HELPER_CODE = '''\
class Node:
    def __init__(self, val=None, successor=None):
        self.val = val
        self.successor = successor
    def __repr__(self):
        parts, cur, seen = [], self, set()
        while cur is not None and id(cur) not in seen:
            parts.append(repr(cur.val))
            seen.add(id(cur))
            cur = cur.successor
        return "Node(" + " -> ".join(parts) + ")"
    def __eq__(self, other):
        a, b = self, other
        seen = set()
        while a is not None and b is not None:
            if a.val != b.val:
                return False
            if id(a) in seen:
                return id(b) in seen
            seen.add(id(a))
            a, b = a.successor, b.successor
        return a is None and b is None

def _list_to_node(lst):
    if not lst:
        return None
    head = Node(lst[0])
    cur = head
    for v in lst[1:]:
        cur.successor = Node(v)
        cur = cur.successor
    return head

def _node_to_list(node):
    result, seen = [], set()
    while node is not None and id(node) not in seen:
        result.append(node.val)
        seen.add(id(node))
        node = node.successor
    return result
'''

# Programs whose test-case inputs/outputs are linked-list Node objects
_NODE_PROGRAMS = {
    "breadth_first_search",
    "depth_first_search",
    "detect_cycle",
    "reverse_linked_list",
    "topological_ordering",
}

# Error prefixes that signal infrastructure / security failures
_INFRA_PREFIXES = ("[SANDBOX]", "[INFRASTRUCTURE", "[NOT TESTABLE", "[NO OUTPUT]")


# ── Error-type classifier ─────────────────────────────────────────────────────

def extract_error_type(output: str, success: bool) -> Optional[str]:
    """
    Classify the error type from sandbox stdout/stderr.

    Returns one of: AssertionError, TypeError, NameError, SyntaxError,
    RecursionError, TimeoutError, InfiniteLoopError, SandboxError, None.
    """
    if success:
        return None
    out_lower = output.lower()
    if "timeout" in out_lower or "time limit" in out_lower or "timed out" in out_lower:
        return "TimeoutError"
    if "превышен" in output:
        return "TimeoutError"
    if "recursionerror" in out_lower or "maximum recursion" in out_lower:
        return "RecursionError"
    if any(output.startswith(p) for p in _INFRA_PREFIXES):
        return "SandboxError"
    for exc in ("SyntaxError", "AssertionError", "TypeError", "NameError",
                "ValueError", "IndexError", "KeyError", "AttributeError",
                "ZeroDivisionError", "StopIteration"):
        if exc.lower() in out_lower:
            return exc
    return "RuntimeError"


# ── Function extractor ────────────────────────────────────────────────────────

def extract_func_from_code(code: str, func_name: str) -> Optional[str]:
    """
    Extract the definition of *func_name* from *code* using AST parsing.

    Returns the unparsed function source, or None if parsing fails or the
    function is not found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                try:
                    return ast.unparse(node)
                except Exception:
                    return None
    return None


# ── Test-harness builder ──────────────────────────────────────────────────────

def _repr_input(value: Any, uses_node: bool) -> str:
    """Return Python repr for a test-case input value."""
    if uses_node and isinstance(value, list):
        return f"_list_to_node({value!r})"
    return repr(value)


def _repr_expected(value: Any, uses_node: bool) -> str:
    """Return Python repr for a test-case expected value."""
    if uses_node and isinstance(value, list):
        return f"_list_to_node({value!r})"
    return repr(value)


def _compare_expr(result_var: str, expected: Any, uses_node: bool) -> str:
    """Return a Python expression that compares result to expected."""
    if uses_node and isinstance(expected, list):
        return f"_node_to_list({result_var}) == {expected!r}"
    return f"{result_var} == {expected!r}"


def build_test_harness(task: Dict[str, Any], solution_code: str) -> Optional[str]:
    """
    Build an executable test script that runs *solution_code* against all
    official QuixBugs test cases for *task*.

    Returns None if the task has no test cases (agent's own verdict is used).
    """
    testcases: List[Tuple[List[Any], Any]] = task.get("testcases", [])
    if not testcases:
        return None

    func_name = task["func_name"]
    uses_node = task.get("uses_node", False)

    lines: List[str] = [_HELPER_CODE, solution_code, ""]

    lines.append("_passed = 0")
    lines.append("_failed = 0")
    lines.append("")

    for i, (inputs, expected) in enumerate(testcases):
        input_args = ", ".join(_repr_input(v, uses_node) for v in inputs)
        call_expr  = f"{func_name}({input_args})"
        result_var = f"_r{i}"
        cmp_expr   = _compare_expr(result_var, expected, uses_node)
        # Readable expected for assertion messages
        if uses_node and isinstance(expected, list):
            exp_display = repr(expected)
        else:
            exp_display = repr(expected)

        lines += [
            f"try:",
            f"    {result_var} = {call_expr}",
            f"    assert {cmp_expr}, (",
            f"        f'Test {i}: expected {exp_display}, "
            f"got {{_node_to_list({result_var}) if isinstance({result_var}, Node) else {result_var}}}'"
            f"    )",
            f"    _passed += 1",
            f"except Exception as _e{i}:",
            f"    _failed += 1",
            f"    print(f'Test {i} FAILED: {{_e{i}}}')",
            "",
        ]

    lines += [
        "if _failed == 0:",
        "    print('ALL TESTS PASSED')",
        "else:",
        f"    raise AssertionError(f'{{_failed}}/{len(testcases)} tests failed')",
    ]

    return "\n".join(lines)


# ── Public verifier ───────────────────────────────────────────────────────────

def verify_solution(
    task: Dict[str, Any],
    solution_code: str,
    container: SandboxContainer,
) -> Tuple[Optional[bool], Optional[str]]:
    """
    Run the official QuixBugs test suite against *solution_code*.

    Returns:
        (True,  None)           — all tests passed
        (False, error_type)     — failed; error_type is a string classification
        (None,  "no_testcases") — task has no test cases; cannot verify
        (None,  "no_solution")  — solution_code is empty
    """
    if not solution_code or not solution_code.strip():
        return None, "no_solution"

    test_code = build_test_harness(task, solution_code)
    if test_code is None:
        return None, "no_testcases"

    success, output = container.execute(test_code, validate=False)

    if success and "ALL TESTS PASSED" in output:
        return True, None

    return False, extract_error_type(output, success)
