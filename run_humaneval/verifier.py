"""
HumanEval solution verifier.

After the agent returns a solution, verify_solution() runs the dataset's
own test suite (task["test"]) inside the sandbox to get an independent pass/fail
verdict — this is the "additional check" mentioned in the benchmark spec.
"""

from typing import Any, Dict, Optional, Tuple

from sandbox.executor import execute_python  # available via sys.path set in __init__


def _imports_from_prompt(prompt: str) -> str:
    """Extract top-level import lines from the task prompt (e.g. ``from typing import List``)."""
    lines = [
        line for line in prompt.split("\n")
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    return "\n".join(lines)


def verify_solution(
    solution_code: str,
    task: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Run the HumanEval test suite against ``solution_code`` in the sandbox.

    Builds the following test script and executes it:

        <imports from task prompt>

        <solution_code>

        <task["test"]>        # contains ``def check(candidate): ...``

        check(<entry_point>)
        print('ALL TESTS PASSED')

    Returns:
        (True,  None)          — all assertions passed
        (False, error_type)    — failure; error_type is one of:
                                   "no_solution"   — solution_code was empty
                                   "timeout"       — sandbox time limit hit
                                   "sandbox_block" — AST validator blocked an import
                                   "test_failure"  — assertion error or runtime exception
    """
    if not solution_code:
        return False, "no_solution"

    imports = _imports_from_prompt(task["prompt"])
    parts = []
    if imports:
        parts.append(imports)
    parts.append(solution_code)
    parts.append(str(task["test"]))
    parts.append(f"check({task['entry_point']})")
    parts.append("print('ALL TESTS PASSED')")

    test_code = "\n\n".join(parts)
    # validate=False: тест-сюита из датасета — доверенный код; AST-валидация
    # ложно блокирует легитимные импорты (sys, os и др.) из тестов HumanEval.
    # Docker-изоляция остаётся и обеспечивает достаточный уровень безопасности.
    success, output = execute_python(test_code, validate=False)

    if success and "ALL TESTS PASSED" in output:
        return True, None

    output_lower = output.lower()
    if "превышен" in output or "timeout" in output_lower or "time limit" in output_lower:
        return False, "timeout"
    if "sandbox" in output_lower and (
        "заблокирован" in output or "blocked" in output_lower
    ):
        return False, "sandbox_block"

    return False, "test_failure"
