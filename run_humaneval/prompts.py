"""
System prompt and tool schema used by the agent during evaluation.
File I/O tools are intentionally omitted — the agent must solve tasks
purely through execute_code + respond_to_user.
"""

from typing import Any, Dict, List

EVAL_SYSTEM_PROMPT = """You are a coding assistant being evaluated on the HumanEval benchmark.

LANGUAGE: Reply in English only. Never use any other language.

YOUR TASK:
You will receive a Python function signature with a docstring.
Implement the function body completely and correctly.

STRICT WORKFLOW — follow this exactly:
1. Call execute_code with your implementation plus assert-based tests derived from the
   docstring examples. The test block MUST end with print("ALL TESTS PASSED").
2. If tests fail (output contains [ERROR]), diagnose, fix, and call execute_code again.
3. Once execute_code returns output containing "ALL TESTS PASSED", call respond_to_user
   with the complete, clean function implementation inside a Python code block.

RULES:
- Do NOT call read_file or write_file — they are not available.
- respond_to_user MUST be the last call and MUST contain ONLY the function code.
- The code block in respond_to_user must include the full function definition (def ...).
- No explanations, prose, or extra text outside the code block.

CORRECT respond_to_user format:
```python
def function_name(args):
    # implementation
    return result
```
"""

EVAL_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "respond_to_user",
            "description": (
                "Return the final function implementation. "
                "Call this ONLY after execute_code confirms tests pass."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": (
                            "The complete Python function wrapped in a ```python``` code block."
                        ),
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python code in an isolated Docker sandbox and return stdout + stderr.\n"
                "\n"
                "Use this to test your implementation with assert-based checks.\n"
                "The sandbox BLOCKS: networking, file I/O, subprocess, eval/exec,\n"
                "  os, sys, pathlib, and other dangerous imports.\n"
                "Write clean algorithmic code only.\n"
                "\n"
                "The test block MUST end with print('ALL TESTS PASSED').\n"
                "Execution limit: 10 seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code with assert checks and print('ALL TESTS PASSED')."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
]
