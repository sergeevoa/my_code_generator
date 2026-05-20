"""
Agent loop for HumanEval evaluation.

run_agent_for_eval() drives the ReACT cycle with a reduced tool set
(execute_code + respond_to_user only — no file I/O).  It returns raw metrics
instead of printing them so the benchmark loop can record everything.

extract_code() parses the function out of the agent's final respond_to_user message.

Trace-augmented self-debugging (B1+trace):
    When trace_debug=True, every failed execute_code call is followed by a
    second sandbox run of the AST-instrumented version of the same code.
    The resulting execution trace (variable values at each step) is appended
    to the tool result so the model can pinpoint the exact divergence.
    trace_iterations counts how many times the trace was successfully attached.
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional, cast

from sandbox.executor import SandboxContainer  # available via sys.path set in __init__

from .client import TrackingLlamaClient
from .config import MAX_REACT_STEPS
from .prompts import EVAL_SYSTEM_PROMPT, EVAL_TOOLS
from trace_instrumenter import instrument, extract_and_compress_trace


# ─────────────────────────────────────────────────────────────────────────────
# Code extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_code(response: str) -> Optional[str]:
    """
    Pull the Python function out of the agent's respond_to_user message.

    Search order:
      1. ```python ... ``` fenced block
      2. ``` ... ``` fenced block that contains a ``def``
      3. Bare ``def`` block anywhere in the response
      4. The whole response stripped (last resort)
    """
    # 1. Typed fenced block
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. Untyped fenced block containing a function
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "def " in code:
            return code

    # 3. Bare function definition
    m = re.search(r"^(def \w+.*?)(?=\n\S|\Z)", response, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()

    # 4. Fallback — return anything non-empty
    stripped = response.strip()
    return stripped if stripped else None


# ─────────────────────────────────────────────────────────────────────────────
# Trace-augmented execution helper
# ─────────────────────────────────────────────────────────────────────────────

# Error prefixes that indicate infrastructure / security failures — traces
# would be uninformative in these cases, so we skip instrumentation.
_SKIP_TRACE_PREFIXES = ("[SANDBOX]", "[INFRASTRUCTURE", "[NOT TESTABLE", "[NO OUTPUT]")

# Max characters kept in a single tool result added to history.
# Typical trace-augmented result is ~2500 chars; 4000 fits it entirely.
_MAX_TOOL_RESULT_CHARS = 4000

_TIMEOUT_HINT = (
    "\n\n[PERFORMANCE] Your code exceeded the time limit. "
    "The current implementation is too slow for the given test cases. "
    "Consider: memoization (@functools.lru_cache), "
    "an iterative dynamic programming table, or a more efficient algorithm."
)


def _is_timeout(output: str) -> bool:
    lower = output.lower()
    return (
        "timeout" in lower
        or "time limit" in lower
        or "timed out" in lower
        or "превышен" in output
    )


def _trim_result(text: str) -> str:
    """Keep head + tail of *text* if it exceeds _MAX_TOOL_RESULT_CHARS."""
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text
    head = _MAX_TOOL_RESULT_CHARS // 2
    tail = _MAX_TOOL_RESULT_CHARS - head
    omitted = len(text) - _MAX_TOOL_RESULT_CHARS
    return text[:head] + f"\n...[{omitted} chars omitted]...\n" + text[-tail:]


def _run_with_trace(code: str, container: SandboxContainer) -> Optional[str]:
    """
    Instrument *code*, execute it in *container*, extract and compress the trace.

    Returns a compact trace string ready for inclusion in the model's context,
    or None if instrumentation failed or the trace was empty.

    validate=False is used because the original code already passed AST
    validation; the instrumented version only adds __trace_log__ calls.
    """
    instrumented = instrument(code)
    if instrumented is None:
        return None

    _, instr_output = container.execute(instrumented, validate=False)
    return extract_and_compress_trace(instr_output)


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent_for_eval(
    client: TrackingLlamaClient,
    task: Dict[str, Any],
    container: SandboxContainer,
    max_tokens: int = 4096,
    trace_debug: bool = False,
) -> Dict[str, Any]:
    """
    Run the ReACT agent on a single HumanEval task and collect metrics.

    Args:
        trace_debug: When True, failed execute_code calls are followed by an
                     instrumented re-run and the resulting trace is appended
                     to the tool result (B1+trace ablation).

    Returns:
        final_completion  : str | None  — Python function extracted from the agent's answer
        iterations        : int         — number of execute_code calls (solve attempts)
        trace_iterations  : int         — number of times a trace was successfully attached
        prompt_tokens     : int         — accumulated LLM prompt tokens across all steps
        completion_tokens : int         — accumulated LLM completion tokens
        agent_error_type  : str | None  — set when the agent loop itself fails
    """
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Implement the following Python function:\n\n{task['prompt']}",
        },
    ]

    iterations = 0
    trace_iterations = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    final_completion: Optional[str] = None
    agent_error_type: Optional[str] = None

    for _step in range(MAX_REACT_STEPS):
        # Up to 3 attempts per step: original call + 2 no-tool-call retries.
        # Retries stay inside the same _step so they don't consume MAX_REACT_STEPS budget.
        for _attempt in range(3):
            tool_calls_received: List[Dict[str, Any]] = []
            try:
                async for event in client.astream_with_tools(
                    history,
                    EVAL_TOOLS,
                    max_tokens=max_tokens,
                    temperature=0.2,
                ):
                    if event["type"] == "tool_use":
                        tool_calls_received.append(cast(Dict[str, Any], event["tool_call"]))
            except Exception as exc:
                agent_error_type = "agent_error"
                print(f"\n  [agent error] {exc}", file=sys.stderr)
                break
            finally:
                total_prompt_tokens    += client._last_prompt_tokens
                total_completion_tokens += client._last_completion_tokens

            if agent_error_type:
                break

            if tool_calls_received:
                break

            if _attempt < 2:
                history.append({
                    "role":    "user",
                    "content": (
                        "You must call one of the available tools: "
                        "execute_code (to test your implementation) or respond_to_user "
                        "(to submit your final answer). "
                        "Plain text responses are not accepted during evaluation."
                    ),
                })

        if agent_error_type:
            break

        if not tool_calls_received:
            agent_error_type = "no_tool_call"
            break

        history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_received,
        })

        done = False
        for tc in tool_calls_received:
            func_name: str = tc["function"]["name"]
            try:
                func_args: Dict[str, Any] = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                func_args = {}

            if func_name == "respond_to_user":
                final_completion = extract_code(func_args.get("message", ""))
                done = True
                break  # no tool result needed for respond_to_user

            if func_name == "execute_code":
                iterations += 1
                code = func_args.get("code", "")
                success, output = container.execute(code)
                prefix = "[OK]" if success else "[ERROR]"
                result = f"{prefix}\n{output}"

                if not success and _is_timeout(output):
                    result += _TIMEOUT_HINT

                # ── Trace-augmented debugging ─────────────────────────────
                # Only run when: test failed + trace mode enabled +
                # failure is a genuine logic error (not infra / security).
                elif (
                    trace_debug
                    and not success
                    and not any(output.startswith(p) for p in _SKIP_TRACE_PREFIXES)
                ):
                    trace = _run_with_trace(code, container)
                    if trace:
                        trace_iterations += 1
                        result += (
                            "\n\n--- EXECUTION TRACE ---\n"
                            + trace
                            + "\n--- END TRACE ---\n"
                            "\nAnalyze the trace above: find the line where the actual value "
                            "diverges from what is expected, then fix the implementation."
                        )
            else:
                result = (
                    f"Error: tool '{func_name}' is not available during evaluation."
                )

            history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": _trim_result(result),
            })

        if done:
            break

    return {
        "final_completion":   final_completion,
        "iterations":         iterations,
        "trace_iterations":   trace_iterations,
        "prompt_tokens":      total_prompt_tokens,
        "completion_tokens":  total_completion_tokens,
        "agent_error_type":   agent_error_type,
    }
