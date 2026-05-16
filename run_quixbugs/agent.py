"""
Debug agent loop for QuixBugs evaluation.

run_agent_for_debug() drives the ReACT debug cycle with two variants:

  B0 (trace_debug=False) — plain self-debugging: the model sees only the error
      output from each execute_code call and tries to fix the bug iteratively.

  B1 (trace_debug=True) — trace-augmented self-debugging: on every failed
      execute_code call the code is re-run with AST instrumentation and the
      resulting variable-level trace is appended to the tool result, giving
      the model a precise observation of where the actual value diverges from
      the expected one.

Key design decision: the buggy QuixBugs program is injected into the history as
if the agent had already made its first execute_code call and received an error.
This means the agent starts the loop already aware of the bug's symptom and can
focus entirely on diagnosis and repair.

Iteration counting starts from 1 at the agent's first genuine execute_code call
(not the injected one).  passed_at_iter_k[i] is the pass/fail verdict from the
official verifier run on the code submitted at iteration i+1.
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional, cast

from sandbox.executor import SandboxContainer  # available via sys.path set in __init__

from .client import SeededTrackingClient
from .config import MAX_REACT_STEPS
from .prompts import DEBUG_SYSTEM_PROMPT, DEBUG_TOOLS
from .verifier import (
    build_test_harness,
    extract_error_type,
    extract_func_from_code,
    verify_solution,
)
from trace_instrumenter import instrument, extract_and_compress_trace


# Error prefixes that indicate infra / security failures — skip trace in these cases
_SKIP_TRACE_PREFIXES = ("[SANDBOX]", "[INFRASTRUCTURE", "[NOT TESTABLE", "[NO OUTPUT]")


# ── Code extractor for respond_to_user messages ───────────────────────────────

def _extract_code_from_message(message: str) -> Optional[str]:
    """Extract a Python function definition from the agent's respond_to_user message."""
    m = re.search(r"```python\s*\n(.*?)```", message, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", message, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "def " in code:
            return code
    m = re.search(r"^(def \w+.*?)(?=\n\S|\Z)", message, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()
    stripped = message.strip()
    return stripped if stripped else None


# ── Trace helper ──────────────────────────────────────────────────────────────

def _run_with_trace(code: str, container: SandboxContainer) -> Optional[str]:
    """
    Instrument *code* and run it in *container*, returning a compact trace.

    Returns None if instrumentation fails or the trace is empty.
    validate=False because the original code already passed AST validation.
    """
    instrumented = instrument(code)
    if instrumented is None:
        return None
    _, instr_output = container.execute(instrumented, validate=False)
    return extract_and_compress_trace(instr_output)


# ── Agent loop ────────────────────────────────────────────────────────────────

async def run_agent_for_debug(
    client: SeededTrackingClient,
    task: Dict[str, Any],
    container: SandboxContainer,
    max_tokens: int,
    trace_debug: bool,
    seed: int,
    max_iter: int,
) -> Dict[str, Any]:
    """
    Run the debug agent on a single QuixBugs task and collect fine-grained metrics.

    The buggy program is injected as the agent's "first generation" so that the
    agent immediately begins the debugging loop.

    Args:
        trace_debug: When True (B1), each failed execute_code is followed by an
                     AST-instrumented re-run; the trace is appended to the tool
                     result that the model sees.
        seed:        Passed to the LLM API for reproducible sampling; different
                     seeds produce meaningfully different model outputs.
        max_iter:    Maximum number of execute_code calls (debug iterations).
                     After this limit the agent receives a forced respond_to_user hint.

    Returns a dict with:
        final_completion     : str | None
        iterations           : int   — number of execute_code calls (agent-initiated)
        trace_iterations     : int   — times a trace was successfully appended (B1 only)
        passed_at_iter_k     : list[bool | None]  — official verifier verdict per iteration
        error_type_at_iter_k : list[str | None]   — error classification per iteration
        prompt_tokens        : int
        completion_tokens    : int
        agent_error_type     : str | None
    """
    func_name   = task["func_name"]
    buggy_code  = task["buggy_code"]
    has_tests   = bool(task.get("testcases"))

    # ── Step 1: execute buggy code + official test harness ────────────────────
    # This simulates the agent's "first generation".
    initial_test_code = build_test_harness(task, buggy_code)
    if initial_test_code is None:
        # No official tests — run the buggy code bare to at least surface syntax errors
        initial_test_code = buggy_code

    initial_success, initial_output = container.execute(initial_test_code, validate=False)
    initial_prefix  = "[OK]" if initial_success else "[ERROR]"
    initial_result  = f"{initial_prefix}\n{initial_output}"

    # ── Step 2 (B1 only): append trace to the initial failure ─────────────────
    trace_iterations = 0
    if trace_debug and not initial_success:
        if not any(initial_output.startswith(p) for p in _SKIP_TRACE_PREFIXES):
            trace = _run_with_trace(initial_test_code, container)
            if trace:
                trace_iterations += 1
                initial_result += (
                    "\n\n--- EXECUTION TRACE ---\n"
                    + trace
                    + "\n--- END TRACE ---\n"
                    "\nAnalyze the trace above: find the line where the actual value "
                    "diverges from what is expected, then fix the implementation."
                )

    # ── Step 3: build initial history with the injection ─────────────────────
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Debug the following Python program from the QuixBugs benchmark.\n"
                f"It contains exactly one bug.\n\n"
                f"```python\n{buggy_code}\n```"
            ),
        },
        # Simulated first generation: agent "called" execute_code with the buggy code
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id":       "call_initial_run",
                    "type":     "function",
                    "function": {
                        "name":      "execute_code",
                        "arguments": json.dumps({"code": initial_test_code}),
                    },
                }
            ],
        },
        {
            "role":         "tool",
            "tool_call_id": "call_initial_run",
            "content":      initial_result,
        },
    ]

    # ── Step 4: debug loop ────────────────────────────────────────────────────
    iterations:            int                      = 0
    passed_at_iter_k:      List[Optional[bool]]     = []
    error_type_at_iter_k:  List[Optional[str]]      = []
    final_completion:      Optional[str]            = None
    agent_error_type:      Optional[str]            = None
    total_prompt_tokens    = 0
    total_completion_tokens = 0
    # Code from the most recent execute_code call (fallback if agent never calls respond_to_user)
    last_executed_code: Optional[str] = None

    for _step in range(MAX_REACT_STEPS):
        tool_calls_received: List[Dict[str, Any]] = []

        try:
            async for event in client.astream_with_tools(
                history,
                DEBUG_TOOLS,
                max_tokens=max_tokens,
                temperature=0.2,
                seed=seed,
            ):
                if event["type"] == "tool_use":
                    tool_calls_received.append(cast(Dict[str, Any], event["tool_call"]))
        except Exception as exc:
            agent_error_type = "agent_error"
            print(f"\n  [agent error] {exc}", file=sys.stderr)
            break
        finally:
            total_prompt_tokens     += client._last_prompt_tokens
            total_completion_tokens += client._last_completion_tokens

        if not tool_calls_received:
            agent_error_type = "no_tool_call"
            break

        history.append({
            "role":       "assistant",
            "content":    None,
            "tool_calls": tool_calls_received,
        })

        done = False
        for tc in tool_calls_received:
            tc_name = tc["function"]["name"]
            try:
                tc_args: Dict[str, Any] = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                tc_args = {}

            # ── respond_to_user → agent is done ───────────────────────────
            if tc_name == "respond_to_user":
                final_completion = _extract_code_from_message(tc_args.get("message", ""))
                done = True
                break

            # ── execute_code → run in sandbox, collect metrics ─────────────
            if tc_name == "execute_code":
                iterations += 1
                code = tc_args.get("code", "")
                last_executed_code = code

                success, output = container.execute(code)
                prefix = "[OK]" if success else "[ERROR]"
                result = f"{prefix}\n{output}"

                # Record error type from agent's own output
                error_type_at_iter_k.append(extract_error_type(output, success))

                # Run official verifier on the function extracted from agent code
                extracted_func = extract_func_from_code(code, func_name)
                if extracted_func and has_tests:
                    iter_passed, _ = verify_solution(task, extracted_func, container)
                    # iter_passed is True/False/None; store bool or None
                    passed_at_iter_k.append(
                        True if iter_passed is True else
                        (None if iter_passed is None else False)
                    )
                else:
                    # Fall back to agent's own test result when verifier unavailable
                    passed_at_iter_k.append(True if (success and "ALL TESTS PASSED" in output) else False)

                # ── Trace augmentation (B1 only) ───────────────────────────
                if (
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
                            "\nAnalyze the trace above: find the line where the actual "
                            "value diverges from what is expected, then fix the implementation."
                        )

                # ── Force respond_to_user after max_iter iterations ────────
                if iterations >= max_iter:
                    result += (
                        "\n\n[MAX ITERATIONS REACHED] "
                        "You must now call respond_to_user with your best current fix."
                    )
            else:
                result = f"Error: tool '{tc_name}' is not available during evaluation."

            history.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      result,
            })

        if done:
            break

        # Give the agent one final chance to respond after hitting max_iter,
        # then break regardless to avoid runaway loops
        if iterations >= max_iter and not done:
            # One more API call is already queued in the next _step iteration
            pass

    # ── Fallback: if agent never called respond_to_user, use last executed code ──
    if final_completion is None and last_executed_code:
        final_completion = extract_func_from_code(last_executed_code, func_name)

    return {
        "final_completion":      final_completion,
        "iterations":            iterations,
        "trace_iterations":      trace_iterations,
        "passed_at_iter_k":      passed_at_iter_k,
        "error_type_at_iter_k":  error_type_at_iter_k,
        "prompt_tokens":         total_prompt_tokens,
        "completion_tokens":     total_completion_tokens,
        "agent_error_type":      agent_error_type,
    }
