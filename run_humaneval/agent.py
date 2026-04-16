"""
Agent loop for HumanEval evaluation.

run_agent_for_eval() drives the ReACT cycle with a reduced tool set
(execute_code + respond_to_user only — no file I/O).  It returns raw metrics
instead of printing them so the benchmark loop can record everything.

extract_code() parses the function out of the agent's final respond_to_user message.
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional, cast

from sandbox.executor import execute_python  # available via sys.path set in __init__

from .client import TrackingLlamaClient
from .config import MAX_REACT_STEPS
from .prompts import EVAL_SYSTEM_PROMPT, EVAL_TOOLS


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
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent_for_eval(
    client: TrackingLlamaClient,
    task: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the ReACT agent on a single HumanEval task and collect metrics.

    Returns:
        final_completion  : str | None  — Python function extracted from the agent's answer
        iterations        : int         — number of execute_code calls (solve attempts)
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
    total_prompt_tokens = 0
    total_completion_tokens = 0
    final_completion: Optional[str] = None
    agent_error_type: Optional[str] = None

    for _step in range(MAX_REACT_STEPS):
        tool_calls_received: List[Dict[str, Any]] = []

        try:
            async for event in client.astream_with_tools(
                history,
                EVAL_TOOLS,
                max_tokens=4096,
                temperature=0.2,
            ):
                if event["type"] == "tool_use":
                    # Cast required: Pylance infers event values as str|Unknown
                    # because the generator also yields text events with str values.
                    tool_calls_received.append(cast(Dict[str, Any], event["tool_call"]))
        except Exception as exc:
            agent_error_type = "agent_error"
            print(f"\n  [agent error] {exc}", file=sys.stderr)
            break
        finally:
            # Usage is set on the client after the generator finishes
            total_prompt_tokens    += client._last_prompt_tokens
            total_completion_tokens += client._last_completion_tokens

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
                success, output = execute_python(func_args.get("code", ""))
                prefix = "[OK]" if success else "[ERROR]"
                result = f"{prefix}\n{output}"
            else:
                result = (
                    f"Error: tool '{func_name}' is not available during evaluation."
                )

            history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        if done:
            break

    return {
        "final_completion":   final_completion,
        "iterations":         iterations,
        "prompt_tokens":      total_prompt_tokens,
        "completion_tokens":  total_completion_tokens,
        "agent_error_type":   agent_error_type,
    }
