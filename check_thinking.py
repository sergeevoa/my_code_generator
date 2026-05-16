#!/usr/bin/env python3
"""
Diagnostic script: runs ONE QuixBugs task with ONE agent call and shows
whether the model produces a <think> block.

Usage:
    python check_thinking.py              # task=bitcount, mode=B0
    python check_thinking.py --task gcd  # specific task
    python check_thinking.py --trace     # B1 mode (with trace)

Output:
    - <think> block printed explicitly if present
    - Final tool call (execute_code or respond_to_user)
    - Completion token count
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from run_quixbugs.client import SeededTrackingClient
from run_quixbugs.config import BASE_URL, MAX_TOKENS, MODEL
from run_quixbugs.dataset import load_quixbugs
from run_quixbugs.prompts import DEBUG_SYSTEM_PROMPT, DEBUG_TOOLS
from run_quixbugs.verifier import build_test_harness
from sandbox.executor import SandboxContainer
from trace_instrumenter import instrument, extract_and_compress_trace

SEP = "─" * 64


def _parse_args():
    p = argparse.ArgumentParser(description="Check if Qwen3 thinking is active.")
    p.add_argument("--task",  default="bitcount", help="QuixBugs task name")
    p.add_argument("--trace", action="store_true",  help="Enable trace (B1 mode)")
    p.add_argument("--seed",  type=int, default=42,  help="LLM seed")
    return p.parse_args()


async def main():
    args = _parse_args()

    # ── Load task ─────────────────────────────────────────────────────────────
    tasks = load_quixbugs()
    task = next((t for t in tasks if t["name"] == args.task), None)
    if task is None:
        print(f"[ERROR] Task '{args.task}' not found. Available: "
              + ", ".join(t["name"] for t in tasks))
        sys.exit(1)

    print(f"\n{SEP}")
    print(f"  Task : {task['name']}  |  mode={'B1+trace' if args.trace else 'B0'}  |  seed={args.seed}")
    print(SEP)

    # ── Run buggy code in sandbox ─────────────────────────────────────────────
    container = SandboxContainer()
    container.start()

    initial_test_code = build_test_harness(task, task["buggy_code"]) or task["buggy_code"]
    success, output = container.execute(initial_test_code, validate=False)
    initial_result  = ("[OK]" if success else "[ERROR]") + "\n" + output

    print(f"\n[INITIAL EXECUTION — injected as first generation]")
    print(initial_result[:800] + ("..." if len(initial_result) > 800 else ""))

    # ── Optionally add trace (B1) ─────────────────────────────────────────────
    if args.trace and not success:
        instrumented = instrument(initial_test_code)
        if instrumented:
            _, instr_out = container.execute(instrumented, validate=False)
            from trace_instrumenter import extract_and_compress_trace
            trace = extract_and_compress_trace(instr_out)
            if trace:
                initial_result += (
                    "\n\n--- EXECUTION TRACE ---\n" + trace + "\n--- END TRACE ---\n"
                    "\nAnalyze the trace above and fix the implementation."
                )
                print(f"\n[TRACE ATTACHED — {len(trace.splitlines())} lines]")

    # ── Build history ─────────────────────────────────────────────────────────
    history = [
        {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Debug the following Python program from the QuixBugs benchmark.\n"
                f"It contains exactly one bug.\n\n"
                f"```python\n{task['buggy_code']}\n```"
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id":       "call_initial_run",
                "type":     "function",
                "function": {
                    "name":      "execute_code",
                    "arguments": json.dumps({"code": initial_test_code}),
                },
            }],
        },
        {
            "role":         "tool",
            "tool_call_id": "call_initial_run",
            "content":      initial_result,
        },
    ]

    # ── Stream ONE agent response, capturing ALL content ──────────────────────
    client = SeededTrackingClient(model=MODEL, base_url=BASE_URL)

    print(f"\n{SEP}")
    print("  STREAMING AGENT RESPONSE (raw)")
    print(SEP)

    full_text   = []
    tool_calls  = []

    async for event in client.astream_with_tools(
        history, DEBUG_TOOLS,
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        seed=args.seed,
    ):
        if event["type"] == "text":
            chunk = event["text"]
            full_text.append(chunk)
            print(chunk, end="", flush=True)
        elif event["type"] == "tool_use":
            tool_calls.append(event["tool_call"])

    print()  # newline after streaming

    # ── Analyse thinking ──────────────────────────────────────────────────────
    combined_text = "".join(full_text)

    print(f"\n{SEP}")

    if "<think>" in combined_text or "</think>" in combined_text:
        # Extract thinking block
        think_start = combined_text.find("<think>")
        think_end   = combined_text.find("</think>")
        if think_start != -1:
            block = combined_text[think_start: think_end + len("</think>") if think_end != -1 else think_start + 200]
            think_lines = block.count("\n") + 1
            think_chars = len(block)
            print(f"  ⚠  THINKING IS ACTIVE")
            print(f"     <think> block: {think_lines} lines, {think_chars} chars")
            print(f"     /no_think marker did NOT suppress thinking.")
        else:
            print(f"  ⚠  Partial <think> tag detected in output.")
    else:
        print(f"  OK  No <think> block found — thinking is suppressed.")

    print(f"\n  Completion tokens : {client._last_completion_tokens}")
    print(f"  Prompt tokens     : {client._last_prompt_tokens}")

    if tool_calls:
        tc = tool_calls[0]
        print(f"\n  Tool called : {tc['function']['name']}")
        try:
            args_parsed = json.loads(tc["function"]["arguments"])
            code = args_parsed.get("code") or args_parsed.get("message", "")
            print(f"  Code preview (first 300 chars):\n")
            print("    " + code[:300].replace("\n", "\n    "))
        except Exception:
            pass
    else:
        print("\n  [WARN] No tool call received.")

    print(f"\n{SEP}\n")
    container.stop()


if __name__ == "__main__":
    asyncio.run(main())
