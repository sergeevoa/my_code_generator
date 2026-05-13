import json
import os
import sys
import asyncio
import time
from typing import List, Dict, Any, Optional

_CODE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.rs', '.swift'}

from system_prompt import build_system_prompt
from tools import TOOLS, execute_tool
from context_manager import compact_history
from trace_instrumenter import instrument, extract_and_compress_trace

MAX_REACT_STEPS = 8

# Output prefixes that indicate infra / security failures — no point tracing these.
_SKIP_TRACE_PREFIXES = ("[SANDBOX]", "[INFRASTRUCTURE", "[NOT TESTABLE", "[NO OUTPUT]")

_TIMEOUT_HINT = (
    "\n\n[PERFORMANCE] Your code exceeded the time limit. "
    "The current implementation is too slow for the given test cases. "
    "Consider: memoization (@functools.lru_cache), "
    "an iterative dynamic programming table, or a more efficient algorithm."
)

# Max characters kept in a single tool result added to history.
_MAX_TOOL_RESULT_CHARS = 4000


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


def _auto_save_session_memory(
    history: List[Dict[str, Any]],
    user_message: str,
    working_dir: str,
) -> None:
    """Fallback: save session memory programmatically if the model forgot to call the tool."""
    files_written = []
    code_executed = False

    for msg in history:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            try:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
            except (KeyError, json.JSONDecodeError):
                continue
            if name == "write_file":
                files_written.append(args.get("path", "?"))
            elif name == "execute_code":
                code_executed = True

    if not files_written and not code_executed:
        return

    task = user_message[:120]
    done_parts = []
    if files_written:
        done_parts.append(f"wrote {', '.join(files_written)}")
    if code_executed and not files_written:
        done_parts.append("tested code in sandbox")
    done = "; ".join(done_parts)

    try:
        from memory import update_session_memory
        update_session_memory(working_dir, task, done, "—")
        print("[context] session memory auto-saved (model skipped update_session_memory)", file=sys.stderr)
    except Exception as exc:
        print(f"[context] session memory auto-save failed: {exc}", file=sys.stderr)


def ensure_system_in_history(history: List[Dict[str, str]], system_prompt: str) -> None:
    """Добавить system prompt в начало истории, если он там не присутствует."""
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": system_prompt})


def _run_with_trace(code: str, container) -> Optional[str]:
    """
    Instrument *code*, run it in *container*, extract and compress the trace.

    Returns a compact trace string, or None if instrumentation failed or
    the sandbox produced no trace markers.

    validate=False is used because the original code already passed AST
    validation; the instrumented version only adds __trace_log__ calls.
    """
    instrumented = instrument(code)
    if instrumented is None:
        return None
    _, instr_output = container.execute(instrumented, validate=False)
    return extract_and_compress_trace(instr_output)


async def run_agent_async(
    client,
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_react_steps: int = MAX_REACT_STEPS,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    working_dir: str = ".",
    container=None,
    on_tool_call=None,
) -> Optional[str]:
    """
    ReACT-агент (Паттерн B): tool_choice="required" + no-op respond_to_user.

    Модель всегда вызывает один из инструментов:
      - read_file / write_file / list_files — рабочий шаг, цикл продолжается
      - respond_to_user                     — финальный ответ, цикл завершается

    Логика агента становится единообразной: всегда читаем tool_calls,
    ветвление идёт только по имени функции.

    Trace-augmented debugging включён всегда: каждый провалившийся execute_code
    дополняется трассой выполнения из AST-инструментированного кода.

    on_tool_call: опциональный async-колбэк(step_num, func_name) для стриминга
    прогресса в UI. Если None — агент работает в CLI-режиме (печатает в stdout).
    Возвращает финальный ответ агента или None при ошибке / превышении шагов.
    """
    if conversation_history is None:
        conversation_history = []

    from memory import project_memory_exists, create_project_memory
    if not project_memory_exists(working_dir):
        create_project_memory(working_dir)

    system_prompt = build_system_prompt(working_dir)
    ensure_system_in_history(conversation_history, system_prompt)
    conversation_history.append({"role": "user", "content": user_message})

    async def _summarizer(text: str) -> str:
        return await client.acomplete(
            [{"role": "user", "content": (
                "Summarize the following agent conversation steps concisely in 3-5 sentences, "
                "preserving key decisions, file paths, and outcomes:\n\n" + text
            )}],
            max_tokens=512,
        )

    session_memory_saved = False
    execute_code_passed = False

    for step in range(max_react_steps):

        # ── Компрессия контекста перед запросом к LLM ────────────────────────
        await compact_history(conversation_history, max_response_tokens=max_tokens, summarizer=_summarizer)

        # ── Таймер ожидания первого токена ──────────────────────────────────
        async def _waiting_ticker(step_num: int, total: int) -> None:
            start = time.monotonic()
            while True:
                elapsed = int(time.monotonic() - start)
                print(
                    f"\r[Step {step_num}/{total}] Waiting for first token... {elapsed}s",
                    end="", flush=True, file=sys.stderr,
                )
                await asyncio.sleep(1)

        # ── Сбор tool_calls от модели (до 2 ретраев при no-tool-call) ──────────
        for _attempt in range(3):
            ticker = asyncio.create_task(_waiting_ticker(step + 1, max_react_steps))
            first_event = True
            tool_calls_received = []

            try:
                async for event in client.astream_with_tools(
                    conversation_history,
                    TOOLS,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    if first_event:
                        ticker.cancel()
                        print(
                            f"\r[Step {step + 1}/{max_react_steps}]" + " " * 40 + "\r",
                            end="", flush=True, file=sys.stderr,
                        )
                        first_event = False

                    if event["type"] == "tool_use":
                        tool_calls_received.append(event["tool_call"])

            except Exception as e:
                ticker.cancel()
                print(f"\n[Agent error while streaming]: {e}", file=sys.stderr)
                return
            finally:
                ticker.cancel()

            if tool_calls_received:
                break

            if _attempt < 2:
                conversation_history.append({
                    "role": "user",
                    "content": (
                        "You must call one of the available tools. "
                        "Plain text responses are not accepted."
                    ),
                })

        if not tool_calls_received:
            print("\n[Agent warning] No tool calls received after retries — stopping.", file=sys.stderr)
            return

        # ── Добавляем ответ ассистента в историю ────────────────────────────
        conversation_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_received,
        })

        # ── Ветвление по имени функции ───────────────────────────────────────
        for tc in tool_calls_received:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                func_args = {}

            # respond_to_user — финальный ответ; выводим и завершаем цикл
            if func_name == "respond_to_user":
                if not session_memory_saved:
                    _auto_save_session_memory(conversation_history, user_message, working_dir)
                message = func_args.get("message", "")
                if on_tool_call is None:
                    print(message)
                return message

            if on_tool_call is not None:
                await on_tool_call(step + 1, func_name)

            # Блокируем write_file для файлов с кодом, если execute_code ещё не прошёл
            if func_name == "write_file" and not execute_code_passed:
                path = func_args.get("path", "")
                ext = os.path.splitext(path)[1].lower()
                if ext in _CODE_EXTENSIONS:
                    result = (
                        "[BLOCKED] write_file was rejected: you MUST call execute_code first "
                        "and receive 'ALL TESTS PASSED' before saving a code file. "
                        f"Call execute_code with your solution for '{path}', then retry write_file."
                    )
                    print(f"[TOOL] {func_name}({func_args}) — BLOCKED (execute_code not yet passed)", file=sys.stderr)
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                    continue

            # Рабочий инструмент — выполняем и возвращаем результат модели
            result = execute_tool(func_name, func_args, working_dir, container=container)

            if func_name == "execute_code" and result.startswith("[OK]"):
                execute_code_passed = True

            if func_name == "execute_code" and result.startswith("[ERROR]"):
                raw_output = result.split("\n", 1)[1] if "\n" in result else ""

                if _is_timeout(raw_output):
                    result += _TIMEOUT_HINT

                # ── Trace-augmented debugging ────────────────────────────────
                # Пропускаем инфраструктурные / sandbox-ошибки: там трасса бесполезна.
                elif container is not None and not any(
                    raw_output.startswith(p) for p in _SKIP_TRACE_PREFIXES
                ):
                    trace = _run_with_trace(func_args.get("code", ""), container)
                    if trace:
                        result += (
                            "\n\n--- EXECUTION TRACE ---\n"
                            + trace
                            + "\n--- END TRACE ---\n"
                            "\nAnalyze the trace above: find the line where the actual "
                            "value diverges from what is expected, then fix the implementation."
                        )
                        print("[TRACE] Execution trace attached to error.", file=sys.stderr)

            if func_name == "update_session_memory":
                session_memory_saved = True
            print(f"[TOOL] {func_name}({func_args})", file=sys.stderr)
            print(f"[RESULT]\n{result}\n", file=sys.stderr)

            if result.startswith("[INFRASTRUCTURE ERROR]"):
                print(f"\n[Agent stopped] Infrastructure error: {result}", file=sys.stderr)
                if on_tool_call is None:
                    print(result)
                return result

            conversation_history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": _trim_result(result),
            })

    print(
        f"[AGENT STOPPED] Reached maximum of {max_react_steps} ReACT steps without final answer.",
        file=sys.stderr,
    )
    return None
