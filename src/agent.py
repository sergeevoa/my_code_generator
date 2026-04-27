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

MAX_REACT_STEPS = 8


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


async def run_agent_async(
    client,
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_react_steps: int = MAX_REACT_STEPS,
    max_tokens: int = 4096,
    working_dir: str = ".",
) -> None:
    """
    ReACT-агент (Паттерн B): tool_choice="required" + no-op respond_to_user.

    Модель всегда вызывает один из инструментов:
      - read_file / write_file / list_files — рабочий шаг, цикл продолжается
      - respond_to_user                     — финальный ответ, цикл завершается

    Логика агента становится единообразной: всегда читаем tool_calls,
    ветвление идёт только по имени функции.
    """
    if conversation_history is None:
        conversation_history = []

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
        # max_response_tokens совпадает с max_tokens ниже — одна переменная
        # используется в обоих местах, поэтому они гарантированно синхронны.
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

        ticker = asyncio.create_task(_waiting_ticker(step + 1, max_react_steps))
        first_event = True

        # ── Сбор tool_calls от модели ────────────────────────────────────────
        tool_calls_received: List[Dict[str, Any]] = []

        try:
            async for event in client.astream_with_tools(
                conversation_history,
                TOOLS,
                max_tokens=max_tokens,
                temperature=0.2,
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

        # ── Модель не вернула ни одного tool_call (неожиданно при required) ──
        if not tool_calls_received:
            print("\n[Agent warning] No tool calls received — stopping.", file=sys.stderr)
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
                print(message)
                return

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
            result = execute_tool(func_name, func_args, working_dir)
            if func_name == "execute_code" and result.startswith("[OK]"):
                execute_code_passed = True
            if func_name == "update_session_memory":
                session_memory_saved = True
            print(f"[TOOL] {func_name}({func_args})", file=sys.stderr)
            print(f"[RESULT]\n{result}\n", file=sys.stderr)

            if result.startswith("[INFRASTRUCTURE ERROR]"):
                print(f"\n[Agent stopped] Infrastructure error: {result}", file=sys.stderr)
                print(result)
                return

            conversation_history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    print(
        f"[AGENT STOPPED] Reached maximum of {max_react_steps} ReACT steps without final answer.",
        file=sys.stderr,
    )
