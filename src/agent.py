import json
import sys
import asyncio
import time
from typing import List, Dict, Any, Optional

from system_prompt import SYSTEM_PROMPT
from tools import TOOLS, execute_tool

MAX_REACT_STEPS = 6


def ensure_system_in_history(history: List[Dict[str, Any]]) -> None:
    """Добавить system prompt в начало истории, если он там ещё не присутствует."""
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})


async def run_agent_async(
    client,
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_react_steps: int = MAX_REACT_STEPS,
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

    ensure_system_in_history(conversation_history)
    conversation_history.append({"role": "user", "content": user_message})

    for step in range(max_react_steps):

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
                max_tokens=8192,
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
                message = func_args.get("message", "")
                print(message)
                return

            # Рабочий инструмент — выполняем и возвращаем результат модели
            result = execute_tool(func_name, func_args)
            short_result = result if len(result) <= 120 else result[:120] + "..."
            print(f"[TOOL] {func_name}({func_args}) → {short_result}", file=sys.stderr)

            conversation_history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    print(
        f"[AGENT STOPPED] Reached maximum of {max_react_steps} ReACT steps without final answer.",
        file=sys.stderr,
    )
