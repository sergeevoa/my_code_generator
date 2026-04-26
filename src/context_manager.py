"""
Context window budget management for local LLMs.

Проблема: локальная модель (llama.cpp/vLLM) имеет фиксированное контекстное
окно — например, 8k или 16k токенов. Без компрессии длинные выводы
execute_code и многооборотный диалог быстро заполняют его, и сервер либо
отклоняет запрос, либо молча обрезает историю с непредсказуемым результатом.

compact_history() вызывается перед каждым запросом к LLM и модифицирует
список истории in-place в два прохода:

  Проход 1 — обрезка tool results.
    Длинный вывод execute_code (трассировки, большие массивы данных)
    сжимается: сохраняется начало и конец, середина заменяется маркером.
    Модель видит, что именно запускалось и чем закончилось; теряется только
    многословная середина. Tool results — текст, обрезка посередине безопасна.

    Аргументы tool_calls (поле "code" в assistant-сообщениях) НЕ обрезаются:
    это исполняемый Python, текстовый маркер посередине ломает синтаксис
    и блокируется валидатором сэндбокса.

  Проход 2 — скользящее окно.
    Если после прохода 1 бюджет всё ещё превышен, удаляется старейший
    полный «ход» диалога: от самого раннего user-сообщения (не считая
    system) до следующего user-сообщения (не включительно). Повторяется,
    пока история не войдёт в бюджет или удалять больше нечего.
    System prompt (индекс 0) никогда не удаляется.

Бюджет вычисляется автоматически:
    VLLM_MAX_MODEL_LEN (из .env) − max_response_tokens − _SAFETY_MARGIN
где max_response_tokens берётся из аргумента compact_history() — того же
значения, что передаётся в astream_with_tools(max_tokens=...).
"""

import json
import os
import sys
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# ── Настройки ─────────────────────────────────────────────────────────────────

# Грубая оценка: 1 токен ≈ 4 символа (работает для кода, русского и английского).
_CHARS_PER_TOKEN = 4

# Полный контекст модели в токенах. Должен совпадать с параметром запуска
# llama-server (--ctx-size) или vLLM (max_model_len).
_MODEL_CONTEXT_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))

# Запас между вычисленным бюджетом и реальным лимитом: компенсирует погрешность
# оценки токенов (~4 chars/token) и служебные токены самого запроса.
_SAFETY_MARGIN: int = 100

# Минимально допустимый бюджет для истории.
# При значениях ниже system prompt (≈1 250 токенов) + хотя бы одно сообщение
# уже не помещаются, и агент деградирует непредсказуемо.
# Особо важен для поимки отрицательного бюджета (VLLM_MAX_MODEL_LEN < max_tokens).
_MIN_BUDGET_TOKENS: int = 1000

# Максимум символов, сохраняемых от одного tool result (голова + хвост).
_TOOL_RESULT_MAX_CHARS: int = 2000


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов по длине строки."""
    return len(text) // _CHARS_PER_TOKEN


def _count_history_tokens(history: List[Dict[str, Any]]) -> int:
    """Суммирует оценки токенов по всем сообщениям истории."""
    total = 0
    for msg in history:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += _estimate_tokens(content)
        elif isinstance(content, list):
            # Anthropic-style content blocks (список словарей)
            for block in content:
                total += _estimate_tokens(str(block))
        # tool_calls находятся в assistant-сообщениях (OpenAI-формат)
        for tc in msg.get("tool_calls", []):
            total += _estimate_tokens(json.dumps(tc))
    return total


def _truncate_tool_result(text: str, max_chars: int = _TOOL_RESULT_MAX_CHARS) -> str:
    """
    Обрезает длинный tool result до max_chars символов.
    Сохраняет первую и последнюю половину, вставляя маркер пропуска.
    """
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return (
        text[:head]
        + f"\n...[{omitted} chars omitted]...\n"
        + text[-tail:]
    )


def _build_summary_input(messages: List[Dict[str, Any]]) -> str:
    """Форматирует список сообщений в читаемый текст для передачи в LLM-суммаризатор."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls", [])

        if role == "assistant" and tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                args_preview = fn.get("arguments", "")[:200]
                parts.append(f"[Assistant] called {fn.get('name', '?')}({args_preview})")
        elif role == "tool":
            preview = content[:300] if isinstance(content, str) else str(content)[:300]
            parts.append(f"[Tool result]: {preview}")
        elif isinstance(content, str) and content:
            parts.append(f"[{role.capitalize()}]: {content[:300]}")
    return "\n".join(parts)


def _oldest_droppable_turn_slice(history: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Возвращает (start, end) — срез старейшего удаляемого хода.

    «Ход» — всё от первого не-system user-сообщения до следующего
    user-сообщения (не включительно). Например, для истории:
        [0] system
        [1] user   ← start
        [2] assistant + tool_calls
        [3] tool result
        [4] user   ← end  (следующий ход, не удаляем)
    вернёт (1, 4) — del history[1:4] уберёт ровно первый ход.

    Возвращает (-1, -1), если удалять нечего (только одно user-сообщение).
    """
    # Собираем индексы всех user-сообщений, пропуская system в [0]
    user_indices = [
        i for i, msg in enumerate(history)
        if msg.get("role") == "user" and i > 0
    ]
    # Нужно минимум два user-сообщения, чтобы безопасно удалить первый ход
    if len(user_indices) < 2:
        return (-1, -1)
    return (user_indices[0], user_indices[1])


# ── Публичный API ─────────────────────────────────────────────────────────────

async def compact_history(
    history: List[Dict[str, Any]],
    max_response_tokens: int = 8192,
    summarizer: Optional[Callable[[str], Awaitable[str]]] = None,
) -> None:
    """
    Сжимает историю диалога in-place, чтобы она уложилась в бюджет.
    Вызывается перед каждым запросом к LLM в цикле агента.

    Бюджет = VLLM_MAX_MODEL_LEN − max_response_tokens − _SAFETY_MARGIN.
    max_response_tokens должен совпадать со значением max_tokens,
    которое передаётся в astream_with_tools() для этого же вызова.

    Args:
        history:             список сообщений в OpenAI-формате (изменяется на месте).
        max_response_tokens: зарезервированный лимит токенов для ответа модели.
        summarizer:          опциональный async-коллбэк (text) -> summary. Если передан,
                             используется в третьем проходе вместо удаления старых ходов.
    """
    budget_tokens = _MODEL_CONTEXT_LEN - max_response_tokens - _SAFETY_MARGIN

    if budget_tokens < _MIN_BUDGET_TOKENS:
        raise ValueError(
            f"Context budget too small: {budget_tokens} tokens "
            f"(VLLM_MAX_MODEL_LEN={_MODEL_CONTEXT_LEN} − max_tokens={max_response_tokens} − margin={_SAFETY_MARGIN}).\n"
            f"Minimum required: {_MIN_BUDGET_TOKENS} tokens.\n"
            f"Fix: increase VLLM_MAX_MODEL_LEN in .env, or decrease max_tokens in agent code."
        )

    tokens_before = _count_history_tokens(history)

    # ── Проход 1: обрезка длинных tool results ────────────────────────────────
    # execute_code может вернуть многотысячный вывод; хвост с ошибкой нам важен,
    # а многословная середина — нет. Tool results — plain text, обрезка безопасна.
    for msg in history:
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            msg["content"] = _truncate_tool_result(msg["content"])

    # ── Проход 2: скользящее окно — удаление старых ходов ────────────────────
    # Продолжаем, пока превышаем бюджет и есть что удалять.
    turns_dropped = 0
    while _count_history_tokens(history) > budget_tokens:
        start, end = _oldest_droppable_turn_slice(history)
        if start == -1:
            # Больше нечего удалять — история несжимаема дальше
            break
        del history[start:end]
        turns_dropped += 1

    # ── Проход 3: LLM-суммаризация старых ходов ──────────────────────────────
    # Запускается только если: бюджет всё ещё превышен, summarizer передан,
    # и в истории есть хотя бы два user-сообщения (есть что суммаризовать
    # при сохранении последнего хода).
    summarized = False
    if _count_history_tokens(history) > budget_tokens and summarizer is not None:
        user_indices = [
            i for i, m in enumerate(history)
            if m.get("role") == "user" and i > 0
        ]
        if len(user_indices) >= 2:
            last_user_idx = user_indices[-1]
            turns_to_summarize = history[1:last_user_idx]
            if turns_to_summarize:
                summary_input = _build_summary_input(turns_to_summarize)
                try:
                    summary = await summarizer(summary_input)
                    history[1:last_user_idx] = [{
                        "role": "system",
                        "content": f"[Summary of previous steps]\n{summary}",
                    }]
                    summarized = True
                except Exception as exc:
                    print(f"[context] summarizer failed: {exc}", file=sys.stderr)

    tokens_after = _count_history_tokens(history)
    if tokens_after < tokens_before:
        dropped_info = f", turns dropped: {turns_dropped}" if turns_dropped else ""
        summarized_info = ", summarized" if summarized else ""
        print(
            f"[context] compacted {tokens_before} → {tokens_after} tokens"
            f" (budget: {budget_tokens}{dropped_info}{summarized_info})",
            file=sys.stderr,
        )
