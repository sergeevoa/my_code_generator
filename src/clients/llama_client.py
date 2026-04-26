import json
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from typing import Any, AsyncGenerator, Dict, List, Optional, cast


class LlamaClient:
    """
    Асинхронный клиент для взаимодействия с llama-server через OpenAI-совместимый API.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080",
        api_key: str = "not-needed",
        default_timeout: Optional[float] = 300.0,
    ):
        self.model = model
        self.default_timeout = default_timeout
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=default_timeout,
        )

    async def acomplete(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Нестриминговый chat-completion для внутренних задач (суммаризация контекста)."""
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def astream_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Стриминг chat-completion с поддержкой native tool calls.

        Генерирует события двух типов:
          {"type": "text",     "text": "..."}          — текстовый фрагмент для пользователя
          {"type": "tool_use", "tool_call": {...}}      — полный вызов инструмента
        """
        stream: AsyncStream[ChatCompletionChunk] = await self._client.chat.completions.create(  # type: ignore[call-overload]
            model=self.model,
            messages=cast(Any, messages),
            tools=cast(Any, tools),
            tool_choice="required",
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        # Буфер для накопления tool_calls по индексу (приходят по кускам)
        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # --- Текстовый контент ---
            if delta.content:
                yield {"type": "text", "text": delta.content}

            # --- Дельта tool_calls ---
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    buf = tool_calls_buffer[idx]
                    if tc_delta.id:
                        buf["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        buf["function"]["name"] += tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        buf["function"]["arguments"] += tc_delta.function.arguments

            # --- Конец генерации с tool_calls ---
            # llama-server может вернуть "stop" вместо "tool_calls" —
            # проверяем буфер в обоих случаях
            if finish_reason in ("tool_calls", "stop") and tool_calls_buffer:
                for idx in sorted(tool_calls_buffer.keys()):
                    yield {"type": "tool_use", "tool_call": tool_calls_buffer[idx]}
                tool_calls_buffer.clear()
