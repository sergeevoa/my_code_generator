"""
TrackingLlamaClient — extends the project's LlamaClient to capture
per-call token usage from the streaming response.

Usage counts accumulate in `_last_prompt_tokens` / `_last_completion_tokens`
and are reset at the start of every `astream_with_tools` call, so the caller
can read them immediately after the generator is exhausted.
"""

from typing import Any, Dict, List, cast

from clients.llama_client import LlamaClient  # available via sys.path set in __init__


class TrackingLlamaClient(LlamaClient):
    """Adds token-usage tracking on top of the base streaming client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0

    async def astream_with_tools(  # type: ignore[override]
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ):
        """
        Streams tool calls, capturing token usage from the final chunk.

        Yields the same event dicts as the parent class:
            {"type": "text",     "text": "..."}
            {"type": "tool_use", "tool_call": {...}}
        """
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0

        # Try with stream_options to get usage in the final chunk.
        # Falls back silently if the server doesn't support the option.
        stream = None
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                tools=cast(Any, tools),
                tool_choice="required",
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception:
            pass

        if stream is None:
            stream = await self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=cast(Any, messages),
                tools=cast(Any, tools),
                tool_choice="required",
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}

        async for chunk in stream:
            # Capture usage when present (last data chunk with stream_options)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                self._last_prompt_tokens = chunk.usage.prompt_tokens or 0
                self._last_completion_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            if delta.content:
                yield {"type": "text", "text": delta.content}

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

            if finish_reason in ("tool_calls", "stop") and tool_calls_buffer:
                for idx in sorted(tool_calls_buffer.keys()):
                    yield {"type": "tool_use", "tool_call": tool_calls_buffer[idx]}
                tool_calls_buffer.clear()
