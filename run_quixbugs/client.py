"""
SeededTrackingClient — LlamaClient extended with:
  - per-call token usage tracking (_last_prompt_tokens / _last_completion_tokens)
  - optional seed parameter forwarded to the LLM API for reproducible sampling

The seed parameter is supported by llama.cpp server since v1.8 and by most
OpenAI-compatible endpoints.  If the server silently ignores it, the benchmark
still runs correctly — individual runs simply have independent randomness.
"""

from typing import Any, Dict, List, Optional, cast

from clients.llama_client import LlamaClient  # available via sys.path set in __init__


class SeededTrackingClient(LlamaClient):
    """LlamaClient with token-usage tracking and per-call seed support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0

    async def astream_with_tools(  # type: ignore[override]
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.2,
        seed: Optional[int] = None,
    ):
        """
        Stream tool calls, capturing token usage from the final chunk.

        seed — passed to the inference server to make sampling deterministic.
               Different seeds produce meaningfully different outputs because
               they shift the multinomial distribution at every sampling step.

        Yields the same event dicts as the parent class:
            {"type": "text",     "text": "..."}
            {"type": "tool_use", "tool_call": {...}}
        """
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0

        extra_kwargs: Dict[str, Any] = {}
        if seed is not None:
            extra_kwargs["seed"] = seed

        stream = None

        # Try with stream_options to get per-call token usage
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
                **extra_kwargs,
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
                **extra_kwargs,
            )

        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}

        async for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                self._last_prompt_tokens    = chunk.usage.prompt_tokens    or 0
                self._last_completion_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue

            choice       = chunk.choices[0]
            delta        = choice.delta
            finish_reason = choice.finish_reason

            if delta.content:
                yield {"type": "text", "text": delta.content}

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id":       "",
                            "type":     "function",
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
