import json
import httpx
from typing import Any, AsyncGenerator, Optional, Dict, List

from utils.stream_util import StreamContext

class OllamaClient:
    """
    Асинхронный клиент для взаимодействия с Ollama LLM сервером.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434", default_timeout: Optional[float] = 60.0):
        self.model = model
        self.host = host.rstrip("/")
        self.api_url = f"{self.host}/api/generate"
        self.default_timeout = default_timeout

    def _create_prompt(self, messages: list[dict]) -> str:
        """
        Преобразует историю сообщений в prompt, понятный Ollama.
        """
        parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        parts.append("Assistant:")
        return "\n".join(parts)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 200,
        temperature: float = 0.2,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Асинхронно отправляет POST /api/generate и возвращает результат.
        Обычно возвращает строку (поле 'response' / 'text' / 'content'), иначе - весь JSON.
        """

        prompt = self._create_prompt(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        timeout = self.default_timeout if timeout is None else timeout

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(self.api_url, json=payload)
                resp.raise_for_status()
            except httpx.RequestError as e:
                raise ConnectionError(f"Error connecting to Ollama: {e}") from e
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = e.response.text
                except Exception:
                    pass
                raise RuntimeError(f"Ollama returned HTTP {e.response.status_code}: {body}") from e

            data = resp.json()

            if isinstance(data, dict):
                return (data.get("response") or "").strip()
            
            raise ValueError(f"Unexpected response format from Ollama")

    # Метод возвращающий StreamContext для асинхронного потокового получения ответа
    async def astream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 200,
        temperature: float = 0.2,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Асинхронный поток генерации.
        Возвращает куски текста по мере генерации.
        """
        prompt = self._create_prompt(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", self.api_url, json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = chunk.get("response")
                    if text:
                        yield text