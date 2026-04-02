import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

import httpx

# LLM клиент, подключенный к приложению
class LLMClient(Protocol):
    api_url: str

# Событие потока
@dataclass
class StreamEvent:
    type: str           # тип события, например "content_block_start", "content_block_delta", "error"
    raw: Dict[str, Any] # необработанный JSON-чанк от Ollama

    content_block: Optional[Dict[str, Any]] = None  # начало или конец блока контента
    delta: Optional[Dict[str, Any]] = None          # Часть данных, пришедшая в текущий момент времени
    error: Optional[str] = None                     # Текст ошибки, если в стриме что-то пошло не так

# Асинхронный контекст-менеджер и итератор событий
class StreamContext:
    def __init__(self, client: LLMClient, payload: Dict, timeout: Optional[float] = None):
        self._client = client
        self._payload = payload # Готовый JSON для отправки в Ollama
        self._timeout = timeout # Максимальное время ожидания для каждого чанка
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._final_message: Optional[Any] = None
        self._accumulated_text: str = ""            # Аккумулятор для ответов от Ollama
        self._closed = False    # Флаг завершения task
        self._exception: Optional[Exception] = None

    async def __aenter__(self) -> "StreamContext":
        # Запускает фоновую задачу, которая читает поток и кладёт события в очередь
        self._task = asyncio.create_task(self._reader_task())
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        # Завершает фоновую задачу и освобождает ресурсы
        self._closed = True
        if self._task:
            try:
                await self._task
            except Exception as e:
                self._exception = e
        await self._queue.put(None)  # Сигнал для итератора о завершении
    
    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        # Итератор по элементам очереди
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
    
    async def get_final_message(self):
        # Дождаться завершения _reader_task если она еще идёт
        if self._task and not self._task.done():
            await self._task
        if self._exception:
            raise self._exception
        return self._final_message
    
    async def _reader_task(self):
        # Фоновая задача, которая читает HTTP стрим и кладёт события StreamEvent в очередь
        url = self._client.api_url
        timeout = self._timeout

        async with httpx.AsyncClient(timeout=timeout) as http:
            try:
                async with http.stream("POST", url, json=self._payload) as resp:
                    resp.raise_for_status()
                    # Буфер для сбора content_blocks
                    # В зависимости от формата финальное сообщение будет собрано из last_obj
                    last_obj = None
                    async for raw_line in resp.aiter_lines():
                        if raw_line is None:
                            continue
                        line = raw_line.strip()
                        if not line:
                            continue
                        # У Ollama иногда иногда встречается префикс "data:"
                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()
                        # Попытка распарсить JSON
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            # если не JSON - положим как полученную строку сырой
                            obj = {"raw": line}

                        last_obj = obj
                        # Преобразуем объект в 0..N событий
                        events = self._parse_ollama_chunks(obj)
                        for e in events:
                            await self._queue.put(e)
                        
                    # Пробуем извлечь финальное сообщение после окончаения потока
                    self._final_message = self._ollama_extract_final_message(last_obj)
            except Exception as e:
                # Отправим событие ошибки и сохраним исключение
                err_event = StreamEvent(type="error", raw={}, error=str(e))
                await self._queue.put(err_event)
                self._exception = e
    
    def _parse_ollama_chunks(self, obj: Dict[str, Any]) -> List[StreamEvent]:
        # Преобразует один Ollama-чанк в 0..N событий StreamEvent
        events: List[StreamEvent] = []

        if not isinstance(obj, dict):
            return events
        
        text_piece = obj.get("response")
        if text_piece:
            # Создаём событие с кусочком текста
            evt = StreamEvent(
                type = "content_block_delta",
                raw = obj,
                delta = {"type": "text_delta", "text": text_piece}
            )
            events.append(evt)

            # Аккумулируем текст для финального сообщения
            if hasattr(self, "_accumulated_text"):
                self._accumulated_text += text_piece
        
        return events
    
    def _ollama_extract_final_message(self, last_obj: Optional[Dict[str, Any]]) -> str:
        # Извлекает финальное сообщение из последнего объекта Ollama.
        # Использует аккумулированный текст self._accumulated_text.
        if hasattr(self, "_accumulated_text") and self._accumulated_text:
            return self._accumulated_text
        
        if last_obj and isinstance(last_obj, dict):
            return last_obj.get("response", "")
        
        return ""

            