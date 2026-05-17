import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional, AsyncGenerator

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))

from clients.llama_client import LlamaClient
from agent import run_agent_async, MAX_REACT_STEPS
from sandbox.executor import SandboxContainer

ROOT = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"
load_dotenv(ROOT / ".env")

BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8080/v1")
MODEL_FALLBACK = os.getenv("LLAMA_MODEL", "unknown")
DEFAULT_MAX_STEPS = int(os.getenv("MAX_REACT_STEPS", str(MAX_REACT_STEPS)))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = 0.2
DEFAULT_WORKING_DIR = str(ROOT)


def _server_base(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


async def _fetch_model_name() -> str:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{_server_base(BASE_URL)}/v1/models")
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    return data[0].get("id", MODEL_FALLBACK)
    except Exception:
        pass
    return MODEL_FALLBACK


async def _fetch_n_ctx_slot() -> Optional[int]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{_server_base(BASE_URL)}/slots")
            if r.status_code == 200:
                slots = r.json()
                if isinstance(slots, list) and slots:
                    return slots[0].get("n_ctx")
    except Exception:
        pass
    return None


# ── Global app state (single-user local app) ──────────────────────────────────

class _AppState:
    def __init__(self):
        self.client = LlamaClient(model=MODEL_FALLBACK, base_url=BASE_URL)
        self.container: Optional[SandboxContainer] = None
        self.working_dir: str = ""
        self.model_name: str = MODEL_FALLBACK
        self.n_ctx_slot: Optional[int] = None

    def ensure_container(self, working_dir: str) -> Optional[str]:
        if self.container is not None and self.working_dir == working_dir:
            return None
        if self.container is not None:
            try:
                self.container.stop()
            except Exception:
                pass
            self.container = None
        self.container = SandboxContainer()
        try:
            self.container.start(working_dir)
            self.working_dir = working_dir
            return None
        except RuntimeError as e:
            self.container = None
            return str(e)


_state = _AppState()

# ── FastAPI app ────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    model_name, n_ctx_slot = await asyncio.gather(_fetch_model_name(), _fetch_n_ctx_slot())
    _state.model_name = model_name
    _state.client = LlamaClient(model=model_name, base_url=BASE_URL)
    _state.n_ctx_slot = n_ctx_slot
    if n_ctx_slot:
        print(f"[server] n_ctx_slot = {n_ctx_slot}", file=sys.stderr)
    else:
        print("[server] n_ctx_slot: не удалось получить", file=sys.stderr)
    yield


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/info")
async def info():
    token_max = max((_state.n_ctx_slot - 1) if _state.n_ctx_slot else 16384, DEFAULT_MAX_TOKENS)
    token_val = min(DEFAULT_MAX_TOKENS, token_max)
    return JSONResponse({
        "model": _state.model_name,
        "n_ctx_slot": _state.n_ctx_slot,
        "default_max_steps": DEFAULT_MAX_STEPS,
        "default_max_tokens": token_val,
        "token_max": token_max,
        "default_temperature": DEFAULT_TEMPERATURE,
        "default_working_dir": DEFAULT_WORKING_DIR,
    })


@app.get("/browse")
async def browse():
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, _browse_directory)
    if path:
        return JSONResponse({"path": path})
    return JSONResponse({"path": None})


def _browse_directory() -> Optional[str]:
    result: list = []
    done = threading.Event()

    def _run() -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root_tk = tk.Tk()
            root_tk.withdraw()
            root_tk.wm_attributes("-topmost", 1)
            folder = filedialog.askdirectory(
                parent=root_tk, title="Выберите директорию проекта"
            )
            root_tk.destroy()
            if folder:
                result.append(folder)
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()
    done.wait(timeout=120)
    return result[0] if result else None


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message: str = body.get("message", "").strip()
    agent_hist: list = body.get("agent_history", [])
    working_dir: str = body.get("working_dir", DEFAULT_WORKING_DIR)
    max_steps: int = int(body.get("max_steps", DEFAULT_MAX_STEPS))
    max_tokens: int = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))
    temperature: float = float(body.get("temperature", DEFAULT_TEMPERATURE))

    return StreamingResponse(
        _stream_agent(request, message, agent_hist, working_dir, max_steps, max_tokens, temperature),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_agent(
    request: Request,
    message: str,
    agent_hist: list,
    working_dir: str,
    max_steps: int,
    max_tokens: int,
    temperature: float,
) -> AsyncGenerator[str, None]:

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    if not message:
        yield _sse("error", {"text": "Пустое сообщение"})
        return

    if not Path(working_dir).is_dir():
        yield _sse("error", {"text": f"Директория не существует: {working_dir}"})
        return

    error = _state.ensure_container(working_dir)
    if error:
        yield _sse("error", {"text": f"Не удалось запустить sandbox: {error}"})
        return

    queue: asyncio.Queue = asyncio.Queue()

    async def on_step(step_num: int, func_name: str, func_args: dict) -> None:
        await queue.put(("call", step_num, func_name, func_args))

    async def on_result(step_num: int, func_name: str, func_args: dict, result: str) -> None:
        await queue.put(("result", step_num, func_name, func_args, result))

    task = asyncio.create_task(
        run_agent_async(
            client=_state.client,
            user_message=message,
            conversation_history=agent_hist,
            max_react_steps=max_steps,
            max_tokens=max_tokens,
            temperature=temperature,
            working_dir=working_dir,
            container=_state.container,
            on_tool_call=on_step,
            on_tool_result=on_result,
        )
    )

    yield _sse("start", {})

    def _emit_item(item: tuple) -> str:
        if item[0] == "call":
            _, step_num, func_name, func_args = item
            return _sse("tool_call", {"step": step_num, "tool": func_name, "args": func_args})
        else:
            _, step_num, func_name, func_args, res = item
            return _sse("tool_result", {"step": step_num, "tool": func_name, "args": func_args, "result": res})

    try:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                return
            try:
                item = await asyncio.wait_for(asyncio.shield(queue.get()), timeout=0.4)
                yield _emit_item(item)
            except asyncio.TimeoutError:
                pass
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return

    # drain remaining events
    while not queue.empty():
        try:
            yield _emit_item(queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    try:
        result = task.result()
    except Exception as e:
        result = f"❌ Ошибка агента: {e}"

    final = result or "_(Нет ответа)_"
    yield _sse("done", {"text": final, "history": agent_hist})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        app_dir=str(Path(__file__).parent),
    )
