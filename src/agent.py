import json
import sys
import asyncio
import time
from typing import List, Dict, Any, Optional


from sandbox.executor import execute_python as _sandbox_execute
from system_prompt import SYSTEM_PROMPT
from tools import TOOLS, execute_tool

MAX_REACT_STEPS = 6


def read_file(path: str) -> str:
    """Прочитать и вернуть содержимое файла с автоматическим определением кодировки."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"

        # Читаем файл в бинарном режиме для детекции кодировки
        raw_data = file_path.read_bytes()
        detection = chardet.detect(raw_data)
        encoding = detection.get("encoding")

        # Декодируем содержимое файла в найденной кодировке
        return raw_data.decode(encoding or "utf-8", errors="replace")

    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str, mode: str = "w") -> str:
    """
    Записать content в файл с учётом кодировки.
    Режимы:
        'w' - перезаписать файл (по умолчанию)
        'a' - добавить в конец файла
    Если файл уже существует, сохраняем его текущую кодировку.
    """
    try:
        if mode not in ("w", "a"):
            return f"Error: Invalid mode '{mode}'. Use 'w' or 'a'."

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Определяем кодировку, если файл существует
        encoding = "utf-8"
        if file_path.exists():
            raw_data = file_path.read_bytes()
            detection = chardet.detect(raw_data)
            encoding = detection.get("encoding", "utf-8")

        with open(file_path, mode, encoding=encoding, errors="replace") as f:
            f.write(content)

        action = "Appended to" if mode == "a" else "Wrote to"
        return f"Successfully {action} {path} (encoding: {encoding})"

    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"

# Точка означает текущую рабочую директорию
def list_files(path: str = ".") -> str:
    """Возвращает список файлов и папок, находящихся по указанному пути."""
    try:
        entries = []
        p = Path(path)
        for entry in sorted(p.iterdir()):
            if entry.is_dir():
                entries.append(f"[DIR] {entry.name}/")
            else:
                entries.append(f"[FILE] {entry.name}")
        if not entries:
           return f"Directory is empty: {path}"
        return "\n".join(entries)
    except FileNotFoundError:
        return f"Error: Directory not found: {path}"
    except NotADirectoryError:
        return f"Error: Not a directory: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"
    
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Воспользоваться инструментом и вернуть результат."""
    try:
        if tool_name == "read_file":
            return read_file(tool_input["path"])
        elif tool_name == "write_file":
            return write_file(tool_input["path"], tool_input["content"], tool_input.get("mode", "w"))
        elif tool_name == "list_files":
            return list_files(tool_input.get("path", "."))
        else:
            return f"Error: Unknown tool: {tool_name}"
    except Exception as e:
        return f"Error executing tool {tool_name} : {e}"

def ensure_system_in_history(history: List[Dict[str, str]]) -> None:
    """Добавить system prompt в начало истории, если он там не присутствует."""
    if not history:
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return
    first = history[0]
    if first.get("role") != "system" or first.get("content") != SYSTEM_PROMPT:
        # если в истории уже есть system, не вставляем второй раз;
        # но если первый элемент не system — добавим system в начало
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
