"""
Определения инструментов в формате OpenAI function calling
и функции их выполнения.
"""

import chardet
from pathlib import Path
from typing import Dict, List, Any

from sandbox.executor import execute_python as _sandbox_execute


# ─── Описания инструментов для модели ───────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "respond_to_user",
            "description": (
                "Отправить текстовый ответ пользователю. "
                "Вызывай этот инструмент, когда задача еще не задана, либо когда она выполнена и ты готов сообщить результат. "
                "Не используй его для промежуточных шагов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Текст ответа для пользователя.",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать содержимое файла. Автоматически определяет кодировку.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь к файлу.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Записать содержимое в файл. "
                "Если файл не существует — создаёт его. "
                "mode='w' — перезаписать, mode='a' — добавить в конец."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь к файлу.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Текст для записи.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["w", "a"],
                        "description": "Режим записи: 'w' — перезапись, 'a' — добавление.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Выполнить Python-код в изолированном Docker-sandbox и вернуть stdout + stderr.\n"
                "\n"
                "Используй этот инструмент чтобы:\n"
                "  • протестировать написанное решение перед показом пользователю;\n"
                "  • проверить корректность алгоритма через assert-тесты;\n"
                "  • запустить вычисления или эксперименты.\n"
                "\n"
                "Sandbox БЛОКИРУЕТ: сеть, файловый I/O, subprocess, eval/exec,\n"
                "  модули os, sys, pathlib и другие опасные импорты.\n"
                "Пиши чистый алгоритмический код — без input(), без open().\n"
                "\n"
                "Тест ОБЯЗАН заканчиваться строкой print('ALL TESTS PASSED').\n"
                "Если вывод не содержит этой строки — тест считается проваленным.\n"
                "\n"
                "Лимит выполнения: 10 секунд."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python-код для выполнения. Должен содержать assert-проверки и print('ALL TESTS PASSED').",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_session_memory",
            "description": (
                "Save a brief summary of the current session to persistent memory. "
                "Call this before respond_to_user whenever you executed code or modified files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What the user asked for (1 sentence).",
                    },
                    "done": {
                        "type": "string",
                        "description": "What you accomplished (1 sentence).",
                    },
                    "pending": {
                        "type": "string",
                        "description": "What remains unfinished. Use '—' if nothing.",
                    },
                },
                "required": ["task", "done", "pending"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Вывести список файлов и папок в указанной директории.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь к директории (по умолчанию '.').",
                    }
                },
                "required": [],
            },
        },
    },
]


# ─── Реализация инструментов ─────────────────────────────────────────────────

def read_file(path: str) -> str:
    """Прочитать и вернуть содержимое файла с автоматическим определением кодировки."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        raw_data = file_path.read_bytes()
        detection = chardet.detect(raw_data)
        encoding = detection.get("encoding") or "utf-8"
        return raw_data.decode(encoding, errors="replace")
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str, mode: str = "w") -> str:
    """Записать content в файл. Режим: 'w' — перезапись, 'a' — добавление."""
    try:
        if mode not in ("w", "a"):
            return f"Error: Invalid mode '{mode}'. Use 'w' or 'a'."
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = "utf-8"
        if file_path.exists():
            raw_data = file_path.read_bytes()
            detection = chardet.detect(raw_data)
            encoding = detection.get("encoding") or "utf-8"
        with open(file_path, mode, encoding=encoding, errors="replace") as f:
            f.write(content)
        action = "Appended to" if mode == "a" else "Wrote to"
        return f"Successfully {action} {path} (encoding: {encoding})"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_files(path: str = ".") -> str:
    """Вернуть список файлов и папок в директории."""
    try:
        entries = []
        p = Path(path)
        for entry in sorted(p.iterdir()):
            if entry.is_dir():
                entries.append(f"[DIR]  {entry.name}/")
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


def execute_code(code: str, working_dir: str = ".") -> str:
    """Выполнить Python-код в Docker-sandbox и вернуть форматированный результат."""
    success, output = _sandbox_execute(code, working_dir=working_dir)
    prefix = "[OK]" if success else "[ERROR]"
    return f"{prefix}\n{output}"


def execute_tool(tool_name: str, tool_input: Dict[str, Any], working_dir: str = ".") -> str:
    """Выполнить инструмент по имени и вернуть результат в виде строки."""
    try:
        if tool_name == "execute_code":
            return execute_code(tool_input["code"], working_dir=working_dir)
        elif tool_name == "read_file":
            return read_file(tool_input["path"])
        elif tool_name == "write_file":
            return write_file(
                tool_input["path"],
                tool_input["content"],
                tool_input.get("mode", "w"),
            )
        elif tool_name == "list_files":
            return list_files(tool_input.get("path", "."))
        elif tool_name == "update_session_memory":
            from memory import update_session_memory as _update_session_memory
            return _update_session_memory(
                working_dir,
                tool_input.get("task", ""),
                tool_input.get("done", ""),
                tool_input.get("pending", "—"),
            )
        else:
            return f"Error: Unknown tool: {tool_name}"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {e}"
