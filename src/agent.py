import json
import sys
import chardet
from pathlib import Path
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = """You are a coding assistant that can read, write, and manage files.

You have access to the following tools:

1) read_file
Description:
Read the contents of a file at the given path.
Input (JSON):
{
  "path": "string"
}

2) write_file
Description:
Write content to a file at the given path.
Creates the file if it does not exist.
Can overwrite or append to existing content.
Input (JSON):
{
  "path": "string",
  "content": "string",
  "mode": "w" | "a"   // optional, default is "w"
}

3) list_files
Description:
List files in the specified directory.
Input (JSON):
{
  "path": "string", // optional, default is "."
}

----------------------------------------

IMPORTANT RULES FOR TOOL USAGE:

• When you want to use a tool, you MUST respond with ONLY a valid JSON object.
• The JSON object MUST have exactly the following structure:

{
  "tool": "<tool_name>",
  "input": {
    ... tool input fields ...
  }
}

• Do NOT include any text before or after the JSON.
• Do NOT wrap the JSON in Markdown.
• Do NOT explain what you are doing.
• Do NOT add comments inside the JSON.

----------------------------------------

RULES FOR NORMAL RESPONSES:

• If no tool is needed, respond with plain text.
• Do NOT return JSON unless you are calling a tool.

----------------------------------------

EXAMPLES:

Correct tool call:
{
  "tool": "read_file",
  "input": {
    "path": "main.py"
  }
}

Correct normal response:
The function should validate input parameters before processing them.

----------------------------------------

You are responsible for deciding when a tool is required.

---------------------------------------
LANGUAGE POLICY (must follow exactly):
- If the user's message is in Russian, reply ONLY in Russian.
- If the user's message is in English, reply ONLY in English.
- Under NO CIRCUMSTANCES reply in Chinese or any other language.
- If you cannot determine whether the user wrote in Russian or English, reply in English.
- If you are calling tools, keep the tool-call JSON in the same language format as the user's request (Russian -> Russian, English -> English).
- If you output any text in a disallowed language, that response should be treated as invalid.
"""

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

def find_tool_calls_from_text(text: str):
    if not text:
        return []
    s = text.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict) and "tool" in obj and "input" in obj:
        return [obj]
    if isinstance(obj, list):
        return [it for it in obj if isinstance(it, dict) and "tool" in it and "input" in it]
    return []

async def run_agent_async(
        client, 
        user_message: str, 
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_react_steps: int = MAX_REACT_STEPS
    ) -> None:
    """
    Запустить агента с заданным user_message с потоковым (по мере генерации) выводом ответа.

    Здесь применяется ReACT (Reason, Act, Observe) цикл:
    1. Модель получает инструкцию проанализировав полученное сообщение (Reason)
    2. Если модель хочет использовать инструмент, она использует его (Act)
    3. Модель анализирует результат применения интсрумента (Observe)
    4. Цикл повторяется, пока модель не сгенерирует окончательный ответ. 
    """

    if conversation_history is None:
        conversation_history = []
    
    # Убедимся, что system prompt в начале истории
    ensure_system_in_history(conversation_history)
    conversation_history.append({"role": "user", "content": user_message})

    for step in range(max_react_steps):
        current_text = ""

        # Потоковый вывод ответа модели
        # Потоковый вывод ответа модели — astream должен быть async generator, yielding str
        try:
            async for chunk in client.astream(conversation_history, max_tokens=1024, temperature=0.2):
                # chunk — кусочек текста из Ollama (поле "response")
                sys.stdout.write(chunk)
                sys.stdout.flush()
                current_text += chunk
        except Exception as e:
            # Логируем и корректно завершаем агент при ошибке стрима
            print(f"\n[Agent error while streaming]: {e}", file=sys.stderr)
            return
        
        print()

        conversation_history.append({"role": "assistant", "content": current_text})
    
        # Поиск вызовов инструментов в сгенерированном ответе
        tool_calls = find_tool_calls_from_text(current_text)
        if not tool_calls:
            return 
        # Выполнение каждого вызова инструмента и добавление результата в историю
        results_for_model = []
        for call in tool_calls:
            tool_name = call["tool"]
            tool_input = call["input"]
            result = execute_tool(tool_name, tool_input)
            results_for_model.append({
                "tool": tool_name,
                "input": tool_input,
                "result": result
            })

            print(f"[TOOL EXECUTED] {tool_name} with input {tool_input} -> result: {result}")

        # Добавление результатов инструментов в историю как сообщение от пользователя (т.е. от внешнего мира)
        conversation_history.append({
            "role": "user",
            "content": json.dumps(results_for_model, ensure_ascii=False, indent=2)
        })

    # Если достигнут максимум шагов ReACT без окончательного ответа
    print(f"[AGENT STOPPED] Reached maximum of {max_react_steps} ReACT steps without final answer.") 