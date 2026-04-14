import json
import re
import ast
import sys
import asyncio
import time
import chardet
from pathlib import Path
from typing import List, Dict, Any, Optional

from sandbox.executor import execute_python as _sandbox_execute
from system_prompt import SYSTEM_PROMPT

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
        elif tool_name == "execute_python":
            success, output = _sandbox_execute(tool_input["code"])
            status = "OK" if success else "ERROR"
            return f"[{status}]\n{output}"
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

# Находит все сегменты {...} с балансом фигурных скобок (включая вложенные)
def extract_brace_objects(text: str) -> List[str]:
    objs = []
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i+1])
                    start = None
    return objs

# Заменить тройные кавычки ("""...""" или '''...''') на валидную JSON-строку
TRIPLE_RE = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', flags=re.S)

def replace_triple_quotes_with_json_string(s: str) -> str:
    def repl(m):
        inner = m.group(0)[3:-3]
        return json.dumps(inner)
    return TRIPLE_RE.sub(repl, s)

# Заменить бэктик-строки (`...`) на валидную JSON-строку.
# Некоторые модели используют JS-стиль template literals вместо JSON-кавычек.
BACKTICK_RE = re.compile(r'`(.*?)`', flags=re.S)

def replace_backticks_with_json_string(s: str) -> str:
    def repl(m):
        return json.dumps(m.group(1))
    return BACKTICK_RE.sub(repl, s)

def try_parse_candidate(s: str) -> Any:
    # 1) чистый json
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) бэктик-строки (`...`) → JSON-строки, затем парсим
    if '`' in s:
        try:
            sanitized = replace_backticks_with_json_string(s)
            return json.loads(sanitized)
        except Exception:
            pass

    # 3) тройные кавычки → JSON-строки, затем парсим
    if '"""' in s or "'''" in s:
        try:
            sanitized = replace_triple_quotes_with_json_string(s)
            return json.loads(sanitized)
        except Exception:
            pass

    # 3) пробуем ast.literal_eval (для Python-литералов, одинарных кавычек и т.д.)
    try:
        obj = ast.literal_eval(s)
        return obj
    except Exception:
        pass

    # ничего не помогло
    return None

def find_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    # 1) сначала — поддержка явных тегов, если вы их используете
    calls = []
    tag_re = re.compile(r'<TOOL_CALL>(.*?)</TOOL_CALL>', flags=re.S)
    for m in tag_re.findall(text):
        # m — содержимое тега (может быть JSON/Python literal)
        cand = m.strip()
        parsed = try_parse_candidate(cand)
        if isinstance(parsed, dict) and "tool" in parsed and "input" in parsed:
            calls.append(parsed)
        elif isinstance(parsed, list):
            for it in parsed:
                if isinstance(it, dict) and "tool" in it and "input" in it:
                    calls.append(it)
    if calls:
        return calls

    # 2) если тегов нет — извлекаем все сбалансированные {...} блоки и пытаемся парсить их
    candidates = extract_brace_objects(text)
    for cand in candidates:
        parsed = try_parse_candidate(cand)
        if isinstance(parsed, dict) and "tool" in parsed and "input" in parsed:
            calls.append(parsed)
        elif isinstance(parsed, list):
            for it in parsed:
                if isinstance(it, dict) and "tool" in it and "input" in it:
                    calls.append(it)

    return calls

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

        async def _waiting_ticker(step_num: int, total: int) -> None:
            """Печатает счётчик секунд пока не придёт первый токен от модели."""
            start = time.monotonic()
            while True:
                elapsed = int(time.monotonic() - start)
                print(
                    f"\r[Step {step_num}/{total}] Waiting for first token... {elapsed}s",
                    end="", flush=True, file=sys.stderr,
                )
                await asyncio.sleep(1)

        ticker = asyncio.create_task(_waiting_ticker(step + 1, max_react_steps))
        first_chunk = True

        try:
            async for chunk in client.astream(conversation_history, max_tokens=8192, temperature=0.2):
                if first_chunk:
                    # Первый токен пришёл — убираем счётчик, дальше модель сама пишет в stdout
                    ticker.cancel()
                    print(f"\r[Step {step + 1}/{max_react_steps}] " + " " * 40 + "\r",
                          end="", flush=True, file=sys.stderr)
                    first_chunk = False
                sys.stdout.write(chunk)
                sys.stdout.flush()
                current_text += chunk
        except Exception as e:
            ticker.cancel()
            print(f"\n[Agent error while streaming]: {e}", file=sys.stderr)
            return
        finally:
            ticker.cancel()
        
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