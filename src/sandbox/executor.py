"""
Слой 2 песочницы: изолированное выполнение кода в отдельном subprocess.

Защитные меры:
  - Код пишется во временный файл и запускается как отдельный процесс.
  - Окружение (env) сведено к минимуму: только PATH + системные переменные Windows.
  - Жёсткий таймаут: процесс убивается через TIMEOUT_SECONDS секунд.
  - Вывод обрезается до MAX_OUTPUT_CHARS символов.
  - Временный файл удаляется всегда, даже при ошибке.

Перед запуском код проходит статический анализ (validator.py).
"""
import os
import sys
import subprocess
import tempfile
from typing import Tuple

from sandbox.validator import validate_code

# ── Настройки ──────────────────────────────────────────────────────────────
TIMEOUT_SECONDS: int   = 10       # максимальное время выполнения (секунд)
MAX_OUTPUT_CHARS: int  = 10_000   # максимальный размер возвращаемого вывода


def execute_python(code: str) -> Tuple[bool, str]:
    """
    Безопасно выполняет Python-код в изолированном subprocess.

    Двухуровневая защита:
      1. AST-валидация (validator.py) — до запуска процесса
      2. Subprocess-изоляция — stripped env, hard timeout

    Args:
        code: строка с Python-кодом

    Returns:
        (success: bool, output: str)
        success=True  — код завершился с кодом 0
        success=False — нарушение безопасности, таймаут или ошибка выполнения
    """

    # ── Слой 1: AST-валидация ───────────────────────────────────────────────
    is_safe, violations = validate_code(code)
    if not is_safe:
        lines = "\n".join(f"  • {v}" for v in violations)
        return False, f"[SANDBOX] Код заблокирован. Обнаружены нарушения:\n{lines}"

    # ── Слой 2: Subprocess-изоляция ─────────────────────────────────────────
    tmp_path: str | None = None
    try:
        # Записываем код во временный .py файл
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        # Минимальное окружение: только то, что нужно Python для запуска на Windows
        safe_env = {
            "PATH":       os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),  # Windows CRT
            "TEMP":       tempfile.gettempdir(),
            "TMP":        tempfile.gettempdir(),
        }

        proc = subprocess.run(
            [sys.executable, tmp_path],   # используем тот же Python, что и агент
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",             # не падаем на нестандартных символах
            timeout=TIMEOUT_SECONDS,
            env=safe_env,
        )

        # Объединяем stdout + stderr
        output = proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + proc.stderr

        # Обрезаем слишком длинный вывод
        if len(output) > MAX_OUTPUT_CHARS:
            output = (
                output[:MAX_OUTPUT_CHARS]
                + f"\n... [вывод обрезан: показано {MAX_OUTPUT_CHARS} из {len(output)} символов]"
            )

        success = proc.returncode == 0
        return success, output if output.strip() else "(нет вывода)"

    except subprocess.TimeoutExpired:
        return False, (
            f"[SANDBOX] Превышен лимит времени выполнения ({TIMEOUT_SECONDS} с). "
            "Процесс принудительно остановлен."
        )
    except FileNotFoundError:
        return False, "[SANDBOX] Ошибка: не найден интерпретатор Python."
    except Exception as e:
        return False, f"[SANDBOX] Внутренняя ошибка выполнения: {e}"
    finally:
        # Временный файл удаляется всегда
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
