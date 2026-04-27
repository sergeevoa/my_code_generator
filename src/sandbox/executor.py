"""
Docker-sandbox: безопасное выполнение Python-кода в изолированном контейнере.

Код передаётся в контейнер через stdin — никаких временных файлов,
никаких volume-монтирований. Контейнер уничтожается сразу после завершения.

Двухуровневая защита:
  1. AST-валидация (validator.py) — отклоняет опасный код до запуска контейнера.
  2. Docker-изоляция:
       --network=none        → нет сети
       --memory / --cpus     → лимиты ресурсов (cgroups)
       --pids-limit          → защита от fork bomb
       --read-only           → файловая система только для чтения
       --tmpfs=/tmp          → единственное место для записи, noexec
       --user=nobody         → непривилегированный пользователь
       --cap-drop=ALL        → убраны все Linux capabilities
       --security-opt        → нельзя поднять привилегии (setuid/setgid)
"""
import subprocess
from pathlib import Path
from typing import Tuple

from sandbox.validator import validate_code

# ── Настройки ──────────────────────────────────────────────────────────────
DOCKER_IMAGE          = "code-sandbox:latest"  
TIMEOUT_SECONDS: int  = 10    # лимит выполнения кода внутри контейнера
STARTUP_OVERHEAD: int = 15    # запас на старт Docker-контейнера
MAX_OUTPUT_CHARS: int = 10_000


def execute_python(code: str, validate: bool = True, working_dir: str = ".") -> Tuple[bool, str]:
    """
    Выполняет Python-код в Docker-контейнере с максимальной изоляцией.

    Args:
        code:     Python-код для выполнения.
        validate: Если True (по умолчанию), перед запуском выполняется
                  AST-валидация — нужна для кода от агента/пользователя.
                  Передайте False только для заведомо доверенного кода
                  (например, тест-сюиты HumanEval), чтобы не блокировать
                  легитимные импорты вроде sys/os.

    Returns:
        (success, output)
        success=True  → код завершился с кодом 0
        success=False → блокировка, таймаут или ошибка выполнения
    """

    # ── Слой 1: AST-валидация (только для недоверенного кода) ───────────────
    if validate:
        is_safe, violations = validate_code(code)
        if not is_safe:
            lines = "\n".join(f"  • {v}" for v in violations)
            return False, f"[SANDBOX] Код заблокирован. Нарушения:\n{lines}"

    # ── Слой 2: Docker-изоляция ─────────────────────────────────────────────
    abs_working_dir = str(Path(working_dir).resolve())

    try:
        proc = subprocess.run(
            [
                "docker", "run",
                "--rm",                                   # удалить контейнер после выполнения
                "--interactive",                          # держать stdin открытым (для передачи кода)
                "-v", f"{abs_working_dir}:/workspace:ro", # рабочая директория проекта (только чтение)
                "-w", "/workspace",                       # CWD внутри контейнера

                # ── Сеть ────────────────────────────────────────────────────
                "--network=none",                         # полное отсутствие сети

                # ── Ресурсы (cgroups) ────────────────────────────────────────
                "--memory=64m",                           # лимит RAM
                "--memory-swap=64m",                      # swap = 0 (memory-swap == memory)
                "--cpus=0.5",                             # не более 50% одного ядра
                "--pids-limit=32",                        # не более 32 процессов (fork bomb)

                # ── Файловая система ─────────────────────────────────────────
                "--read-only",                            # вся FS контейнера только для чтения
                "--tmpfs=/tmp:size=8m,noexec,nosuid",     # /tmp — единственное место для записи
                                                          # noexec — нельзя запускать файлы из /tmp
                                                          # nosuid — suid-биты в /tmp игнорируются

                # ── Привилегии ───────────────────────────────────────────────
                "--user=nobody",                          # непривилегированный пользователь
                "--cap-drop=ALL",                         # убрать все Linux capabilities
                "--security-opt=no-new-privileges",       # нельзя поднять привилегии через setuid
                "--no-healthcheck",                       # не запускать healthcheck

                # ── Переменные окружения Python ──────────────────────────────
                "-e", "PYTHONDONTWRITEBYTECODE=1",        # не писать .pyc файлы (нужно для --read-only)
                "-e", "PYTHONUNBUFFERED=1",               # вывод без буферизации (сразу в stdout)

                DOCKER_IMAGE,

                # Команда внутри контейнера:
                # timeout <N> python -
                #   timeout — убивает процесс через N секунд (coreutils, есть в debian slim)
                #   python - — читает код из stdin
                "timeout", str(TIMEOUT_SECONDS), "python", "-",
            ],
            input=code,                                           # код передаётся через stdin
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS + STARTUP_OVERHEAD,          # общий таймаут включает старт контейнера
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

        # Код 124 — стандартный exit code команды `timeout` при срабатывании таймаута
        if proc.returncode == 124:
            return False, (
                f"[SANDBOX] Превышен лимит времени выполнения ({TIMEOUT_SECONDS} с). "
                "Процесс принудительно остановлен."
            )

        success = proc.returncode == 0
        if not output.strip():
            if success:
                return False, (
                    "[NO OUTPUT] The code ran without errors but printed nothing.\n"
                    "This means the tests were NOT verified.\n"
                    "You MUST add assert statements and print('ALL TESTS PASSED') to confirm correctness."
                )
            else:
                return False, "[ERROR] Process exited with an error but produced no output."
        return success, output

    except subprocess.TimeoutExpired:
        return False, (
            f"[INFRASTRUCTURE ERROR] Превышен общий таймаут "
            f"({TIMEOUT_SECONDS + STARTUP_OVERHEAD} с, включая старт контейнера)."
        )
    except FileNotFoundError:
        return False, "[INFRASTRUCTURE ERROR] Docker не найден. Установите и запустите Docker Desktop."
    except Exception as e:
        return False, f"[INFRASTRUCTURE ERROR] Внутренняя ошибка sandbox: {e}"
