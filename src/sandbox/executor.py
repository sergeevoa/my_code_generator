"""
Docker-sandbox: безопасное выполнение Python-кода в изолированном контейнере.

Код передаётся в контейнер через stdin — никаких временных файлов,
никаких volume-монтирований.

Архитектура: один контейнер живёт всю сессию агента (SandboxContainer),
каждое выполнение — отдельный `docker exec ... python -`.
Изоляция сохраняется: read-only FS, отдельный Python-процесс на каждый вызов.

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
from typing import Optional, Tuple

from sandbox.validator import validate_code, check_sandbox_compatibility

# ── Настройки ──────────────────────────────────────────────────────────────
DOCKER_IMAGE          = "code-sandbox:latest"
TIMEOUT_SECONDS: int  = 10    # лимит выполнения кода внутри контейнера
EXEC_OVERHEAD: int    = 5     # запас для docker exec (контейнер уже запущен)
MAX_OUTPUT_CHARS: int = 10_000

# Флаги безопасности, общие для docker run и SandboxContainer
_SECURITY_FLAGS = [
    "--network=none",
    "--memory=64m",
    "--memory-swap=64m",
    "--cpus=0.5",
    "--pids-limit=32",
    "--read-only",
    "--tmpfs=/tmp:size=8m,noexec,nosuid",
    "--user=nobody",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--no-healthcheck",
    "-e", "PYTHONDONTWRITEBYTECODE=1",
    "-e", "PYTHONUNBUFFERED=1",
]


def _process_docker_output(proc: subprocess.CompletedProcess) -> Tuple[bool, str]:
    """Общая обработка вывода после docker exec."""
    output = proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + proc.stderr

    if len(output) > MAX_OUTPUT_CHARS:
        output = (
            output[:MAX_OUTPUT_CHARS]
            + f"\n... [вывод обрезан: показано {MAX_OUTPUT_CHARS} из {len(output)} символов]"
        )

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


class SandboxContainer:
    """
    Persistent Docker sandbox. Запускается один раз на сессию агента,
    принимает произвольное число выполнений через docker exec.
    """

    def __init__(self) -> None:
        self._container_id: Optional[str] = None

    def start(self, working_dir: str = ".") -> None:
        """Запустить контейнер в фоне. Бросает RuntimeError при ошибке."""
        abs_working_dir = str(Path(working_dir).resolve())
        proc = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "-v", f"{abs_working_dir}:/workspace:ro",
                "-w", "/workspace",
            ]
            + _SECURITY_FLAGS
            + [DOCKER_IMAGE, "sleep", "infinity"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "docker run -d failed")
        self._container_id = proc.stdout.strip()

    def stop(self) -> None:
        """Остановить контейнер. Безопасно вызывать повторно."""
        if self._container_id:
            subprocess.run(["docker", "stop", self._container_id], capture_output=True)
            self._container_id = None

    def execute(self, code: str, validate: bool = True) -> Tuple[bool, str]:
        """
        Выполнить код в контейнере.

        Слой 1: AST-валидация (если validate=True).
        Слой 2: docker exec — изолированный Python-процесс.
        """
        if not self._container_id:
            return False, "[INFRASTRUCTURE ERROR] Sandbox container is not running."

        if validate:
            is_safe, violations = validate_code(code)
            if not is_safe:
                lines = "\n".join(f"  • {v}" for v in violations)
                return False, f"[SANDBOX] Код заблокирован. Нарушения:\n{lines}"

            is_compatible, reason = check_sandbox_compatibility(code)
            if not is_compatible:
                return False, reason

        try:
            proc = subprocess.run(
                [
                    "docker", "exec", "-i", self._container_id,
                    "timeout", str(TIMEOUT_SECONDS), "python", "-",
                ],
                input=code,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_SECONDS + EXEC_OVERHEAD,
            )
            return _process_docker_output(proc)
        except subprocess.TimeoutExpired:
            return False, (
                f"[INFRASTRUCTURE ERROR] Превышен общий таймаут "
                f"({TIMEOUT_SECONDS + EXEC_OVERHEAD} с)."
            )
        except Exception as e:
            return False, f"[INFRASTRUCTURE ERROR] Внутренняя ошибка sandbox: {e}"

    def __enter__(self) -> "SandboxContainer":
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ── Совместимость с run_humaneval/ ──────────────────────────────────────────
# execute_python оставлен для HumanEval-runner, который управляет контейнерами
# самостоятельно и не использует SandboxContainer.

STARTUP_OVERHEAD: int = 15  # запас на старт контейнера (только для execute_python)


def execute_python(code: str, validate: bool = True, working_dir: str = ".") -> Tuple[bool, str]:
    """
    Выполняет Python-код в одноразовом Docker-контейнере.
    Используется только HumanEval-runner'ом.
    """
    if validate:
        is_safe, violations = validate_code(code)
        if not is_safe:
            lines = "\n".join(f"  • {v}" for v in violations)
            return False, f"[SANDBOX] Код заблокирован. Нарушения:\n{lines}"

        is_compatible, reason = check_sandbox_compatibility(code)
        if not is_compatible:
            return False, reason

    abs_working_dir = str(Path(working_dir).resolve())
    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm", "--interactive",
                "-v", f"{abs_working_dir}:/workspace:ro",
                "-w", "/workspace",
            ]
            + _SECURITY_FLAGS
            + [DOCKER_IMAGE, "timeout", str(TIMEOUT_SECONDS), "python", "-"],
            input=code,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS + STARTUP_OVERHEAD,
        )
        return _process_docker_output(proc)
    except subprocess.TimeoutExpired:
        return False, (
            f"[INFRASTRUCTURE ERROR] Превышен общий таймаут "
            f"({TIMEOUT_SECONDS + STARTUP_OVERHEAD} с, включая старт контейнера)."
        )
    except FileNotFoundError:
        return False, "[INFRASTRUCTURE ERROR] Docker не найден. Установите и запустите Docker Desktop."
    except Exception as e:
        return False, f"[INFRASTRUCTURE ERROR] Внутренняя ошибка sandbox: {e}"
