"""
Persistent memory for the coding assistant.

Two files per project:
  PROJECT_MEMORY.md  — written by developer or agent on first run;
                       describes project structure, stack, conventions.
                       Stored inside the project: <working_dir>/memory/

  SESSION_MEMORY.md  — written by agent after each productive session;
                       keeps the last MAX_SESSIONS entries.
                       Stored in the agent home directory, outside the project:
                       <AGENT_DATA_DIR>/memory/<project_name>/SESSION_MEMORY.md
"""

import os
import re
import shutil
import time
from datetime import date
from pathlib import Path
from typing import Optional

_MEMORY_DIR = "memory"
_PROJECT_FILE = "PROJECT_MEMORY.md"
_SESSION_FILE = "SESSION_MEMORY.md"

# Выделяем 1/40 контекста (2.5%) под session memory.
# 1 токен ≈ 4 символа; одна запись сессии ≈ 150 символов.
_CONTEXT_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
_MAX_SESSIONS = max(3, min((_CONTEXT_LEN // 40 * 4) // 150, 15))

# Директория агента для хранения SESSION_MEMORY вне проектов.
_AGENT_DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", Path.home() / ".coding_agent"))

# Записи сессий старше этого числа дней удаляются при очистке.
_SESSION_MAX_AGE_DAYS = int(os.getenv("SESSION_MEMORY_MAX_AGE_DAYS", "90"))

_SESSION_SPLIT = re.compile(r'(?=^## \d{4}-\d{2}-\d{2})', re.MULTILINE)


def _project_mdir(working_dir: str) -> Path:
    """Директория для PROJECT_MEMORY внутри проекта."""
    return Path(working_dir) / _MEMORY_DIR


def _session_dir(working_dir: str) -> Path:
    """Директория для SESSION_MEMORY в домашней директории агента."""
    project_name = Path(working_dir).resolve().name
    return _AGENT_DATA_DIR / _MEMORY_DIR / project_name


def project_memory_exists(working_dir: str) -> bool:
    return (_project_mdir(working_dir) / _PROJECT_FILE).exists()


def load_memory(working_dir: str) -> Optional[str]:
    """Return combined content of both memory files, or None if neither exists."""
    parts = []

    project_file = _project_mdir(working_dir) / _PROJECT_FILE
    if project_file.exists():
        content = project_file.read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"### Project Memory\n{content}")

    session_file = _session_dir(working_dir) / _SESSION_FILE
    if session_file.exists():
        content = session_file.read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"### Recent Sessions\n{content}")

    return "\n\n".join(parts) if parts else None


def create_project_memory(working_dir: str) -> None:
    """Scan project structure and write PROJECT_MEMORY.md without using the LLM."""
    root = Path(working_dir)
    mdir = _mdir(working_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    dirs  = [p.name + "/" for p in entries if p.is_dir()  and not p.name.startswith(".")]
    files = [p.name       for p in entries if p.is_file() and not p.name.startswith(".")]

    deps = ""
    req_path = root / "requirements.txt"
    if req_path.exists():
        lines = req_path.read_text(encoding="utf-8").strip().splitlines()
        packages = [
            ln.split("==")[0].split(">=")[0].split("~=")[0].strip()
            for ln in lines if ln and not ln.startswith("#")
        ]
        deps = ", ".join(packages[:15])

    project_name = root.resolve().name

    structure = "\n".join(f"- {e}" for e in dirs + files)
    content = (
        f"# Project: {project_name}\n\n"
        f"## Purpose\nCode generation and sandboxed Python execution assistant.\n\n"
        f"## Tech Stack\nLanguage: Python\n"
        f"Dependencies: {deps or 'see requirements.txt'}\n\n"
        f"## Structure\n{structure}\n\n"
        '## Coding Conventions\n'
        '- Assert-based tests; always end with print("ALL TESTS PASSED")\n'
        '- execute_code must succeed before write_file for .py files\n'
        '- No interactive input (input() is blocked in sandbox)\n'
    )

    (mdir / _PROJECT_FILE).write_text(content, encoding="utf-8")


def update_session_memory(working_dir: str, task: str, done: str, pending: str) -> str:
    """Append a new entry to SESSION_MEMORY.md, keeping at most _MAX_SESSIONS entries."""
    sdir = _session_dir(working_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    fpath = sdir / _SESSION_FILE

    today = date.today().isoformat()
    new_entry = f"## {today}\nTask: {task}\nDone: {done}\nPending: {pending}"

    existing = fpath.read_text(encoding="utf-8") if fpath.exists() else ""
    entries = [e.strip() for e in _SESSION_SPLIT.split(existing) if e.strip()]
    entries.append(new_entry)
    entries = entries[-_MAX_SESSIONS:]

    fpath.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return f"Session memory updated ({len(entries)}/{_MAX_SESSIONS} sessions kept)."


def cleanup_old_sessions(max_age_days: int = _SESSION_MAX_AGE_DAYS) -> None:
    """Remove per-project session directories not touched for more than max_age_days."""
    sessions_root = _AGENT_DATA_DIR / _MEMORY_DIR
    if not sessions_root.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    for project_dir in sessions_root.iterdir():
        if not project_dir.is_dir():
            continue
        session_file = project_dir / _SESSION_FILE
        mtime = session_file.stat().st_mtime if session_file.exists() else project_dir.stat().st_mtime
        if mtime < cutoff:
            shutil.rmtree(project_dir)
            print(f"[memory] removed stale session directory: {project_dir}", flush=True)
