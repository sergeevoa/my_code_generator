"""
Persistent memory for the coding assistant.

Two files per project (stored in memory/ subdirectory):
  PROJECT_MEMORY.md  — written by developer or agent on first run;
                       describes project structure, stack, conventions.
  SESSION_MEMORY.md  — written by agent after each productive session;
                       keeps the last MAX_SESSIONS entries.
"""

import os
import re
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

_SESSION_SPLIT = re.compile(r'(?=^## \d{4}-\d{2}-\d{2})', re.MULTILINE)


def _mdir(working_dir: str) -> Path:
    return Path(working_dir) / _MEMORY_DIR


def project_memory_exists(working_dir: str) -> bool:
    return (_mdir(working_dir) / _PROJECT_FILE).exists()


def load_memory(working_dir: str) -> Optional[str]:
    """Return combined content of both memory files, or None if neither exists."""
    parts = []
    mdir = _mdir(working_dir)
    for fname, header in (
        (_PROJECT_FILE, "### Project Memory"),
        (_SESSION_FILE, "### Recent Sessions"),
    ):
        fpath = mdir / fname
        if fpath.exists():
            content = fpath.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"{header}\n{content}")
    return "\n\n".join(parts) if parts else None


def update_session_memory(working_dir: str, task: str, done: str, pending: str) -> str:
    """Append a new entry to SESSION_MEMORY.md, keeping at most _MAX_SESSIONS entries."""
    mdir = _mdir(working_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    fpath = mdir / _SESSION_FILE

    today = date.today().isoformat()
    new_entry = f"## {today}\nTask: {task}\nDone: {done}\nPending: {pending}"

    existing = fpath.read_text(encoding="utf-8") if fpath.exists() else ""
    entries = [e.strip() for e in _SESSION_SPLIT.split(existing) if e.strip()]
    entries.append(new_entry)
    entries = entries[-_MAX_SESSIONS:]

    fpath.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return f"Session memory updated ({len(entries)}/{_MAX_SESSIONS} sessions kept)."
