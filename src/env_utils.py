"""
Shared environment-file helpers for local scripts.
"""

from __future__ import annotations

import os

from pipeline_contract import resolve_repo_path


DEFAULT_ENV_FILES = (".env", ".env.local")


def resolve_env_value(env_name: str) -> str | None:
    value = os.getenv(env_name, "").strip()
    if value:
        return value

    for env_file in DEFAULT_ENV_FILES:
        path = resolve_repo_path(env_file)
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, file_value = line.split("=", 1)
            if key.strip() != env_name:
                continue
            cleaned = file_value.strip().strip("'").strip('"')
            return cleaned or None

    return None
