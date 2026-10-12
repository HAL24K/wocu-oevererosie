"""Records of what a stage used and produced, shared by both stages.

A record is one small JSON file: input files (path, size, date), settings, the
code version and results. It is written next to the output and copied to
``docs/runs/`` (tracked in git), so it outlives the output's data.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import src.paths as PATHS

LEDGER_DIR = PATHS.ROOT_DIR / "docs" / "runs"


def git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PATHS.ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_changes": bool(git("status", "--porcelain", "--", "src")),
    }


def rel(path: Path, data_dir: Path) -> str:
    try:
        return str(path.relative_to(data_dir))
    except ValueError:
        return str(path)


def describe_files(paths: dict[str, Path], data_dir: Path) -> dict[str, dict[str, Any]]:
    """Path (relative to data_dir), size and modification date per input file."""
    out = {}
    for name, p in paths.items():
        entry: dict[str, Any] = {"path": rel(p, data_dir)}
        if p.is_file():
            st = p.stat()
            entry["bytes"] = st.st_size
            entry["modified"] = datetime.fromtimestamp(st.st_mtime).isoformat(
                timespec="seconds"
            )
        elif not p.exists():
            entry["missing"] = True
        out[name] = entry
    return out


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_record(
    record: dict[str, Any], path: Path, ledger_name: str, data_dir: Path
) -> Path:
    """Write ``record`` to ``path`` and to ``docs/runs/<ledger_name>.json``.

    The tracked copy is only written for the project's own data tree, so test
    runs on a temporary data dir leave the repository alone.
    """
    text = json.dumps(record, indent=2, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if data_dir.resolve() == PATHS.DATA_DIR.resolve() and LEDGER_DIR.exists():
        target = LEDGER_DIR / f"{ledger_name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return path
