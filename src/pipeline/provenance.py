"""Run records: what a run used and what it scored, in one small JSON file.

Every run writes ``run.json`` into its run folder and a copy to
``docs/runs/<experiment>.json``, which is tracked in git. The copy outlives the
run's data, so a number quoted in a deck or in Notion can always be traced to
its inputs, settings and code version.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import src.paths as PATHS
from src.pipeline.config import ExperimentConfig

LEDGER_DIR = PATHS.ROOT_DIR / "docs" / "runs"


def _git() -> dict[str, Any]:
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


def _rel(path: Path, data_dir: Path) -> str:
    try:
        return str(path.relative_to(data_dir))
    except ValueError:
        return str(path)


def describe_inputs(cfg: ExperimentConfig) -> dict[str, dict[str, Any]]:
    """Path, size and modification date of every input file."""
    out = {}
    for name in cfg._DEFAULTS:
        p = getattr(cfg, name)
        entry: dict[str, Any] = {"path": _rel(p, cfg.data_dir)}
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


def settings(cfg: ExperimentConfig) -> dict[str, Any]:
    """Every config field that is not an input path."""
    out = {}
    for f in fields(cfg):
        if f.name in cfg._DEFAULTS or f.name in ("experiment", "data_dir"):
            continue
        v = getattr(cfg, f.name)
        out[f.name] = (
            _rel(v, cfg.data_dir)
            if isinstance(v, Path)
            else list(v)
            if isinstance(v, tuple)
            else v
        )
    return out


def run_record(
    cfg: ExperimentConfig,
    region_results: dict[str, dict[str, float]] | None,
    segment_metrics: dict[str, Any] | None,
    counts: dict[str, int] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "experiment": cfg.experiment,
        "created": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        "git": _git(),
        "inputs": describe_inputs(cfg),
        "settings": settings(cfg),
        "counts": counts or {},
        "metrics": {
            "region": region_results or {},
            "segment": segment_metrics or {},
        },
    }


def write_run_record(cfg: ExperimentConfig, record: dict[str, Any]) -> Path:
    """Write ``run.json`` in the run folder, plus the tracked copy in docs/runs.

    The tracked copy is only written for runs on the project's own data tree,
    so test runs on a temporary data dir leave the repository alone.
    """
    text = json.dumps(record, indent=2, default=str) + "\n"
    path = cfg.run_dir / "run.json"
    path.write_text(text)
    if cfg.data_dir.resolve() == PATHS.DATA_DIR.resolve() and LEDGER_DIR.exists():
        (LEDGER_DIR / f"{cfg.experiment}.json").write_text(text)
    return path
