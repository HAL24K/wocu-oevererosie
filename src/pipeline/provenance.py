"""The model stage's run record: ``04_runs/<experiment>/run.json``.

Also copied to ``docs/runs/<experiment>.json`` (tracked; see ``src.provenance``).
When the hybrid line set was built by ``python -m src.hybrid``, the record
carries that hybrid's own record too, so a run traces back to the raw files.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from src.pipeline.config import ExperimentConfig
from src.provenance import describe_files, git_info, now, rel, write_record


def describe_inputs(cfg: ExperimentConfig) -> dict[str, dict[str, Any]]:
    """Path, size and modification date of every input file."""
    out = describe_files({n: getattr(cfg, n) for n in cfg._DEFAULTS}, cfg.data_dir)
    hybrid_record = cfg.hybrid_gpkg.parent / "hybrid.json"
    if hybrid_record.exists():
        rec = json.loads(hybrid_record.read_text())
        out["hybrid_gpkg"]["hybrid_record"] = {
            k: rec.get(k)
            for k in ("name", "created", "notes", "git", "inputs", "settings")
        }
    return out


def settings(cfg: ExperimentConfig) -> dict[str, Any]:
    """Every config field that is not an input path."""
    out = {}
    for f in fields(cfg):
        if f.name in cfg._DEFAULTS or f.name in ("experiment", "data_dir"):
            continue
        v = getattr(cfg, f.name)
        out[f.name] = (
            rel(v, cfg.data_dir)
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
        "created": now(),
        "notes": notes,
        "git": git_info(),
        "inputs": describe_inputs(cfg),
        "settings": settings(cfg),
        "counts": counts or {},
        "metrics": {
            "region": region_results or {},
            "segment": segment_metrics or {},
        },
    }


def write_run_record(cfg: ExperimentConfig, record: dict[str, Any]) -> Path:
    return write_record(record, cfg.run_dir / "run.json", cfg.experiment, cfg.data_dir)
