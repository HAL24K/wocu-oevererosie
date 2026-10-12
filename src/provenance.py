"""Records of what a stage used and produced, shared by both stages.

A record is one small JSON file: input files (path, size, date), settings, the
code version and results. It is written next to the output (``run.json`` or
``hybrid.json``) and added as one line to ``docs/runs/ledger.jsonl``, which is
tracked in git, so it outlives the output's data. One file, however many
experiments: ``python -m src.provenance`` prints the runs as a table.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import src.paths as PATHS

LEDGER = PATHS.ROOT_DIR / "docs" / "runs" / "ledger.jsonl"


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


def read_ledger(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or LEDGER
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_to_ledger(entry: dict[str, Any], path: Path | None = None) -> None:
    """Add ``entry`` to the ledger, replacing an earlier entry with the same kind and id."""
    path = path or LEDGER
    rows = [
        r
        for r in read_ledger(path)
        if (r.get("kind"), r.get("id")) != (entry["kind"], entry["id"])
    ]
    rows.append(entry)
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def write_record(
    record: dict[str, Any], path: Path, kind: str, record_id: str, data_dir: Path
) -> Path:
    """Write ``record`` to ``path`` and add it to the ledger in git.

    ``kind`` is ``"run"`` or ``"hybrid"``. The ledger entry is only written for
    the project's own data tree, so test runs on a temporary data dir leave the
    repository alone.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    if data_dir.resolve() == PATHS.DATA_DIR.resolve() and LEDGER.parent.exists():
        append_to_ledger({"kind": kind, "id": record_id, **record})
    return path


def ledger_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table of the model runs in the ledger, oldest first."""
    out = [
        "| Run | Created | Segment MAE | Skill | Tail MAE | Region MAE | Regions | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(
        (r for r in rows if r["kind"] == "run"), key=lambda r: r["created"]
    ):
        seg = r["metrics"]["segment"]
        reg = r["metrics"]["region"].get("5 – LightGBM", {})
        c = r.get("counts", {})
        n = c.get("regions_train", 0) + c.get("regions_test", 0)
        note = (r.get("notes") or "").split(". ")[0][:90]
        out.append(
            f"| {r['id']} | {r['created'][:10]} | {seg.get('test_mae', float('nan')):.3f} "
            f"| {1 - seg['test_mae'] / seg['naive_mae']:+.0%} | {seg.get('test_tail_mae', float('nan')):.2f} "
            f"| {reg.get('test_mae', float('nan')):.2f} | {n:,} | {note} |"
        )
    return "\n".join(out)


if __name__ == "__main__":
    print(ledger_table(read_ledger()))
