"""Copy the light, irreplaceable part of every run and hybrid to Google Drive.

data/ is not in git and not backed up. What a run needs to be understood again
— its record, exports, figures, comparisons, report — is small; the rest
(features, model files, the hybrid GeoPackages) can be rebuilt from the code
and the record. Copies only new or changed files; never deletes in Drive.

Run: ``uv run python scripts/backup_to_drive.py [--dry-run]``
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import src.paths as PATHS

DRIVE = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-alexander.bakker@morainewater.com"
    / "My Drive/WOCU Oevererosie/runs"
)
KEEP = {
    "04_runs": [
        "run.json",
        "notes.md",
        "exports",
        "figures",
        "compare",
        "model/report.html",
        "model/segment_metrics.json",
    ],
    "03_hybrid": [
        "hybrid.json",
        "notes.md",
        "sam_cleaning_stats.csv",
        "compare",
        "delivery_format",
    ],
}

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--drive", type=Path, default=DRIVE)
args = ap.parse_args()

copied = skipped = 0
for stage, items in KEEP.items():
    for folder in sorted((PATHS.DATA_DIR / stage).iterdir()):
        if not folder.is_dir():
            continue
        for item in items:
            src = folder / item
            files = sorted(p for p in src.rglob("*") if p.is_file()) if src.is_dir() else [src]
            for f in files:
                if not f.exists():
                    continue
                dst = args.drive / stage / f.relative_to(PATHS.DATA_DIR / stage)
                if dst.exists() and dst.stat().st_size == f.stat().st_size and dst.stat().st_mtime >= f.stat().st_mtime:
                    skipped += 1
                    continue
                copied += 1
                print(("would copy " if args.dry_run else "copy ") + str(f.relative_to(PATHS.DATA_DIR)))
                if not args.dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
print(f"{copied} copied, {skipped} unchanged → {args.drive}")
