"""How much do forecasts and crossing years move when only the seed changes?

Runs that differ only in ``--seed`` retrain the model on a different draw of
strips (the region split is frozen, the strip split and the trees are not).
Per run: test skill against the historical average on its own test strips.
Across runs: spread of the forecast speed per strip and of the crossing year
and class per VVR scope area (the crossing exports are made if missing).

Writes ``compare/seed_spread_regions.csv`` (one row per scope area) and
``compare/seed_spread.txt`` under the first run.

Run: ``PYTHONPATH=. uv run python scripts/evaluate/seed_spread.py
        --runs 20261014-stable 20261015-seed1 … 20261015-seed5``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import geopandas as gpd
import pandas as pd

from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
args = ap.parse_args()
cfgs = {r: ExperimentConfig.from_run(r) for r in args.runs}
out_dir = cfgs[args.runs[0]].compare_dir
out_dir.mkdir(parents=True, exist_ok=True)
lines = []


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


# 1 · test skill per run
say("test error per run (own test strips; skill = 1 - MAE / historical-average MAE)")
skill = {}
for r, cfg in cfgs.items():
    m = json.loads((cfg.model_out_dir / "segment_metrics.json").read_text())
    skill[r] = 1 - m["test_mae"] / m["naive_mae"]
    say(
        f"  {r:22s} MAE {m['test_mae']:.3f}  baseline {m['naive_mae']:.3f}  "
        f"skill {skill[r]:+.1%}  n_test {m['n_test']:,}"
    )
s = pd.Series(skill)
say(f"  skill: mean {s.mean():+.1%}, range {s.min():+.1%} to {s.max():+.1%}")

# 2 · forecast speed per strip
v = pd.concat(
    {
        r: pd.read_parquet(cfg.model_out_dir / "segment_predictions.parquet").set_index(
            [LOCATION_ID, "seg"]
        )["v_pred"]
        for r, cfg in cfgs.items()
    },
    axis=1,
).dropna()
sd = v.std(axis=1)
say()
say(f"forecast speed per strip, {len(v):,} strips in every run (m/yr)")
say(
    f"  spread across runs (sd): median {sd.median():.3f}, p90 {sd.quantile(0.9):.3f}, "
    f"p99 {sd.quantile(0.99):.3f}"
)
say(
    f"  strips whose speed differs > 0.5 m/yr between runs: {(v.max(axis=1) - v.min(axis=1) > 0.5).mean():.1%}"
)

# 3 · crossing year and class per scope area
cross = {}
for r, cfg in cfgs.items():
    f = cfg.run_dir / "exports" / f"vvr_kruising_{r}.gpkg"
    if not f.exists():
        subprocess.run(
            [sys.executable, "scripts/export/export_vvr_crossing.py", "--run", r],
            check=True,
            capture_output=True,
        )
    cross[r] = gpd.read_file(f, layer="vvr_kruising_vlakken").set_index(LOCATION_ID)
year = pd.concat({r: c["kruisingsjaar"] for r, c in cross.items()}, axis=1)
cls = pd.concat({r: c["kruising_klasse"] for r, c in cross.items()}, axis=1)
first = args.runs[0]
reg = pd.DataFrame(
    {
        "klasse_" + first: cls[first],
        "klassen": cls.apply(lambda c: " | ".join(sorted(c.dropna().unique())), axis=1),
        "n_klassen": cls.nunique(axis=1),
        "jaar_min": year.min(axis=1),
        "jaar_max": year.max(axis=1),
        "runs_met_kruising": year.notna().sum(axis=1),
    }
)
reg["jaar_spreiding"] = reg["jaar_max"] - reg["jaar_min"]
reg.to_csv(out_dir / "seed_spread_regions.csv")
n = len(reg)
say()
say(f"crossing per VVR scope area ({n:,} areas, {len(args.runs)} runs)")
say(
    f"  same class in every run: {(reg['n_klassen'] == 1).sum():,} ({(reg['n_klassen'] == 1).mean():.1%})"
)
moving = reg[reg["n_klassen"] > 1]
say(f"  class differs between runs: {len(moving):,}")
for k, g in moving.groupby("klasse_" + first):
    say(f"    of which {first} says {k}: {len(g)}")
some = reg[(reg["runs_met_kruising"] > 0) & (reg["runs_met_kruising"] < len(args.runs))]
say(f"  crosses before 2050 in some runs but not all: {len(some)}")
both = reg[reg["runs_met_kruising"] == len(args.runs)]
both = both[both["jaar_min"] > 2026]
if len(both):
    say(
        f"  future crossing in every run ({len(both)} areas): year spread median "
        f"{both['jaar_spreiding'].median():.0f}, max {both['jaar_spreiding'].max():.0f} years"
    )
say("  class counts per run:")
say(
    cls.apply(lambda c: c.value_counts()).fillna(0).astype(int).sort_index().to_string()
)
(out_dir / "seed_spread.txt").write_text("\n".join(lines) + "\n")
print(f"→ {out_dir / 'seed_spread_regions.csv'}")
