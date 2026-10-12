"""Compare a run with a baseline run, fairly: on the strips both were tested on, and on the rest.

Runs on different inputs (another hybrid, a looser quality gate) test on different
strips, so their headline MAE is not comparable. This splits the challenger's
test strips into the baseline's test regions (same targets, paired comparison)
and regions new to the challenger (main channel and side channels separately).
The report is kept in ``<run>/compare/vs_<base>.txt``.

Run: ``PYTHONPATH=. uv run python scripts/evaluate/compare_runs.py --base 20261012-hm0310 --run 20261013-gate-sam``
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.analysis.tee import tee_to
from src.pipeline.config import ExperimentConfig

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="baseline experiment")
ap.add_argument("--run", required=True, help="challenger experiment (owns the report)")
args = ap.parse_args()

base, run = ExperimentConfig(experiment=args.base), ExperimentConfig(experiment=args.run)
tee_to(run.compare_dir / f"vs_{args.base}.txt")


def record(cfg):
    return json.loads((cfg.run_dir / "run.json").read_text())


def stats(d: pd.DataFrame) -> str:
    if d.empty:
        return "—"
    mae = (d["pred"] - d["v_test"]).abs().mean()
    naive = (d["naive"] - d["v_test"]).abs().mean()
    return (
        f"MAE {mae:.3f} · naive {naive:.3f} · skill {1 - mae / naive:+.0%} "
        f"(n {len(d):,}, {d['location_id'].nunique():,} regions)"
    )


print(f"{args.run} vs {args.base}\n")
for name, cfg in (("base", base), ("run", run)):
    r = record(cfg)
    c = r["counts"]
    reg = r["metrics"]["region"].get("5 – LightGBM", {})
    print(
        f"{name:4s} {cfg.experiment}: regions train {c.get('regions_train', 0):,} · test "
        f"{c.get('regions_test', 0):,} · inference-only {c.get('regions_inference_only', 0):,} · "
        f"region LGB test MAE {reg.get('test_mae', float('nan')):.2f}"
    )
    print(f"     {r.get('notes', '')}\n")

a = pd.read_parquet(base.model_out_dir / "segment_test.parquet")
b = pd.read_parquet(run.model_out_dir / "segment_test.parquet")
known = set(a["location_id"])
side = b["location_id"].str.startswith("ngeul_")

print("Segment model, test strips (m/yr):")
print(f"  base, all its test strips          {stats(a)}")
print(f"  run, the base's test regions       {stats(b[b['location_id'].isin(known)])}")
print(f"  run, new main-channel regions      {stats(b[~b['location_id'].isin(known) & ~side])}")
print(f"  run, new side-channel regions      {stats(b[~b['location_id'].isin(known) & side])}")

j = a.merge(b, on=["location_id", "seg"], suffixes=("_base", "_run"))
same = np.allclose(j["v_test_base"], j["v_test_run"])
print(
    f"\nPaired, identical strips (n {len(j):,}, same targets: {same}): "
    f"MAE base {(j['pred_base'] - j['v_test_base']).abs().mean():.3f} · "
    f"run {(j['pred_run'] - j['v_test_run']).abs().mean():.3f}"
)
if not same:
    print("  targets differ: the runs observed these strips differently (other hybrid or cleaning)")
