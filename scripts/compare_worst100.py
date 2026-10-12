"""Compare worst-predicted v2 exports of two runs (top-100 by rank_horizon).

Run: ``uv run python scripts/compare_worst100.py 20260907-raw-len 20260907-near-fz``
"""

from __future__ import annotations

import sys

import geopandas as gpd
import pandas as pd

from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID, normalise_location_id

MODES = ["doolhof", "kronkelend", "tijd_uitschieter", "fragment", "verkeerde_oever"]


def load(run: str) -> pd.DataFrame:
    cfg = ExperimentConfig(experiment=run)
    p = cfg.data_dir / "02_processed/triage" / f"worst_predicted_v2_{run}.gpkg"
    return normalise_location_id(gpd.read_file(p, layer="vlakken")).set_index(LOCATION_ID)


def summary(vl: pd.DataFrame) -> dict:
    top = vl.sort_values("rank_horizon").head(100)
    pref = top["model_preference"].value_counts()
    return {
        "regions": len(vl),
        "horizon-scored": int(vl["rank_horizon"].notna().sum()),
        "top100 SAM": int(pref.get("segmentation", 0)),
        "top100 height": int(pref.get("height", 0)),
        "top100 geen_flags": int(top["geen_flags"].sum()),
        "top100 in_train": int(top["horizon_in_train"].sum()),
        "horizon err #1": round(float(top["err_horizon_mean"].iloc[0]), 1),
        "horizon err #10": round(float(top["err_horizon_mean"].iloc[9]), 1),
        "horizon err #100": round(float(top["err_horizon_mean"].iloc[99]), 1),
        "top100 median 1jr CV": round(float(top["err_1jr_cv"].median()), 2),
        "all: CV MAE": round(float(vl["err_1jr_cv"].mean()), 3),
        "all: CV err > 5 m/yr": int((vl["err_1jr_cv"] > 5).sum()),
        "all: CV err > 10 m/yr": int((vl["err_1jr_cv"] > 10).sum()),
        "all: horizon err > 10": int((vl["err_horizon_mean"] > 10).sum()),
        "pop SAM share": round(float((vl["model_preference"] == "segmentation").mean()), 3),
        **{f"top100 {m}": int((top[f"n_{m}"] > 0).sum()) for m in MODES},
    }


def main(a: str, b: str) -> None:
    va, vb = load(a), load(b)
    print(pd.DataFrame({a: summary(va), b: summary(vb)}).to_string())
    ta = set(va.sort_values("rank_horizon").head(100).index)
    tb = set(vb.sort_values("rank_horizon").head(100).index)
    print(f"\ntop-100 overlap: {len(ta & tb)}/100")
    gone = [r for r in ta if r not in vb.index]
    print(f"raw top-100 regions absent from {b}: {len(gone)}")
    moved = va.loc[list(ta & set(vb.index)), ["rank_horizon", "err_horizon_mean"]].join(
        vb[["rank_horizon", "err_horizon_mean"]], rsuffix="_b"
    )
    print("\nraw top-100 → their rank/err in the other run (first 15 by raw rank):")
    print(moved.sort_values("rank_horizon").head(15).round(1).to_string())
    print("\nCV error by preference, all regions:")
    for name, v in [(a, va), (b, vb)]:
        g = v.groupby("model_preference")["err_1jr_cv"].agg(["mean", "median", lambda s: (s > 2).mean()])
        g.columns = ["mean", "median", "share>2"]
        print(name)
        print(g.round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
