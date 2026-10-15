"""Error of the strip model per bank type (asked for in the Maas session, 15 Oct).

Test strips of a run (``model/segment_test.parquet``) joined with the region context
(nature-friendly bank, river, soil, vegetation, land use, bend) and the hybrid's chosen
source; per group the model's error, the historical average's error and the skill.
Writes ``compare/error_by_bank_type.csv``.

Run: ``PYTHONPATH=. uv run python scripts/evaluate/error_by_bank_type.py --run 20261014-stable``
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import pandas as pd

from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
args = ap.parse_args()
cfg = ExperimentConfig.from_run(args.run)
t = pd.read_parquet(cfg.model_out_dir / "segment_test.parquet")
rf = pd.read_parquet(cfg.features_dir / "region_features.parquet")
if LOCATION_ID in rf.columns:
    rf = rf.set_index(LOCATION_ID)
pref = gpd.read_file(cfg.hybrid_gpkg, layer="model_preference", ignore_geometry=True)
pref = pref.set_index(LOCATION_ID)["model_preference"]
t = t.join(rf.drop(columns=[c for c in t.columns if c in rf.columns]), on=LOCATION_ID)
t["bron"] = (
    t[LOCATION_ID]
    .map(pref)
    .map({"height": "hoogtemodel", "segmentation": "satelliet (SAM)"})
)
t["nvo"] = (
    t["is_nvo"].astype(bool).map({True: "natuurvriendelijke oever", False: "overig"})
)
b = t["bend_exposure_n8"]
t["bocht"] = pd.cut(
    b,
    [-float("inf"), b.quantile(1 / 3), b.quantile(2 / 3), float("inf")],
    labels=["binnenbocht-achtig", "recht", "buitenbocht-achtig"],
)
t["err"] = (t["pred"] - t["v_test"]).abs()
t["err_naive"] = (t["naive"] - t["v_test"]).abs()

rows = []
for dim in [
    "bron",
    "nvo",
    "river",
    "bocht",
    "soil_group",
    "vegetation_class",
    "land_use",
]:
    for val, g in t.groupby(dim, observed=True):
        if g[LOCATION_ID].nunique() < 15:
            continue
        rows.append(
            {
                "kenmerk": dim,
                "waarde": val,
                "strookjes": len(g),
                "vlakken": g[LOCATION_ID].nunique(),
                "fout_model": round(g["err"].mean(), 2),
                "fout_hist_gem": round(g["err_naive"].mean(), 2),
                "skill_pct": round((1 - g["err"].mean() / g["err_naive"].mean()) * 100),
                "gem_verplaatsing": round(g["v_test"].mean(), 2),
            }
        )
out = pd.DataFrame(rows)
path = cfg.compare_dir / "error_by_bank_type.csv"
out.to_csv(path, index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
print(f"→ {path}")
