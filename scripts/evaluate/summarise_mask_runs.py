"""One row per run: model choice vs the baseline hybrid, segment MAE and skill (all, VVR, per chosen model),
and VVR crossings. Runs share the frozen split; each is scored on its own test strips.

Run: ``PYTHONPATH=. uv run python scripts/evaluate/summarise_mask_runs.py --base 20261006-base:…/hybrid_repro.gpkg
        --run "side channels:20261007-side:…/hybrid_side.gpkg" … --out …/mask_runs.csv``
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID, ScopeGeometry, normalise_location_id

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="experiment:hybrid_gpkg")
ap.add_argument("--run", action="append", default=[], help="label:experiment:hybrid_gpkg")
ap.add_argument("--out", help="CSV (default: <base run>/compare/mask_runs.csv)")
args = ap.parse_args()

cfg = ExperimentConfig(experiment="summary")
ours = set(ScopeGeometry(centreline_gpkg=cfg.raw_gpkg).centrelines.index)
scope = normalise_location_id(gpd.read_file(cfg.scope_gpkg, layer="vlakken_scope")).to_crs(28992)
vvr = gpd.read_file(cfg.signalering_gpkg, layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl").to_crs(28992)
vvr_regions = set(gpd.sjoin(scope[[LOCATION_ID, "geometry"]], vvr[["geometry"]], predicate="intersects")[LOCATION_ID])


def preference(path: str) -> pd.Series:
    p = normalise_location_id(pyogrio.read_dataframe(path, layer="model_preference", read_geometry=False))
    p = p.set_index(LOCATION_ID)["model_preference"]
    return p[p.index.isin(ours)]


def mae_skill(t: pd.DataFrame) -> tuple[float, float]:
    return t["err"].mean(), 1 - t["err"].mean() / t["nerr"].mean()


base_exp, base_gpkg = args.base.split(":", 1)
base_pref = preference(base_gpkg)
rows = []
for spec in [f"baseline (Luke's hybrid):{args.base}", *args.run]:
    label, exp, gpkg = spec.split(":", 2)
    pref = preference(gpkg)
    both = base_pref.index.intersection(pref.index)
    t = pd.read_parquet(ExperimentConfig(experiment=exp).model_out_dir / "segment_test.parquet")
    t["err"], t["nerr"] = (t["v_test"] - t["pred"]).abs(), (t["v_test"] - t["naive"]).abs()
    t["pref"] = t[LOCATION_ID].map(pref)
    vt = t[t[LOCATION_ID].isin(vvr_regions)]
    out = ExperimentConfig(experiment=exp).output_gpkg
    y = pyogrio.read_dataframe(out, layer="vvr_rates_of_change", read_geometry=False)["predicted_vvr_crossing_year"]
    y = y[y.notna() & (y != 9999)]
    rows.append(
        {
            "variant": label,
            "SAM-preferred regions": int((pref == "segmentation").sum()),
            "SAM→height": int(((base_pref[both] == "segmentation") & (pref[both] == "height")).sum()),
            "height→SAM": int(((base_pref[both] == "height") & (pref[both] == "segmentation")).sum()),
            "MAE all": mae_skill(t)[0],
            "skill all": mae_skill(t)[1],
            "MAE VVR": mae_skill(vt)[0],
            "skill VVR": mae_skill(vt)[1],
            "MAE height-preferred": t.loc[t["pref"] == "height", "err"].mean(),
            "MAE SAM-preferred": t.loc[t["pref"] == "segmentation", "err"].mean(),
            "test strips": len(t),
            "VVR crossing by 2027": int((y <= 2027).sum()),
            "VVR crossing by 2031": int((y <= 2031).sum()),
        }
    )
res = pd.DataFrame(rows)
out = args.out or ExperimentConfig(experiment=args.base.split(":")[0]).compare_dir / "mask_runs.csv"
Path(out).parent.mkdir(parents=True, exist_ok=True)
res.to_csv(out, index=False)
print(res.round(3).to_string(index=False))
print("→", out)
