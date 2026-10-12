"""Effect of cleaning SAM before the hybrid: model choice, segment MAE per model and on VVR regions.

Both runs must share a frozen split (``--split-from``). Errors are each run's own test error, so a
strip whose lines changed is graded against changed observations too.

Run: ``PYTHONPATH=. uv run python scripts/evaluate/compare_prehybrid.py --base 20261006-base --base-hybrid …
        --clean 20261006-preclean --clean-hybrid …``
"""

from __future__ import annotations

import argparse
import json
import warnings

import geopandas as gpd
import pandas as pd
import pyogrio

from src.analysis.tee import tee_to
from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID, normalise_location_id

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--base-hybrid", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--clean-hybrid", required=True)
ap.add_argument("--out", help="per-strip table (default: <clean run>/compare/)")
args = ap.parse_args()
compare_dir = ExperimentConfig(experiment=args.clean).compare_dir
tee_to(compare_dir / f"compare_prehybrid_vs_{args.base}.txt")

cfg = ExperimentConfig(experiment=args.base)
scope = normalise_location_id(gpd.read_file(cfg.scope_gpkg)).to_crs(28992)[[LOCATION_ID, "geometry"]]
vvr = gpd.read_file(cfg.signalering_gpkg, layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl").to_crs(28992)
vvr_regions = set(gpd.sjoin(scope, vvr[["geometry"]], predicate="intersects")[LOCATION_ID])


def preference(path: str) -> pd.Series:
    p = pyogrio.read_dataframe(path, layer="model_preference", read_geometry=False)
    return normalise_location_id(p).set_index(LOCATION_ID)["model_preference"]


def test_errors(run: str) -> pd.DataFrame:
    """Segment-model test strips (the model behind the presented MAE), keyed by region and strip."""
    t = pd.read_parquet(ExperimentConfig(experiment=run).model_out_dir / "segment_test.parquet")
    t["err"], t["naive"] = (t["v_test"] - t["pred"]).abs(), (t["v_test"] - t["naive"]).abs()
    return t.set_index([LOCATION_ID, "seg"])[["err", "naive", "v_test"]]


pb, pc = preference(args.base_hybrid), preference(args.clean_hybrid)
both = pb.index.intersection(pc.index)
print(f"== model choice: {len(both):,} regions in both hybrids ==")
print(pd.crosstab(pb[both], pc[both], rownames=["before"], colnames=["after cleaning"], margins=True).to_string())

eb, ec = test_errors(args.base), test_errors(args.clean)
common = eb.index.intersection(ec.index)
t = eb.loc[common].join(ec.loc[common], lsuffix="_base", rsuffix="_clean")
loc = t.index.get_level_values(LOCATION_ID)
t["pref_base"], t["pref_clean"] = pb.reindex(loc).values, pc.reindex(loc).values
t["vvr"] = loc.isin(vvr_regions)
t["switched"] = t["pref_base"].ne(t["pref_clean"])
print(f"\n== segment model, frozen test split: {len(eb):,} / {len(ec):,} test strips, {len(common):,} in both ==")


def row(m: pd.Series) -> dict:
    d = t[m]
    out = {"n": len(d), "regions": d.index.get_level_values(LOCATION_ID).nunique()}
    for k in ("base", "clean"):
        out[f"MAE {k}"] = d[f"err_{k}"].mean()
        out[f"skill {k}"] = 1 - d[f"err_{k}"].mean() / d[f"naive_{k}"].mean()
    return out


groups = {
    "all": t.index == t.index,
    "VVR regions": t["vvr"],
    "height-preferred (before)": t["pref_base"].eq("height"),
    "SAM-preferred (before)": t["pref_base"].eq("segmentation"),
    "height-preferred (after)": t["pref_clean"].eq("height"),
    "SAM-preferred (after)": t["pref_clean"].eq("segmentation"),
    "switched SAM→height": t["pref_base"].eq("segmentation") & t["pref_clean"].eq("height"),
    "switched height→SAM": t["pref_base"].eq("height") & t["pref_clean"].eq("segmentation"),
    "unchanged choice": ~t["switched"],
    "VVR, SAM-preferred (before)": t["vvr"] & t["pref_base"].eq("segmentation"),
}
res = pd.DataFrame({k: row(m) for k, m in groups.items()}).T
fmt = res.copy()
for c in fmt.columns:
    fmt[c] = fmt[c].map(lambda v, c=c: f"{v:+.0%}" if c.startswith("skill") else (f"{v:,.0f}" if c in ("n", "regions") else f"{v:.2f}"))
print(fmt.to_string())

print("\n== segment horizon metrics (own test strips) ==")
for run in (args.base, args.clean):
    m = json.loads((ExperimentConfig(experiment=run).model_out_dir / "segment_metrics.json").read_text())
    print(f"  {run}: MAE {m['test_mae']:.3f} · tail {m['test_tail_mae']:.2f} · naive {m['naive_mae']:.3f} · "
          f"skill {1 - m['test_mae'] / m['naive_mae']:+.0%} · R² {m['test_r2']:.2f} · n_test {m['n_test']:,}")

print("\n== VVR polygons predicted to cross ==")
for run in (args.base, args.clean):
    gp = ExperimentConfig(experiment=run).output_gpkg
    if gp.exists():
        y = pyogrio.read_dataframe(gp, layer="vvr_rates_of_change", read_geometry=False)["predicted_vvr_crossing_year"]
        y = y[y.notna() & (y != 9999)]
        print(f"  {run}: by 2027 {int((y <= 2027).sum())} · by 2031 {int((y <= 2031).sum())} · by 2050 {int((y <= 2050).sum())}")

out = args.out or compare_dir / f"compare_prehybrid_vs_{args.base}.parquet"
t.to_parquet(out)
print("→", out)
