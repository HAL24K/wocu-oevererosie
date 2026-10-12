"""VVR lens: how well do we predict the regions that have a VVR?

Per run: errors for VVR regions vs the rest (region model on its test split,
1-yr CV error and horizon error from the worst-predicted v2 layer), the
worst-N restricted to VVR regions, and — from the export — the VVRs predicted
to cross soon, cross-checked against the host region's error and flags.

Run: ``uv run python scripts/vvr_lens.py --run 20260907-near-fz [--cross-by 2027]``
"""

from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID, normalise_location_id

warnings.filterwarnings("ignore")
MODES = ["doolhof", "kronkelend", "tijd_uitschieter", "fragment", "verkeerde_oever"]

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--cross-by", type=int, default=2027)
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()
cfg = ExperimentConfig(experiment=args.run)
TRIAGE = cfg.data_dir / "02_processed/triage"

# ── region ↔ VVR mapping (spatial) ───────────────────────────────────────────
scope = normalise_location_id(gpd.read_file(cfg.scope_gpkg)).to_crs(28992)[[LOCATION_ID, "geometry"]]
vvr = gpd.read_file(cfg.signalering_gpkg, layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl").to_crs(28992)
vvr["vvr_idx"] = range(len(vvr))
hit = gpd.sjoin(scope, vvr[["vvr_idx", "geometry"]], how="inner", predicate="intersects")
vvr_regions = set(hit[LOCATION_ID])
print(f"scope regions {len(scope):,} · VVR polygons {len(vvr):,} · regions intersecting a VVR: {len(vvr_regions):,}")

# ── errors per region (worst-predicted v2 layer) ─────────────────────────────
vl = normalise_location_id(gpd.read_file(TRIAGE / f"worst_predicted_v2_{args.run}.gpkg", layer="vlakken")).set_index(LOCATION_ID)
vl["vvr"] = vl.index.isin(vvr_regions)
print(f"modelled regions {len(vl):,} · with VVR {int(vl['vvr'].sum()):,} ({vl['vvr'].mean():.0%})")


def block(d):
    e, h = d["err_1jr_cv"].dropna(), d["err_horizon_mean"].dropna()
    return dict(
        n=len(d),
        cv_mae=e.mean(), cv_median=e.median(), cv_share_gt2=(e > 2).mean(), cv_share_gt5=(e > 5).mean(),
        horizon_n=len(h), horizon_mae=h.mean(), horizon_median=h.median(),
        sam_share=(d["model_preference"] == "segmentation").mean(),
        no_flag_share=d["geen_flags"].mean(),
    )


print("\n== errors: VVR regions vs the rest (all modelled regions) ==")
print(pd.DataFrame({"VVR": block(vl[vl["vvr"]]), "no VVR": block(vl[~vl["vvr"]]), "all": block(vl)}).round(3).to_string())

# region model on its own test split, VVR-only
rf = pd.read_parquet(cfg.features_dir / "region_features.parquet")
bundle = joblib.load(cfg.model_out_dir / "bundle.joblib")
model = joblib.load(cfg.model_out_dir / "model_lgb.joblib")
feats = bundle["config"]["FEATS_LGB"]
test = rf[rf["split"] == "test"]
err = pd.Series(np.abs(model.predict(test[feats].astype(float)) - test["v_test"]), index=test.index)
naive = (rf[rf["split"] == "train"]["v_test"].mean() - test["v_test"]).abs()
isv = test.index.isin(vvr_regions)
print("\n== region model, own test split ==")
for lab, m in [("VVR", isv), ("no VVR", ~isv), ("all", np.ones(len(test), bool))]:
    tail = (test["v_test"].abs() > 2) & m
    print(f"  {lab:7s} n={int(m.sum()):5d}  MAE {err[m].mean():.3f}  tail {err[tail].mean() if tail.any() else float('nan'):.3f} (n={int(tail.sum())})  naive {naive[m].mean():.3f}  skill {1 - err[m].mean() / naive[m].mean():+.0%}")

# ── worst-N among VVR regions ────────────────────────────────────────────────
top_v = vl[vl["vvr"] & vl["rank_horizon"].notna()].sort_values("rank_horizon").head(args.n)
top_all = vl[vl["rank_horizon"].notna()].sort_values("rank_horizon").head(args.n)
print(f"\n== worst-{args.n} ==")
print(f"  general worst-{args.n}: {int(top_all['vvr'].sum())} have a VVR")
print(f"  VVR-only worst-{args.n}: overall ranks #{int(top_v['rank_horizon'].min())}–#{int(top_v['rank_horizon'].max())} · "
      f"SAM {int((top_v['model_preference'] == 'segmentation').sum())} · no flag {int(top_v['geen_flags'].sum())} · "
      f"horizon err #1 {top_v['err_horizon_mean'].iloc[0]:.1f} · #{args.n} {top_v['err_horizon_mean'].iloc[-1]:.1f}")
riv = top_v.index.str.extract(r"^([a-z]+\d*)", expand=False).value_counts()
print("  rivers:", riv.to_dict())

# ── VVRs predicted to cross soon vs host-region error ────────────────────────
gp = cfg.output_gpkg
if gp.exists():
    rc = gpd.read_file(gp, layer="vvr_rates_of_change").to_crs(28992)
    soon = rc[(rc["predicted_vvr_crossing_year"] <= args.cross_by) & (rc["predicted_vvr_crossing_year"] != 9999)].copy()
    j = gpd.sjoin(soon[["predicted_vvr_crossing_year", "geometry"]], scope, how="left", predicate="intersects")
    j = j.join(vl[["err_1jr_cv", "rank_1jr", "err_horizon_mean", "rank_horizon", "geen_flags", "model_preference", *[f"n_{m}" for m in MODES]]], on=LOCATION_ID)
    per = j.groupby(j.index).agg(
        crossing=("predicted_vvr_crossing_year", "first"),
        regions=(LOCATION_ID, lambda s: ",".join(sorted(set(s.dropna())))),
        max_cv_err=("err_1jr_cv", "max"),
        best_rank_horizon=("rank_horizon", "min"),
        any_flag=("geen_flags", lambda s: int((s == 0).any())),
        sam=("model_preference", lambda s: int((s == "segmentation").any())),
    )
    print(f"\n== VVRs predicted to cross by {args.cross_by}: {len(per)} ==")
    print(f"  host region CV error > 5 m/yr: {int((per['max_cv_err'] > 5).sum())} · > 10: {int((per['max_cv_err'] > 10).sum())} · "
          f"host in worst-100: {int((per['best_rank_horizon'] <= 100).sum())} · host has a cleaning flag: {int(per['any_flag'].sum())} · "
          f"host SAM-preferred: {int(per['sam'].sum())} · no modelled host region: {int(per['max_cv_err'].isna().sum())}")
    print(per.sort_values("max_cv_err", ascending=False).head(12).round(1).to_string())
    out = TRIAGE / f"vvr_kruising_check_{args.run}.gpkg"
    soon.join(per.drop(columns="crossing")).to_file(out, layer="vvr_kruising_check", driver="GPKG")
    print("→", out)
