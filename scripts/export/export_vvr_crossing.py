"""Predicted year the bank crosses the signaleringslijn, per scope region.

Uses the segment-resolution forecast (R = 20): each strip is projected forward
from its own last observation, the strips are joined into a bank line per year,
and the first year that line touches the signaleringslijn
(``Vlak_vrije_ruimte_natuurvriendelijke_oever_ln``) is the crossing year.
"Any strip crosses" is the alert, which is what the segment artifact was for.

Output layer ``vvr_kruising_vlakken``: scope polygons of the regions that have
a VVR, with ``kruisingsjaar``, an ordered ``kruising_klasse`` for styling, and
``model_preference`` so the border can carry which model was preferred.

Run: ``uv run python scripts/export/export_vvr_crossing.py --run 20260907-near-fz
        --hybrid-gpkg …_nearest_20260710.gpkg --until 2050``
"""

from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")

from src.analysis.region_inspector import RegionInspector  # noqa: E402
from src.pipeline.config import ExperimentConfig  # noqa: E402
from src.sources.geometry import LOCATION_ID, normalise_location_id  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--hybrid-gpkg", default=None)
ap.add_argument("--start", type=int, default=2026)
ap.add_argument("--until", type=int, default=2050)
ap.add_argument("--out", default=None)
args = ap.parse_args()

cfg = ExperimentConfig(experiment=args.run, hybrid_gpkg=args.hybrid_gpkg)
YEARS = list(range(args.start, args.until + 1))

vvr = gpd.read_file(cfg.signalering_gpkg, layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl").to_crs(28992)
print(f"VVR polygons: {len(vvr):,}")
# NB the "_ln" layer is exactly the polygon outline (both edges), so "line
# touches signaleringslijn" fires on the river-side edge and is meaningless.
# Crossing = the bank reaches the LANDWARD edge, measured per segment by
# casting the perpendicular ray and taking the far intersection with the VVR.
vvr_tree = STRtree(vvr.geometry.values)
RAY = 400.0

ri = RegionInspector(hybrid_gpkg=cfg.hybrid_gpkg)
scope = ri.geometry.polygons
clines = ri.geometry.centrelines
segp = pd.read_parquet(cfg.model_out_dir / "segment_predictions.parquet")
samples = pd.read_parquet(cfg.features_dir / "samples.parquet", columns=[LOCATION_ID, "x", "y"])
side = samples.groupby(LOCATION_ID)[["x", "y"]].mean()

# regions that actually have a VVR
sc = gpd.GeoDataFrame(geometry=scope.values, index=scope.index, crs=28992).reset_index(names=LOCATION_ID)
hit = gpd.sjoin(sc, vvr[["geometry"]], how="inner", predicate="intersects")
vvr_regions = set(hit[LOCATION_ID])
print(f"scope regions intersecting a VVR: {len(vvr_regions):,}")

pref = normalise_location_id(gpd.read_file(cfg.hybrid_gpkg, layer="model_preference")).set_index(LOCATION_ID)["model_preference"]

rows = []
for loc, g in segp.groupby(LOCATION_ID, sort=False):
    if loc not in vvr_regions:
        continue
    cl = clines.get(loc)
    if cl is None or loc not in side.index or len(g) < 2:
        continue
    g = g.sort_values("seg")
    st = g["seg_center"].to_numpy()
    eps = 0.02
    p0 = np.array([cl.interpolate(max(s - eps, 0.0), normalized=True).coords[0] for s in st])
    p1 = np.array([cl.interpolate(min(s + eps, 1.0), normalized=True).coords[0] for s in st])
    base = np.array([cl.interpolate(s, normalized=True).coords[0] for s in st])
    tx, ty = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
    norm = np.hypot(tx, ty); norm[norm == 0] = 1.0
    nx, ny = -ty / norm, tx / norm
    sign = np.sign((side.loc[loc, "x"] - base[:, 0]) * nx + (side.loc[loc, "y"] - base[:, 1]) * ny)
    sign[sign == 0] = 1.0
    last_year, last_dist, vpred = g["last_year"].to_numpy(), g["last_dist_m"].to_numpy(), g["v_pred"].to_numpy()

    # per segment: where does the VVR start and end along the outward ray?
    near = np.full(len(g), np.nan)
    far = np.full(len(g), np.nan)
    for i in range(len(g)):
        ray = LineString([(base[i, 0], base[i, 1]),
                          (base[i, 0] + sign[i] * nx[i] * RAY, base[i, 1] + sign[i] * ny[i] * RAY)])
        idx = vvr_tree.query(ray)
        if not len(idx):
            continue
        inter = [vvr.geometry.values[j].intersection(ray) for j in idx]
        inter = [q for q in inter if not q.is_empty]
        if not inter:
            continue
        pts = np.vstack([np.asarray(q.coords) if q.geom_type == "LineString"
                         else np.vstack([np.asarray(p.coords) for p in q.geoms])
                         for q in inter])
        dd = np.hypot(pts[:, 0] - base[i, 0], pts[:, 1] - base[i, 1])
        near[i], far[i] = dd.min(), dd.max()
    has_vvr = ~np.isnan(far)
    if not has_vvr.any():
        continue

    kruising = None
    for year in YEARS:
        d = last_dist + vpred * (year - last_year)
        if np.any(d[has_vvr] >= far[has_vvr]):
            kruising = year
            break
    d0 = last_dist + vpred * (args.start - last_year)
    span = np.where(far - near > 0, far - near, np.nan)
    consumed = np.clip((d0 - near) / span, 0, 1)
    rows.append({
        LOCATION_ID: loc,
        "kruisingsjaar": kruising,
        "pct_vvr_op_2026": round(float(np.nanmean(consumed) * 100), 1),
        "vvr_breedte_m": round(float(np.nanmean(span)), 1),
        "v_mean_m_per_yr": round(float(vpred.mean()), 2),
        "v_max_m_per_yr": round(float(vpred.max()), 2),
        "n_seg_vvr": int(has_vvr.sum()),
        "model_preference": pref.get(loc, "onbekend"),
        "geometry": scope.get(loc),
    })

out = gpd.GeoDataFrame(rows, crs=28992)

def klasse(y):
    if y is None or pd.isna(y): return f"geen kruising voor {YEARS[-1]}"
    if y <= args.start: return f"reeds gekruist ({args.start})"
    if y <= 2027: return "2027"
    if y <= 2029: return "2028-2029"
    if y <= 2031: return "2030-2031"
    if y <= 2035: return "2032-2035"
    if y <= 2040: return "2036-2040"
    return f"2041-{YEARS[-1]}"

out["kruising_klasse"] = out["kruisingsjaar"].map(klasse)
cfg.exports_dir.mkdir(parents=True, exist_ok=True)
path = args.out or str(cfg.exports_dir / f"vvr_kruising_{args.run}.gpkg")
out.to_file(path, layer="vvr_kruising_vlakken", driver="GPKG")
print(f"\nregions scored: {len(out):,}")
print(out["kruising_klasse"].value_counts().reindex(
    [f"reeds gekruist ({args.start})", "2027", "2028-2029", "2030-2031", "2032-2035",
     "2036-2040", f"2041-{YEARS[-1]}", f"geen kruising voor {YEARS[-1]}"]).dropna().to_string())
print("\nmodel preference:", out["model_preference"].value_counts().to_dict())
print("→", path)
