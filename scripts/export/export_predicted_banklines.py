"""Predicted bank LINES per year at segment resolution (R = 20).

The pipeline's ``predicted_bank_positions`` layer carries one point per
(region, year) — the region-level scalar. This script turns the segment
artifact into an actual bank line per (region, year): each of the R segments
is projected forward from its own last observation,

    dist(year) = last_dist_m + v_pred * (year - last_year)

anchored perpendicular to the centreline on the bank side the samples sit.
Strips that follow each other form one line piece; a strip without data breaks
the line (``src.analysis.predicted_lines``), so each (region, year) is a
MultiLineString with ``n_stukken`` pieces.

Run: ``uv run python scripts/export/export_predicted_banklines.py --run 20260907-near-fz
        --hybrid-gpkg …_nearest_20260710.gpkg --years 2026 2031``
"""

from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")

from src.analysis.predicted_lines import region_lines  # noqa: E402
from src.analysis.region_inspector import RegionInspector  # noqa: E402
from src.pipeline.config import ExperimentConfig  # noqa: E402
from src.sources.geometry import LOCATION_ID  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--hybrid-gpkg", default=None)
ap.add_argument("--years", nargs=2, type=int, default=[2026, 2031])
ap.add_argument("--out", default=None)
args = ap.parse_args()

cfg = ExperimentConfig.from_run(args.run, hybrid_gpkg=args.hybrid_gpkg)
YEARS = list(range(args.years[0], args.years[1] + 1))
R = cfg.segment_R

segp = pd.read_parquet(cfg.model_out_dir / "segment_predictions.parquet")
print(f"segment predictions: {len(segp):,} rows · {segp[LOCATION_ID].nunique():,} regions · R={R}")

ri = RegionInspector(hybrid_gpkg=cfg.hybrid_gpkg)
clines = ri.geometry.centrelines
samples = pd.read_parquet(cfg.features_dir / "samples.parquet", columns=[LOCATION_ID, "x", "y"])
side = samples.groupby(LOCATION_ID)[["x", "y"]].mean()

rows = []
skipped = 0
for loc, g in segp.groupby(LOCATION_ID, sort=False):
    cl = clines.get(loc)
    if cl is None or loc not in side.index:
        skipped += 1
        continue
    side_xy = (side.loc[loc, "x"], side.loc[loc, "y"])
    vpred = g["v_pred"].to_numpy()
    for year in YEARS:
        geom = region_lines(g, cl, side_xy, R, year)
        dist = g["last_dist_m"] + g["v_pred"] * (year - g["last_year"])
        rows.append(
            {
                LOCATION_ID: loc,
                "year": year,
                "n_seg": int(len(g)),
                "n_stukken": len(geom.geoms),
                "v_mean_m_per_yr": round(float(vpred.mean()), 3),
                "dist_mean_m": round(float(dist.mean()), 2),
                "verplaatsing_tov_2026_m": round(float((vpred * (year - YEARS[0])).mean()), 2),
                "geometry": geom,
            }
        )

out = gpd.GeoDataFrame(rows, crs=28992)
cfg.exports_dir.mkdir(parents=True, exist_ok=True)
path = args.out or str(
    cfg.exports_dir / f"voorspelde_oeverlijnen_{YEARS[0]}_{YEARS[-1]}_{args.run}.gpkg"
)
out.to_file(path, layer="voorspelde_oever_per_jaar", driver="GPKG")
print(f"regions skipped (no centreline): {skipped}")
print(f"lines: {len(out):,} ({out[LOCATION_ID].nunique():,} regions x {len(YEARS)} years)")
print(f"median segments per line: {out['n_seg'].median():.0f} · median length {out.geometry.length.median():.0f} m")
print("→", path)
