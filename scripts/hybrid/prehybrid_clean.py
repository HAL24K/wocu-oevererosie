"""Clean the SAM ``_nearest`` lines before the hybrid, ready for ``reproduce_hybrid.py --nearest``.

Input: the ``lines_all`` layer of a ``reproduce_hybrid.py`` run; its segmentation rows are exactly the
SAM lines that went into that hybrid (after the nearest-crossing step, before the points count).
Steps, each optional: our rule stack (``--rules``), the side-channel mask, the jetty mask.

Run: ``PYTHONPATH=. uv run python scripts/hybrid/prehybrid_clean.py --repro …/hybrid_repro.gpkg
        --out …/sam_nearest_masked.gpkg [--rules] [--side-channel-m 10] [--jetty-m 100]``
"""

from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import pyogrio

from src.hybrid.prehybrid_clean import clean_sam_lines, jetty_mask, side_channel_mask
from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID, ScopeGeometry

ap = argparse.ArgumentParser()
ap.add_argument("--repro", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rules", action="store_true", help="apply our cleaning rule stack")
ap.add_argument("--side-channel-m", type=float, help="side-channel register mask, buffered by this many metres")
ap.add_argument("--jetty-m", type=float, help="mask this many metres around mooring jetties")
ap.add_argument("--sample-spacing-m", type=float, default=5.0)
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")

cfg = ExperimentConfig(experiment="prehybrid-clean", sample_spacing_m=args.sample_spacing_m)
centrelines = ScopeGeometry(centreline_gpkg=cfg.raw_gpkg).centrelines
lines = pyogrio.read_dataframe(args.repro, layer="lines_all")
seg = gpd.GeoDataFrame(lines[lines["model"] == "segmentation"], geometry="geometry", crs=lines.crs).reset_index(drop=True)
before = seg[LOCATION_ID].nunique()

if args.rules:
    seg, stats = clean_sam_lines(seg, centrelines, cfg)
    print(stats.to_string(index=False))
    stats.to_csv(f"{args.out}.rule_stats.csv", index=False)
if args.side_channel_m is not None:
    legger = gpd.read_file(cfg.data_dir / "01_raw/scope/20260107_Nevengeulen_RWS_Legger.gpkg").to_crs(28992)
    seg = side_channel_mask(seg, legger, centrelines, args.side_channel_m)
if args.jetty_m is not None:
    kw = gpd.read_file(cfg.structures_gpkg, layer=cfg.kunstwerken_layer).to_crs(28992)
    seg = jetty_mask(seg, kw[kw["categorie"] == "steiger_afmeer"], args.jetty_m)

print(f"regions with SAM lines: {before} → {seg[LOCATION_ID].nunique()}")
seg.to_file(args.out, layer="lines", driver="GPKG")
print("→", args.out)
