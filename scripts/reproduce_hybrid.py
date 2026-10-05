"""Regenerate Luke's ``hybrid_model_results_nearest_<date>.gpkg`` from his inputs.

Step 1 — ``post_processing_segmentation_lines``: raw SAM lines → ``_nearest`` lines.
Step 2 — ``run_hybrid_model`` on the ``_nearest`` lines: preference per region.
Writes ``lines`` (winner only, as his file), ``lines_all`` (both models) and
``model_preference`` to ``--out``; compare with ``scripts/compare_hybrid.py``.

Run: ``PYTHONPATH=. uv run python scripts/reproduce_hybrid.py --sam …full_aoi.geojson
        --wocu-output …wocu_output_fase2_20260310.gpkg --out …/hybrid_repro.gpkg [--workers 8]``
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import pandas as pd
import pyogrio

from src.hybrid.luke_v2 import (
    HEIGHT_MODEL,
    LOCATION_ID,
    add_location_ids,
    clean_location,
    hybrid_location,
    validate_segmentation_lines,
)

ap = argparse.ArgumentParser()
ap.add_argument("--sam", required=True)
ap.add_argument("--wocu-output", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit", type=int, default=None, help="first N locations only (smoke test)")
args = ap.parse_args()
logging.basicConfig(level=logging.WARNING)

_CTX: dict = {}


def _init(ctx):
    _CTX.update(ctx)


def _clean_chunk(locs):
    out = []
    for loc in locs:
        out.extend(clean_location(loc, _CTX["lines"].get(loc), _CTX["centrelines"]))
    return out


def _hybrid_chunk(locs):
    rats, height = [], []
    for loc in locs:
        cl = _CTX["centrelines"].get(loc)
        if cl is None or loc not in _CTX["polygons"]:
            continue
        pts = _CTX["points"].get(loc, _CTX["empty_points"])
        rat, lines_h = hybrid_location(loc, pts, _CTX["lines"][loc], cl)
        rat["geometry"] = _CTX["polygons"][loc]
        rats.append(rat)
        if len(lines_h):
            height.append(lines_h)
    return rats, height


def _chunks(seq, n):
    size = max(1, len(seq) // (n * 8))
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _run(fn, locs, ctx):
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(ctx,)) as ex:
        return list(ex.map(fn, _chunks(locs, args.workers)))


if __name__ == "__main__":
    t0 = time.time()
    polys = pyogrio.read_dataframe(args.wocu_output, layer="vlakken_scope").rename(columns={"position_id": LOCATION_ID})
    cls = pyogrio.read_dataframe(args.wocu_output, layer="centrelines").rename(columns={"position_id": LOCATION_ID})
    centrelines = dict(zip(cls[LOCATION_ID], cls.geometry, strict=True))
    polygons = dict(zip(polys[LOCATION_ID], polys.geometry, strict=True))
    print(f"scope: {len(polys)} polygons, {len(cls)} centrelines", flush=True)

    raw = validate_segmentation_lines(gpd.read_file(args.sam))
    raw = add_location_ids(raw, polys)
    print(f"step 1: {len(raw)} raw line-region rows, {raw[LOCATION_ID].isna().sum()} lines in no region", flush=True)
    raw = raw[raw[LOCATION_ID].notna()]
    locs = sorted(raw[LOCATION_ID].unique())[: args.limit]
    by_loc = dict(list(raw.groupby(LOCATION_ID)))
    parts = _run(_clean_chunk, locs, {"lines": by_loc, "centrelines": centrelines})
    nearest = gpd.GeoDataFrame([r for p in parts for r in p], geometry="geometry", crs=raw.crs)
    print(f"step 1: {len(nearest)} _nearest pieces from {len(locs)} regions ({time.time() - t0:.0f} s)", flush=True)

    # As his second run: the _nearest file is read back and joined to the regions again.
    seg = add_location_ids(validate_segmentation_lines(nearest), polys)
    seg = seg[seg[LOCATION_ID].notna()]
    seg_by_loc = dict(list(seg.groupby(LOCATION_ID)))
    pts = pyogrio.read_dataframe(args.wocu_output, layer="punten_oever")
    pts_by_loc = dict(list(pts.groupby(LOCATION_ID, sort=False)))
    hlocs = sorted(seg_by_loc)
    print(f"step 2: {len(pts)} height points, {len(hlocs)} regions with SAM lines", flush=True)
    parts = _run(
        _hybrid_chunk,
        hlocs,
        {"lines": seg_by_loc, "centrelines": centrelines, "polygons": polygons,
         "points": pts_by_loc, "empty_points": pts.iloc[:0]},
    )
    rationale = gpd.GeoDataFrame([r for p in parts for r in p[0]], geometry="geometry", crs=polys.crs)
    height = pd.concat([h for p in parts for h in p[1]], ignore_index=True)
    height = gpd.GeoDataFrame(height, geometry="geometry", crs=polys.crs)

    seg_out = seg[[LOCATION_ID, "date", "model", "geometry"]]
    all_lines = gpd.GeoDataFrame(pd.concat([height, seg_out], ignore_index=True), geometry="geometry", crs=polys.crs)
    all_lines["date"] = pd.to_datetime(all_lines["date"])
    pref = rationale.set_index(LOCATION_ID)["model_preference"]
    winner = all_lines[
        ((all_lines["model"] == HEIGHT_MODEL) & all_lines[LOCATION_ID].map(pref).eq("height"))
        | ((all_lines["model"] == "segmentation") & all_lines[LOCATION_ID].map(pref).eq("segmentation"))
    ]
    winner.to_file(args.out, layer="lines", driver="GPKG")
    all_lines.to_file(args.out, layer="lines_all", driver="GPKG")
    rationale.to_file(args.out, layer="model_preference", driver="GPKG")
    print(f"→ {args.out}: {len(rationale)} regions, {len(winner)} winner lines, {len(all_lines)} lines in all "
          f"({time.time() - t0:.0f} s)", flush=True)
