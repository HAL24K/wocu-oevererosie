"""Our cleaning stack applied to the SAM ``_nearest`` lines before the hybrid points count.

The rules work on sampled points (``src.cleaning.rules``); a line survives with at
least one kept sample, and is then clipped by the structure / secondary-water mask
so the hybrid counts only the bank that the pipeline would also keep.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from src.cleaning.rules import RuleContext, apply_rules
from src.pipeline.config import ExperimentConfig
from src.pipeline.hybrid_prep import build_structures_geom, cleaning_rules, line_metrics, sample_lines
from src.sources.geometry import LOCATION_ID

logger = logging.getLogger(__name__)


def clean_sam_lines(
    seg: gpd.GeoDataFrame, centrelines: gpd.GeoSeries, cfg: ExperimentConfig
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Return the surviving, mask-clipped SAM pieces and the per-rule stats.

    Regions without a pipeline centreline (side-channel regions in Luke's scope) pass through as-is."""
    outside = seg[~seg[LOCATION_ID].isin(centrelines.index)]
    seg = seg[seg[LOCATION_ID].isin(centrelines.index)].reset_index(drop=True)
    seg["date"] = pd.to_datetime(seg["date"])
    seg["year"] = seg["date"].dt.year
    samples = sample_lines(
        seg, centrelines, cfg.n_samples, spacing_m=cfg.sample_spacing_m, min_per_line=cfg.min_samples_per_line
    )
    structures = build_structures_geom(cfg, centrelines)
    ctx = RuleContext(line_metrics=line_metrics(seg), cl_len=centrelines.length, structures=structures)
    kept = apply_rules(samples, ctx, cleaning_rules(cfg))
    stats = pd.DataFrame(ctx.stats).drop(columns="params")

    out = seg.loc[sorted(kept["line_idx"].unique())].copy()
    out["geometry"] = out.geometry.difference(structures)
    out = out[~out.geometry.is_empty].explode(index_parts=False)
    out = out[out.geometry.geom_type == "LineString"].drop(columns="year")
    logger.info("pre-hybrid cleaning: %d → %d SAM pieces; %d passed through", len(seg), len(out), len(outside))
    out = pd.concat([out, outside.assign(date=pd.to_datetime(outside["date"]))], ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=seg.crs), stats


def side_channel_mask(
    seg: gpd.GeoDataFrame, legger: gpd.GeoDataFrame, centrelines: gpd.GeoSeries, buffer_m: float
) -> gpd.GeoDataFrame:
    """Clip SAM pieces by the RWS side-channel register, except the channel a region itself lies on."""
    poly = gpd.GeoDataFrame(geometry=legger.geometry.buffer(buffer_m).values, crs=28992)
    poly["pid"] = range(len(poly))
    cl = gpd.GeoDataFrame({LOCATION_ID: centrelines.index}, geometry=centrelines.buffer(2.0).values, crs=28992)
    own = set(map(tuple, gpd.sjoin(cl, poly, predicate="intersects")[[LOCATION_ID, "pid"]].values))
    hit = gpd.sjoin(seg[[LOCATION_ID, "geometry"]], poly, predicate="intersects")
    hit = hit[[(loc, pid) not in own for loc, pid in zip(hit[LOCATION_ID], hit["pid"], strict=True)]]
    cut = hit.groupby(level=0)["pid"].agg(list)
    out = seg.copy()
    for i, pids in cut.items():
        out.at[i, "geometry"] = out.at[i, "geometry"].difference(poly.geometry.iloc[pids].union_all())
    logger.info("side-channel mask +%.0f m: %d of %d pieces clipped", buffer_m, len(cut), len(seg))
    return _singlepart(out)


def jetty_mask(seg: gpd.GeoDataFrame, jetties: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    """Clip SAM pieces within ``buffer_m`` of a mooring jetty (moored ships read as bank)."""
    zone = jetties.geometry.buffer(buffer_m).union_all()
    out = seg.copy()
    touched = out.geometry.intersects(zone)
    out.loc[touched, "geometry"] = out.loc[touched].geometry.difference(zone)
    logger.info("jetty mask %.0f m: %d of %d pieces clipped", buffer_m, int(touched.sum()), len(seg))
    return _singlepart(out)


def _singlepart(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[~gdf.geometry.is_empty].explode(index_parts=False)
    return gdf[gdf.geometry.geom_type == "LineString"].reset_index(drop=True)
