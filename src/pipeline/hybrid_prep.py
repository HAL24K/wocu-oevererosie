"""Step 01 (hybrid source) — cleaned bank observations from the line delivery.

The graduated replacement for the point-cloud step when ``cfg.source ==
"hybrid"``: sample every delivered line against its region centreline, run
the e8 cleaning stack (experiments/loop/TRACK1_REPORT.md), aggregate with
the furthest-N convention, collapse to per-year distances and apply the
region-level |v| guard.

Outputs (under ``cfg.features_dir``):
  samples.parquet        every surviving sampled point (segment step reuses
                         the kept-line set)
  observations.parquet   one row per (region, date) — trajectory features
                         and the segment-horizon step consume this
  dist_per_year.parquet  the legacy per-year shape for region_split
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src.cleaning.rules import RuleContext, apply_rules
from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID, ScopeGeometry, normalise_location_id
from src.sources.observations import BankObservations

logger = logging.getLogger(__name__)


def cleaning_rules(cfg: ExperimentConfig) -> list[tuple[str, dict]]:
    """The graduated e8 stack, parameterized from the config."""
    return [
        ("structure_mask", {}),
        ("max_tortuosity_line", {"max_tort": cfg.tortuosity_max}),
        (
            "near_bank_line",
            {"frac": cfg.nearbank_frac, "min_ref": cfg.nearbank_min_ref},
        ),
        (
            "maze_survey",
            {"max_ratio": cfg.maze_max_ratio, "min_iqr": cfg.maze_min_iqr},
        ),
        ("fragment_survey", {"min_cov": cfg.fragment_min_cov}),
        ("min_samples_survey", {"min_n": cfg.min_samples_per_obs}),
        (
            "temporal_outlier_survey",
            {
                "max_dev": cfg.temporal_max_dev,
                "min_surveys": cfg.temporal_min_surveys,
                "detrend": True,
                "protect_min_years": cfg.temporal_protect_years,
            },
        ),
    ]


def load_lines(cfg: ExperimentConfig, geometry: ScopeGeometry) -> gpd.GeoDataFrame:
    lines = normalise_location_id(gpd.read_file(cfg.hybrid_gpkg, layer="lines"))
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
    lines["date"] = pd.to_datetime(lines["date"])
    lines["year"] = lines["date"].dt.year
    usable = lines[lines[LOCATION_ID].isin(geometry.centrelines.index)]
    logger.info(
        "hybrid lines: %d delivered, %d measurable (%d regions)",
        len(lines),
        len(usable),
        usable[LOCATION_ID].nunique(),
    )
    return usable


def sample_lines(
    lines: gpd.GeoDataFrame, centrelines: gpd.GeoSeries, n_samples: int
) -> pd.DataFrame:
    """Evenly sample each line; measure station and distance to centreline."""
    fractions = np.linspace(0.0, 1.0, n_samples)
    line_geoms = np.repeat(lines.geometry.values, n_samples)
    cline_geoms = np.repeat(centrelines.reindex(lines[LOCATION_ID]).values, n_samples)
    pts = shapely.line_interpolate_point(
        line_geoms, np.tile(fractions, len(lines)), normalized=True
    )
    coords = shapely.get_coordinates(pts)

    def rep(col):
        return np.repeat(lines[col].values, n_samples)

    return pd.DataFrame(
        {
            "line_idx": np.repeat(lines.index.values, n_samples),
            LOCATION_ID: rep(LOCATION_ID),
            "date": rep("date"),
            "year": rep("year"),
            "model": rep("model"),
            "station": shapely.line_locate_point(cline_geoms, pts, normalized=True),
            "dist": shapely.distance(pts, cline_geoms),
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    ).dropna(subset=["dist"])


def line_metrics(lines: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-line length, endpoint chord and tortuosity (for the maze rules)."""
    length = lines.geometry.length
    first = lines.geometry.apply(lambda g: shapely.get_coordinates(g)[0])
    last = lines.geometry.apply(lambda g: shapely.get_coordinates(g)[-1])
    chord = np.hypot(
        first.str[0].astype(float) - last.str[0].astype(float),
        first.str[1].astype(float) - last.str[1].astype(float),
    )
    return pd.DataFrame(
        {"length": length, "chord": chord, "tortuosity": length / np.maximum(chord, 1)},
        index=lines.index,
    )


def build_structures_geom(cfg: ExperimentConfig, centrelines: gpd.GeoSeries):
    """Kribben, masked kunstwerken and (optionally) secondary water, buffered.

    Kribben and kunstwerken come from ``structures.gpkg`` (BKN deliveries,
    scripts/prep_structures.py); only the kunstwerken categories listed in
    ``cfg.kunstwerk_categories`` are masked.

    Secondary water: vegetatielegger 'Water' parts that touch no centreline —
    marinas, floodplain pools, side gullies without their own scope region.
    Their edges are genuine waterlines but not riverbanks.
    """
    parts = []
    kribs = gpd.read_file(cfg.structures_gpkg, layer=cfg.structures_layer).to_crs(28992)
    parts.append(shapely.union_all(kribs.geometry.buffer(cfg.mask_buffer_m).values))
    logger.info("structures: %d kribben", len(kribs))

    if cfg.kunstwerk_categories:
        kw = gpd.read_file(cfg.structures_gpkg, layer=cfg.kunstwerken_layer).to_crs(
            28992
        )
        kw = kw[kw["categorie"].isin(cfg.kunstwerk_categories)]
        parts.append(shapely.union_all(kw.geometry.buffer(cfg.mask_buffer_m).values))
        logger.info(
            "structures: %d kunstwerken (%s)",
            len(kw),
            ", ".join(cfg.kunstwerk_categories),
        )

    if cfg.water_mask:
        veg = gpd.read_file(
            cfg.veg_gpkg, layer="rws_vegetatielegger:vegetatieklassen"
        ).to_crs(28992)
        water = veg[veg["vlklasse"] == "Water"][["geometry"]].explode(index_parts=False)
        water["geometry"] = shapely.force_2d(water.geometry.values)
        cline_buf = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(centrelines.values, crs=28992).buffer(2.0)
        )
        hit = gpd.sjoin(
            water.reset_index(drop=True), cline_buf, predicate="intersects", how="left"
        )
        is_channel = hit.groupby(level=0)["index_right"].first().notna()
        mask_polys = water.reset_index(drop=True)[~is_channel.values]
        parts.append(
            shapely.union_all(mask_polys.geometry.buffer(cfg.mask_buffer_m).values)
        )
        logger.info(
            "structures: %d secondary-water parts (%d channel parts kept)",
            len(mask_polys),
            int(is_channel.sum()),
        )

    geom = shapely.union_all(parts)
    shapely.prepare(geom)
    return geom


def build_observations(
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full hybrid prep. Returns (samples, observations, dist_per_year)."""
    geometry = ScopeGeometry(centreline_gpkg=cfg.raw_gpkg)
    lines = load_lines(cfg, geometry)
    samples = sample_lines(lines, geometry.centrelines, cfg.n_samples)

    ctx = RuleContext(
        line_metrics=line_metrics(lines),
        cl_len=geometry.centrelines.length,
        structures=build_structures_geom(cfg, geometry.centrelines),
    )
    kept = apply_rules(samples, ctx, cleaning_rules(cfg))
    for st in ctx.stats:
        logger.info(
            "   %s: -%d samples, -%d lines, -%d surveys, -%d regions",
            st["rule"],
            st["samples_dropped"],
            st["lines_dropped"],
            st["surveys_dropped"],
            st["regions_dropped"],
        )

    grouped = kept.groupby([LOCATION_ID, "date"], sort=False)
    top = (
        grouped["dist"]
        .nlargest(cfg.n_points)
        .groupby(level=[0, 1], sort=False)
        .mean()
        .rename("dist_m")
        .reset_index()
    )
    counts = grouped.agg(
        n_candidates=("dist", "size"), source=("model", "first")
    ).reset_index()
    obs = top.merge(counts, on=[LOCATION_ID, "date"], how="left")
    obs["n_selected"] = np.minimum(obs["n_candidates"], cfg.n_points).astype("float64")
    obs["n_candidates"] = obs["n_candidates"].astype("float64")
    obs = obs.sort_values([LOCATION_ID, "date"]).reset_index(drop=True)

    dpy = BankObservations(obs).to_dist_per_year(within_year="median")

    d = dpy.sort_values([LOCATION_ID, "year"])
    v = d.groupby(LOCATION_ID)["dist_m"].diff() / d.groupby(LOCATION_ID)["year"].diff()
    vmax = v.abs().groupby(d[LOCATION_ID]).max()
    bad = set(vmax[vmax > cfg.farbank_v_limit].index)
    dpy = dpy[~dpy[LOCATION_ID].isin(bad)].reset_index(drop=True)
    logger.info(
        "far-bank guard (|v| > %.0f): %d regions excluded, %s remain",
        cfg.farbank_v_limit,
        len(bad),
        f"{dpy[LOCATION_ID].nunique():,}",
    )
    return kept, obs, dpy
