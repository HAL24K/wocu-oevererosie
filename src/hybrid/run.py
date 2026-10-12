"""Hybrid stage: raw SAM lines + height model in, a hybrid line set out.

Command line::

    uv run python -m src.hybrid --name 20261013-luke                 # Luke's _nearest hybrid
    uv run python -m src.hybrid --name 20261013-rules --nearest-from 20261013-luke --sam-rules
    uv run python -m src.hybrid --name 20261013-rules ... --run-model 20261013-rules

Steps:

1. Luke's ``post_processing_segmentation_lines``: per region, keep only the SAM
   crossing nearest the centreline → ``nearest.gpkg``. Slow; reusable by other
   hybrids through ``--nearest-from``.
2. Optional SAM cleaning (``src.hybrid.sam_cleaning``): our rule stack, the
   side-channel mask, the jetty mask.
3. Luke's points count per region: SAM or height lines (``src.hybrid.luke_v2``).
   Regions that lost all SAM pieces in step 2 take their height lines.

Writes ``03_hybrid/<name>/hybrid.gpkg`` (``lines``, ``lines_all``,
``model_preference``) and ``hybrid.json`` (also added to ``docs/runs/ledger.jsonl``).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd
import pyogrio

from src.hybrid.config import HybridConfig
from src.hybrid.luke_v2 import (
    HEIGHT_MODEL,
    LOCATION_ID,
    SEGMENTATION,
    add_location_ids,
    clean_location,
    hybrid_location,
    validate_segmentation_lines,
)
from src.provenance import describe_files, git_info, now, write_record

logger = logging.getLogger(__name__)

LINE_COLUMNS = [LOCATION_ID, "date", "model", "geometry"]


@dataclass
class HybridResult:
    lines: gpd.GeoDataFrame  # winner lines per region
    lines_all: gpd.GeoDataFrame  # both models' lines
    preference: gpd.GeoDataFrame  # rationale + preference per region
    counts: dict[str, int] = field(default_factory=dict)
    cleaning_stats: pd.DataFrame | None = None


# ── scope ──────────────────────────────────────────────────────────────────────


@dataclass
class Scope:
    polygons_gdf: gpd.GeoDataFrame
    polygons: dict
    centrelines: dict

    @classmethod
    def read(cls, height_gpkg) -> Scope:
        polys = pyogrio.read_dataframe(height_gpkg, layer="vlakken_scope").rename(
            columns={"position_id": LOCATION_ID}
        )
        cls_ = pyogrio.read_dataframe(height_gpkg, layer="centrelines").rename(
            columns={"position_id": LOCATION_ID}
        )
        return cls(
            polygons_gdf=polys,
            polygons=dict(zip(polys[LOCATION_ID], polys.geometry, strict=True)),
            centrelines=dict(zip(cls_[LOCATION_ID], cls_.geometry, strict=True)),
        )


# ── parallel workers (module level so they pickle) ─────────────────────────────

_CTX: dict = {}


def _init(ctx):
    _CTX.update(ctx)


def _nearest_chunk(locs):
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
        seg = _CTX["lines"].get(loc, _CTX["empty_lines"])
        rat, lines_h = hybrid_location(loc, pts, seg, cl)
        rat["geometry"] = _CTX["polygons"][loc]
        rats.append(rat)
        if len(lines_h):
            height.append(lines_h)
    return rats, height


def _chunks(seq, n):
    size = max(1, len(seq) // (n * 8))
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _run_parallel(fn, locs, ctx, workers):
    if workers <= 1:
        _init(ctx)
        return [fn(c) for c in _chunks(locs, 1)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(ctx,)) as ex:
        return list(ex.map(fn, _chunks(locs, workers)))


# ── steps ──────────────────────────────────────────────────────────────────────


def nearest_pieces(cfg: HybridConfig, scope: Scope) -> gpd.GeoDataFrame:
    """Step 1: raw SAM lines → Luke's ``_nearest`` pieces, joined to regions."""
    raw = validate_segmentation_lines(gpd.read_file(cfg.sam_path))
    raw = add_location_ids(raw, scope.polygons_gdf)
    logger.info(
        "step 1: %d line-region rows, %d lines in no region",
        len(raw),
        raw[LOCATION_ID].isna().sum(),
    )
    raw = raw[raw[LOCATION_ID].notna()]
    locs = sorted(raw[LOCATION_ID].unique())[: cfg.limit]
    parts = _run_parallel(
        _nearest_chunk,
        locs,
        {
            "lines": dict(list(raw.groupby(LOCATION_ID))),
            "centrelines": scope.centrelines,
        },
        cfg.workers,
    )
    nearest = gpd.GeoDataFrame(
        [r for p in parts for r in p], geometry="geometry", crs=raw.crs
    )
    # As in Luke's second run: the _nearest pieces are read back and joined to
    # the regions again, so a piece can land in a neighbouring region too.
    seg = add_location_ids(validate_segmentation_lines(nearest), scope.polygons_gdf)
    seg = seg[seg[LOCATION_ID].notna()][LINE_COLUMNS]
    logger.info(
        "step 1: %d nearest pieces in %d regions", len(seg), seg[LOCATION_ID].nunique()
    )
    return gpd.GeoDataFrame(seg, geometry="geometry", crs=raw.crs)


def read_nearest(path) -> gpd.GeoDataFrame:
    seg = pyogrio.read_dataframe(path, layer="lines")
    seg["date"] = pd.to_datetime(seg["date"]).dt.date
    return seg[LINE_COLUMNS]


def clean_sam(
    cfg: HybridConfig, seg: gpd.GeoDataFrame, scope: Scope
) -> tuple[gpd.GeoDataFrame, pd.DataFrame | None]:
    """Step 2: the SAM cleaning options, in a fixed order."""
    from src.hybrid.sam_cleaning import clean_sam_lines, jetty_mask, side_channel_mask
    from src.pipeline.config import ExperimentConfig

    centrelines = gpd.GeoSeries(scope.centrelines, crs=28992)
    centrelines.index.name = LOCATION_ID
    seg = gpd.GeoDataFrame(seg, geometry="geometry", crs=28992).reset_index(drop=True)
    stats = None
    if cfg.sam_rules:
        # The rule parameters are the model stage's defaults.
        rules_cfg = ExperimentConfig(
            experiment=f"hybrid-{cfg.name}",
            data_dir=cfg.data_dir,
            raw_gpkg=cfg.height_gpkg,
            structures_gpkg=cfg.structures_gpkg,
            sample_spacing_m=cfg.sample_spacing_m,
        )
        seg, stats = clean_sam_lines(seg, centrelines, rules_cfg)
    if cfg.side_channel_m is not None:
        legger = gpd.read_file(cfg.legger_gpkg).to_crs(28992)
        seg = side_channel_mask(seg, legger, centrelines, cfg.side_channel_m)
    if cfg.jetty_m is not None:
        kw = gpd.read_file(cfg.structures_gpkg, layer="kunstwerken").to_crs(28992)
        seg = jetty_mask(seg, kw[kw["categorie"] == "steiger_afmeer"], cfg.jetty_m)
    seg = seg[LINE_COLUMNS].copy()
    seg["date"] = pd.to_datetime(seg["date"]).dt.date
    return seg, stats


def points_count(
    cfg: HybridConfig, seg: gpd.GeoDataFrame, regions: list[str], scope: Scope
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Step 3: Luke's preference per region, plus the height lines it built."""
    pts = pyogrio.read_dataframe(cfg.height_gpkg, layer="punten_oever")
    logger.info("step 3: %d height points, %d regions", len(pts), len(regions))
    parts = _run_parallel(
        _hybrid_chunk,
        regions,
        {
            "lines": dict(list(seg.groupby(LOCATION_ID))),
            "empty_lines": seg.iloc[:0],
            "centrelines": scope.centrelines,
            "polygons": scope.polygons,
            "points": dict(list(pts.groupby(LOCATION_ID, sort=False))),
            "empty_points": pts.iloc[:0],
        },
        cfg.workers,
    )
    crs = scope.polygons_gdf.crs
    rationale = gpd.GeoDataFrame(
        [r for p in parts for r in p[0]], geometry="geometry", crs=crs
    )
    height = [h for p in parts for h in p[1]]
    height = (
        gpd.GeoDataFrame(
            pd.concat(height, ignore_index=True), geometry="geometry", crs=crs
        )
        if height
        else gpd.GeoDataFrame(columns=LINE_COLUMNS, geometry="geometry", crs=crs)
    )
    return rationale, height


def apply_height_fallback(rationale: pd.DataFrame, lost: set[str]) -> pd.DataFrame:
    """Mark regions that lost all SAM pieces; drop those the height model can't carry.

    With no SAM lines, Luke's count prefers height whenever the height model has
    at least two years. With fewer it prefers segmentation, which these regions
    no longer have, so they leave the hybrid.
    """
    rationale = rationale.copy()
    rationale["height_fallback"] = rationale[LOCATION_ID].isin(lost)
    unusable = rationale["height_fallback"] & (
        rationale["model_preference"] != "height"
    )
    return rationale[~unusable].reset_index(drop=True)


def winner_lines(
    lines_all: gpd.GeoDataFrame, preference: pd.Series
) -> gpd.GeoDataFrame:
    """Per region, the lines of the preferred model only."""
    pref = lines_all[LOCATION_ID].map(preference)
    keep = ((lines_all["model"] == HEIGHT_MODEL) & pref.eq("height")) | (
        (lines_all["model"] == SEGMENTATION) & pref.eq("segmentation")
    )
    return lines_all[keep]


# ── the stage ──────────────────────────────────────────────────────────────────


def build_hybrid(cfg: HybridConfig, notes: str = "") -> HybridResult:
    missing = cfg.missing_inputs()
    if missing:
        raise FileNotFoundError(
            "Missing inputs:\n" + "\n".join(f"  {p}" for p in missing)
        )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings: dict[str, float] = {}
    mark = [t0]

    def lap(name: str) -> None:
        now_ = time.time()
        timings[name] = round(now_ - mark[0], 1)
        mark[0] = now_
        logger.info("   %s: %.0f s", name, timings[name])

    scope = Scope.read(cfg.height_gpkg)

    if cfg.nearest_from is None:
        seg = nearest_pieces(cfg, scope)
        pyogrio.write_dataframe(seg, cfg.nearest_gpkg, layer="lines", driver="GPKG")
        lap("1 · nearest pieces (Luke step 1)")
    else:
        seg = read_nearest(cfg.nearest_source)
        if cfg.limit:
            keep = sorted(seg[LOCATION_ID].unique())[: cfg.limit]
            seg = seg[seg[LOCATION_ID].isin(keep)]
        logger.info("step 1 reused: %d pieces from %s", len(seg), cfg.nearest_source)
        lap("1 · nearest pieces (reused)")
    regions_before = sorted(seg[LOCATION_ID].unique())

    stats = None
    if cfg.cleans_sam:
        seg, stats = clean_sam(cfg, seg, scope)
        lap("2 · SAM cleaning")
    regions_after = set(seg[LOCATION_ID].unique())
    lost = [r for r in regions_before if r not in regions_after]
    regions = regions_before if cfg.height_fallback else sorted(regions_after)
    logger.info(
        "step 2: %d of %d regions keep SAM pieces; %d lost all%s",
        len(regions_after),
        len(regions_before),
        len(lost),
        " (height fallback)" if cfg.height_fallback and lost else "",
    )

    rationale, height = points_count(cfg, seg, regions, scope)
    lap("3 · points count (Luke step 2)")
    rationale = apply_height_fallback(rationale, set(lost))
    height = height[height[LOCATION_ID].isin(set(rationale[LOCATION_ID]))]
    crs = scope.polygons_gdf.crs
    lines_all = gpd.GeoDataFrame(
        pd.concat([height[LINE_COLUMNS], seg[LINE_COLUMNS]], ignore_index=True),
        geometry="geometry",
        crs=crs,
    )
    lines_all["date"] = pd.to_datetime(lines_all["date"])
    lines = winner_lines(
        lines_all, rationale.set_index(LOCATION_ID)["model_preference"]
    )

    res = HybridResult(
        lines=lines,
        lines_all=lines_all,
        preference=rationale,
        cleaning_stats=stats,
        counts={
            "regions_with_sam_before_cleaning": len(regions_before),
            "regions_with_sam_after_cleaning": len(regions_after),
            "regions_in_hybrid": len(rationale),
            "regions_height_fallback": int(rationale["height_fallback"].sum()),
            "regions_preferring_segmentation": int(
                (rationale["model_preference"] == "segmentation").sum()
            ),
            "regions_preferring_height": int(
                (rationale["model_preference"] == "height").sum()
            ),
            "sam_pieces": len(seg),
            "winner_lines": len(lines),
        },
    )
    write_hybrid(cfg, res)
    lap("4 · write GeoPackage")
    record = {
        "name": cfg.name,
        "created": now(),
        "notes": notes,
        "git": git_info(),
        "inputs": describe_files(cfg.inputs(), cfg.data_dir),
        "settings": cfg.settings(),
        "counts": res.counts,
        "cleaning_rules": stats.to_dict("records") if stats is not None else None,
        "seconds": round(time.time() - t0),
        "timings_s": timings,
    }
    path = write_record(
        record, cfg.out_dir / "hybrid.json", "hybrid", cfg.name, cfg.data_dir
    )
    logger.info("hybrid → %s (%s); record → %s", cfg.hybrid_gpkg, res.counts, path)
    return res


def write_hybrid(cfg: HybridConfig, res: HybridResult) -> None:
    if cfg.hybrid_gpkg.exists():
        cfg.hybrid_gpkg.unlink()
    res.lines.to_file(cfg.hybrid_gpkg, layer="lines", driver="GPKG")
    res.lines_all.to_file(cfg.hybrid_gpkg, layer="lines_all", driver="GPKG")
    res.preference.to_file(cfg.hybrid_gpkg, layer="model_preference", driver="GPKG")
    if res.cleaning_stats is not None:
        res.cleaning_stats.to_csv(cfg.out_dir / "sam_cleaning_stats.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build a hybrid line set from raw SAM + height model."
    )
    ap.add_argument("--name", required=True, help="output folder under data/03_hybrid/")
    ap.add_argument(
        "--notes", default="", help="what this hybrid is for (goes into hybrid.json)"
    )
    ap.add_argument(
        "--sam", help="raw SAM lines (default: 01_raw/sam/…full_aoi.geojson)"
    )
    ap.add_argument("--height-gpkg", help="height-model delivery (default: 0310)")
    ap.add_argument("--nearest-from", help="reuse step 1 of this earlier hybrid")
    ap.add_argument(
        "--sam-rules", action="store_true", help="apply our cleaning rules to SAM"
    )
    ap.add_argument(
        "--side-channel-m",
        type=float,
        help="mask the side-channel register, + buffer (m)",
    )
    ap.add_argument(
        "--jetty-m", type=float, help="mask this many metres around mooring jetties"
    )
    ap.add_argument(
        "--no-height-fallback",
        action="store_true",
        help="drop regions that lose all SAM pieces instead of using their height lines",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N regions only (smoke test)")
    ap.add_argument(
        "--run-model",
        metavar="EXPERIMENT",
        help="then run the model stage on this hybrid as this experiment",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = HybridConfig(
        name=args.name,
        sam_path=args.sam,
        height_gpkg=args.height_gpkg,
        nearest_from=args.nearest_from,
        sam_rules=args.sam_rules,
        side_channel_m=args.side_channel_m,
        jetty_m=args.jetty_m,
        height_fallback=not args.no_height_fallback,
        workers=args.workers,
        limit=args.limit,
    )
    build_hybrid(cfg, notes=args.notes)

    if args.run_model:
        from src.pipeline.config import ExperimentConfig
        from src.pipeline.run import run

        result = run(
            ExperimentConfig(
                experiment=args.run_model,
                hybrid_gpkg=cfg.hybrid_gpkg,
                raw_gpkg=cfg.height_gpkg,
            ),
            notes=args.notes,
        )
        return 0 if result.all_checks_passed else 1
    return 0
