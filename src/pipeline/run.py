"""Run the full erosion prediction pipeline from an :class:`ExperimentConfig`.

This lifts the orchestration that previously lived only in
``notebooks/04_model/20260617a/00_master.ipynb`` — including the four steps
that existed nowhere else: building the prediction start points, assembling
the combined feature table, projecting predicted distances back to
coordinates, and the acceptance checks. Logic is kept identical to the
notebook so a run reproduces the 20260617a reference.

Command line::

    uv run python -m src.pipeline --experiment 20260817a
    uv run python -m src.pipeline --experiment quick --end-year 2035 --no-export

Each run writes its per-step parquets to ``03_features/<experiment>/``, the
model bundle and predictions to ``04_model_outputs/<experiment>/``, and a
self-contained HTML report (figures + tables, browsable like the notebook
output) next to them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.erosion.centerline_utils import (
    build_ref_geom_lookup,
    compute_vvr_crossing_year,
    ensure_location_id_column,
    get_nvo_location_ids,
    point_from_offset,
)
from src.erosion.export import export_predictions
from src.model.export_utils import load_model_bundle
from src.model.predictor import predict_iterative_ml
from src.pipeline.config import ExperimentConfig
from src.pipeline.feature_engineering import build_features
from src.pipeline.region_split import build_region_split
from src.pipeline.train import train_and_save_models
from src.sources import HeightModelPointSource

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Everything a run produced, for inspection or reporting."""

    config: ExperimentConfig
    dist_per_year: pd.DataFrame | None = None
    region_split: pd.DataFrame | None = None
    region_inference: pd.DataFrame | None = None
    region_features: pd.DataFrame | None = None
    region_inference_features: pd.DataFrame | None = None
    train_results: dict[str, Any] | None = None
    predictions: pd.DataFrame | None = None
    vvr_crossing: pd.DataFrame | None = None
    segment_metrics: dict[str, Any] | None = None
    acceptance: list[tuple[str, bool, str]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    output_gpkg: Path | None = None

    @property
    def all_checks_passed(self) -> bool:
        return bool(self.acceptance) and all(ok for _, ok, _ in self.acceptance)


# ── steps lifted from the master notebook ─────────────────────────────────────


def build_start_points(
    region_features: pd.DataFrame,
    region_split_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    inf_meta: pd.DataFrame,
) -> dict[str, dict]:
    """Starting position per location for the iterative prediction.

    Train/test regions start from their last observation (t3); inference-only
    regions from t2. The fallback years (2025 / 2022) match the notebook and
    only apply when a region is missing from the lookup table, which does not
    happen for a normal run.
    """
    start_points: dict[str, dict] = {}
    for loc_id, row in region_features.iterrows():
        t3 = (
            int(region_split_df.loc[loc_id, "t3"])
            if loc_id in region_split_df.index
            else 2025
        )
        start_points[loc_id] = {
            "last_dist": row["dist_t3"],
            "last_year": t3,
            "v_hist": row["v_train"],
        }
    for loc_id, row in inference_df.iterrows():
        if loc_id not in start_points:
            t2 = int(inf_meta.loc[loc_id, "t2"]) if loc_id in inf_meta.index else 2022
            start_points[loc_id] = {
                "last_dist": row["dist_t2"],
                "last_year": t2,
                "v_hist": row["v_train"],
            }
    return start_points


def combine_feature_tables(
    region_features: pd.DataFrame, inference_features: pd.DataFrame
) -> pd.DataFrame:
    """One feature table covering both train/test and inference-only regions.

    Inference-only regions lack ``test_span_yr`` (no t3 observation); the
    notebook backfills it from ``train_span_yr`` so the model sees a complete
    row. Only columns present in both tables are kept.
    """
    filled = inference_features.copy()
    if "test_span_yr" not in filled.columns:
        filled["test_span_yr"] = filled["train_span_yr"]
    shared = [c for c in region_features.columns if c in filled.columns]
    return pd.concat([region_features[shared], filled[shared]])


def project_predictions(
    predictions: pd.DataFrame,
    cl_lookup: dict,
    ref_geom_lookup: dict,
    nvo_ids: set[str],
    crs,
) -> gpd.GeoDataFrame:
    """Turn predicted distances into bank-position points.

    One point per (location_id, year): the centreline offset by the predicted
    distance, on the bank side indicated by the reference geometry.
    """
    records = []
    for row in predictions.itertuples(index=False):
        cline = cl_lookup.get(row.location_id)
        ref = ref_geom_lookup.get(row.location_id)
        point = (
            point_from_offset(cline, row.predicted_dist_m, ref)
            if cline is not None and ref is not None and len(ref) > 0
            else None
        )
        records.append(
            {
                "location_id": row.location_id,
                "year": row.year,
                "predicted_dist_m": row.predicted_dist_m,
                "velocity_m_per_yr": row.velocity_m_per_yr,
                "is_nvo": int(row.location_id in nvo_ids),
                "geometry": point,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def acceptance_checks(
    result: RunResult, bank_positions: gpd.GeoDataFrame
) -> list[tuple[str, bool, str]]:
    """The four checks from the master notebook, as (name, passed, detail)."""
    from pyogrio import list_layers

    cfg = result.config
    checks: list[tuple[str, bool, str]] = []

    n_pred = bank_positions["location_id"].nunique()
    n_expected = len(result.region_features) + len(result.region_inference_features)
    checks.append(
        (
            "prediction locations",
            n_pred >= len(result.region_features),
            f"{n_pred:,} unique (expected ~{n_expected:,})",
        )
    )

    checks.append(
        (
            "is_nvo dtype",
            str(bank_positions["is_nvo"].dtype) in ("int64", "int32"),
            str(bank_positions["is_nvo"].dtype),
        )
    )

    if cfg.export_gpkg and result.output_gpkg and result.output_gpkg.exists():
        layers = {name for name, _ in list_layers(result.output_gpkg)}
        required = {
            "predicted_bank_positions",
            "vvr_rates_of_change",
            "summary_scope",
            "signaleringslijn",
        }
        missing = required - layers
        checks.append(
            (
                "required layers",
                not missing,
                f"missing: {sorted(missing)}" if missing else "all present",
            )
        )
        vvr_out = gpd.read_file(result.output_gpkg, layer="vvr_rates_of_change")
        has_col = "predicted_vvr_crossing_year" in vvr_out.columns
        n_filled = (
            int(vvr_out["predicted_vvr_crossing_year"].notna().sum()) if has_col else 0
        )
        checks.append(
            ("crossing year column", has_col, f"{n_filled:,}/{len(vvr_out):,} filled")
        )
    return checks


# ── the run itself ─────────────────────────────────────────────────────────────


def run(cfg: ExperimentConfig, write_report: bool = True) -> RunResult:
    """Execute the pipeline end to end. Returns everything it produced."""
    missing = cfg.missing_inputs()
    if missing:
        raise FileNotFoundError(
            "Missing inputs:\n" + "\n".join(f"  {p}" for p in missing)
        )
    cfg.features_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_out_dir.mkdir(parents=True, exist_ok=True)
    res = RunResult(config=cfg)

    def step(name):
        logger.info("── %s", name)
        return time.time()

    def done(name, t0):
        res.timings[name] = round(time.time() - t0, 1)
        logger.info("   %s done in %.0fs", name, res.timings[name])

    def cached(path: Path) -> pd.DataFrame | None:
        if cfg.resume and path.exists():
            logger.info("   resume: loading %s", path.name)
            return pd.read_parquet(path)
        return None

    # 1 · bank distances ------------------------------------------------------
    t = step("1 · bank distances")
    dpy_path = cfg.features_dir / "dist_per_year.parquet"
    obs_path = cfg.features_dir / "observations.parquet"
    samples_path = cfg.features_dir / "samples.parquet"
    res.dist_per_year = cached(dpy_path)
    if res.dist_per_year is None:
        if cfg.source == "hybrid":
            from src.pipeline.hybrid_prep import build_observations

            samples, observations, res.dist_per_year = build_observations(cfg)
            samples.to_parquet(samples_path, index=False)
            observations.to_parquet(obs_path, index=False)
        else:
            source = HeightModelPointSource(cfg.raw_gpkg, n_points=cfg.n_points)
            res.dist_per_year = source.load().to_dist_per_year()
        res.dist_per_year.to_parquet(dpy_path, index=False)
    done("1 · bank distances", t)

    # 2 · region split --------------------------------------------------------
    t = step("2 · region split")
    split_path = cfg.features_dir / "region_split.parquet"
    inf_path = cfg.features_dir / "region_inference_only.parquet"
    res.region_split = cached(split_path)
    res.region_inference = cached(inf_path)
    if res.region_split is None or res.region_inference is None:
        res.region_split, res.region_inference = build_region_split(
            res.dist_per_year,
            proc_gpkg=cfg.proc_gpkg,
            test_size=cfg.test_size,
            random_seed=cfg.seed,
        )
        res.region_split.to_parquet(split_path)
        res.region_inference.to_parquet(inf_path)
    done("2 · region split", t)

    # 3 · features -------------------------------------------------------------
    t = step("3 · feature engineering")
    feat_path = cfg.features_dir / "region_features.parquet"
    inf_feat_path = cfg.features_dir / "region_inference_features.parquet"
    res.region_features = cached(feat_path)
    res.region_inference_features = cached(inf_feat_path)
    if res.region_features is None or res.region_inference_features is None:
        res.region_features, res.region_inference_features = build_features(
            res.region_split,
            res.region_inference,
            scope_gpkg=cfg.scope_gpkg,
            veg_gpkg=cfg.veg_gpkg,
            lu_gpkg=cfg.lu_gpkg,
            soil_gpkg=cfg.soil_gpkg,
            stations_gpkg=cfg.stations_gpkg,
            discharge_dir=cfg.discharge_dir,
            reference_features_v2=cfg.reference_features_v2,
        )
        res.region_features.to_parquet(feat_path)
        res.region_inference_features.to_parquet(inf_feat_path)
    done("3 · feature engineering", t)

    # 3b · trajectory features (hybrid source) ---------------------------------
    extra_features: list[str] = []
    if cfg.source == "hybrid" and cfg.trajectory_features:
        t = step("3b · trajectory features")
        from src.pipeline.trajectory import TRAJ_FEATS2, trajectory_features

        observations = pd.read_parquet(cfg.features_dir / "observations.parquet")
        for frame, meta in (
            (res.region_features, res.region_split),
            (res.region_inference_features, res.region_inference),
        ):
            missing = [c for c in TRAJ_FEATS2 if c not in frame.columns]
            if not missing:
                continue
            origins = meta["t2"].reindex(frame.index).astype(int)
            traj = trajectory_features(observations, origins)
            frame[TRAJ_FEATS2] = traj.reindex(frame.index)[TRAJ_FEATS2].fillna(0.0)
        extra_features = list(TRAJ_FEATS2)
        done("3b · trajectory features", t)

    if cfg.source == "hybrid":
        # hybrid regions extend beyond the reference feature coverage; the
        # linear baselines cannot digest the resulting NaNs (the experiment
        # harness and prep script filled them the same way).
        for frame in (res.region_features, res.region_inference_features):
            num = frame.select_dtypes("number").columns
            frame[num] = frame[num].fillna(0.0)
        res.region_features.to_parquet(feat_path)
        res.region_inference_features.to_parquet(inf_feat_path)

    # 4 · train -----------------------------------------------------------------
    t = step("4 · train")
    res.train_results = train_and_save_models(
        res.region_features,
        cfg.model_out_dir,
        seed=cfg.seed,
        extra_features=extra_features,
        val_frac=cfg.val_frac,
    )
    done("4 · train", t)

    # 5 · predict ----------------------------------------------------------------
    t = step("5 · iterative prediction")
    bundle = load_model_bundle(cfg.model_out_dir)
    start_points = build_start_points(
        res.region_features,
        res.region_split,
        res.region_inference_features,
        res.region_inference,
    )
    all_features = combine_feature_tables(
        res.region_features, res.region_inference_features
    )
    res.predictions = predict_iterative_ml(
        bundle=bundle,
        features_df=all_features,
        start_points=start_points,
        model_name=cfg.model_name,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        step=1,
        rolling=True,
    )
    res.predictions.to_parquet(cfg.model_out_dir / "predictions.parquet", index=False)
    done("5 · iterative prediction", t)

    # 6 · geometry & crossing years ----------------------------------------------
    t = step("6 · geometry")
    scope_raw = ensure_location_id_column(
        gpd.read_file(cfg.raw_gpkg, layer="vlakken_scope")
    )
    centerlines = ensure_location_id_column(
        gpd.read_file(cfg.raw_gpkg, layer="centrelines")
    )
    bank_points = ensure_location_id_column(
        gpd.read_file(cfg.raw_gpkg, layer="punten_oever")
    )
    vvr = gpd.read_file(cfg.proc_gpkg, layer="vvr_rates_of_change")
    scope = ensure_location_id_column(
        gpd.read_file(cfg.proc_gpkg, layer="summary_scope")
    )
    signalering = gpd.read_file(
        cfg.signalering_gpkg, layer=cfg.signalering_layer
    ).to_crs(28992)

    cl_lookup = centerlines.set_index("location_id")["geometry"].to_dict()
    ref_geom_lookup = build_ref_geom_lookup(bank_points, n_points=cfg.n_points)
    nvo_ids = get_nvo_location_ids(vvr, scope)

    bank_positions = project_predictions(
        res.predictions, cl_lookup, ref_geom_lookup, nvo_ids, centerlines.crs
    )
    res.vvr_crossing = compute_vvr_crossing_year(
        res.predictions,
        nvo_ids,
        centerlines,
        scope_raw,
        signalering,
        reference_year=cfg.start_year - 1,
    )
    done("6 · geometry", t)

    # 7 · export -------------------------------------------------------------------
    if cfg.export_gpkg:
        t = step("7 · export GeoPackage")
        res.output_gpkg = export_predictions(
            base_gpkg=cfg.proc_gpkg,
            output_gpkg=cfg.output_gpkg,
            predicted_bank_positions=bank_positions,
            vvr_crossing=res.vvr_crossing,
            scope_raw=scope_raw,
            signaleringslijn=signalering,
        )
        done("7 · export GeoPackage", t)

    # 8 · segment-horizon artifact (hybrid source) -------------------------------
    if cfg.source == "hybrid" and cfg.build_segments:
        t = step("8 · segment-horizon artifact")
        from src.pipeline.segments import build_segment_artifact

        kept_line_idx = pd.read_parquet(
            cfg.features_dir / "samples.parquet", columns=["line_idx"]
        )["line_idx"].unique()
        res.segment_metrics = build_segment_artifact(
            cfg, res.region_split, res.region_features, kept_line_idx
        )
        done("8 · segment-horizon artifact", t)

    res.acceptance = acceptance_checks(res, bank_positions)
    for name, ok, detail in res.acceptance:
        logger.info("   [%s] %s — %s", "OK" if ok else "FAIL", name, detail)

    if write_report:
        from src.pipeline.report import write_run_report

        write_run_report(res, bank_positions)
        logger.info("report → %s", cfg.report_path)

    return res


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run the erosion prediction pipeline.")
    ap.add_argument("--experiment", required=True, help="run name, e.g. 20260817a")
    ap.add_argument(
        "--source",
        choices=["hybrid", "points"],
        default="hybrid",
        help="'hybrid' (default) = line delivery, cleaning rules, trajectory "
        "features, honest validation, segment artifact; 'points' = the "
        "pre-2026-08 point-cloud pipeline",
    )
    ap.add_argument("--start-year", type=int, default=2026)
    ap.add_argument("--end-year", type=int, default=2050)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-export", action="store_true", help="skip the large output GeoPackage"
    )
    ap.add_argument(
        "--resume", action="store_true", help="reuse existing per-step parquets"
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = ExperimentConfig(
        experiment=args.experiment,
        source=args.source,
        start_year=args.start_year,
        end_year=args.end_year,
        seed=args.seed,
        export_gpkg=not args.no_export,
        resume=args.resume,
    )
    result = run(cfg)
    return 0 if result.all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
