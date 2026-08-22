"""Step 08 (hybrid source) — the segment-horizon artifact.

Graduated from tracks 2+3 of the loop-engineering experiment (k6): each
region is cut into ``segment_R`` station bins; every segment gets its own
cleaned year series (furthest-3 of dense samples, medians within a year),
and a LightGBM model is trained on all >= ``horizon_min_years`` increments
of the train regions (span-weighted, honest validation split). Evaluation
uses one held-out increment per test region-segment; the operational output
is a forward forecast from each segment's last observation.

Outputs (under ``cfg.model_out_dir``):
  segment_predictions.parquet  per (region, segment): last observed position
                               and the predicted position horizon years out
                               — the input for a per-segment VVR alert
  segment_metrics.json         honest test metrics of the segment model
"""

from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import shapely
from sklearn.metrics import mean_absolute_error, r2_score

from src.pipeline.config import ExperimentConfig
from src.pipeline.hybrid_prep import build_structures_geom, load_lines, sample_lines
from src.pipeline.trajectory import TRAJ_FEATS2, traj_row
from src.sources.geometry import LOCATION_ID, ScopeGeometry

logger = logging.getLogger(__name__)

SEG_KEY = [LOCATION_ID, "seg"]
SEG_FEATS = [
    "dist_t2",
    "v_train",
    "train_span_yr",
    "test_span_yr",
    "seg_center",
    "seg_edge",
    "n_seg_years",
]
REGION_CONTEXT = [
    "is_nvo",
    "river_enc",
    "vegetation_class_enc",
    "land_use_enc",
    "soil_group_enc",
    "bend_exposure_n5",
    "bend_exposure_n8",
    "erosion_vol_rate_t1",
    "n_events_t2",
    "max_rise_rate_t2",
    "drawdown_index_t2",
    "flood_days_t2",
]


def _segment_yearly(cfg: ExperimentConfig, kept_line_idx) -> pd.DataFrame:
    """Dense-resample the kept lines; per (region, seg, year) median scalar."""
    geometry = ScopeGeometry(centreline_gpkg=cfg.raw_gpkg)
    lines = load_lines(cfg, geometry)
    lines = lines.loc[lines.index.intersection(pd.Index(kept_line_idx))]
    dense = sample_lines(lines, geometry.centrelines, cfg.segment_n_samples)

    geom = build_structures_geom(cfg, geometry.centrelines)
    pts = shapely.points(dense["x"].values, dense["y"].values)
    dense = dense[~shapely.contains(geom, pts)]

    R = cfg.segment_R
    dense = dense.assign(seg=np.minimum((dense["station"] * R).astype(int), R - 1))
    g = dense.groupby([*SEG_KEY, "date"], sort=False)
    top = (
        g["dist"]
        .nlargest(3)
        .groupby(level=[0, 1, 2], sort=False)
        .mean()
        .rename("dist_m")
        .reset_index()
    )
    counts = g.agg(n=("dist", "size"), model=("model", "first")).reset_index()
    seg_obs = top.merge(counts, on=[*SEG_KEY, "date"], how="left")
    seg_obs = seg_obs[seg_obs["n"] >= 3].reset_index(drop=True)
    seg_obs["year"] = seg_obs["date"].dt.year
    return seg_obs


def _traj_for(seg_obs: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """traj2 per (region, seg) row, from that segment's series year <= t2."""
    o = seg_obs.copy()
    o["t"] = o["date"].map(pd.Timestamp.toordinal) / 365.25
    o["sam"] = (o["model"] == "segmentation").astype(float)
    o = o.sort_values("t")
    by_key = {k: g for k, g in o.groupby(SEG_KEY, sort=False)}  # noqa: C416

    rows = []
    for loc, seg, t2 in zip(
        keys[LOCATION_ID].values, keys["seg"].values, keys["t2"].values, strict=True
    ):
        g = by_key.get((loc, seg))
        if g is None:
            rows.append(dict.fromkeys(TRAJ_FEATS2, np.nan))
            continue
        h = g[g["year"] <= t2]
        rows.append(traj_row(h["t"].values, h["dist_m"].values, h["sam"].values))
    return pd.DataFrame(rows, index=keys.index)[TRAJ_FEATS2]


def _increment_rows(
    yearly: pd.DataFrame, H: int, test_ids: set, train_ids: set
) -> pd.DataFrame:
    """Training rows: all >=H-year increments; test: latest >=H increment."""
    rows = []
    for (loc, seg), g in yearly.groupby(SEG_KEY, sort=False):
        ys = g["year"].values
        ds = g["dist_m"].values
        n = len(ys)
        if n < 3:
            continue
        if loc in test_ids:
            cand = [i for i in range(1, n - 1) if ys[i] <= ys[-1] - H]
            pairs = [(cand[-1], n - 1)] if cand else []
            split = "test"
        elif loc in train_ids:
            pairs = [
                (i, j)
                for i in range(1, n - 1)
                for j in range(i + 1, n)
                if ys[j] - ys[i] >= H
            ]
            split = "train"
        else:
            continue
        for i, j in pairs:
            rows.append(
                {
                    LOCATION_ID: loc,
                    "seg": seg,
                    "t2": int(ys[i]),
                    "dist_t2": ds[i],
                    "train_span_yr": ys[i] - ys[i - 1],
                    "test_span_yr": ys[j] - ys[i],
                    "v_train": (ds[i] - ds[i - 1]) / (ys[i] - ys[i - 1]),
                    "v_test": (ds[j] - ds[i]) / (ys[j] - ys[i]),
                    "n_seg_years": n,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def build_segment_artifact(
    cfg: ExperimentConfig,
    region_split: pd.DataFrame,
    region_features: pd.DataFrame,
    kept_line_idx,
) -> dict:
    """Train the segment-horizon model, evaluate honestly, forecast forward."""
    R, H = cfg.segment_R, cfg.horizon_min_years
    seg_obs = _segment_yearly(cfg, kept_line_idx)
    yearly = seg_obs.groupby([*SEG_KEY, "year"])["dist_m"].median().reset_index()

    test_ids = set(region_split.index[region_split["split"] == "test"])
    train_ids = set(region_split.index[region_split["split"] == "train"])
    frame = _increment_rows(yearly, H, test_ids, train_ids)
    frame["seg_center"] = (frame["seg"] + 0.5) / R
    frame["seg_edge"] = np.minimum(frame["seg_center"], 1 - frame["seg_center"])

    ctx_cols = [c for c in REGION_CONTEXT if c in region_features.columns]
    ctx = region_features[ctx_cols].copy()
    if "is_nvo" in ctx:
        ctx["is_nvo"] = ctx["is_nvo"].astype(float)
    frame = frame.join(ctx, on=LOCATION_ID)

    traj = _traj_for(seg_obs, frame)
    frame = pd.concat([frame, traj], axis=1)
    feat_names = SEG_FEATS + ctx_cols + TRAJ_FEATS2
    frame[feat_names] = frame[feat_names].astype(float).fillna(0.0)

    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    logger.info(
        "segment-horizon: R=%d H>=%d · %s train rows · %s test rows",
        R,
        H,
        f"{len(train):,}",
        f"{len(test):,}",
    )

    rng = np.random.default_rng(cfg.seed)
    train_locs = train[LOCATION_ID].unique()
    val_locs = set(rng.choice(train_locs, int(len(train_locs) * cfg.val_frac), False))
    val_mask = train[LOCATION_ID].isin(val_locs)
    fit_rows, val_rows = train[~val_mask], train[val_mask]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=cfg.seed,
        verbose=-1,
    )
    model.fit(
        fit_rows[feat_names],
        fit_rows["v_test"],
        sample_weight=np.clip(fit_rows["test_span_yr"], 2, 5),
        eval_set=[(val_rows[feat_names], val_rows["v_test"])],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    pred = model.predict(test[feat_names])
    y = test["v_test"].values
    tail = y > 2.0
    pos_err = np.abs((y - pred) * test["test_span_yr"].values)
    metrics = {
        "R": R,
        "horizon_min_years": H,
        "n_train": int(len(train)),
        "n_val": int(len(val_rows)),
        "n_test": int(len(test)),
        "test_mae": float(mean_absolute_error(y, pred)),
        "test_r2": float(r2_score(y, pred)),
        "test_tail_mae": (
            float(mean_absolute_error(y[tail], pred[tail])) if tail.any() else None
        ),
        "tail_n": int(tail.sum()),
        "naive_mae": float(np.abs(y - fit_rows["v_test"].mean()).mean()),
        "persist_mae": float(mean_absolute_error(y, test["v_train"])),
        "pos_err_median_m": float(np.median(pos_err)),
        "pos_err_p90_m": float(np.percentile(pos_err, 90)),
    }
    logger.info(
        "segment-horizon test: MAE %.3f · tail %.3f (n=%d) · R² %.3f"
        " · pos-err med %.2f m",
        metrics["test_mae"],
        metrics["test_tail_mae"] or float("nan"),
        metrics["tail_n"],
        metrics["test_r2"],
        metrics["pos_err_median_m"],
    )

    # ── forward forecast from each segment's last observation ────────────────
    fwd_rows = []
    for (loc, seg), g in yearly.groupby(SEG_KEY, sort=False):
        ys = g["year"].values
        ds = g["dist_m"].values
        if len(ys) < 2:
            continue
        fwd_rows.append(
            {
                LOCATION_ID: loc,
                "seg": seg,
                "t2": int(ys[-1]),
                "dist_t2": ds[-1],
                "train_span_yr": ys[-1] - ys[-2],
                "test_span_yr": float(H),
                "v_train": (ds[-1] - ds[-2]) / (ys[-1] - ys[-2]),
                "n_seg_years": len(ys),
            }
        )
    fwd = pd.DataFrame(fwd_rows)
    fwd["seg_center"] = (fwd["seg"] + 0.5) / R
    fwd["seg_edge"] = np.minimum(fwd["seg_center"], 1 - fwd["seg_center"])
    fwd = fwd.join(ctx, on=LOCATION_ID)
    fwd = pd.concat([fwd, _traj_for(seg_obs, fwd)], axis=1)
    fwd[feat_names] = fwd.reindex(columns=feat_names).astype(float).fillna(0.0)
    fwd["v_pred"] = model.predict(fwd[feat_names])
    fwd["last_year"] = fwd["t2"]
    fwd["last_dist_m"] = fwd["dist_t2"]
    fwd["pred_dist_m_horizon"] = fwd["dist_t2"] + fwd["v_pred"] * H
    out_cols = [
        LOCATION_ID,
        "seg",
        "seg_center",
        "last_year",
        "last_dist_m",
        "v_pred",
        "pred_dist_m_horizon",
        "n_seg_years",
    ]
    out_path = cfg.model_out_dir / "segment_predictions.parquet"
    fwd[out_cols].to_parquet(out_path, index=False)
    (cfg.model_out_dir / "segment_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    logger.info(
        "segment forecast → %s (%s segments, %s regions)",
        out_path.name,
        f"{len(fwd):,}",
        f"{fwd[LOCATION_ID].nunique():,}",
    )
    return metrics
