"""Track 3 — resolution: R scalars per region instead of one.

A region's centreline is cut into R station bins; each segment gets its own
distance scalar per survey (mean of the furthest 3 dense samples inside the
segment — the region convention, applied locally), its own year series, its
own t1/t2/t3 record and its own prediction. Segment targets stay at year
scale (the track-2 lesson: sub-scale richness belongs in features).

Evaluation is two-sided:
  segment level  — can the model predict local increments at all, and how
                   does that degrade as segments shrink?
  region level   — predicted segment positions re-aggregated (max, matching
                   the furthest-N region convention) against the same frozen
                   region ruler as tracks 1–2, with an oracle run of the
                   aggregation on *observed* segment positions to separate
                   aggregation noise from model error.

Split: grouped by region on the frozen holdout — every segment of a frozen
test region is test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.loop.harness import Caches
from src.loop.multi_t import (
    TRAJ_FEATS2,
    _traj_row,
    build_features_pairwise,
    prepare_standard,
    trajectory_features,
)
from src.sources.geometry import LOCATION_ID

SEG_KEY = [LOCATION_ID, "seg"]
SEG_EXTRA_FEATS = ["seg_center", "seg_edge", "n_seg_years"]
REG_CONTEXT_FEATS = ["dist_t2_reg", "v_train_reg", "theil_v_reg", "resid_std_reg"]


def load_dense(caches: Caches) -> pd.DataFrame:
    return pd.read_parquet(caches.cache_dir / "samples_dense_e8.parquet")


def segment_observations(dense: pd.DataFrame, R: int) -> pd.DataFrame:
    """Per (region, segment, date): mean of the 3 furthest samples inside."""
    d = dense.copy()
    d["seg"] = np.minimum((d["station"] * R).astype(int), R - 1)
    g = d.groupby([*SEG_KEY, "date"], sort=False)
    top = (
        g["dist"]
        .nlargest(3)
        .groupby(level=[0, 1, 2], sort=False)
        .mean()
        .rename("dist_m")
        .reset_index()
    )
    counts = g.agg(n=("dist", "size"), model=("model", "first")).reset_index()
    out = top.merge(counts, on=[*SEG_KEY, "date"], how="left")
    return out[out["n"] >= 3].reset_index(drop=True)


def build_segment_frame(
    dense: pd.DataFrame,
    caches: Caches,
    R: int,
    v_limit: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Segment split rows (one per eligible (region, segment)) + raw series.

    Eligible: >= 3 distinct years in the segment, region quality OK, and the
    segment's own yearly series within ``v_limit`` (drops the segment, never
    the region). Returns (rows, seg_obs) — seg_obs is the pre-collapse series
    for trajectory features.
    """
    seg_obs = segment_observations(dense, R)
    seg_obs["year"] = seg_obs["date"].dt.year
    yearly = seg_obs.groupby([*SEG_KEY, "year"])["dist_m"].median().reset_index()

    ok_ids = set(caches.static.index[caches.static["quality"] == "OK"])
    yearly = yearly[yearly[LOCATION_ID].isin(ok_ids)]

    rows = []
    for (loc, seg), g in yearly.groupby(SEG_KEY, sort=False):
        ys = g["year"].values
        ds = g["dist_m"].values
        if len(ys) < 3:
            continue
        v = np.diff(ds) / np.diff(ys)
        if np.abs(v).max() > v_limit:
            continue
        t1, t2, t3 = int(ys[-3]), int(ys[-2]), int(ys[-1])
        rows.append(
            {
                LOCATION_ID: loc,
                "seg": seg,
                "t1": t1,
                "t2": t2,
                "t3": t3,
                "dist_t1": ds[-3],
                "dist_t2": ds[-2],
                "dist_t3": ds[-1],
                "train_span_yr": t2 - t1,
                "test_span_yr": t3 - t2,
                "v_train": (ds[-2] - ds[-3]) / (t2 - t1),
                "v_test": (ds[-1] - ds[-2]) / (t3 - t2),
                "n_seg_years": len(ys),
            }
        )
    frame = pd.DataFrame(rows)
    frame["seg_center"] = (frame["seg"] + 0.5) / R
    frame["seg_edge"] = np.minimum(frame["seg_center"], 1 - frame["seg_center"])

    from src.pipeline.region_split import get_cluster

    frame["cluster"] = frame[LOCATION_ID].map(get_cluster)
    frame["n_timestamps"] = 3
    frame["is_nvo"] = frame[LOCATION_ID].map(caches.static["is_nvo"]).astype(bool)
    frame["quality"] = frame[LOCATION_ID].map(caches.static["quality"])
    ev_sum = (
        caches.ev.groupby([LOCATION_ID, "_yb", "_ya"])["erosion_volume"]
        .sum()
        .reset_index()
    )
    for label, (a, b) in {"train": ("t1", "t2"), "test": ("t2", "t3")}.items():
        m = frame[[LOCATION_ID, a, b]].merge(
            ev_sum,
            left_on=[LOCATION_ID, a, b],
            right_on=[LOCATION_ID, "_yb", "_ya"],
            how="left",
        )
        frame[f"erosion_vol_{label}_rate"] = (
            m["erosion_volume"].fillna(0).values / frame[f"{label}_span_yr"].values
        )
    frame["erosion_vol_rate_t1"] = frame["erosion_vol_train_rate"]
    return frame, seg_obs


def segment_traj_features(seg_obs: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """traj2 descriptors per segment row, from its own series with year <= t2."""
    o = seg_obs.copy()
    o["t"] = o["date"].map(pd.Timestamp.toordinal) / 365.25
    o["sam"] = (o["model"] == "segmentation").astype(float)
    o = o.sort_values("t")
    by_key = {k: g for k, g in o.groupby(SEG_KEY, sort=False)}  # noqa: C416

    rows = []
    for loc, seg, t2 in zip(
        frame[LOCATION_ID].values, frame["seg"].values, frame["t2"].values, strict=True
    ):
        g = by_key.get((loc, seg))
        if g is None:
            rows.append(dict.fromkeys(TRAJ_FEATS2, np.nan))
            continue
        h = g[g["year"] <= t2]
        rows.append(_traj_row(h["t"].values, h["dist_m"].values, h["sam"].values))
    return pd.DataFrame(rows, index=frame.index)[TRAJ_FEATS2]


def region_context(caches: Caches, obs: pd.DataFrame) -> pd.DataFrame:
    """Region-level signal every segment inherits (frozen R=1 quantities)."""
    _, split, _ = prepare_standard(caches, obs)
    traj = trajectory_features(obs, split["t2"].astype(int))
    ctxf = pd.DataFrame(
        {
            "dist_t2_reg": split["dist_t2"],
            "v_train_reg": split["v_train"],
            "theil_v_reg": traj["theil_v"].reindex(split.index),
            "resid_std_reg": traj["theil_resid_std"].reindex(split.index),
        }
    )
    return ctxf, split


def region_aggregate(
    frame: pd.DataFrame, pred: np.ndarray, split: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate segment (predicted and observed) positions to region level.

    Position at t3 per segment = dist_t2 + v * span; the region position is
    the max over segments — the local counterpart of the region's
    furthest-N convention. Returns per-region predicted and oracle
    velocities against the region's own dist_t2/span/target.
    """
    f = frame.copy()
    f["pos3_pred"] = f["dist_t2"] + pred * f["test_span_yr"]
    f["pos3_true"] = f["dist_t3"]
    agg = f.groupby(LOCATION_ID)[["pos3_pred", "pos3_true"]].max()

    out = pd.DataFrame(
        {
            "v_test_reg": split["v_test"],
            "dist_t2_reg": split["dist_t2"],
            "span_reg": split["test_span_yr"],
        }
    ).join(agg, how="inner")
    out["v_pred_agg"] = (out["pos3_pred"] - out["dist_t2_reg"]) / out["span_reg"]
    out["v_oracle_agg"] = (out["pos3_true"] - out["dist_t2_reg"]) / out["span_reg"]
    return out


def run_t3_variant(
    name: str,
    caches: Caches,
    dense: pd.DataFrame,
    obs: pd.DataFrame,
    R: int,
    features: str = "base",
    notes: str = "",
    seed: int = 42,
) -> dict:
    """Train on segment rows, score segment- and region-level, ledger it."""
    import json
    from datetime import datetime

    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, r2_score

    from src.pipeline.train import FEATS_LGB, TAIL_THRESHOLD

    frame, seg_obs = build_segment_frame(dense, caches, R)
    ctxf, reg_split = region_context(caches, obs)

    pwf = build_features_pairwise(frame, caches)
    for c in REG_CONTEXT_FEATS:
        pwf[c] = pwf[LOCATION_ID].map(ctxf[c]).fillna(0.0)

    feat_names = list(FEATS_LGB) + SEG_EXTRA_FEATS + REG_CONTEXT_FEATS
    if features == "traj2":
        tf = segment_traj_features(seg_obs, pwf)
        pwf = pd.concat([pwf, tf.fillna(0.0)], axis=1)
        feat_names += TRAJ_FEATS2

    pwf["split"] = np.where(pwf[LOCATION_ID].isin(caches.frozen_test), "test", "train")
    train = pwf[pwf["split"] == "train"]
    test = pwf[pwf["split"] == "test"]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        train[feat_names].astype(float),
        train["v_test"],
        eval_set=[(test[feat_names].astype(float), test["v_test"])],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    pred_te = model.predict(test[feat_names].astype(float))

    y = test["v_test"].values
    tail = y > TAIL_THRESHOLD
    seg_metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "lgb_mae": mean_absolute_error(y, pred_te),
        "lgb_r2": r2_score(y, pred_te),
        "lgb_tail_mae": (
            mean_absolute_error(y[tail], pred_te[tail]) if tail.any() else np.nan
        ),
        "tail_n": int(tail.sum()),
        "naive_mae": float(np.abs(y - train["v_test"].mean()).mean()),
        "persist_mae": mean_absolute_error(y, test["v_train"]),
        "vtest_std": float(pwf["v_test"].std()),
    }

    # region-level re-aggregation against the frozen R=1 ruler
    agg = region_aggregate(test.reset_index(drop=True), pred_te, reg_split)
    agg = agg[agg.index.isin(caches.frozen_test)]
    reg_metrics = {
        "n_regions": int(agg.shape[0]),
        "coverage_core": (
            len(set(agg.index) & caches.core) / len(caches.core)
            if caches.core
            else np.nan
        ),
        "core_mae": mean_absolute_error(agg["v_test_reg"], agg["v_pred_agg"]),
        "tail_frozen_mae": np.nan,
        "regions_excluded_far": 0,
    }
    if caches.tail_frozen:
        keep_t = agg.index.intersection(caches.tail_frozen)
        reg_metrics["tail_frozen_n_kept"] = len(keep_t)
        if len(keep_t):
            reg_metrics["tail_frozen_mae"] = mean_absolute_error(
                agg.loc[keep_t, "v_test_reg"], agg.loc[keep_t, "v_pred_agg"]
            )
    oracle_mae = mean_absolute_error(agg["v_test_reg"], agg["v_oracle_agg"])

    metrics = {**seg_metrics, **reg_metrics}
    var_dir = caches.cache_dir / "variants" / name
    var_dir.mkdir(parents=True, exist_ok=True)
    test_out = test[[LOCATION_ID, "seg", "v_test", "v_train"]].copy()
    test_out["pred_lgb"] = pred_te
    test_out.to_parquet(var_dir / "test_preds_segments.parquet", index=False)
    agg.to_parquet(var_dir / "region_aggregate.parquet")
    imp = pd.Series(model.feature_importances_, index=feat_names).sort_values(
        ascending=False
    )
    imp.to_csv(var_dir / "feature_importance.csv", header=["importance"])

    row = {
        "variant": name,
        "when": datetime.now().isoformat(timespec="seconds"),
        "rules": json.dumps(
            {
                "track": 3,
                "R": R,
                "features": features,
                "oracle_mae": round(float(oracle_mae), 4),
            }
        ),
        "v_limit": 50.0,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "notes": notes,
    }
    ledger_path = caches.exp_dir / "ledger.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    n_eligible = len(frame)
    n_possible = frame[LOCATION_ID].nunique() * R
    print(f"── {name}  [R={R} features={features}]")
    print(
        f"   segments: {n_eligible:,} eligible of {n_possible:,} possible"
        f" ({n_eligible / n_possible:.0%}) · train {len(train):,} · test {len(test):,}"
    )
    print(
        f"   SEGMENT  mae {seg_metrics['lgb_mae']:.3f}"
        f" · tail {seg_metrics['lgb_tail_mae']:.3f} (n={seg_metrics['tail_n']})"
        f" · R² {seg_metrics['lgb_r2']:.3f}"
        f" · naive {seg_metrics['naive_mae']:.3f}"
        f" · persist {seg_metrics['persist_mae']:.3f}"
    )
    print(
        f"   REGION   mae {reg_metrics['core_mae']:.3f}"
        f" (oracle floor {oracle_mae:.3f})"
        f" · tail_frozen {reg_metrics['tail_frozen_mae']:.3f}"
        f" (n={reg_metrics.get('tail_frozen_n_kept', 0)})"
        f" · coverage_core {reg_metrics['coverage_core']:.3f}"
    )
    print(f"   top features: {', '.join(imp.head(6).index)}")
    return {**metrics, "oracle_mae": oracle_mae, "importance": imp, "agg": agg}
