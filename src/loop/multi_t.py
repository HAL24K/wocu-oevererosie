"""Track 2 — multi-t: use the whole survey history, not just t1/t2/t3.

Three levers, separable so the ledger can attribute gains:

  features  "base" | "traj"   — add trajectory descriptors computed from all
                                surveys at or before the forecast origin
                                (Theil–Sen slope, residual spread,
                                acceleration, span, density, recency, source
                                mix). Leakage-safe by construction: only
                                surveys with year <= t2 enter.
  training  "standard" | "pairwise" — one row per region (the t2→t3
                                increment, as today) or one row per
                                consecutive increment in the region's
                                history (train regions only; the test rows
                                stay the standard t2→t3 increments).
  target    "year" | "date"   — the year-collapsed v_test (comparable with
                                track 1) or the date-true last increment:
                                position change between the last pre-cutoff
                                survey and the t3-year surveys, divided by
                                the actual time span.

All runs keep the track-1 e8 cleaning, the frozen split, the |v|>50 region
filter and the ledger contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.loop.harness import (
    Caches,
    aggregate_observations,
    build_features_fast,
    build_split_frame,
    farbank_region_filter,
    to_dist_per_year,
)
from src.loop.rules import RuleContext, apply_rules
from src.sources.geometry import LOCATION_ID

E8_RULES = [
    ("structure_mask", {}),
    ("max_tortuosity_line", {"max_tort": 3.0}),
    ("near_bank_line", {"frac": 0.25, "min_ref": 30.0}),
    ("maze_survey", {"max_ratio": 1.8, "min_iqr": 20.0}),
    ("fragment_survey", {"min_cov": 0.25}),
    ("min_samples_survey", {"min_n": 8}),
    (
        "temporal_outlier_survey",
        {"max_dev": 15.0, "min_surveys": 3, "detrend": True, "protect_min_years": 3},
    ),
]

TRAJ_FEATS = [
    "n_hist",
    "nyears_hist",
    "span_hist_yr",
    "theil_v",
    "theil_resid_std",
    "accel_v",
    "last_gap_yr",
    "frac_sam",
]


def load_obs_e8(caches: Caches, structures) -> pd.DataFrame:
    """e8-cleaned observations (one row per region-survey), cached on disk."""
    path = caches.cache_dir / "obs_e8.parquet"
    if path.exists():
        return pd.read_parquet(path)
    ctx = RuleContext(
        line_metrics=caches.line_metrics,
        cl_len=caches.static["cl_len"],
        structures=structures,
    )
    s = apply_rules(caches.samples, ctx, E8_RULES)
    obs = aggregate_observations(s)
    obs.to_parquet(path, index=False)
    return obs


def _theil(t: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Theil–Sen slope and residuals; (nan, zeros) when underdetermined."""
    if len(t) < 2 or t.max() == t.min():
        return np.nan, np.zeros_like(y)
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    ok = dt != 0
    slope = float(np.median((y[j] - y[i])[ok] / dt[ok]))
    resid = y - (slope * t + np.median(y - slope * t))
    return slope, resid


def _traj_row(t: np.ndarray, y: np.ndarray, sam: np.ndarray) -> dict:
    slope, resid = _theil(t, y)
    half = t.min() + (t.max() - t.min()) / 2
    lo, hi = t <= half, t > half
    accel = np.nan
    if lo.sum() >= 2 and hi.sum() >= 2:
        s1, _ = _theil(t[lo], y[lo])
        s2, _ = _theil(t[hi], y[hi])
        accel = s2 - s1
    return {
        "n_hist": len(t),
        "nyears_hist": len(np.unique(np.floor(t))),
        "span_hist_yr": float(t.max() - t.min()),
        "theil_v": slope,
        "theil_resid_std": float(resid.std()) if len(t) >= 3 else np.nan,
        "accel_v": accel,
        "last_gap_yr": float(t[-1] - t[-2]) if len(t) >= 2 else np.nan,
        "frac_sam": float(sam.mean()),
    }


def trajectory_features(obs: pd.DataFrame, origin_year: pd.Series) -> pd.DataFrame:
    """Per-region trajectory descriptors from surveys with year <= origin.

    ``origin_year``: forecast origin (t2) per location_id. Rows for regions
    absent from it are dropped.
    """
    o = obs[[LOCATION_ID, "date", "dist_m", "source"]].copy()
    o["origin"] = o[LOCATION_ID].map(origin_year)
    o = o[o["date"].dt.year <= o["origin"]].sort_values([LOCATION_ID, "date"])
    o["t"] = o["date"].map(pd.Timestamp.toordinal) / 365.25
    o["sam"] = (o["source"] == "segmentation").astype(float)

    rows = {}
    for loc, g in o.groupby(LOCATION_ID, sort=False):
        rows[loc] = _traj_row(g["t"].values, g["dist_m"].values, g["sam"].values)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = LOCATION_ID
    return out


def date_true_target(obs: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Replace the year-collapsed test increment with the date-true one.

    Origin: the last survey with year <= t2 (position and date). End: the
    median position and date of the t3-year surveys. v_test becomes the
    change over the actual span; dist_t2/dist_t3/test_span_yr follow.
    """
    o = obs[[LOCATION_ID, "date", "dist_m"]].copy()
    o["year"] = o["date"].dt.year
    o["t"] = o["date"].map(pd.Timestamp.toordinal) / 365.25
    out = split.copy()

    t2 = split["t2"].astype(int)
    t3 = split["t3"].astype(int)
    o["t2"] = o[LOCATION_ID].map(t2)
    o["t3"] = o[LOCATION_ID].map(t3)

    pre = o[o["year"] <= o["t2"]].sort_values("t").groupby(LOCATION_ID).last()
    end_grp = o[o["year"] == o["t3"]].groupby(LOCATION_ID)
    end = pd.DataFrame({"dist": end_grp["dist_m"].median(), "t": end_grp["t"].median()})

    span = (end["t"] - pre["t"]).reindex(out.index)
    v = ((end["dist"] - pre["dist_m"]).reindex(out.index)) / span
    ok = span > 0.2  # guard against near-zero spans blowing up the target
    out = out[ok.fillna(False)].copy()
    out["dist_t2"] = pre["dist_m"].reindex(out.index)
    out["dist_t3"] = end["dist"].reindex(out.index)
    out["test_span_yr"] = span.reindex(out.index)
    out["v_test"] = v.reindex(out.index)
    return out


def build_pairwise_train(
    dpy: pd.DataFrame, caches: Caches, split: pd.DataFrame
) -> pd.DataFrame:
    """One training row per consecutive increment, train regions only.

    For a region with years y0..y_{m-1}, origins k = 1..m-2 give rows with
    t1 = y_{k-1}, t2 = y_k, t3 = y_{k+1} — the standard row is the k = m-2
    case, so this is a strict superset of today's training set. Features for
    each row are built exactly like the standard ones (same HW windows, same
    erosion-volume rates) from its own t-triple.
    """
    train_ids = set(split.index[split["split"] == "train"])
    d = dpy[dpy[LOCATION_ID].isin(train_ids)].sort_values([LOCATION_ID, "year"])

    rows = []
    for loc, g in d.groupby(LOCATION_ID, sort=False):
        ys = g["year"].values
        ds = g["dist_m"].values
        for k in range(1, len(ys) - 1):
            t1, t2, t3 = int(ys[k - 1]), int(ys[k]), int(ys[k + 1])
            rows.append(
                {
                    LOCATION_ID: loc,
                    "t1": t1,
                    "t2": t2,
                    "t3": t3,
                    "dist_t1": ds[k - 1],
                    "dist_t2": ds[k],
                    "dist_t3": ds[k + 1],
                    "train_span_yr": t2 - t1,
                    "test_span_yr": t3 - t2,
                    "v_train": (ds[k] - ds[k - 1]) / (t2 - t1),
                    "v_test": (ds[k + 1] - ds[k]) / (t3 - t2),
                }
            )
    pw = pd.DataFrame(rows)

    from src.pipeline.region_split import get_cluster

    pw["cluster"] = pw[LOCATION_ID].map(get_cluster)
    pw["n_timestamps"] = 3
    pw["is_nvo"] = pw[LOCATION_ID].map(caches.static["is_nvo"]).astype(bool)
    pw["quality"] = pw[LOCATION_ID].map(caches.static["quality"])

    ev_sum = (
        caches.ev.groupby([LOCATION_ID, "_yb", "_ya"])["erosion_volume"]
        .sum()
        .reset_index()
    )
    for label, (a, b) in {"train": ("t1", "t2"), "test": ("t2", "t3")}.items():
        m = pw[[LOCATION_ID, a, b]].merge(
            ev_sum,
            left_on=[LOCATION_ID, a, b],
            right_on=[LOCATION_ID, "_yb", "_ya"],
            how="left",
        )
        pw[f"erosion_vol_{label}_rate"] = (
            m["erosion_volume"].fillna(0).values / pw[f"{label}_span_yr"].values
        )
    pw["erosion_vol_rate_t1"] = pw["erosion_vol_train_rate"]
    pw["split"] = "train"
    return pw


def build_features_pairwise(pw: pd.DataFrame, caches: Caches) -> pd.DataFrame:
    """build_features_fast for a frame with duplicate regions (column key)."""
    tmp2 = pw.reset_index(drop=True)
    st = caches.static
    for col in [
        "river",
        "vegetation_class",
        "land_use",
        "soil_group",
        "nearest_station",
        "bend_exposure_n5",
        "bend_exposure_n8",
    ]:
        tmp2[col] = tmp2[LOCATION_ID].map(st[col])

    from src.pipeline.feature_engineering import (
        HW_WINDOW_COLS,
        _hw_window_stats,
        encode_col,
    )

    hw = tmp2.apply(
        _hw_window_stats,
        axis=1,
        hw_metrics=caches.hw_metrics,
        station_annual=caches.station_annual,
        has_t3=True,
    )
    no_ev = hw[HW_WINDOW_COLS].isna().all(axis=1)
    hw.loc[no_ev, HW_WINDOW_COLS] = 0
    tmp2[HW_WINDOW_COLS] = hw[HW_WINDOW_COLS]

    from src.loop.harness import CAT_KEYS

    for col, key in CAT_KEYS.items():
        tmp2[f"{col}_enc"] = encode_col(tmp2[col], key)
    num = tmp2.select_dtypes("number").columns
    tmp2[num] = tmp2[num].fillna(0.0)
    return tmp2


def trajectory_features_pairwise(obs: pd.DataFrame, pw: pd.DataFrame) -> pd.DataFrame:
    """Trajectory descriptors per pairwise row, from surveys year <= its t2."""
    o = obs[[LOCATION_ID, "date", "dist_m", "source"]].copy()
    o["year"] = o["date"].dt.year
    o["t"] = o["date"].map(pd.Timestamp.toordinal) / 365.25
    o["sam"] = (o["source"] == "segmentation").astype(float)
    o = o.sort_values("t")
    by_loc = dict(o.groupby(LOCATION_ID, sort=False))

    rows = []
    for loc, t2 in zip(pw[LOCATION_ID].values, pw["t2"].values, strict=True):
        g = by_loc.get(loc)
        if g is None:
            rows.append(dict.fromkeys(TRAJ_FEATS, np.nan))
            continue
        h = g[g["year"] <= t2]
        rows.append(_traj_row(h["t"].values, h["dist_m"].values, h["sam"].values))
    return pd.DataFrame(rows, index=pw.index)


def prepare_standard(caches: Caches, obs: pd.DataFrame, v_limit: float = 50.0):
    """obs → (dpy, split frame) exactly as the track-1 harness does."""
    dpy = to_dist_per_year(obs)
    dpy, n_far = farbank_region_filter(dpy, v_limit)
    split = build_split_frame(dpy, caches)
    return dpy, split, n_far


def assemble(
    caches: Caches,
    obs: pd.DataFrame,
    features: str = "base",
    training: str = "standard",
    target: str = "year",
    v_limit: float = 50.0,
) -> tuple[pd.DataFrame, list[str], int]:
    """Build the modelling frame for one (features, training, target) combo.

    Returns (frame, feature_names, n_far_excluded).
    """
    from src.pipeline.train import FEATS_LGB

    dpy, split, n_far = prepare_standard(caches, obs, v_limit)
    if target == "date":
        split = date_true_target(obs, split)

    feats = build_features_fast(split, caches)
    feat_names = list(FEATS_LGB)

    if features == "traj":
        traj = trajectory_features(obs, split["t2"].astype(int))
        feats = feats.join(traj)
        feat_names += TRAJ_FEATS
        feats[TRAJ_FEATS] = feats[TRAJ_FEATS].fillna(0.0)

    if training == "pairwise":
        pw = build_pairwise_train(dpy, caches, split)
        pwf = build_features_pairwise(pw, caches)
        if features == "traj":
            pwf = pd.concat(
                [pwf, trajectory_features_pairwise(obs, pwf).fillna(0.0)], axis=1
            )
        test = feats[feats["split"] == "test"].reset_index()
        cols = [LOCATION_ID, "split", "v_test"] + [
            c for c in feat_names if c != LOCATION_ID
        ]
        frame = pd.concat([pwf[cols], test[cols]], ignore_index=True)
        return frame, feat_names, n_far

    return feats.reset_index(), feat_names, n_far


def run_t2_variant(
    name: str,
    caches: Caches,
    obs: pd.DataFrame,
    features: str = "base",
    training: str = "standard",
    target: str = "year",
    notes: str = "",
    seed: int = 42,
) -> dict:
    """Assemble, train, score against the frozen views, append to the ledger."""
    import json
    from datetime import datetime

    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, r2_score

    from src.loop.harness import score_frozen_views
    from src.pipeline.train import TAIL_THRESHOLD

    frame, feat_names, n_far = assemble(
        caches, obs, features=features, training=training, target=target
    )
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]

    X_tr = train[feat_names].astype(float)
    X_te = test[feat_names].astype(float)
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X_tr,
        train["v_test"],
        eval_set=[(X_te, test["v_test"])],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    preds = pd.DataFrame(
        {
            "v_test": test["v_test"].values,
            "v_train": test["v_train"].values,
            "pred_naive": train["v_test"].mean(),
            "pred_persist": test["v_train"].values,
            "pred_lgb": model.predict(X_te),
        },
        index=pd.Index(test[LOCATION_ID].values, name=LOCATION_ID),
    )

    y = preds["v_test"]
    tail = y > TAIL_THRESHOLD
    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "lgb_mae": mean_absolute_error(y, preds["pred_lgb"]),
        "lgb_r2": r2_score(y, preds["pred_lgb"]),
        "lgb_tail_mae": (
            mean_absolute_error(y[tail], preds.loc[tail, "pred_lgb"])
            if tail.any()
            else np.nan
        ),
        "tail_n": int(tail.sum()),
        "naive_mae": mean_absolute_error(y, preds["pred_naive"]),
        "persist_mae": mean_absolute_error(y, preds["pred_persist"]),
        "vtest_std": float(frame["v_test"].std()),
    }
    metrics.update(score_frozen_views(preds, caches))
    metrics["n_regions"] = int(frame[LOCATION_ID].nunique())
    metrics["regions_excluded_far"] = n_far

    var_dir = caches.cache_dir / "variants" / name
    var_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(var_dir / "test_preds.parquet")
    imp = pd.Series(model.feature_importances_, index=feat_names).sort_values(
        ascending=False
    )
    imp.to_csv(var_dir / "feature_importance.csv", header=["importance"])

    row = {
        "variant": name,
        "when": datetime.now().isoformat(timespec="seconds"),
        "rules": json.dumps(
            {"track": 2, "features": features, "training": training, "target": target}
        ),
        "v_limit": 50.0,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "notes": notes,
    }
    ledger_path = caches.exp_dir / "ledger.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(f"── {name}  [features={features} training={training} target={target}]")
    print(f"   train rows {len(train):,} · test {len(test)}")
    print(
        f"   LGB mae {metrics['lgb_mae']:.3f} · tail {metrics['lgb_tail_mae']:.3f}"
        f" (n={metrics['tail_n']}) · R² {metrics['lgb_r2']:.3f}"
        f" · naive {metrics['naive_mae']:.3f} · persist {metrics['persist_mae']:.3f}"
    )
    for k in ("coverage_core", "core_mae", "tail_frozen_mae", "tail_frozen_n_kept"):
        if k in metrics:
            val = metrics[k]
            print(f"   {k}: {val:.4f}" if isinstance(val, float) else f"   {k}: {val}")
    print(f"   top features: {', '.join(imp.head(6).index)}")
    return {**metrics, "preds": preds, "importance": imp}
