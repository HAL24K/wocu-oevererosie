"""Segment × horizon sweep: the k6 recipe (>=2-yr pairs, all origin/end pairs,
span-weighted) for R = 1, 2, 5, 10, 20 (50 if not starved) on the frozen
holdout. R=1 is the region-level k5-H2-multi-w and must reproduce it.

Usage: uv run python experiments/loop/scripts/loop_horizon_sweep.py [R ...]
"""

import json
import sys
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from experiments.loop.harness.harness import load_caches
from experiments.loop.harness.multi_t import (
    TRAJ_FEATS2,
    build_features_pairwise,
    prepare_standard,
    trajectory_features_pairwise,
)
from experiments.loop.harness.resolution import load_dense, segment_observations
from src.pipeline.region_split import get_cluster
from src.pipeline.train import FEATS_LGB
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")

H = 2
caches = load_caches()
obs = pd.read_parquet(caches.cache_dir / "obs_e8.parquet")
ok_ids = set(caches.static.index[caches.static["quality"] == "OK"])
frozen = caches.frozen_test


def horizon_rows(yearly: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    rows = []
    for key, g in yearly.groupby(key_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        loc = key[0]
        if loc not in ok_ids:
            continue
        ys = g["year"].values
        ds = g["dist_m"].values
        n = len(ys)
        if n < 3:
            continue
        if loc in frozen:
            cand = [i for i in range(1, n - 1) if ys[i] <= ys[-1] - H]
            pairs = [(cand[-1], n - 1)] if cand else []
            split = "test"
        else:
            pairs = [
                (i, j)
                for i in range(1, n - 1)
                for j in range(i + 1, n)
                if ys[j] - ys[i] >= H
            ]
            split = "train"
        for i, j in pairs:
            row = {
                LOCATION_ID: loc,
                "t1": int(ys[i - 1]),
                "t2": int(ys[i]),
                "t3": int(ys[j]),
                "dist_t1": ds[i - 1],
                "dist_t2": ds[i],
                "dist_t3": ds[j],
                "train_span_yr": ys[i] - ys[i - 1],
                "test_span_yr": ys[j] - ys[i],
                "v_train": (ds[i] - ds[i - 1]) / (ys[i] - ys[i - 1]),
                "v_test": (ds[j] - ds[i]) / (ys[j] - ys[i]),
                "split": split,
            }
            if len(key_cols) > 1:
                row["seg"] = key[1]
            rows.append(row)
    return pd.DataFrame(rows)


def decorate(pw: pd.DataFrame) -> pd.DataFrame:
    pw = pw.copy()
    pw["cluster"] = pw[LOCATION_ID].map(get_cluster)
    pw["n_timestamps"] = 3
    pw["is_nvo"] = pw[LOCATION_ID].map(caches.static["is_nvo"]).astype(bool)
    pw["quality"] = "OK"
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
    return pw


def fit_score(name, pw, series, starved_frac, notes):
    fx = build_features_pairwise(decorate(pw), caches)
    if "seg" in pw.columns:
        key = (pw[LOCATION_ID] + "#" + pw["seg"].astype(str)).values
        tf = trajectory_features_pairwise(series, fx.assign(**{LOCATION_ID: key}))
    else:
        tf = trajectory_features_pairwise(series, fx)
    fx = pd.concat([fx, tf[TRAJ_FEATS2].fillna(0.0)], axis=1)
    feat_names = list(FEATS_LGB) + TRAJ_FEATS2
    fx["split"] = pw["split"].values
    train, test = fx[fx["split"] == "train"], fx[fx["split"] == "test"]
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
    )
    sw = np.clip(train["test_span_yr"].astype(float), 2, 5)
    model.fit(
        train[feat_names].astype(float),
        train["v_test"],
        sample_weight=sw,
        eval_set=[(test[feat_names].astype(float), test["v_test"])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    pred = model.predict(test[feat_names].astype(float))
    y = test["v_test"].values
    t = y > 2.0
    tfz = test[LOCATION_ID].isin(caches.tail_frozen).values
    pos_err = np.abs((y - pred) * test["test_span_yr"].values)
    naive = float(np.abs(y - train["v_test"].mean()).mean())
    m = {
        "n_train": len(train),
        "n_test": len(test),
        "lgb_mae": mean_absolute_error(y, pred),
        "lgb_r2": r2_score(y, pred),
        "lgb_tail_mae": mean_absolute_error(y[t], pred[t]) if t.any() else np.nan,
        "tail_n": int(t.sum()),
        "tail_frozen_mae": mean_absolute_error(y[tfz], pred[tfz]) if tfz.any() else np.nan,
        "tail_frozen_n_kept": int(tfz.sum()),
        "naive_mae": naive,
        "persist_mae": mean_absolute_error(y, test["v_train"]),
        "pos_err_med": float(np.median(pos_err)),
        "pos_err_p90": float(np.percentile(pos_err, 90)),
        "n_regions": int(pw[LOCATION_ID].nunique()),
        "coverage_core": round(1.0 - starved_frac, 4),
    }
    row = {
        "variant": name,
        "when": datetime.now().isoformat(timespec="seconds"),
        "rules": json.dumps({"track": "horizon-sweep", "H": H, "weight": "span 2-5"}),
        "v_limit": 50.0,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
        "notes": notes,
    }
    ledger_path = caches.exp_dir / "ledger.csv"
    ledger = pd.read_csv(ledger_path)
    pd.concat([ledger, pd.DataFrame([row])], ignore_index=True).to_csv(
        ledger_path, index=False
    )
    skill = 1 - m["lgb_mae"] / naive
    print(
        f"── {name}: train {m['n_train']:,} · test {m['n_test']:,} · starved {starved_frac:.1%}\n"
        f"   MAE {m['lgb_mae']:.3f} · tail {m['lgb_tail_mae']:.3f} (n={m['tail_n']})"
        f" · naive {naive:.3f} · skill {skill:.2f} · R² {m['lgb_r2']:.3f}"
        f" · pos-err med {m['pos_err_med']:.2f} m · regions {m['n_regions']:,}",
        flush=True,
    )
    return m


def run_R(R: int, dense: pd.DataFrame):
    if R == 1:
        dpy, _, _ = prepare_standard(caches, obs)
        yearly = dpy[[LOCATION_ID, "year", "dist_m"]]
        pw = horizon_rows(yearly, [LOCATION_ID])
        series = obs
        return fit_score(f"hz-R1", pw, series, 0.0, "region level = k5 recipe, span-weighted")
    seg_obs = segment_observations(dense, R)
    seg_obs["year"] = seg_obs["date"].dt.year
    # starvation: (region, date) combos × R vs kept (region, seg, date) rows
    n_rd = dense.groupby([LOCATION_ID, "date"]).ngroups
    starved = 1.0 - len(seg_obs) / (n_rd * R)
    print(f"   R={R}: kept {len(seg_obs):,} of {n_rd * R:,} (region,seg,date) → starved {starved:.1%}", flush=True)
    if R >= 50 and starved > 0.10:
        print(f"   R={R} skipped: starvation {starved:.1%} > 10 %", flush=True)
        return None
    yearly = seg_obs.groupby([LOCATION_ID, "seg", "year"])["dist_m"].median().reset_index()
    pw = horizon_rows(yearly, [LOCATION_ID, "seg"])
    series = seg_obs.rename(columns={"model": "source"})[
        [LOCATION_ID, "seg", "date", "dist_m", "source"]
    ].copy()
    series[LOCATION_ID] = series[LOCATION_ID] + "#" + series["seg"].astype(str)
    return fit_score(f"hz-R{R}", pw, series, starved, f"segments R={R} × horizon>=2, span-weighted")


if __name__ == "__main__":
    Rs = [int(a) for a in sys.argv[1:]] or [1, 2, 5, 10, 20, 50]
    dense = load_dense(caches)
    for R in Rs:
        run_R(R, dense)
