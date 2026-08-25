"""Loop batch 8 — gully mask + horizon-ruler optimization.

e10: new_in_hybrid (side gully) scope polygons join kribben+water in the
structures mask — 5.9% of normal-region samples sit on gully banks.

k-variants: the >=H-year horizon ruler, optimized. Training rows are all
(origin, end) year pairs with span >= H in train regions ("multi") or just
the latest one ("single"); test rows are always one per frozen region:
origin = latest year >= H before the last measurement, end = the last
measurement. All trajectory features are computed at each row's own origin
(leakage-safe). k6 combines horizons with resolution: segment-level series
(R=5) under the same >=2-yr contract — sample growth from both directions.
"""

import json
import logging
import warnings
from datetime import datetime

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
import shapely
from sklearn.metrics import mean_absolute_error, r2_score

from experiments.loop.harness.harness import load_caches, run_variant
from experiments.loop.harness.multi_t import (
    E8_RULES,
    TRAJ_FEATS2,
    build_features_pairwise,
    load_obs_e8,
    prepare_standard,
    trajectory_features_pairwise,
)
from experiments.loop.harness.resolution import load_dense, segment_observations
from src.pipeline.region_split import get_cluster
from src.pipeline.train import FEATS_LGB
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

caches = load_caches()
data_dir = caches.cache_dir.parents[1]

kribs = gpd.read_file(
    data_dir / "01_raw/scope/Levering_erosie_data.gpkg", layer="Kribben_BKN"
).to_crs(28992)
water = gpd.read_file(
    data_dir / "02_processed/triage/secondary_water_mask.gpkg", layer="secondary_water"
).to_crs(28992)
sc = gpd.read_file(data_dir / "02_processed/scope_coverage.gpkg").to_crs(28992)
gully = sc[sc["coverage"] == "new_in_hybrid"]

MASK_ALL = shapely.union_all(
    [
        shapely.union_all(kribs.geometry.buffer(10.0).values),
        shapely.union_all(water.geometry.buffer(10.0).values),
        shapely.union_all(gully.geometry.buffer(5.0).values),
    ]
)
shapely.prepare(MASK_ALL)

SKIP_E10 = True
# ── e10: gully mask on top of e8 + water ─────────────────────────────────────
if not SKIP_E10:
    run_variant(
        "e10-gully-mask",
        caches,
        E8_RULES,
        structures=MASK_ALL,
        v_limit=50.0,
        notes="kribben + secondary water + new_in_hybrid gully polygons",
    )

obs = load_obs_e8(caches, None)  # cached parquet; structures arg unused on load

# ── horizon machinery ────────────────────────────────────────────────────────
ok_ids = set(caches.static.index[caches.static["quality"] == "OK"])
frozen = caches.frozen_test
train_ids = ok_ids - frozen


def horizon_rows(yearly: pd.DataFrame, key_cols: list, H: int, multi: bool):
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
            if multi:
                pairs = [
                    (i, j)
                    for i in range(1, n - 1)
                    for j in range(i + 1, n)
                    if ys[j] - ys[i] >= H
                ]
            else:
                cand = [i for i in range(1, n - 1) if ys[i] <= ys[-1] - H]
                pairs = [(cand[-1], n - 1)] if cand else []
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


def run_horizon(name, pw, traj_source_obs, notes, weight=False, extra_traj=None):
    feats = build_features_pairwise(decorate(pw), caches)
    tf = trajectory_features_pairwise(traj_source_obs, feats)[TRAJ_FEATS2].fillna(0.0)
    feats = pd.concat([feats, tf], axis=1)
    feat_names = list(FEATS_LGB) + TRAJ_FEATS2
    if extra_traj is not None:
        reg_tf = extra_traj(feats)
        feats = pd.concat([feats, reg_tf], axis=1)
        feat_names += list(reg_tf.columns)
    feats["split"] = pw["split"].values

    train = feats[feats["split"] == "train"]
    test = feats[feats["split"] == "test"]
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    sw = np.clip(train["test_span_yr"].astype(float), 2, 5) if weight else None
    model.fit(
        train[feat_names].astype(float),
        train["v_test"],
        sample_weight=sw,
        eval_set=[(test[feat_names].astype(float), test["v_test"])],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    pred = model.predict(test[feat_names].astype(float))
    y = test["v_test"].values
    t = y > 2.0
    tfz = test[LOCATION_ID].isin(caches.tail_frozen).values
    pos_err = np.abs((y - pred) * test["test_span_yr"].values)
    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "lgb_mae": mean_absolute_error(y, pred),
        "lgb_r2": r2_score(y, pred),
        "lgb_tail_mae": mean_absolute_error(y[t], pred[t]) if t.any() else np.nan,
        "tail_n": int(t.sum()),
        "tail_frozen_mae": (
            mean_absolute_error(y[tfz], pred[tfz]) if tfz.any() else np.nan
        ),
        "tail_frozen_n_kept": int(tfz.sum()),
        "naive_mae": float(np.abs(y - train["v_test"].mean()).mean()),
        "persist_mae": mean_absolute_error(y, test["v_train"]),
        "pos_err_med": float(np.median(pos_err)),
        "pos_err_p90": float(np.percentile(pos_err, 90)),
        "n_regions": int(pw[LOCATION_ID].nunique()),
    }
    row = {
        "variant": name,
        "when": datetime.now().isoformat(timespec="seconds"),
        "rules": json.dumps({"track": "horizon", "notes_cfg": notes}),
        "v_limit": 50.0,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "notes": notes,
    }
    ledger_path = caches.exp_dir / "ledger.csv"
    ledger = pd.read_csv(ledger_path)
    pd.concat([ledger, pd.DataFrame([row])], ignore_index=True).to_csv(
        ledger_path, index=False
    )
    print(f"── {name}")
    print(
        f"   train {len(train):,} · test {len(test):,}"
        f" · median horizon {test['test_span_yr'].median():.1f} jr"
    )
    print(
        f"   MAE {metrics['lgb_mae']:.3f} · tail {metrics['lgb_tail_mae']:.3f}"
        f" (n={metrics['tail_n']}) · tail_frozen {metrics['tail_frozen_mae']:.3f}"
        f" (n={metrics['tail_frozen_n_kept']}) · R² {metrics['lgb_r2']:.3f}"
    )
    print(
        f"   naive {metrics['naive_mae']:.3f} · persist {metrics['persist_mae']:.3f}"
        f" · pos-err med {metrics['pos_err_med']:.2f} m / p90 {metrics['pos_err_p90']:.2f} m"
    )
    return metrics


# region-level year series from e8 observations
dpy, _, _ = prepare_standard(caches, obs)
yearly_reg = dpy.rename(columns={"dist_m": "dist_m"})[[LOCATION_ID, "year", "dist_m"]]

run_horizon(
    "k1-H2-single",
    horizon_rows(yearly_reg, [LOCATION_ID], 2, False),
    obs,
    "H=2, latest origin only",
)
run_horizon(
    "k2-H2-multi",
    horizon_rows(yearly_reg, [LOCATION_ID], 2, True),
    obs,
    "H=2, all >=2yr origin/end pairs",
)
run_horizon(
    "k3-H3-single",
    horizon_rows(yearly_reg, [LOCATION_ID], 3, False),
    obs,
    "H=3, latest origin only",
)
run_horizon(
    "k4-H3-multi",
    horizon_rows(yearly_reg, [LOCATION_ID], 3, True),
    obs,
    "H=3, all >=3yr pairs",
)
run_horizon(
    "k5-H2-multi-w",
    horizon_rows(yearly_reg, [LOCATION_ID], 2, True),
    obs,
    "k2 + span weighting",
    weight=True,
)

# k6: segments (R=5) under the >=2yr horizon contract
dense = load_dense(caches)
seg_obs = segment_observations(dense, 5)
seg_obs["year"] = seg_obs["date"].dt.year
yearly_seg = (
    seg_obs.groupby([LOCATION_ID, "seg", "year"])["dist_m"].median().reset_index()
)
pw6 = horizon_rows(yearly_seg, [LOCATION_ID, "seg"], 2, True)

# the shared traj machinery groups one series per LOCATION_ID value — give it
# a composite (region#segment) key on both sides, decoration stays on the
# true region id.
seg_series = seg_obs.rename(columns={"model": "source"})[
    [LOCATION_ID, "seg", "date", "dist_m", "source"]
].copy()
seg_series[LOCATION_ID] = seg_series[LOCATION_ID] + "#" + seg_series["seg"].astype(str)
seg_key = (pw6[LOCATION_ID] + "#" + pw6["seg"].astype(str)).values

fx = build_features_pairwise(decorate(pw6), caches)
tf6 = trajectory_features_pairwise(seg_series, fx.assign(**{LOCATION_ID: seg_key}))[
    TRAJ_FEATS2
].fillna(0.0)
fx = pd.concat([fx, tf6], axis=1)
feat_names6 = list(FEATS_LGB) + TRAJ_FEATS2
fx["split"] = pw6["split"].values
train6, test6 = fx[fx["split"] == "train"], fx[fx["split"] == "test"]
model6 = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)
model6.fit(
    train6[feat_names6].astype(float),
    train6["v_test"],
    eval_set=[(test6[feat_names6].astype(float), test6["v_test"])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
)
pred6 = model6.predict(test6[feat_names6].astype(float))
y6 = test6["v_test"].values
t6 = y6 > 2.0
pos_err6 = np.abs((y6 - pred6) * test6["test_span_yr"].values)
print("── k6-R5-H2-multi  (segments × horizons)")
print(
    f"   train {len(train6):,} · test {len(test6):,}"
    f" · median horizon {test6['test_span_yr'].median():.1f} jr"
)
print(
    f"   MAE {mean_absolute_error(y6, pred6):.3f}"
    f" · tail {mean_absolute_error(y6[t6], pred6[t6]):.3f} (n={int(t6.sum())})"
    f" · R² {r2_score(y6, pred6):.3f}"
    f" · naive {np.abs(y6 - train6['v_test'].mean()).mean():.3f}"
    f" · persist {mean_absolute_error(y6, test6['v_train']):.3f}"
)
print(
    f"   pos-err med {np.median(pos_err6):.2f} m"
    f" / p90 {np.percentile(pos_err6, 90):.2f} m"
)
