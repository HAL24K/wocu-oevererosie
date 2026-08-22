"""The horizon ruler: score the model the way October will use it.

Part 1 — stratify the existing champion's errors by test span: shows how
much of the reported error is short-span measurement noise, no retraining.

Part 2 — the >=2-year-horizon variant: forecast origin = the latest year at
least 2 years before the last measurement; target = observed change from
origin to last, in m/yr over that span. Features (incl. traj2) use only data
at or before the origin. Same frozen split, same LGB. This is the
operationally honest ruler: position change over a multi-year window.
"""

import logging
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from src.loop.harness import load_caches
from src.loop.multi_t import (
    TRAJ_FEATS2,
    build_features_pairwise,
    load_obs_e8,
    prepare_standard,
    trajectory_features,
)
from src.loop.rules import make_structure_geom
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

MIN_HORIZON = 2  # years between forecast origin and last measurement

caches = load_caches()
data_dir = caches.cache_dir.parents[1]
kribs = gpd.read_file(
    data_dir / "01_raw/scope/Levering_erosie_data.gpkg", layer="Kribben_BKN"
).to_crs(28992)
obs = load_obs_e8(caches, make_structure_geom(kribs, 10.0))
dpy, split_std, _ = prepare_standard(caches, obs)

# ── part 1: span stratification of the current champion ─────────────────────
i1 = pd.read_parquet(caches.cache_dir / "variants/i1-traj2/test_preds.parquet")
spans = split_std["test_span_yr"].reindex(i1.index)
err = (i1["v_test"] - i1["pred_lgb"]).abs()
tail = i1["v_test"] > 2.0
print("── part 1 · i1-traj2 error by test span (the ruler effect, no retraining)")
for lo, hi, label in [(0, 1, "1 jr"), (1, 2, "2 jr"), (2, 99, ">=3 jr")]:
    m = (spans > lo) & (spans <= hi)
    if not m.any():
        continue
    mt = m & tail
    print(
        f"   span {label:>6}: n={int(m.sum()):>4} · MAE {err[m].mean():.3f}"
        f" · tail-MAE {err[mt].mean() if mt.any() else float('nan'):.3f}"
        f" (n={int(mt.sum())})"
    )

# ── part 2: the >=2-year-horizon variant ─────────────────────────────────────
d = dpy.sort_values([LOCATION_ID, "year"])
rows = []
for loc, g in d.groupby(LOCATION_ID, sort=False):
    ys = g["year"].values
    ds = g["dist_m"].values
    y_n = ys[-1]
    cand = [i for i in range(1, len(ys) - 1) if ys[i] <= y_n - MIN_HORIZON]
    if not cand:
        continue
    i = cand[-1]
    rows.append(
        {
            LOCATION_ID: loc,
            "t1": int(ys[i - 1]),
            "t2": int(ys[i]),
            "t3": int(y_n),
            "dist_t1": ds[i - 1],
            "dist_t2": ds[i],
            "dist_t3": ds[-1],
            "train_span_yr": ys[i] - ys[i - 1],
            "test_span_yr": y_n - ys[i],
            "v_train": (ds[i] - ds[i - 1]) / (ys[i] - ys[i - 1]),
            "v_test": (ds[-1] - ds[i]) / (y_n - ys[i]),
        }
    )
pw = pd.DataFrame(rows)
pw["quality"] = pw[LOCATION_ID].map(caches.static["quality"])
pw = pw[pw["quality"] == "OK"].copy()

from src.pipeline.region_split import get_cluster  # noqa: E402

pw["cluster"] = pw[LOCATION_ID].map(get_cluster)
pw["n_timestamps"] = 3
pw["is_nvo"] = pw[LOCATION_ID].map(caches.static["is_nvo"]).astype(bool)
ev_sum = (
    caches.ev.groupby([LOCATION_ID, "_yb", "_ya"])["erosion_volume"].sum().reset_index()
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

feats = build_features_pairwise(pw, caches)
traj = trajectory_features(obs, pw.set_index(LOCATION_ID)["t2"].astype(int))
feats = feats.merge(traj[TRAJ_FEATS2].reset_index(), on=LOCATION_ID, how="left").fillna(
    dict.fromkeys(TRAJ_FEATS2, 0.0)
)
feats["split"] = np.where(feats[LOCATION_ID].isin(caches.frozen_test), "test", "train")

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

from src.pipeline.train import FEATS_LGB  # noqa: E402

feat_names = list(FEATS_LGB) + TRAJ_FEATS2
train = feats[feats["split"] == "train"]
test = feats[feats["split"] == "test"]
model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)
model.fit(
    train[feat_names].astype(float),
    train["v_test"],
    eval_set=[(test[feat_names].astype(float), test["v_test"])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
)
pred = model.predict(test[feat_names].astype(float))
y = test["v_test"].values
t = y > 2.0
tf = test[LOCATION_ID].isin(caches.tail_frozen).values

print(f"\n── part 2 · >= {MIN_HORIZON}-yr horizon ruler")
print(
    f"   rows: {len(pw):,} regions ({len(train):,} train / {len(test):,} test)"
    f" · median horizon {test['test_span_yr'].median():.1f} jr"
)
print(
    f"   MAE {mean_absolute_error(y, pred):.3f} m/jr"
    f" · tail-MAE {mean_absolute_error(y[t], pred[t]):.3f} (n={int(t.sum())})"
    f" · tail_frozen {mean_absolute_error(y[tf], pred[tf]):.3f} (n={int(tf.sum())})"
    f" · R² {r2_score(y, pred):.3f}"
)
print(
    f"   naive {np.abs(y - train['v_test'].mean()).mean():.3f}"
    f" · persist {mean_absolute_error(y, test['v_train']):.3f}"
)
pos_err = np.abs((y - pred) * test["test_span_yr"].values)
print(
    f"   position error at horizon: mean {pos_err.mean():.2f} m"
    f" · median {np.median(pos_err):.2f} m · p90 {np.percentile(pos_err, 90):.2f} m"
)
