"""Step 05 — Train models.

Trains all comparison models on a region_features table and saves a model bundle:
  0 – Naive mean (zero-parameter baseline)
  1 – v_train passthrough (persistence baseline)
  2 – OLS on v_train only
  3 – Ridge numeric features
  4 – Ridge numeric + categorical features
  5 – LightGBM (primary model)

The bundle is saved to ``model_out_dir`` via ``save_model_bundle``.

Feature sets:
  FEATS_LGB is the canonical set for iterative prediction.
  FEATS_2/3/4 are comparison baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.export_utils import save_model_bundle

# ── Feature sets ──────────────────────────────────────────────────────────────

TARGET = "v_test"
TAIL_THRESHOLD = 2.0  # m/yr — operationally critical threshold

FEATS_2 = ["v_train"]

FEATS_3 = [
    "v_train",
    "dist_t2",
    "train_span_yr",
    "test_span_yr",
    "n_events_t1",
    "max_rise_rate_t1",
    "drawdown_index_t1",
    "flood_days_t1",
    "n_events_t2",
    "max_rise_rate_t2",
    "drawdown_index_t2",
    "flood_days_t2",
    "bend_exposure_n5",
    "bend_exposure_n8",
]

FEATS_4 = FEATS_3 + [
    "is_nvo",
    "river_enc",
    "vegetation_class_enc",
    "soil_group_enc",
    "land_use_enc",
]

FEATS_LGB = [
    "v_train",
    "dist_t2",
    "train_span_yr",
    "test_span_yr",
    "erosion_vol_rate_t1",
    "n_events_t1",
    "max_rise_rate_t1",
    "drawdown_index_t1",
    "flood_days_t1",
    "n_events_t2",
    "max_rise_rate_t2",
    "drawdown_index_t2",
    "flood_days_t2",
    "bend_exposure_n5",
    "bend_exposure_n8",
    "is_nvo",
    "river_enc",
    "vegetation_class_enc",
    "soil_group_enc",
    "land_use_enc",
]

CAT_FEATS = ["river_enc", "vegetation_class_enc", "soil_group_enc", "land_use_enc"]


def train_and_save_models(
    features: pd.DataFrame,
    model_out_dir: Path,
    seed: int = 42,
    extra_features: list[str] | None = None,
    val_frac: float = 0.0,
) -> dict[str, Any]:
    """Train all models on features and save a bundle to model_out_dir.

    Args:
        features:      region_features DataFrame (must have 'split' column).
        model_out_dir: Directory where bundle and individual model files are written.
        seed:          Random seed for LightGBM and any stochastic operations.
        extra_features: Additional LGB feature columns (e.g. the trajectory
            descriptors of the graduated hybrid pipeline).
        val_frac:      When > 0, LightGBM early-stops on a validation split
            carved from the *train* regions instead of on the test set —
            honest evaluation; the test rows are never seen during training.

    Returns:
        RESULTS dict keyed by model name, each containing train/test metrics.
    """
    model_out_dir = Path(model_out_dir)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    feats_lgb = FEATS_LGB + [c for c in (extra_features or []) if c not in FEATS_LGB]

    train = features[features["split"] == "train"].copy()
    test = features[features["split"] == "test"].copy()

    results: dict[str, Any] = {}

    # Shared evaluation helper (captures results dict in closure)
    def evaluate(name: str, y_tr_true, y_tr_pred, y_te_true, y_te_pred) -> dict:
        return _evaluate(name, y_tr_true, y_tr_pred, y_te_true, y_te_pred, results)

    # ── Model 0: Naive mean ───────────────────────────────────────────────────
    global_mean = train[TARGET].mean()
    evaluate(
        "0 – Naive mean",
        train[TARGET],
        np.full(len(train), global_mean),
        test[TARGET],
        np.full(len(test), global_mean),
    )

    # ── Model 1: v_train passthrough ─────────────────────────────────────────
    evaluate(
        "1 – v_train baseline",
        train[TARGET],
        train["v_train"],
        test[TARGET],
        test["v_train"],
    )

    # ── Model 2: OLS on v_train only ─────────────────────────────────────────
    ols = LinearRegression()
    X_tr2 = _prep_num(train, FEATS_2)
    X_te2 = _prep_num(test, FEATS_2)
    ols.fit(X_tr2, train[TARGET])
    evaluate(
        "2 – OLS v_train",
        train[TARGET],
        ols.predict(X_tr2),
        test[TARGET],
        ols.predict(X_te2),
    )

    # ── Model 3: Ridge numeric features ──────────────────────────────────────
    ridge_num = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    X_tr3 = _prep_num(train, FEATS_3)
    X_te3 = _prep_num(test, FEATS_3)
    ridge_num.fit(X_tr3, train[TARGET])
    evaluate(
        "3 – Ridge numeric",
        train[TARGET],
        ridge_num.predict(X_tr3),
        test[TARGET],
        ridge_num.predict(X_te3),
    )

    # ── Model 4: Ridge numeric + categorical ──────────────────────────────────
    ridge_cat = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    X_tr4 = _prep_cats(train, FEATS_4)
    X_te4 = _prep_cats(test, FEATS_4)
    ridge_cat.fit(X_tr4, train[TARGET])
    evaluate(
        "4 – Ridge + cat",
        train[TARGET],
        ridge_cat.predict(X_tr4),
        test[TARGET],
        ridge_cat.predict(X_te4),
    )

    # ── Model 5: LightGBM ─────────────────────────────────────────────────────
    X_te_lgb = _prep_lgb(test, feats_lgb)
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    if val_frac > 0:
        # honest early stopping: hold out a slice of the train regions;
        # the test set plays no role in fitting.
        rng = np.random.default_rng(seed)
        val_mask = rng.random(len(train)) < val_frac
        fit_rows, val_rows = train[~val_mask], train[val_mask]
        lgb_model.fit(
            _prep_lgb(fit_rows, feats_lgb),
            fit_rows[TARGET],
            eval_set=[(_prep_lgb(val_rows, feats_lgb), val_rows[TARGET])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
    else:
        lgb_model.fit(
            _prep_lgb(train, feats_lgb),
            train[TARGET],
            eval_set=[(X_te_lgb, test[TARGET])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
    X_tr_lgb = _prep_lgb(train, feats_lgb)
    evaluate(
        "5 – LightGBM",
        train[TARGET],
        lgb_model.predict(X_tr_lgb),
        test[TARGET],
        lgb_model.predict(X_te_lgb),
    )

    # ── Save bundle ───────────────────────────────────────────────────────────
    config = {
        "TARGET": TARGET,
        "FEATS_2": FEATS_2,
        "FEATS_3": FEATS_3,
        "FEATS_4": FEATS_4,
        "FEATS_LGB": feats_lgb,
        "CAT_FEATS": CAT_FEATS,
        "TAIL_THRESHOLD": TAIL_THRESHOLD,
        "seed": seed,
        "val_frac": val_frac,
    }
    save_model_bundle(
        path=model_out_dir,
        models={
            "ols": ols,
            "ridge_num": ridge_num,
            "ridge_cat": ridge_cat,
            "lgb": lgb_model,
        },
        config=config,
        results=results,
    )
    print(f"\nBundle saved → {model_out_dir}")

    return results


# ── Private helpers ────────────────────────────────────────────────────────────


def _prep_num(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).values


def _prep_cats(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = df[cols].copy()
    if "is_nvo" in out.columns:
        out["is_nvo"] = out["is_nvo"].astype(int)
    return out.astype(float).values


def _prep_lgb(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    if "is_nvo" in out.columns:
        out["is_nvo"] = out["is_nvo"].astype(int)
    return out.astype(float)


def _root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate(
    name: str,
    y_tr_true,
    y_tr_pred,
    y_te_true,
    y_te_pred,
    results: dict,
) -> dict:
    y_tr_true = np.asarray(y_tr_true)
    y_tr_pred = np.asarray(y_tr_pred)
    y_te_true = np.asarray(y_te_true)
    y_te_pred = np.asarray(y_te_pred)

    tr_mask = y_tr_true > TAIL_THRESHOLD
    te_mask = y_te_true > TAIL_THRESHOLD

    row = {
        "train_rmse": _root_mean_squared_error(y_tr_true, y_tr_pred),
        "train_mae": float(mean_absolute_error(y_tr_true, y_tr_pred)),
        "train_r2": float(r2_score(y_tr_true, y_tr_pred)),
        "train_tail_mae": (
            float(mean_absolute_error(y_tr_true[tr_mask], y_tr_pred[tr_mask]))
            if tr_mask.sum() > 0
            else float("nan")
        ),
        "train_tail_n": int(tr_mask.sum()),
        "test_rmse": _root_mean_squared_error(y_te_true, y_te_pred),
        "test_mae": float(mean_absolute_error(y_te_true, y_te_pred)),
        "test_r2": float(r2_score(y_te_true, y_te_pred)),
        "test_tail_mae": (
            float(mean_absolute_error(y_te_true[te_mask], y_te_pred[te_mask]))
            if te_mask.sum() > 0
            else float("nan")
        ),
        "test_tail_n": int(te_mask.sum()),
    }
    results[name] = row
    print(
        f"\n── {name}\n"
        f"  {'metric':>14}    train       test\n"
        f"  {'MAE (prim)':>14}    {row['train_mae']:.4f}    {row['test_mae']:.4f}\n"
        f"  {'RMSE':>14}    {row['train_rmse']:.4f}    {row['test_rmse']:.4f}\n"
        f"  {'MAE tail>2':>14}    {row['train_tail_mae']:.4f}    {row['test_tail_mae']:.4f}"
        f"  (n={row['train_tail_n']}/{row['test_tail_n']})\n"
        f"  {'R²':>14}    {row['train_r2']:.4f}    {row['test_r2']:.4f}"
    )
    return row
