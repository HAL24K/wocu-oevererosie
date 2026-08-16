"""Save and load trained models with their config for erosion prediction.

Models are serialized with joblib (sklearn's standard). The config stores
feature lists, target name, and preprocessing metadata so you can run
predictions in a new notebook or script without re-training.

Usage in training notebook:
    from src.model.export_utils import save_model_bundle

    save_model_bundle(
        path=Path("artifacts/models_v2"),
        models={"ols": ols_vt, "ridge_num": ridge_num, "ridge_cat": ridge_cat, "lgb": lgb_model},
        config={"FEATS_2": FEATS_2, "FEATS_3": FEATS_3, "FEATS_4": FEATS_4, "FEATS_LGB": FEATS_LGB, ...},
        results=RESULTS,
    )

Usage when loading:
    from src.model.export_utils import load_model_bundle, predict

    bundle = load_model_bundle(Path("artifacts/models_v2"))
    preds = predict(bundle, df, model_name="lgb")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _prep_for_ridge_cats(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Prepare dataframe for Ridge + categorical model (matches notebook prep_cats)."""
    out = df[cols].copy()
    if "is_nvo" in out.columns:
        out["is_nvo"] = out["is_nvo"].astype(int)
    return out.astype(float).values


def _prep_for_lgb(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Prepare dataframe for LightGBM (matches notebook prep_lgb)."""
    out = df[cols].copy()
    if "is_nvo" in out.columns:
        out["is_nvo"] = out["is_nvo"].astype(int)
    return out.astype(float)


def save_model_bundle(
    path: Path,
    *,
    models: dict[str, Any],
    config: dict[str, Any],
    results: dict[str, Any] | None = None,
) -> Path:
    """Save models, config, and optional results to a directory.

    Args:
        path: Directory to write artifacts (created if missing).
        models: Dict of model_name -> fitted model (sklearn Pipeline, LightGBM Booster, etc.).
        config: Dict with FEATS_2, FEATS_3, FEATS_4, FEATS_LGB, CAT_FEATS, TARGET, etc.
        results: Optional RESULTS dict from evaluate() for analysis.

    Returns:
        Path to the saved bundle directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save each model separately (easier to load individual models)
    for name, model in models.items():
        joblib.dump(model, path / f"model_{name}.joblib")

    # Config and results are plain dicts; joblib handles them well
    bundle = {"config": config, "results": results or {}}
    joblib.dump(bundle, path / "bundle.joblib")

    return path


def load_model_bundle(path: Path) -> dict[str, Any]:
    """Load the full model bundle (config + results). Models are loaded on demand.

    Returns:
        Dict with keys: config, results, models (lazy-loaded on first access).
    """
    path = Path(path)
    bundle = joblib.load(path / "bundle.joblib")

    # Lazy-load models when accessed via get_model()
    bundle["_models_path"] = path
    bundle["_models_cache"] = {}
    return bundle


def get_model(bundle: dict[str, Any], name: str) -> Any:
    """Load a single model from the bundle by name (e.g. 'ols', 'ridge_num', 'lgb')."""
    cache = bundle.get("_models_cache", {})
    if name in cache:
        return cache[name]
    path = bundle.get("_models_path")
    if not path:
        raise ValueError(
            "Bundle was not loaded via load_model_bundle (missing _models_path)"
        )
    model = joblib.load(Path(path) / f"model_{name}.joblib")
    cache[name] = model
    return model


def predict(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    model_name: str,
) -> np.ndarray:
    """Run prediction for a given model on a dataframe.

    Handles feature selection and preprocessing according to the model's config.

    Args:
        bundle: From load_model_bundle().
        df: DataFrame with required columns.
        model_name: One of 'ols', 'ridge_num', 'ridge_cat', 'lgb'.

    Returns:
        1D array of predictions.
    """
    config = bundle["config"]
    model = get_model(bundle, model_name)

    if model_name == "ols":
        feats = config["FEATS_2"]
        X = df[feats].values
        return model.predict(X)

    if model_name == "ridge_num":
        feats = config["FEATS_3"]
        X = df[feats].values
        return model.predict(X)

    if model_name == "ridge_cat":
        feats = config["FEATS_4"]
        X = _prep_for_ridge_cats(df, feats)
        return model.predict(X)

    if model_name == "lgb":
        feats = config["FEATS_LGB"]
        X = _prep_for_lgb(df, feats)
        return model.predict(X)

    raise ValueError(f"Unknown model_name: {model_name}")
