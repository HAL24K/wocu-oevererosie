"""Iterative multi-step erosion prediction with a trained model bundle.

Output schema: location_id, year, predicted_dist_m, velocity_m_per_yr

Usage::

    from src.pipeline.predict import predict_iterative_ml

    df = predict_iterative_ml(bundle, features_df, start_points, model_name="lgb")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.feature_shifter import FeatureShiftConfig, FeatureShifter
from src.pipeline.model_bundle import predict as _bundle_predict


def predict_iterative_ml(
    bundle: dict,
    features_df: pd.DataFrame,
    start_points: dict[str, dict],
    model_name: str,
    start_year: int = 2026,
    end_year: int = 2035,
    step: int = 1,
    rolling: bool = True,
    shift_config: FeatureShiftConfig | None = None,
) -> pd.DataFrame:
    """Iterative prediction using an ML model from a saved bundle.

    At each step:
      1. Predict velocity from the current feature state.
      2. Compute new dist = previous dist + velocity × step.
      3. Advance the feature state via ``FeatureShifter.step()``.

    With ``rolling=True`` (default), ``FeatureShifter`` updates:

    * ``dist_t2`` and ``v_train`` from the previous prediction
    * ``train_span_yr`` / ``test_span_yr`` (rolling window)
    * Hydrology t1 features ← t2 (shift); t2 held as best future forecast

    With ``rolling=False``, only ``dist_t2`` updates (legacy behaviour,
    useful for comparison).

    Args:
        bundle: Loaded bundle from ``load_model_bundle()``.
        features_df: Static feature DataFrame indexed by location_id.
            Must contain all columns required by the chosen model.
        start_points: Mapping ``{location_id: {'last_dist', 'last_year', 'v_hist'}}``.
        model_name: One of ``'ols'``, ``'ridge_num'``, ``'ridge_cat'``, ``'lgb'``.
        start_year: First year to predict (inclusive).
        end_year: Last year to predict (inclusive).
        step: Interval between prediction years.
        rolling: Whether to use full ``FeatureShifter`` rolling updates.
        shift_config: Override the default ``FeatureShiftConfig`` (e.g. different column
            names).

    Returns:
        DataFrame with columns: location_id, year, predicted_dist_m, velocity_m_per_yr.
    """
    prediction_years = list(range(start_year, end_year + 1, step))
    locs = [loc for loc in features_df.index if loc in start_points]
    config = shift_config or FeatureShiftConfig()

    if rolling:
        shifter = FeatureShifter(features_df.loc[locs], config=config)
        shifter.initialize(start_points, start_year)
    else:
        features_iter = features_df.loc[locs].copy()
        for loc_id in locs:
            sp = start_points[loc_id]
            v_hist = sp.get("v_hist") or 0.0
            dist_at_start = (
                sp["last_dist"] + v_hist * (start_year - sp["last_year"])
                if sp["last_year"] < start_year
                else sp["last_dist"]
            )
            features_iter.loc[loc_id, config.dist_feature] = dist_at_start

    rows = []
    for year in prediction_years:
        feats = shifter.get(locs) if rolling else features_iter
        velocities = _bundle_predict(bundle, feats, model_name)
        new_dists = feats[config.dist_feature].values + np.asarray(velocities) * step

        for i, loc_id in enumerate(locs):
            rows.append(
                {
                    "location_id": loc_id,
                    "year": year,
                    "predicted_dist_m": float(new_dists[i]),
                    "velocity_m_per_yr": float(velocities[i]),
                }
            )

        if rolling:
            shifter.step(locs, velocities=velocities, dists=new_dists, step_size=step)
        else:
            for i, loc_id in enumerate(locs):
                features_iter.loc[loc_id, config.dist_feature] = float(new_dists[i])

    return pd.DataFrame(rows)
