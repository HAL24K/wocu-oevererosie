"""Unified model loader and prediction dispatcher.

Wraps both the baseline model and ML bundles behind a single interface.
Configure once, call predict — the loader handles model loading, feature
preparation, and dispatches to the right predictor function.

Usage::

    from src.model.model_loader import ModelLoader

    # Baseline
    loader = ModelLoader(model='baseline', model_path=MODEL_PKL,
                         start_year=2026, end_year=2035)
    df = loader.predict(start_points=start_points)

    # ML model
    loader = ModelLoader(model='lgb', model_path=BUNDLE_DIR,
                         start_year=2026, end_year=2035, rolling=True)
    df = loader.predict(features_df=features_df, start_points=start_points)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.model.export_utils import load_model_bundle
from src.model.feature_shifter import FeatureShiftConfig
from src.model.predictor import predict_iterative_baseline, predict_iterative_ml

#: ML model names supported by the bundle
ML_MODELS = ("ols", "ridge_num", "ridge_cat", "lgb")


@dataclass
class ModelLoader:
    """Load a model and run iterative predictions with a unified interface.

    Args:
        model:      ``'baseline'`` or one of ``'ols'``, ``'ridge_num'``,
                    ``'ridge_cat'``, ``'lgb'``.
        model_path: Path to the baseline ``.pkl`` file or to the bundle
                    directory (for ML models).
        start_year: First prediction year (inclusive).
        end_year:   Last prediction year (inclusive).
        step:       Interval between prediction years (1 = every year).
        rolling:    ML only — whether to use full ``FeatureShifter`` rolling
                    updates (``True``) or only update ``dist_t2`` (``False``).
        shift_config: Override the default ``FeatureShiftConfig``.
    """

    model: str
    model_path: Path
    start_year: int = 2026
    end_year: int = 2035
    step: int = 1
    rolling: bool = True
    shift_config: Optional[FeatureShiftConfig] = field(default=None)

    # Internal — loaded lazily
    _baseline_velocities: Optional[dict] = field(default=None, repr=False)
    _bundle: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if self.model not in ("baseline",) + ML_MODELS:
            raise ValueError(
                f"Unknown model '{self.model}'. "
                f"Choose 'baseline' or one of {ML_MODELS}."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        start_points: dict[str, dict],
        features_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Run iterative prediction for all configured years.

        Args:
            start_points: Mapping ``{location_id: {'last_dist', 'last_year', 'v_hist'}}``.
            features_df:  Required for ML models; ignored for baseline.

        Returns:
            DataFrame with columns: location_id, year, predicted_dist_m, velocity_m_per_yr.
        """
        if self.model == "baseline":
            return predict_iterative_baseline(
                model_velocities=self._load_baseline(),
                start_points=start_points,
                start_year=self.start_year,
                end_year=self.end_year,
                step=self.step,
            )

        if features_df is None:
            raise ValueError(
                f"features_df is required for ML model '{self.model}'."
            )
        return predict_iterative_ml(
            bundle=self._load_bundle(),
            features_df=features_df,
            start_points=start_points,
            model_name=self.model,
            start_year=self.start_year,
            end_year=self.end_year,
            step=self.step,
            rolling=self.rolling,
            shift_config=self.shift_config,
        )

    @property
    def prediction_years(self) -> list[int]:
        """List of years this loader will predict."""
        return list(range(self.start_year, self.end_year + 1, self.step))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_baseline(self) -> dict[str, float]:
        if self._baseline_velocities is None:
            from src.model.baseline_model import BaselineErosionModel
            bm = BaselineErosionModel.load_model(self.model_path)
            self._baseline_velocities = bm.model
        return self._baseline_velocities

    def _load_bundle(self) -> dict:
        if self._bundle is None:
            self._bundle = load_model_bundle(self.model_path)
        return self._bundle
