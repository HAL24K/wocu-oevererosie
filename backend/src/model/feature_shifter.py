"""Rolling feature state manager for iterative multi-step erosion prediction.

At each prediction step the model re-uses its own output as input for the
next step.  Different features need different update rules:

  - dist_feature (dist_t2)      : ← previous predicted dist
  - vel_feature  (v_train)      : ← previous predicted velocity  → drives mean-reversion
  - span features               : train_span ← test_span; test_span ← step_size
  - hydrology t1 (n_events, …)  : ← hydrology t2  (shift: recent → historical)
  - hydrology t2 (n_events, …)  : unchanged         (hold: repeat last known as best forecast)
  - everything else             : unchanged         (static: geometry, soil, vegetation)

Assumption for t2 hydrology: future hydraulic regime ≈ most recently observed period.
If a future hydrology forecast becomes available, inject it into t2 columns before
calling step().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureShiftConfig:
    """Describes which columns update at every prediction step.

    Defaults match the 20260312 model bundle feature names.  Override any
    field when working with a differently-named feature set.
    """

    # Columns derived from the previous prediction
    dist_feature: str = "dist_t2"
    vel_feature: str = "v_train"

    # Span columns: train_span ← test_span; test_span ← step_size
    train_span_feature: str = "train_span_yr"
    test_span_feature: str = "test_span_yr"

    # Hydrology pairs (t1_col, t2_col): t1 ← t2, t2 held
    hydrology_pairs: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("n_events_t1",       "n_events_t2"),
            ("max_rise_rate_t1",  "max_rise_rate_t2"),
            ("drawdown_index_t1", "drawdown_index_t2"),
            ("flood_days_t1",     "flood_days_t2"),
        ]
    )


#: Singleton default config — avoids mutable-default-argument issues
DEFAULT_CONFIG = FeatureShiftConfig()


class FeatureShifter:
    """Stateful feature matrix for iterative multi-step prediction.

    Usage::

        shifter = FeatureShifter(features_df)
        shifter.initialize(start_points, start_year=2026)

        for year in range(2026, 2036):
            feats   = shifter.get(loc_ids)
            vels    = model.predict(feats)
            new_dist = feats[shifter.config.dist_feature].values + vels * step
            shifter.step(loc_ids, velocities=vels, dists=new_dist, step_size=step)
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        config: FeatureShiftConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config
        self._state = features_df.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(
        self,
        start_points: dict[str, dict],
        start_year: int,
    ) -> None:
        """Set dist_feature to each location's starting position.

        If a location's last observation is before start_year the position is
        linearly extrapolated using the historical velocity (v_hist).

        Args:
            start_points: Mapping loc_id → {'last_dist', 'last_year', 'v_hist'}.
            start_year:   First prediction year (e.g. 2026).
        """
        cfg = self.config
        for loc_id, sp in start_points.items():
            if loc_id not in self._state.index:
                continue
            last_dist = sp["last_dist"]
            last_year = sp["last_year"]
            v_hist = sp.get("v_hist") or 0.0
            dist_at_start = (
                last_dist + v_hist * (start_year - last_year)
                if last_year < start_year
                else last_dist
            )
            self._state.loc[loc_id, cfg.dist_feature] = dist_at_start

    def step(
        self,
        loc_ids: list[str],
        velocities: np.ndarray,
        dists: np.ndarray,
        step_size: int = 1,
    ) -> None:
        """Advance the feature state by one prediction step.

        All updates are vectorised over ``loc_ids``.

        Args:
            loc_ids:    Ordered list of location identifiers (same order as arrays).
            velocities: 1-D array of predicted velocities (m/yr).
            dists:      1-D array of new predicted distances (m).
            step_size:  Prediction horizon in years for this step.
        """
        cfg = self.config
        loc_index = pd.Index(loc_ids)

        # Snapshot current test_span before overwriting
        prev_test_span = self._state.loc[loc_index, cfg.test_span_feature].values

        # 1. Prediction-derived updates
        self._state.loc[loc_index, cfg.vel_feature] = np.asarray(velocities)
        self._state.loc[loc_index, cfg.dist_feature] = np.asarray(dists)

        # 2. Span rolling
        self._state.loc[loc_index, cfg.train_span_feature] = prev_test_span
        self._state.loc[loc_index, cfg.test_span_feature] = step_size

        # 3. Hydrology shift: t1 ← t2;  t2 unchanged (hold last known)
        for t1_feat, t2_feat in cfg.hydrology_pairs:
            if t1_feat in self._state.columns and t2_feat in self._state.columns:
                self._state.loc[loc_index, t1_feat] = (
                    self._state.loc[loc_index, t2_feat].values
                )

    def get(self, loc_ids: Optional[list[str]] = None) -> pd.DataFrame:
        """Return the current feature state, optionally filtered to loc_ids."""
        if loc_ids is None:
            return self._state
        return self._state.loc[loc_ids]

    @property
    def state(self) -> pd.DataFrame:
        """Full current feature state (all locations)."""
        return self._state
