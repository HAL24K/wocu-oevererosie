"""Trajectory descriptors of a region's survey history (the traj2 set).

Graduated from track 2 of the loop-engineering experiment
(experiments/loop/TRACK2_REPORT.md): the winning way to consume the hybrid
delivery's irregular multi-t history is to keep one horizon-normalized
target per region and describe the history's *shape* as features — robust
velocity, spread, acceleration, density, recency, source mix, and the
mean-reversion signals.

All descriptors are computed from surveys at or before the forecast origin
(calendar year <= t2), so they are leakage-safe by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sources.geometry import LOCATION_ID

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

#: traj2: TRAJ_FEATS plus mean-reversion/recency signals.
TRAJ_FEATS2 = [*TRAJ_FEATS, "resid_last", "v_recent", "slope_recent"]


def theil(t: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Theil–Sen slope and residuals; (nan, zeros) when underdetermined."""
    if len(t) < 2 or t.max() == t.min():
        return np.nan, np.zeros_like(y)
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    ok = dt != 0
    slope = float(np.median((y[j] - y[i])[ok] / dt[ok]))
    resid = y - (slope * t + np.median(y - slope * t))
    return slope, resid


def traj_row(t: np.ndarray, y: np.ndarray, sam: np.ndarray) -> dict:
    """One region-history descriptor row from (year-fraction, dist) series."""
    slope, resid = theil(t, y)
    half = t.min() + (t.max() - t.min()) / 2
    lo, hi = t <= half, t > half
    accel = np.nan
    if lo.sum() >= 2 and hi.sum() >= 2:
        s1, _ = theil(t[lo], y[lo])
        s2, _ = theil(t[hi], y[hi])
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
        "resid_last": float(resid[-1]) if len(t) >= 3 else np.nan,
        "v_recent": (
            float((y[-1] - y[-2]) / (t[-1] - t[-2]))
            if len(t) >= 2 and t[-1] - t[-2] > 0.1
            else np.nan
        ),
        "slope_recent": theil(t[-3:], y[-3:])[0] if len(t) >= 3 else np.nan,
    }


def trajectory_features(obs: pd.DataFrame, origin_year: pd.Series) -> pd.DataFrame:
    """Per-region traj2 descriptors from surveys with year <= origin.

    Args:
        obs: observation table with location_id, date, dist_m, source.
        origin_year: forecast origin (t2) per location_id; regions absent
            from it are dropped.
    """
    o = obs[[LOCATION_ID, "date", "dist_m", "source"]].copy()
    o["origin"] = o[LOCATION_ID].map(origin_year)
    o = o[o["date"].dt.year <= o["origin"]].sort_values([LOCATION_ID, "date"])
    o["t"] = o["date"].map(pd.Timestamp.toordinal) / 365.25
    o["sam"] = (o["source"] == "segmentation").astype(float)

    rows = {}
    for loc, g in o.groupby(LOCATION_ID, sort=False):
        rows[loc] = traj_row(g["t"].values, g["dist_m"].values, g["sam"].values)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = LOCATION_ID
    return out
