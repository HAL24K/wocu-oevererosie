"""Turn the strip model's speeds into a forecast that holds up over the years.

The strip model predicts how far a strip moves over the next period (>= 2 years),
and much of that is a correction: a SAM survey taken at high water or with a cloud
puts the bank too far inland, and the next survey is back. Drawn as a constant speed
for 5-25 years, that correction carried banks into the river and made neighbouring
strips zigzag (13 Oct 2026). The strip histories show why
(``scripts/evaluate/forecast_lab.py``):

- after a SAM jump of > 20 m the next survey goes back in 91 % of strips;
- after a height-model jump of > 20 m the bank stays (77 % continue, slowly);
- over >= 3 years "no change" beats any strip trend, and the model's speed is only
  better than "no change" for the first couple of years after a SAM survey.

So a strip's position in a future year is

    start + v_short * min(dt, short_years) + v_long * dt

- ``start``: the last measured position, as a median over neighbouring strips;
- ``v_short``: the model's speed, for SAM strips only and for ``short_years`` at most,
  as a median over neighbouring strips;
- ``v_long``: the lasting drift, ``alpha`` x the median over the region's strips of
  each strip's long-term slope with outlying years left out (jump-free Theil–Sen).

Backtest on held-out regions (hiding each strip's last year): position error 2.59 m
against 2.57 m for the plain model and 2.77 m for "no change"; extreme speeds
(> 3 m/yr) 0.7 % instead of 11.7 %, and no zigzag between neighbouring strips.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.sources.geometry import LOCATION_ID

KEY = [LOCATION_ID, "seg"]


# ── strip histories as matrices ──────────────────────────────────────────────
class Hist:
    """Yearly strip positions as NaN-padded matrices (one row per strip)."""

    def __init__(self, h: pd.DataFrame):
        h = h.sort_values([*KEY, "year"])
        pos = h.groupby(KEY).cumcount().to_numpy()
        row = h.groupby(KEY, sort=False).ngroup().to_numpy()
        self.keys = h[KEY].drop_duplicates().reset_index(drop=True)
        n, m = len(self.keys), int(pos.max()) + 1
        self.T = np.full((n, m), np.nan)
        self.D = np.full((n, m), np.nan)
        self.S = np.full((n, m), np.nan)
        self.T[row, pos] = h["year"].to_numpy(float)
        self.D[row, pos] = h["dist"].to_numpy(float)
        self.S[row, pos] = h["sam"].to_numpy(float)
        self.cnt = np.bincount(row, minlength=n)
        r = np.arange(n)
        self.t_last = self.T[r, self.cnt - 1]
        self.d_last = self.D[r, self.cnt - 1]
        prev = np.maximum(self.cnt - 2, 0)
        self.t_prev, self.d_prev = self.T[r, prev], self.D[r, prev]
        self.sam_last = self.S[r, self.cnt - 1] == 1


def yearly(seg_obs: pd.DataFrame) -> pd.DataFrame:
    """Per strip and year: median position over that year's surveys, and its source."""
    y = (
        seg_obs.groupby([*KEY, "year"])
        .agg(dist=("dist_m", "median"), model=("model", "first"))
        .reset_index()
    )
    y["sam"] = (y["model"] == "segmentation").astype(float)
    return y


def theil_mat(T, D, valid):
    """Theil–Sen slope and intercept per row over the ``valid`` cells."""
    i, j = np.triu_indices(T.shape[1], k=1)
    ok = valid[:, i] & valid[:, j]
    dt = T[:, j] - T[:, i]
    ok &= dt != 0
    sl = np.where(ok, (D[:, j] - D[:, i]) / np.where(dt == 0, 1, dt), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope = np.nanmedian(sl, axis=1)
        icpt = np.nanmedian(np.where(valid, D - slope[:, None] * T, np.nan), axis=1)
    return slope, icpt


def fit(H: Hist, valid):
    """(level at the last year, slope) per strip; last value and 0 when undetermined."""
    slope, icpt = theil_mat(H.T, H.D, valid)
    good = np.isfinite(slope) & np.isfinite(icpt)
    level = np.where(good, slope * H.t_last + icpt, H.d_last)
    return level, np.where(good, slope, 0.0), slope, icpt


def jumpfree_valid(H: Hist, k: float):
    """Years within ``k`` m of the robust line (strips with >= 3 years, >= 2 kept)."""
    valid = np.isfinite(H.D)
    _, _, slope, icpt = fit(H, valid)
    resid = H.D - (slope[:, None] * H.T + icpt[:, None])
    keep = valid & (np.abs(resid) <= k)
    use = (H.cnt >= 3) & (keep.sum(1) >= 2)
    return np.where(use[:, None], keep, valid)


def jumpfree_slope(H: Hist, k: float) -> np.ndarray:
    return fit(H, jumpfree_valid(H, k))[1]


def smooth(fc: pd.DataFrame, width: int, cols=("v",)) -> pd.DataFrame:
    """Rolling median over ``width`` neighbouring strips; a missing strip breaks it."""
    if width <= 1:
        return fc
    fc = fc.sort_values(KEY).copy()
    run = (fc.groupby(LOCATION_ID)["seg"].diff() != 1).cumsum()
    for c in cols:
        fc[c] = fc.groupby(run)[c].transform(
            lambda s: s.rolling(width, center=True, min_periods=1).median()
        )
    return fc


def regional_drift(H: Hist, alpha: float, k: float) -> pd.Series:
    """``alpha`` x the median jump-free slope over each region's strips."""
    s = H.keys.assign(v=jumpfree_slope(H, k))
    return s.groupby(LOCATION_ID)["v"].median() * alpha


# ── the forecast ─────────────────────────────────────────────────────────────
def stable_forecast(
    fwd: pd.DataFrame,
    seg_obs: pd.DataFrame,
    short_years: float,
    alpha: float,
    smooth_v: int,
    smooth_start: int,
    jump_k: float,
) -> pd.DataFrame:
    """Add ``last_dist_m``, ``v_pred``, ``short_years`` and ``v_long`` to ``fwd``.

    ``fwd`` has one row per strip with ``dist_t2`` (last measured position), ``t2``
    and the model's ``v_pred``; the raw values are kept as ``last_dist_raw_m`` and
    ``v_model``. See the module docstring for the method.
    """
    H = Hist(yearly(seg_obs))
    src = H.keys.assign(sam_last=H.sam_last)
    f = fwd.merge(src, on=KEY, how="left")
    f["sam_last"] = f["sam_last"].fillna(False).astype(bool)
    f["last_dist_raw_m"] = f["dist_t2"]
    f["v_model"] = f["v_pred"]
    f["start"] = f["dist_t2"]
    f["v"] = np.where(f["sam_last"], f["v_pred"], 0.0)
    f = smooth(f, smooth_v, cols=("v",))
    f = smooth(f, smooth_start, cols=("start",))
    f["short_years"] = np.where(f["sam_last"], float(short_years), 0.0)
    drift = regional_drift(H, alpha, jump_k)
    f["v_long"] = f[LOCATION_ID].map(drift).fillna(0.0)
    f["last_dist_m"] = f["start"]
    f["v_pred"] = f["v"]
    return f.drop(columns=["start", "v"])
