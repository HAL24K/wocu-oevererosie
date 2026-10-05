"""Along-bank consensus between the surveys of one region.

A survey is compared with the other surveys station bin by station bin (the
``fragment_survey`` bins), not as one median distance: a cloud or a shadow
usually displaces part of a line, which a single number hides. The reference
is the largest group of surveys that agree with each other within ``tol`` —
a leave-one-out median is masked as soon as two of five surveys are bad.
Positive deviation = landward (further from the river centreline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.cleaning.rules import N_STATION_BINS

Q = 0.8  # a pair disagrees when it is off over at least ~20 % of the shared bank
MIN_SHARED_BINS = 5
MIN_SHARE_OFF = 0.3  # partial displacements shorter than this are left alone


@dataclass
class Consensus:
    profiles: pd.DataFrame  # survey date x station bin -> median dist (NaN where not covered)
    core: list  # dates of the agreeing group
    off: pd.DataFrame  # per flagged date: dev_m (signed median), share_off, bins_off (list)
    reason: str  # "ok", "no majority", "too few surveys"
    consensus: pd.Series | None = None  # median profile of the core


def survey_profiles(g: pd.DataFrame) -> pd.DataFrame:
    """Median dist per (survey date, station bin) for one region's samples."""
    b = np.minimum((g["station"] * N_STATION_BINS).astype(int), N_STATION_BINS - 1)
    prof = g.assign(bin=b).groupby(["date", "bin"])["dist"].median().unstack("bin")
    prof.index = pd.to_datetime(prof.index)
    return prof.reindex(columns=range(N_STATION_BINS)).sort_index()


def pair_distance(prof: pd.DataFrame) -> pd.DataFrame:
    """Q-quantile of |dist difference| over the bins two surveys share (NaN if too few)."""
    v = prof.values
    n = len(v)
    D = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            m = ~np.isnan(v[i]) & ~np.isnan(v[j])
            if m.sum() >= MIN_SHARED_BINS:
                D[i, j] = D[j, i] = np.quantile(np.abs(v[i, m] - v[j, m]), Q)
    return pd.DataFrame(D, index=prof.index, columns=prof.index)


def survey_consensus(
    g: pd.DataFrame, tol: float = 6.0, abs_min: float = 8.0, min_surveys: int = 3
) -> Consensus:
    """Find the agreeing core of a region's surveys and the surveys clearly off it."""
    prof = survey_profiles(g)
    empty = pd.DataFrame(columns=["dev_m", "share_off", "bins_off"])
    if len(prof) < min_surveys:
        return Consensus(prof, list(prof.index), empty, "too few surveys")
    D = pair_distance(prof)
    far = np.nanmax(D.values) if np.isfinite(D.values).any() else 0.0
    Dm = D.fillna(far + tol).values.copy()
    np.fill_diagonal(Dm, 0.0)
    labels = fcluster(linkage(squareform(Dm, checks=False), "complete"), t=tol, criterion="distance")
    sizes = pd.Series(labels).value_counts()
    tie = len(sizes) > 1 and sizes.iloc[0] == sizes.iloc[1]
    # A reference of 2 out of 6 is not a consensus; it just paints the rest red.
    if sizes.iloc[0] < max(2, len(prof) / 2) or tie:
        return Consensus(prof, [], empty, "no majority")
    core = list(prof.index[labels == sizes.index[0]])
    ref = prof.loc[core].median()
    rows = {}
    for d in prof.index.difference(core):
        dev = prof.loc[d] - ref
        shared = dev.dropna()
        if len(shared) < MIN_SHARED_BINS:
            continue
        bad = shared[shared.abs() >= abs_min]
        if len(bad) / len(shared) >= MIN_SHARE_OFF:
            rows[d] = {
                "dev_m": float(bad.median()) if len(bad) else float(shared.median()),
                "share_off": len(bad) / len(shared),
                "bins_off": list(bad.index),
            }
    off = pd.DataFrame.from_dict(rows, orient="index") if rows else empty
    return Consensus(prof, core, off, "ok", ref)
