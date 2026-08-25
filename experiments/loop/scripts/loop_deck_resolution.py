"""Deck experiment B — resolution sweep R = 1..100 on the graduated recipe
(e8 obs + traj2 segment features), frozen holdout, test-ES as in track 3.
Stops when segments starve (reports empty-segment share)."""

import logging
import sys
import warnings

import pandas as pd

from experiments.loop.harness.harness import load_caches
from experiments.loop.harness.resolution import (
    load_dense,
    run_t3_variant,
    segment_observations,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

caches = load_caches()
obs = pd.read_parquet(caches.cache_dir / "obs_e8.parquet")
dense = load_dense(caches)
print(f"dense: {len(dense):,} samples · {dense.location_id.nunique():,} regions")

Rs = [int(a) for a in sys.argv[1:]] or [1, 2, 5, 10, 20, 50, 100]
for R in Rs:
    seg = segment_observations(dense, R)
    n_possible = dense.location_id.nunique() * R
    n_seen = seg.groupby(["location_id", "seg"]).ngroups
    yrs = seg.groupby(["location_id", "seg"])["date"].apply(
        lambda d: d.dt.year.nunique()
    )
    print(
        f"R={R}: segments with any data {n_seen:,}/{n_possible:,} "
        f"({1 - n_seen / n_possible:.1%} empty) · ≥3 years {(yrs >= 3).sum():,}"
    )
    run_t3_variant(
        f"deck-R{R}-traj2",
        caches,
        dense,
        obs,
        R=R,
        features="traj2",
        notes=f"deck resolution sweep R={R}",
    )
