"""Step 01 — Bank distances.

Reduces raw WOCU bank point clouds (``punten_oever``) to a single representative
distance per scope region per survey year: the mean of the N furthest OK-status
points per (location_id, year).

The algorithm lives in :class:`src.sources.observations.HeightModelPointSource`;
this module is the pipeline-facing wrapper kept for the step-01 name and the
notebook call sites. Output is verified byte-identical to the 20260617a
reference.
"""

from pathlib import Path

import pandas as pd

from src.sources import HeightModelPointSource


def compute_bank_distances(raw_gpkg: Path, n_points: int = 3) -> pd.DataFrame:
    """Compute mean bank distance per region per year from raw point data.

    Args:
        raw_gpkg: Path to the raw WOCU GeoPackage containing ``punten_oever``.
        n_points: Number of furthest OK points to average per (location_id, year).

    Returns:
        DataFrame with columns: location_id, year, dist_m, n_ok_pts, n_selected.
        Sorted by (location_id, year).
    """
    return HeightModelPointSource(raw_gpkg, n_points=n_points).load().to_dist_per_year()
