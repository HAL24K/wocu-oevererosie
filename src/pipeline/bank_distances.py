"""Step 01 — Bank distances.

Reduces raw WOCU bank point clouds (`punten_oever`) to a single representative
distance per scope region per survey year.

Algorithm:
  1. Keep only status == 'OK' points.
  2. For each (location_id, year), select the N_POINTS furthest OK points.
  3. Take their mean distance from the centreline → dist_m.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def compute_bank_distances(raw_gpkg: Path, n_points: int = 3) -> pd.DataFrame:
    """Compute mean bank distance per region per year from raw point data.

    Args:
        raw_gpkg: Path to the raw WOCU GeoPackage containing the ``punten_oever`` layer.
        n_points: Number of furthest OK points to average per (location_id, year).

    Returns:
        DataFrame with columns: location_id, year, dist_m, n_ok_pts, n_selected.
        Sorted by (location_id, year).
    """
    pts = gpd.read_file(raw_gpkg, layer="punten_oever")

    ok = pts[pts["status"] == "OK"].copy()
    ok = ok.rename(columns={"dtm_date": "year"})
    ok["year"] = ok["year"].astype(int)

    dist_per_year = (
        ok.groupby(["location_id", "year"], group_keys=False)
        .apply(_top_n_mean_dist, n=n_points, include_groups=False)
        .reset_index()
    )

    return dist_per_year.sort_values(["location_id", "year"]).reset_index(drop=True)


def _top_n_mean_dist(group: pd.DataFrame, n: int) -> pd.Series:
    """Aggregate a single (location_id, year) group into one distance row."""
    chosen = group.nlargest(n, "dist")
    return pd.Series(
        {
            "dist_m": chosen["dist"].mean(),
            "n_ok_pts": len(group),
            "n_selected": len(chosen),
        }
    )
