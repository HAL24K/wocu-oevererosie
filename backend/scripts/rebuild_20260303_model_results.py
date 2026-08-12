#!/usr/bin/env python3
"""
Rebuild the 20260303_model_results.gpkg predicted_bank_positions and vvr_rates_of_change layers.

This script reverse-engineers the pipeline that produced the model output GeoPackage.
It loads the baseline model, processed erosion data, and bank points to regenerate:
- predicted_bank_positions: bank position predictions for 2025-2035 per location
- vvr_rates_of_change: updates predicted_vvr_crossing_year using model velocities

Run from backend/: python scripts/rebuild_20260303_model_results.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# Add backend to path for src imports
BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import src.paths as PATHS
import src.constants as CONST

# Paths
DATA_DIR = PATHS.DATA_DIR
MODEL_PKL = DATA_DIR / "04_model_outputs/20260217/baseline_model_full_dataset.pkl"
PARQUET_PATH = DATA_DIR / "02_processed/erosion/processed_erosion_data.parquet"
RAW_GPKG = DATA_DIR / "01_raw/erosion/wocu_output_fase2_20260210.gpkg"
OUTPUT_GPKG = DATA_DIR / "04_model_outputs/20260303/20260303_model_results.gpkg"
# Alexander used Luke's wocu_post_processed_fase2_20260223.gpkg as base (per his email)
POSTPROC_20260223 = DATA_DIR / "02_processed/erosion/wocu_post_processed_fase2_20260223.gpkg"
POSTPROC_20260310 = DATA_DIR / "02_processed/erosion/wocu_post_processed_fase2_20260310.gpkg"
POSTPROC_GPKG = POSTPROC_20260223 if POSTPROC_20260223.exists() else POSTPROC_20260310

REFERENCE_YEAR = 2025
PREDICTION_YEARS = list(range(2025, 2036))  # 2025-2035
DIST_COL = CONST.DISTANCE_TO_CENTERLINE


def load_baseline_model() -> dict:
    """Load baseline model and return velocity dict (location_id -> m/yr)."""
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    return model.model


def get_last_dist_and_year(processed: pd.DataFrame) -> pd.DataFrame:
    """
    For each location, get last observed distance and year.
    Extrapolate to REFERENCE_YEAR if last year < REFERENCE_YEAR.
    """
    records = []
    for location_id, group in processed.groupby(level=0):
        group_sorted = group.sort_index()
        years = group_sorted.index.get_level_values(1).tolist()
        dists = group_sorted[DIST_COL].tolist()
        last_year = years[-1]
        last_dist = dists[-1]
        records.append({
            "location_id": location_id,
            "last_year": last_year,
            "last_dist": last_dist,
        })
    df = pd.DataFrame(records).set_index("location_id")
    return df


def build_predicted_bank_positions(
    model_velocities: dict,
    last_dist_df: pd.DataFrame,
    bank_points: gpd.GeoDataFrame,
    nvo_location_ids: set[str],
) -> gpd.GeoDataFrame:
    """
    Build predicted_bank_positions layer.
    - last_dist at REFERENCE_YEAR (extrapolated if needed)
    - predicted_dist = last_dist + velocity * (year - REFERENCE_YEAR)
    - geometry from bank_points (centroid of most recent observation per location)
    """
    rows = []
    for location_id, velocity in model_velocities.items():
        if location_id not in last_dist_df.index:
            continue
        row = last_dist_df.loc[location_id]
        last_year = row["last_year"]
        last_dist = row["last_dist"]

        # Extrapolate to REFERENCE_YEAR if last observation is before
        if last_year < REFERENCE_YEAR:
            dist_at_ref = last_dist + velocity * (REFERENCE_YEAR - last_year)
        else:
            dist_at_ref = last_dist

        is_nvo = location_id in nvo_location_ids

        for year in PREDICTION_YEARS:
            predicted_dist = dist_at_ref + velocity * (year - REFERENCE_YEAR)
            rows.append({
                "location_id": location_id,
                "year": year,
                "predicted_dist_m": predicted_dist,
                "velocity_m_per_yr": velocity,
                "is_nvo": is_nvo,
            })

    df = pd.DataFrame(rows)

    # Get geometry: one point per location from most recent bank observation
    bank_points = bank_points.copy()
    if "dtm_date" in bank_points.columns:
        bank_points["_dtm"] = bank_points["dtm_date"]
    else:
        bank_points["_dtm"] = 0
    last_obs = bank_points.loc[bank_points.groupby("location_id")["_dtm"].idxmax()]
    geom_per_loc = last_obs.groupby("location_id").agg(
        geometry=("geometry", lambda g: g.unary_union.centroid if len(g) > 1 else g.iloc[0])
    )
    df = df.merge(geom_per_loc, left_on="location_id", right_index=True, how="left")
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=bank_points.crs)
    return gdf


def get_nvo_location_ids(
    vvr_gdf: gpd.GeoDataFrame,
    scope_gdf: gpd.GeoDataFrame,
) -> set[str]:
    """
    Spatial join vvr_rates_of_change with summary_scope to get location_ids.
    Locations with VVR coverage are NVO.
    """
    if vvr_gdf.crs != scope_gdf.crs:
        vvr_gdf = vvr_gdf.to_crs(scope_gdf.crs)
    scope_renamed = scope_gdf.rename(columns={"position_id": "location_id"})
    joined = gpd.sjoin(
        vvr_gdf[["geometry"]],
        scope_renamed[["location_id", "geometry"]],
        how="left",
        predicate="intersects",
    )
    # Deduplicate: take first match per vvr row (drop index_right)
    joined = joined.drop(columns=["index_right"] if "index_right" in joined.columns else [])
    joined = joined[~joined["location_id"].isna()]
    # One vvr row per location - take unique
    return set(joined["location_id"].astype(str).unique())


def update_vvr_predicted_crossing_year(
    vvr_gdf: gpd.GeoDataFrame,
    scope_gdf: gpd.GeoDataFrame,
    model_velocities: dict,
) -> gpd.GeoDataFrame:
    """
    Add location_id to vvr via spatial join, then recompute predicted_vvr_crossing_year.
    Formula: crossing_year = REFERENCE_YEAR + distance_to_signaleringlijn / velocity
    For velocity <= 0 (accretion): no crossing -> NaN
    Preserves original vvr row count (one row per vvr polygon).
    """
    dist_col = "distance_to_signaleringlijn_most_recent"
    if dist_col not in vvr_gdf.columns:
        return vvr_gdf

    vvr = vvr_gdf.copy()
    vvr["_vvr_idx"] = range(len(vvr))
    if vvr.crs != scope_gdf.crs:
        vvr = vvr.to_crs(scope_gdf.crs)
    scope_renamed = scope_gdf.rename(columns={"position_id": "location_id"})
    joined = gpd.sjoin(
        vvr,
        scope_renamed[["location_id", "geometry"]],
        how="left",
        predicate="intersects",
    )
    # One location_id per vvr row: take first match
    first_match = joined.groupby("_vvr_idx").first().reset_index(drop=True)
    first_match = first_match.drop(columns=["index_right"], errors="ignore")

    def crossing_year(row):
        loc_id = row.get("location_id")
        if pd.isna(loc_id) or str(loc_id) not in model_velocities:
            return np.nan
        vel = model_velocities[str(loc_id)]
        dist = row.get(dist_col)
        if pd.isna(dist) or vel <= 0:
            return np.nan
        yr = REFERENCE_YEAR + dist / vel
        return np.clip(yr, 2020, 2099)

    crossing = first_match.apply(crossing_year, axis=1)
    result = vvr_gdf.copy()
    result["predicted_vvr_crossing_year"] = crossing.values
    return result


def main() -> None:
    print("Loading baseline model...")
    model_velocities = load_baseline_model()
    print(f"  {len(model_velocities):,} locations with velocities")

    print("Loading processed_erosion_data.parquet...")
    processed = pd.read_parquet(PARQUET_PATH)
    processed.index.names = ["location_id", "year"]
    if DIST_COL not in processed.columns:
        raise ValueError(f"Expected column {DIST_COL} in parquet")
    last_dist_df = get_last_dist_and_year(processed)
    print(f"  {len(last_dist_df):,} locations")

    print("Loading punten_oever for geometry...")
    bank_points = gpd.read_file(RAW_GPKG, layer="punten_oever")
    print(f"  {len(bank_points):,} points, {bank_points['location_id'].nunique():,} locations")

    print("Loading vvr_rates_of_change and summary_scope...")
    # Use post-processed as source for vvr (canonical 1359 rows); scope from output
    vvr = gpd.read_file(POSTPROC_GPKG, layer="vvr_rates_of_change")
    scope = gpd.read_file(OUTPUT_GPKG, layer="summary_scope")
    print(f"  vvr: {len(vvr)} rows, scope: {len(scope)} rows")

    print("Computing NVO location_ids from vvr spatial join...")
    nvo_ids = get_nvo_location_ids(vvr, scope)
    print(f"  {len(nvo_ids):,} NVO locations")

    print("Building predicted_bank_positions...")
    pbp = build_predicted_bank_positions(
        model_velocities, last_dist_df, bank_points, nvo_ids
    )
    print(f"  {len(pbp):,} rows, {pbp['location_id'].nunique():,} locations")

    print("Updating vvr_rates_of_change predicted_vvr_crossing_year...")
    vvr_updated = update_vvr_predicted_crossing_year(vvr, scope, model_velocities)

    print(f"Writing to {OUTPUT_GPKG}...")
    pbp.to_file(OUTPUT_GPKG, layer="predicted_bank_positions", driver="GPKG")
    vvr_updated.to_file(OUTPUT_GPKG, layer="vvr_rates_of_change", driver="GPKG")
    print("Done.")


if __name__ == "__main__":
    main()
