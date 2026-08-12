#!/usr/bin/env python3
"""
Acceptance tests for the rebuilt 20260303_model_results.gpkg.

Validates predicted_bank_positions and vvr_rates_of_change layers against
expected structure and consistency rules.

Run from backend/: python scripts/validate_rebuilt_model_output.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import src.paths as PATHS

DATA_DIR = PATHS.DATA_DIR
MODEL_PKL = DATA_DIR / "04_model_outputs/20260217/baseline_model_full_dataset.pkl"
GPKG_PATH = DATA_DIR / "04_model_outputs/20260303/20260303_model_results.gpkg"
REFERENCE_YEAR = 2025
PREDICTION_YEARS = list(range(2025, 2036))


def load_baseline_model() -> dict:
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    return model.model


def test_predicted_bank_positions(
    pbp: gpd.GeoDataFrame,
    model_velocities: dict,
) -> list[str]:
    """Check predicted_bank_positions against acceptance criteria. Returns list of failures."""
    failures = []

    # 1. Required columns
    required = ["location_id", "year", "predicted_dist_m", "velocity_m_per_yr", "is_nvo", "geometry"]
    for col in required:
        if col not in pbp.columns:
            failures.append(f"Missing column: {col}")

    # 2. Row count: 10,484 locations × 11 years = 115,324
    n_locs = pbp["location_id"].nunique()
    n_rows = len(pbp)
    expected_rows = len(model_velocities) * 11
    if n_rows != expected_rows:
        failures.append(f"Row count: got {n_rows:,}, expected {expected_rows:,}")
    if n_locs != len(model_velocities):
        failures.append(f"Location count: got {n_locs:,}, expected {len(model_velocities):,}")

    # 3. Each location has exactly 11 rows (years 2025-2035)
    years_per_loc = pbp.groupby("location_id")["year"].apply(lambda s: sorted(s.unique().tolist()))
    wrong_years = years_per_loc[years_per_loc != years_per_loc.apply(lambda y: y == PREDICTION_YEARS)]
    if len(wrong_years) > 0:
        failures.append(f"Locations with wrong years: {len(wrong_years)} (e.g. {wrong_years.index[0]})")

    # 4. velocity_m_per_yr matches baseline model and is constant per location
    for loc_id, grp in pbp.groupby("location_id"):
        vels = grp["velocity_m_per_yr"].unique()
        if len(vels) != 1:
            failures.append(f"Location {loc_id}: velocity not constant across years")
            break
        expected_vel = model_velocities.get(loc_id)
        if expected_vel is None:
            failures.append(f"Location {loc_id}: not in baseline model")
            break
        if not np.isclose(vels[0], expected_vel, rtol=1e-9):
            failures.append(f"Location {loc_id}: velocity {vels[0]} != model {expected_vel}")
            break

    # 5. predicted_dist_m follows arithmetic: dist(y+1) = dist(y) + velocity
    for loc_id, grp in pbp.groupby("location_id"):
        grp = grp.sort_values("year")
        dists = grp["predicted_dist_m"].values
        vel = grp["velocity_m_per_yr"].iloc[0]
        for i in range(len(dists) - 1):
            expected = dists[i] + vel
            if not np.isclose(dists[i + 1], expected, rtol=1e-9):
                failures.append(
                    f"Location {loc_id}: dist[{i+1}]={dists[i+1]:.4f} != dist[{i}]+vel={expected:.4f}"
                )
                break
        if failures:
            break

    # 6. is_nvo is constant per location
    nvo_per_loc = pbp.groupby("location_id")["is_nvo"].nunique()
    if (nvo_per_loc > 1).any():
        bad = nvo_per_loc[nvo_per_loc > 1].index[0]
        failures.append(f"Location {bad}: is_nvo not constant across years")

    # 7. Geometry: valid Point, not null
    if pbp["geometry"].isna().any():
        failures.append(f"Null geometry: {pbp['geometry'].isna().sum()} rows")
    if not pbp["geometry"].geom_type.str.contains("Point", na=False).all():
        failures.append("Some geometries are not Point type")

    return failures


def test_vvr_rates_of_change(
    vvr: gpd.GeoDataFrame,
    model_velocities: dict,
) -> list[str]:
    """Check vvr_rates_of_change and predicted_vvr_crossing_year. Returns list of failures."""
    failures = []

    if "predicted_vvr_crossing_year" not in vvr.columns:
        failures.append("Missing column: predicted_vvr_crossing_year")
        return failures

    if "distance_to_signaleringlijn_most_recent" not in vvr.columns:
        failures.append("Missing column: distance_to_signaleringlijn_most_recent")
        return failures

    # VVR should have ~1,359 rows (NVO regions only)
    if len(vvr) < 1000 or len(vvr) > 2000:
        failures.append(f"vvr row count: {len(vvr)} (expected ~1,359)")

    # predicted_vvr_crossing_year: when present, should be in [2020, 2099]
    valid = vvr["predicted_vvr_crossing_year"].dropna()
    if len(valid) > 0:
        if valid.min() < 2020 or valid.max() > 2099:
            failures.append(f"predicted_vvr_crossing_year out of range: min={valid.min()}, max={valid.max()}")

    return failures


def main() -> int:
    print("Loading baseline model...")
    model_velocities = load_baseline_model()
    print(f"  {len(model_velocities):,} locations")

    print(f"Loading {GPKG_PATH}...")
    if not GPKG_PATH.exists():
        print(f"  ERROR: File not found")
        return 1

    pbp = gpd.read_file(GPKG_PATH, layer="predicted_bank_positions")
    vvr = gpd.read_file(GPKG_PATH, layer="vvr_rates_of_change")

    print("\n--- predicted_bank_positions acceptance tests ---")
    pbp_failures = test_predicted_bank_positions(pbp, model_velocities)
    if pbp_failures:
        for f in pbp_failures:
            print(f"  FAIL: {f}")
    else:
        print("  All passed.")

    print("\n--- vvr_rates_of_change acceptance tests ---")
    vvr_failures = test_vvr_rates_of_change(vvr, model_velocities)
    if vvr_failures:
        for f in vvr_failures:
            print(f"  FAIL: {f}")
    else:
        print("  All passed.")

    all_failures = pbp_failures + vvr_failures
    if all_failures:
        print(f"\n{len(all_failures)} failure(s) total.")
        return 1
    print("\nAll acceptance tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
