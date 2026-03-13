"""GeoPackage export for erosion prediction results.

Creates a dated copy of a base GeoPackage and appends prediction outputs:

  - ``predicted_bank_positions``: new GeoPoint layer — one row per (location_id, year)
    with predicted dist, velocity, and is_nvo flag.
  - ``predicted_vvr_crossing_year``: new column on the existing
    ``vvr_rates_of_change`` layer — earliest predicted crossing year per VVR polygon,
    derived via spatial join with scope regions.

Usage::

    from src.erosion.export import export_predictions

    output_path = export_predictions(
        base_gpkg=POSTPROC_GPKG,
        output_gpkg=OUTPUT_GPKG,
        predicted_bank_positions=predicted_bank_positions,
        vvr_crossing=vvr_crossing,
        scope_raw=scope_raw,
    )
"""

from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd


def export_predictions(
    base_gpkg: Path,
    output_gpkg: Path,
    predicted_bank_positions: gpd.GeoDataFrame,
    vvr_crossing: pd.DataFrame,
    scope_raw: gpd.GeoDataFrame,
    bank_layer: str = "predicted_bank_positions",
    vvr_layer: str = "vvr_rates_of_change",
    crossing_col: str = "predicted_vvr_crossing_year",
    target_crs: int = 28992,
) -> Path:
    """Copy base GeoPackage and append prediction results.

    Args:
        base_gpkg:   Path to the original (unmodified) base GeoPackage.
        output_gpkg: Destination path. Will be overwritten if it exists.
        predicted_bank_positions: GeoDataFrame with columns
            location_id, year, predicted_dist_m, velocity_m_per_yr, is_nvo, geometry.
        vvr_crossing: DataFrame from ``compute_vvr_crossing_year`` with columns
            location_id, crossing_year (and optionally dist_to_vvr_m, years_to_crossing).
        scope_raw: GeoDataFrame with location_id + geometry (vlakken_scope).
            Used to spatially link scope regions to VVR polygons.
        bank_layer: Name for the new bank positions layer.
        vvr_layer:  Name of the existing VVR layer to patch.
        crossing_col: Name of the new attribute added to ``vvr_layer``.
        target_crs: EPSG code to reproject outputs to (default: RD New / 28992).

    Returns:
        Path to the output GeoPackage.
    """
    output_gpkg = Path(output_gpkg)
    shutil.copy2(base_gpkg, output_gpkg)

    # --- Layer 1: predicted_bank_positions ---
    bank_export = predicted_bank_positions[
        predicted_bank_positions.geometry.notna()
    ].copy()
    bank_export = bank_export.to_crs(target_crs)
    bank_export.to_file(output_gpkg, layer=bank_layer, driver="GPKG")

    # --- Layer 2: add crossing_col to vvr_layer ---
    # Each VVR polygon may overlap multiple scope regions.
    # We take the minimum (earliest / most urgent) crossing year.
    crossing_lookup = (
        vvr_crossing.set_index("location_id")["crossing_year"].dropna()
    )
    scope_matched = scope_raw[
        scope_raw["location_id"].isin(crossing_lookup.index)
    ][["location_id", "geometry"]]

    vvr_gdf = gpd.read_file(output_gpkg, layer=vvr_layer)
    vvr_proj = vvr_gdf.to_crs(scope_matched.crs)

    joined = gpd.sjoin(
        vvr_proj[["geometry"]], scope_matched, how="left", predicate="intersects"
    )
    joined[crossing_col] = joined["location_id"].map(crossing_lookup)
    earliest = joined.groupby(joined.index)[crossing_col].min()

    vvr_gdf[crossing_col] = earliest.reindex(vvr_gdf.index).values
    vvr_gdf.to_file(output_gpkg, layer=vvr_layer, driver="GPKG")

    return output_gpkg
