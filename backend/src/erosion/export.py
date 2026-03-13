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
import sqlite3
import time
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
    verbose: bool = True,
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
        verbose: Print progress steps.

    Returns:
        Path to the output GeoPackage.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    output_gpkg = Path(output_gpkg)

    # --- Step 1: Copy base GPKG ---
    _log(f"[1/4] Copying base GPKG → {output_gpkg.name} ...")
    t = time.time()
    shutil.copy2(base_gpkg, output_gpkg)
    for _ext in ("-wal", "-shm"):
        _sidecar = Path(str(output_gpkg) + _ext)
        if _sidecar.exists():
            _sidecar.unlink()
            _log(f"      removed stale sidecar: {_sidecar.name}")
    _log(f"      done  ({output_gpkg.stat().st_size / 1e6:.1f} MB)  {time.time()-t:.1f}s")

    # --- Step 2: Write predicted_bank_positions via temp GPKG + SQLite ATTACH ---
    bank_export = predicted_bank_positions[
        predicted_bank_positions.geometry.notna()
    ].copy()

    _log(f"[2/4] Writing {bank_layer} ({len(bank_export):,} rows) ...")
    t = time.time()
    bank_export = bank_export.to_crs(target_crs)

    tmp_gpkg = output_gpkg.with_suffix(".tmp.gpkg")
    bank_export.to_file(tmp_gpkg, layer=bank_layer, driver="GPKG")
    _log(f"      to_file done  {time.time()-t:.1f}s  ({tmp_gpkg.stat().st_size/1e6:.1f} MB)")

    t = time.time()
    for _ext in ("-wal", "-shm"):
        _sidecar = Path(str(tmp_gpkg) + _ext)
        if _sidecar.exists():
            _sidecar.unlink()
    _attach_copy_layer(tmp_gpkg, output_gpkg, bank_layer)
    tmp_gpkg.unlink()
    _log(f"      sqlite ATTACH copy done  {time.time()-t:.1f}s")

    # --- Step 3: Compute crossing year per VVR polygon ---
    _log(f"[3/4] Computing crossing year per VVR polygon ...")
    t = time.time()
    crossing_lookup = vvr_crossing.set_index("location_id")["crossing_year"].dropna()
    scope_matched = scope_raw[
        scope_raw["location_id"].isin(crossing_lookup.index)
    ][["location_id", "geometry"]].copy()

    vvr_geom = gpd.read_file(output_gpkg, layer=vvr_layer)
    vvr_pts = vvr_geom.to_crs(scope_matched.crs).copy()
    vvr_pts["geometry"] = vvr_pts.geometry.representative_point()

    joined = gpd.sjoin(vvr_pts[["geometry"]], scope_matched, how="left", predicate="within")
    joined[crossing_col] = joined["location_id"].map(crossing_lookup)
    earliest = joined.groupby(joined.index)[crossing_col].min()
    n_filled = earliest.notna().sum()
    _log(f"      done  {time.time()-t:.1f}s  → {n_filled:,}/{len(vvr_geom):,} assigned")

    # --- Step 4: Patch crossing_col into vvr_layer ---
    _log(f"[4/4] Patching {vvr_layer}.{crossing_col} ...")
    t = time.time()
    vvr_geom[crossing_col] = vvr_geom.index.map(earliest)
    vvr_geom.to_file(output_gpkg, layer=vvr_layer, driver="GPKG")
    _log(f"      done  {time.time()-t:.1f}s")

    return output_gpkg


def _attach_copy_layer(src_gpkg: Path, dst_gpkg: Path, layer: str) -> None:
    """Copy a layer from src_gpkg into dst_gpkg using SQLite ATTACH.

    Copies the feature table and the three GPKG metadata rows
    (gpkg_contents, gpkg_geometry_columns, gpkg_spatial_ref_sys).
    No GDAL/fiona involved — pure SQL.
    """
    con = sqlite3.connect(dst_gpkg)
    con.execute(f"ATTACH DATABASE '{src_gpkg}' AS src")
    con.execute("""
        INSERT OR IGNORE INTO gpkg_spatial_ref_sys
        SELECT * FROM src.gpkg_spatial_ref_sys
    """)
    con.execute(f"""
        INSERT OR REPLACE INTO gpkg_contents
        SELECT * FROM src.gpkg_contents WHERE table_name = '{layer}'
    """)
    con.execute(f"""
        INSERT OR REPLACE INTO gpkg_geometry_columns
        SELECT * FROM src.gpkg_geometry_columns WHERE table_name = '{layer}'
    """)
    con.execute(f'CREATE TABLE "{layer}" AS SELECT * FROM src."{layer}"')
    con.commit()
    con.execute("DETACH src")
    con.close()