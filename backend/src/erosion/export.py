"""GeoPackage export for erosion prediction results.

Creates a dated copy of a base GeoPackage and appends prediction outputs:

  - ``predicted_bank_positions``: new GeoPoint layer — one row per (location_id, year)
    with predicted dist, velocity, and is_nvo flag (stored as integer 0/1).
  - ``predicted_vvr_crossing_year``: new column on the existing
    ``vvr_rates_of_change`` layer — earliest predicted crossing year per VVR polygon,
    derived via spatial join with scope regions. Three possible values:

      * 2026–end_year  — bank crosses the signaleringslijn within the prediction window
      * 9999           — bank was predicted but does not cross before the end year ("safe")
      * NULL           — no scope region with predictions matched this VVR polygon
  - ``signaleringslijn`` (optional): new LineString layer with the VVR boundary lines
    used as crossing threshold reference.

Usage::

    from src.erosion.export import export_predictions

    output_path = export_predictions(
        base_gpkg=POSTPROC_GPKG,
        output_gpkg=OUTPUT_GPKG,
        predicted_bank_positions=predicted_bank_positions,
        vvr_crossing=vvr_crossing,
        scope_raw=scope_raw,
        signaleringslijn=signalering,  # optional
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
    signaleringslijn: gpd.GeoDataFrame | None = None,
    bank_layer: str = "predicted_bank_positions",
    vvr_layer: str = "vvr_rates_of_change",
    crossing_col: str = "predicted_vvr_crossing_year",
    signalering_layer: str = "signaleringslijn",
    target_crs: int = 28992,
    verbose: bool = True,
) -> Path:
    """Copy base GeoPackage and append prediction results.

    Args:
        base_gpkg:   Path to the original (unmodified) base GeoPackage.
        output_gpkg: Destination path. Will be overwritten if it exists.
        predicted_bank_positions: GeoDataFrame with columns
            location_id, year, predicted_dist_m, velocity_m_per_yr, is_nvo, geometry.
            ``is_nvo`` will be cast to int (0/1) before writing to avoid pyogrio
            misinterpreting a bool column as a geometry field.
        vvr_crossing: DataFrame from ``compute_vvr_crossing_year`` with columns
            location_id, crossing_year (and optionally dist_to_vvr_m, years_to_crossing).
        scope_raw: GeoDataFrame with location_id + geometry (vlakken_scope).
            Used to spatially link scope regions to VVR polygons.
        signaleringslijn: Optional GeoDataFrame of VVR boundary lines to include as
            a standalone layer in the output GPKG.
        bank_layer: Name for the new bank positions layer.
        vvr_layer:  Name of the existing VVR layer to patch.
        crossing_col: Name of the new attribute added to ``vvr_layer``.
        signalering_layer: Name for the optional signaleringslijn layer.
        target_crs: EPSG code to reproject outputs to (default: RD New / 28992).
        verbose: Print progress steps.

    Returns:
        Path to the output GeoPackage.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    output_gpkg = Path(output_gpkg)
    n_steps = 5 if signaleringslijn is not None else 4

    # --- Step 1: Copy base GPKG ---
    _log(f"[1/{n_steps}] Copying base GPKG → {output_gpkg.name} ...")
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
    # Cast is_nvo to int to prevent pyogrio from misidentifying a bool column as geometry
    if "is_nvo" in bank_export.columns:
        bank_export["is_nvo"] = bank_export["is_nvo"].astype(int)

    _log(f"[2/{n_steps}] Writing {bank_layer} ({len(bank_export):,} rows) ...")
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
    _log(f"[3/{n_steps}] Computing crossing year per VVR polygon ...")
    t = time.time()
    crossing_lookup = vvr_crossing.set_index("location_id")["crossing_year"].dropna()
    scope_matched = scope_raw[
        scope_raw["location_id"].isin(crossing_lookup.index)
    ][["location_id", "geometry"]].copy()

    vvr_geom = gpd.read_file(output_gpkg, layer=vvr_layer)
    vvr_pts = vvr_geom.to_crs(scope_matched.crs).copy()
    vvr_pts["geometry"] = vvr_pts.geometry.representative_point()

    # Left-join: every VVR polygon gets matched scope region(s) if any overlap.
    # scope_matched only contains locations that have a crossing_year, so rows
    # that join but have no crossing_year means the scope region was predicted
    # but the bank never crosses before the end year.
    scope_all_predicted = scope_raw[
        scope_raw["location_id"].isin(vvr_crossing["location_id"])
    ][["location_id", "geometry"]].copy()

    joined_all = gpd.sjoin(vvr_pts[["geometry"]], scope_all_predicted, how="left", predicate="within")
    has_any_prediction = joined_all.groupby(joined_all.index)["location_id"].count() > 0

    joined_all[crossing_col] = joined_all["location_id"].map(crossing_lookup)
    earliest = joined_all.groupby(joined_all.index)[crossing_col].min()

    # Sentinel logic:
    #   crosses before end year  → actual year (2026–end_year)
    #   predicted but no crossing → NO_CROSSING_SENTINEL (9999)
    #   no matching scope region  → NULL (no prediction available)
    NO_CROSSING_SENTINEL = 9999
    earliest = earliest.where(earliest.notna(), other=pd.Series(
        {idx: NO_CROSSING_SENTINEL if has_any_prediction.get(idx, False) else float("nan")
         for idx in earliest.index}
    ))

    n_crossing = (earliest < NO_CROSSING_SENTINEL).sum()
    n_safe     = (earliest == NO_CROSSING_SENTINEL).sum()
    n_null     = earliest.isna().sum()
    _log(f"      done  {time.time()-t:.1f}s  → {n_crossing:,} crossing  "
         f"{n_safe:,} safe (={NO_CROSSING_SENTINEL})  {n_null:,} no-prediction (NULL)")

    # --- Step 4: Patch crossing_col into vvr_layer ---
    _log(f"[4/{n_steps}] Patching {vvr_layer}.{crossing_col} ...")
    t = time.time()
    vvr_geom[crossing_col] = vvr_geom.index.map(earliest)
    vvr_geom.to_file(output_gpkg, layer=vvr_layer, driver="GPKG")
    _log(f"      done  {time.time()-t:.1f}s")

    # --- Step 5 (optional): Write signaleringslijn layer ---
    if signaleringslijn is not None:
        _log(f"[5/{n_steps}] Writing {signalering_layer} ({len(signaleringslijn):,} features) ...")
        t = time.time()
        sig_export = signaleringslijn.to_crs(target_crs)
        tmp_sig = output_gpkg.with_suffix(".sig.tmp.gpkg")
        sig_export.to_file(tmp_sig, layer=signalering_layer, driver="GPKG")
        for _ext in ("-wal", "-shm"):
            _sidecar = Path(str(tmp_sig) + _ext)
            if _sidecar.exists():
                _sidecar.unlink()
        _attach_copy_layer(tmp_sig, output_gpkg, signalering_layer)
        tmp_sig.unlink()
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