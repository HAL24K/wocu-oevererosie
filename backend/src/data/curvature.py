"""
src/features/curvature.py

Menger curvature and bend-exposure features for river bank scope regions.

Usage
-----
from src.features.curvature import add_curvature_features

scope_with_curvature = add_curvature_features(scope)

Output columns (per scale N in scales):
    curvature_n{N}      float  1/m, unsigned magnitude
    bend_side_n{N}      float  +1 = outer bend, -1 = inner bend, 0 = straight
    bend_exposure_n{N}  float  signed: positive = outer (erosion prone)

Sign convention verified empirically: outer arc scope regions are geometrically
larger than inner arc regions, consistent with meander geometry.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


# ── Core geometry ──────────────────────────────────────────────────────────


def _menger_curvature_signed(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> tuple[float, float]:
    """
    Menger curvature magnitude at p2 given three ordered 2-D points.

    Returns
    -------
    kappa : float
        Curvature in 1/m (0 = straight).
    cross_sign : float
        Sign of the cross product (+1 = left turn, -1 = right turn, 0 = collinear).
    """
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    if a * b * c == 0:
        return 0.0, 0.0
    cross = np.cross(p2 - p1, p3 - p1)  # positive = left turn
    area = abs(cross) / 2.0
    kappa = (2.0 * area) / (a * b * c)
    return float(kappa), float(np.sign(cross))


def _compute_curvature_arrays(
    coords: np.ndarray,
    bank: str,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute curvature, bend_side, bend_exposure for an ordered sequence of
    2-D centroids belonging to a single (river, bank) group.

    Parameters
    ----------
    coords : (N, 2) array of centroid coordinates in a metric CRS
    bank : 'l' or 'r'
    n_neighbors : window half-width (use centroid i±n for curvature at i)

    Returns
    -------
    curvatures, bend_side, bend_exposure : (N,) arrays
    """
    n = len(coords)
    curvatures = np.zeros(n)
    cross_signs = np.zeros(n)

    for i in range(n):
        i_left = max(0, i - n_neighbors)
        i_right = min(n - 1, i + n_neighbors)
        if i_left == i or i_right == i:
            continue  # edge region — not enough context
        kappa, sign = _menger_curvature_signed(
            coords[i_left], coords[i], coords[i_right]
        )
        curvatures[i] = kappa
        cross_signs[i] = sign

    # Determine inner / outer from cross product sign and bank side.
    # cross_sign > 0 → bend curves left (towards left bank).
    # Raw convention is inverted after computation based on area sanity check
    # (outer arc must be geometrically larger than inner arc).
    if bank == "l":
        bend_side = np.where(cross_signs > 0, 1.0, np.where(cross_signs < 0, -1.0, 0.0))
    else:
        bend_side = np.where(cross_signs < 0, 1.0, np.where(cross_signs > 0, -1.0, 0.0))

    # Invert signs — verified correct by area sanity check
    bend_side = -bend_side
    bend_exposure = curvatures * bend_side

    return curvatures, bend_side, bend_exposure


# ── Public API ─────────────────────────────────────────────────────────────


def add_curvature_features(
    scope: gpd.GeoDataFrame,
    scales: list[int] | None = None,
    river_col: str = "river",
    bank_col: str = "bank",
    chainage_col: str = "chainage",
    crs_metric: str = "EPSG:28992",
) -> gpd.GeoDataFrame:
    """
    Add Menger curvature and bend-exposure features to a scope region GeoDataFrame.

    Each scope region is characterised by its position along a river meander
    using the centroids of neighbouring regions as a proxy for local channel
    curvature. Three spatial scales are supported simultaneously, capturing
    local irregularities (n=3, ~300 m), primary meander geometry (n=5, ~500 m),
    and regional sinuosity (n=8, ~800 m).

    Parameters
    ----------
    scope : GeoDataFrame
        Must contain polygon geometries and columns for river, bank, and chainage.
        Index should be the location_id.
    scales : list of int, optional
        Neighbor window half-widths. Default [3, 5, 8].
    river_col : str
        Column identifying the river segment (e.g. 'maas1', 'nederrijn').
    bank_col : str
        Column identifying bank side: 'l' (left) or 'r' (right).
    chainage_col : str
        Column with along-river distance in metres, used for ordering.
    crs_metric : str
        EPSG code of a metric CRS for distance calculations.
        Default EPSG:28992 (RD New, suitable for the Netherlands).

    Returns
    -------
    GeoDataFrame
        Input GeoDataFrame with added columns:
            curvature_n{N}      unsigned curvature magnitude in 1/m
            bend_side_n{N}      +1 = outer bend, -1 = inner bend, 0 = straight
            bend_exposure_n{N}  signed feature: positive = outer (erosion prone)
        Original CRS is restored after computation.
        Any pre-existing curvature columns are overwritten.

    Notes
    -----
    Recommended modelling features:
        bend_exposure_n5    primary scale, best for Maas-style meanders
        curvature_n5        unsigned magnitude if sign is not needed
        bend_exposure_n8    large-scale sinuosity complement
    Drop curvature_n3 / bend_exposure_n3 if noisy — n=3 is sensitive to
    digitisation irregularities in scope region boundaries.
    """
    if scales is None:
        scales = [3, 5, 8]

    original_crs = scope.crs

    # Reproject to metric CRS for distance calculations
    if scope.crs is None or scope.crs.to_epsg() != int(crs_metric.split(":")[1]):
        logger.info("Reprojecting scope to %s for curvature computation.", crs_metric)
        scope_m = scope.to_crs(crs_metric)
    else:
        scope_m = scope.copy()

    # Initialise output columns
    for n in scales:
        scope_m[f"curvature_n{n}"] = np.nan
        scope_m[f"bend_side_n{n}"] = np.nan
        scope_m[f"bend_exposure_n{n}"] = np.nan

    groups = scope_m.groupby([river_col, bank_col])
    n_groups = groups.ngroups
    logger.info(
        "Computing curvature at scales %s for %d (river, bank) groups.",
        scales,
        n_groups,
    )

    for (river, bank), grp in groups:
        grp_sorted = grp.sort_values(chainage_col)
        coords = np.array(
            [(geom.centroid.x, geom.centroid.y) for geom in grp_sorted.geometry]
        )

        if len(coords) < 3:
            logger.debug(
                "Skipping (%s, %s) — fewer than 3 regions, curvature undefined.",
                river,
                bank,
            )
            continue

        for n in scales:
            curvatures, bend_side, bend_exposure = _compute_curvature_arrays(
                coords, bank=bank, n_neighbors=n
            )
            scope_m.loc[grp_sorted.index, f"curvature_n{n}"] = curvatures
            scope_m.loc[grp_sorted.index, f"bend_side_n{n}"] = bend_side
            scope_m.loc[grp_sorted.index, f"bend_exposure_n{n}"] = bend_exposure

    # Log summary
    for n in scales:
        col = f"curvature_n{n}"
        logger.info(
            "  n=%d: range [%.6f, %.6f] 1/m, non-null=%d/%d",
            n,
            scope_m[col].min(),
            scope_m[col].max(),
            scope_m[col].notna().sum(),
            len(scope_m),
        )

    # Restore original CRS
    if original_crs is not None and original_crs != scope_m.crs:
        scope_m = scope_m.to_crs(original_crs)

    return scope_m


# ── Convenience ────────────────────────────────────────────────────────────


def curvature_feature_cols(scales: list[int] | None = None) -> list[str]:
    """Return the list of column names produced by add_curvature_features."""
    if scales is None:
        scales = [3, 5, 8]
    return [
        f"{prefix}_n{n}"
        for prefix in ["curvature", "bend_side", "bend_exposure"]
        for n in scales
    ]
