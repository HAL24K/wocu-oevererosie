"""Centerline and geometry utilities for the erosion prediction pipeline."""

import json
from pathlib import Path
from typing import List, Optional, Set, Union

import geopandas as gpd
import numpy as np
import pandas as pd


def ensure_location_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename position_id to location_id if present. Returns a copy."""
    if "position_id" in df.columns and "location_id" not in df.columns:
        return df.rename(columns={"position_id": "location_id"})
    return df


from shapely import offset_curve
from shapely.geometry import MultiPoint, Point


def _scalar_distance(p, geom) -> float:
    """Get scalar distance, handling Shapely returning arrays."""
    d = p.distance(geom)
    arr = np.atleast_1d(np.asarray(d))
    return float(arr.min()) if arr.size > 0 else float("inf")


def _ref_centroid(ref) -> Point:
    """Get centroid from reference geometry (geometry with .centroid or iterable of Points)."""
    # Numpy array (ref_geom_lookup uses g.values) or GeoSeries — use MultiPoint
    if isinstance(ref, np.ndarray) or hasattr(ref, "iloc"):
        return MultiPoint(list(ref)).centroid
    # Single Shapely geometry
    if hasattr(ref, "geom_type"):
        return ref.centroid
    return MultiPoint(list(ref)).centroid


def offset_line_toward(
    centerline,
    dist: float,
    reference_geom,
) -> Optional[object]:
    """
    Offset centerline by dist toward the side where reference_geom lies.

    Args:
        centerline: LineString
        dist: offset distance (meters)
        reference_geom: geometry with .centroid (LineString, Polygon, etc.)
            or iterable of Points (e.g. GeoSeries of bank points)

    Returns:
        Offset LineString or None if both sides fail
    """
    left = offset_curve(centerline, dist)
    right = offset_curve(centerline, -dist)
    if left is None or left.is_empty:
        return right
    if right is None or right.is_empty:
        return left
    centroid = _ref_centroid(reference_geom)
    mid_l = left.interpolate(0.5, normalized=True)
    mid_r = right.interpolate(0.5, normalized=True)
    d_l = _scalar_distance(mid_l, centroid)
    d_r = _scalar_distance(mid_r, centroid)
    return left if d_l < d_r else right


def point_from_offset(
    centerline,
    dist: float,
    ref_geom,
) -> Optional[Point]:
    """
    Compute point at midpoint of centerline offset by dist toward ref_geom.

    Used for historical bank position and predicted bank position.
    """
    offset_line = offset_line_toward(centerline, dist, ref_geom)
    if offset_line is None or offset_line.is_empty:
        return None
    return offset_line.interpolate(0.5, normalized=True)


def dist_signaleringslijn(centerline, vvr_geom, n_samples: int = 200) -> float:
    """
    Minimum perpendicular distance from centerline to VVR boundary.

    Samples n_points along the centerline and returns the minimum distance
    to the VVR geometry (or its boundary for Polygons).
    """
    if vvr_geom.geom_type == "Polygon":
        vvr_boundary = vvr_geom.boundary
    else:
        vvr_boundary = vvr_geom
    distances = np.linspace(0, centerline.length, n_samples)
    perp_dists = [_scalar_distance(centerline.interpolate(d), vvr_boundary) for d in distances]
    return min(perp_dists) if perp_dists else float("inf")


def parallel_line_from_vvr(centerline, vvr_geom):
    """
    Create a line parallel to centerline at dist_signaleringslijn distance,
    offset toward the VVR side.
    """
    dist = dist_signaleringslijn(centerline, vvr_geom)
    return offset_line_toward(centerline, dist, vvr_geom)


def build_ref_geom_lookup(
    bank_points: gpd.GeoDataFrame,
    n_points: int = 3,
    status_filter: str = "OK",
) -> dict:
    """
    For each location_id, return numpy array of N furthest OK points from latest date.
    Used as reference geometry for predicted bank positions.
    """
    ok = bank_points[bank_points["status"] == status_filter].copy()
    max_dates = ok.groupby("location_id")["dtm_date"].transform("max")
    latest = ok[ok["dtm_date"] == max_dates]
    return (
        latest.sort_values("dist", ascending=False)
        .groupby("location_id")
        .head(n_points)
        .groupby("location_id")["geometry"]
        .apply(lambda g: g.values)
        .to_dict()
    )


def compute_qualifying_regions(
    bank_points: gpd.GeoDataFrame,
    min_shift: float = 5.0,
    n_points: int = 3,
    n_timepoints: int = 3,
) -> pd.DataFrame:
    """
    Regions with >= min_shift in dist between consecutive timepoints.
    Returns DataFrame indexed by location_id with d_t1, d_t2, d_t3, shift_t1t2, shift_t2t3.
    """
    def mean_dist_furthest(group):
        return group.nlargest(n_points, "dist")["dist"].mean()

    agg = (
        bank_points.groupby(["location_id", "dtm_date"])
        .apply(mean_dist_furthest, include_groups=False)
        .reset_index(name="mean_dist")
    )
    ts_counts = agg.groupby("location_id")["dtm_date"].nunique()
    agg = agg[agg["location_id"].isin(ts_counts[ts_counts == n_timepoints].index)].sort_values(
        ["location_id", "dtm_date"]
    )
    agg["rank"] = agg.groupby("location_id").cumcount()
    wide = agg.pivot(index="location_id", columns="rank", values="mean_dist").rename(
        columns={0: "d_t1", 1: "d_t2", 2: "d_t3"}
    )
    wide["shift_t1t2"] = (wide["d_t2"] - wide["d_t1"]).abs()
    wide["shift_t2t3"] = (wide["d_t3"] - wide["d_t2"]).abs()
    return wide[
        (wide["shift_t1t2"] >= min_shift) & (wide["shift_t2t3"] >= min_shift)
    ].sort_values("shift_t1t2", ascending=False)


def pick_across_clusters(
    ids: List[str],
    clusters: List[str] = None,
    n: int = 4,
) -> List[str]:
    """Pick one id per cluster first, then fill remaining slots."""
    if clusters is None:
        clusters = ["rijn", "ijssel", "maas", "neder"]
    selected = []
    for cluster in clusters:
        match = next((lid for lid in ids if cluster in lid and lid not in selected), None)
        if match:
            selected.append(match)
    for lid in ids:
        if lid not in selected:
            selected.append(lid)
        if len(selected) == n:
            break
    return selected


def get_nvo_location_ids(vvr: gpd.GeoDataFrame, scope: gpd.GeoDataFrame) -> Set[str]:
    """Location IDs where VVR intersects scope (from spatial join)."""
    scope_for_join = scope[["location_id", "geometry"]].copy()
    vvr_for_join = vvr.to_crs(scope_for_join.crs) if vvr.crs != scope_for_join.crs else vvr
    joined = gpd.sjoin(vvr_for_join[["geometry"]], scope_for_join, how="left", predicate="intersects")
    return set(joined["location_id"].dropna().astype(str).unique())


def flatten_geom_to_lines(geom) -> list:
    """Flatten any geometry to a list of plottable LineString parts."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "Polygon":
        return [geom.exterior]
    if geom.geom_type == "MultiPolygon":
        return [p.exterior for p in geom.geoms]
    if geom.geom_type == "GeometryCollection":
        return [p for g in geom.geoms for p in flatten_geom_to_lines(g)]
    return [geom]


def ensure_axes_list(axes) -> list:
    """
    Ensure axes from plt.subplots(1, n) is always a list.
    When n==1, subplots returns a single Axes; when n>1, an array. This normalizes to list.
    """
    return list(np.atleast_1d(axes))


def compute_vvr_crossing_year(
    predicted_dist_df: pd.DataFrame,
    nvo_location_ids: Set[str],
    centerlines: gpd.GeoDataFrame,
    scope: gpd.GeoDataFrame,
    signaleringslijn: gpd.GeoDataFrame,
    dist_column: str = "predicted_dist_m",
    reference_year: int = 2025,
) -> pd.DataFrame:
    """
    For each NVO region: when does predicted bank distance exceed signaleringslijn?
    Returns: location_id, dist_to_vvr_m, crossing_year, velocity_m_per_yr, years_to_crossing.
    """
    sig = signaleringslijn.to_crs(28992) if signaleringslijn.crs.to_epsg() != 28992 else signaleringslijn
    cl, sc = centerlines.set_index("location_id")["geometry"], scope.set_index("location_id")["geometry"]
    has_vel = "velocity_m_per_yr" in predicted_dist_df.columns
    rows = []
    for loc_id in nvo_location_ids:
        cline, sgeom = cl.get(loc_id), sc.get(loc_id)
        if cline is None or sgeom is None:
            continue
        pred = predicted_dist_df[predicted_dist_df["location_id"] == loc_id].sort_values("year")
        if pred.empty:
            continue
        vvr = sig[sig.intersects(sgeom.buffer(10))].geometry.union_all().intersection(sgeom)
        if vvr is None or vvr.is_empty:
            continue
        try:
            dist_to_vvr = dist_signaleringslijn(cline, vvr)
        except (ValueError, TypeError):
            continue
        crossing_year = None
        for i in range(len(pred) - 1):
            d1, d2 = pred.iloc[i][dist_column], pred.iloc[i + 1][dist_column]
            t1, t2 = pred.iloc[i]["year"], pred.iloc[i + 1]["year"]
            if d1 >= dist_to_vvr:
                crossing_year = float(t1)
                break
            if d2 >= dist_to_vvr:
                crossing_year = t1 + (dist_to_vvr - d1) / (d2 - d1) * (t2 - t1) if d2 > d1 else float(t2)
                break
        if crossing_year is None and pred.iloc[-1][dist_column] >= dist_to_vvr:
            crossing_year = float(pred.iloc[-1]["year"])
        vel = pred.iloc[0]["velocity_m_per_yr"] if has_vel else None
        rows.append({
            "location_id": loc_id,
            "dist_to_vvr_m": dist_to_vvr,
            "crossing_year": crossing_year,
            "velocity_m_per_yr": vel,
            "years_to_crossing": crossing_year - reference_year if crossing_year else None,
        })
    return pd.DataFrame(rows)


def count_notebook_loc(notebook_path: Path) -> dict:
    """
    Count lines of code in a Jupyter notebook.

    Returns dict with:
        total: total lines in code cells (including empty, comments)
        non_empty: lines that are not blank
    """
    with open(notebook_path) as f:
        nb = json.load(f)

    total = 0
    non_empty = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in cell.get("source", []):
            total += 1
            if line.strip():
                non_empty += 1

    return {"total": total, "non_empty": non_empty}
