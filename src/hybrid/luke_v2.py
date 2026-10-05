"""Port of Luke Moth's hybrid model v2 (SAM line cleaning + model preference).

Source: ``luke-moth/wocu-erosion-dtm-tipping-point`` @ 69bfe56 (branch
``postprocessing-and-comparison``): ``src/wocu/hybrid_model_v2.py``,
``src/wocu/post_processing_utils.create_lines_from_points`` and
``src/utils/line_utils.calculate_perpendicular_distances``.

The logic is copied as-is, quirks included, so the output can be compared
one-to-one with his deliveries. Only the data access differs: his PostGIS
tables ``vlakken_scope`` / ``centrelines`` / ``punten_oever`` are read from
the matching layers of a ``wocu_output_fase2_*.gpkg`` (keyed ``position_id``
there, ``location_id`` in his database).
"""

from __future__ import annotations

import datetime as dt
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import nearest_points

LOCATION_ID = "location_id"
HEIGHT_MODEL = "hoogtemodel"
SEGMENTATION = "segmentation"


# ── line_utils ────────────────────────────────────────────────────────────────


def calculate_perpendicular_distances(points, input_line):
    """Distance from each centreline point to the nearest crossing of its 400 m perpendicular."""
    distances = []
    points_out = []
    for i in range(1, len(points) - 1):
        prev_point, next_point, current_point = points[i - 1], points[i + 1], points[i]
        dx = next_point.x - prev_point.x
        dy = next_point.y - prev_point.y
        perp_dx, perp_dy = -dy, dx
        perp_line = LineString(
            [
                (current_point.x - perp_dx, current_point.y - perp_dy),
                (current_point.x + perp_dx, current_point.y + perp_dy),
            ]
        )
        length = perp_line.length
        if length < 400:
            scale = 400 / length
            mid_x = (perp_line.coords[0][0] + perp_line.coords[1][0]) / 2
            mid_y = (perp_line.coords[0][1] + perp_line.coords[1][1]) / 2
            perp_line = LineString(
                [
                    (
                        mid_x + (perp_line.coords[0][0] - mid_x) * scale,
                        mid_y + (perp_line.coords[0][1] - mid_y) * scale,
                    ),
                    (
                        mid_x + (perp_line.coords[1][0] - mid_x) * scale,
                        mid_y + (perp_line.coords[1][1] - mid_y) * scale,
                    ),
                ]
            )
        intersection = perp_line.intersection(input_line)
        if not intersection.is_empty:
            if intersection.geom_type == "MultiPoint":
                intersection = nearest_points(current_point, intersection)[1]
            distances.append(current_point.distance(intersection))
            points_out.append(intersection)
    return distances, points_out


def _centreline_points(centreline):
    return list(centreline.interpolate(np.arange(0, centreline.length, 1)))


# ── SAM lines ─────────────────────────────────────────────────────────────────


def validate_segmentation_lines(lines_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """``SegmentationLines._validate_lines``: keep date, model, geometry."""
    lines_gdf = lines_gdf.copy()
    if "date" not in lines_gdf.columns:
        if "datum" in lines_gdf.columns:
            lines_gdf["date"] = pd.to_datetime(lines_gdf["datum"]).dt.date
        elif "jaar" in lines_gdf.columns:
            lines_gdf["date"] = pd.to_datetime(lines_gdf["jaar"], format="%Y").dt.date
        else:
            raise ValueError("Lines gdf must have either 'date', 'datum' or 'jaar' column")
    lines_gdf["model"] = SEGMENTATION
    return lines_gdf[["date", "model", "geometry"]]


def add_location_ids(lines_gdf: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A line gets every region whose polygon (+1 m) contains it entirely; others get NaN."""
    poly = polygons[[LOCATION_ID, "geometry"]].copy()
    poly["geometry"] = poly.geometry.buffer(1)
    out = gpd.sjoin(lines_gdf, poly, how="left", predicate="within")
    return out.drop(columns=["index_right"])


def get_clean_line(lines_location_id: gpd.GeoDataFrame, centreline) -> list[dict]:
    """Keep, per perpendicular, only the crossing nearest the centreline; rejoin within 5 m."""
    new_lines_location_id = []
    location_id = lines_location_id[LOCATION_ID].unique()[0]
    centreline_points = _centreline_points(centreline)
    for date_str in lines_location_id["date"].unique():
        lines = list(lines_location_id[lines_location_id["date"] == date_str].geometry)
        for line in lines:
            _, proj_points = calculate_perpendicular_distances(centreline_points, line)
            if len(proj_points) < 2:
                continue
            new_lines, current_line = [], []
            for i in range(len(proj_points)):
                if i == 0:
                    current_line.append(proj_points[i])
                elif proj_points[i].distance(proj_points[i - 1]) < 5:
                    current_line.append(proj_points[i])
                else:
                    if len(current_line) > 1:
                        new_lines.append(LineString(current_line))
                    current_line = [proj_points[i]]
            if len(current_line) > 1:
                new_lines.append(LineString(current_line))
            for new_line in new_lines:
                new_lines_location_id.append(
                    {"date": date_str, "model": SEGMENTATION, LOCATION_ID: location_id, "geometry": new_line}
                )
    return new_lines_location_id


def clean_location(location_id, lines_location_id, centrelines: dict) -> list[dict]:
    """One location of ``post_processing_segmentation_lines``, including its fallback."""
    try:
        return get_clean_line(lines_location_id, centrelines[location_id])
    except BaseException as e:  # noqa: BLE001 — his fallback: on any error, keep the raw lines
        logging.error("Skipping location_id %s: %s", location_id, e)
        return [
            {"date": r["date"], "model": r["model"], LOCATION_ID: r[LOCATION_ID], "geometry": r["geometry"]}
            for _, r in lines_location_id.iterrows()
        ]


# ── height lines ──────────────────────────────────────────────────────────────


def create_lines_from_points(gdf: gpd.GeoDataFrame, min_points: int = 6) -> gpd.GeoDataFrame:
    """OK points < 10 m apart in row order, per (location, year); the first point of a run is not status-checked."""
    lines = []
    for (location_id, annum), group_gdf in gdf.groupby([LOCATION_ID, "dtm_date"]):
        current = []
        for _, row in group_gdf.iterrows():
            # Precedence as in the source: (OK and < 10 m) if a run is open, else always accept.
            if row["status"] == "OK" and row["geometry"].distance(current[-1]) < 10 if current else True:
                current.append(row["geometry"])
            else:
                if len(current) >= min_points:
                    lines.append(_height_line(location_id, annum, current))
                current = []
        if len(current) >= min_points:
            lines.append(_height_line(location_id, annum, current))
    return gpd.GeoDataFrame(lines, columns=[LOCATION_ID, "date", "model", "geometry"], geometry="geometry", crs=gdf.crs)


def _height_line(location_id, annum, points) -> dict:
    return {
        LOCATION_ID: location_id,
        "date": dt.date(int(annum), 1, 1),
        "model": HEIGHT_MODEL,
        "geometry": LineString([p.coords[0] for p in points]),
    }


def get_slope_kpi(points: pd.DataFrame) -> float:
    lookup = {"CLIFF": 1, "STEEP_SLOPE": 0.9, "MEDIUM_SLOPE": 0.4, "MILD_SLOPE": 0.1}
    return points["type_oever"].map(lookup).fillna(0).mean()


def height_data_for_location(points_location: gpd.GeoDataFrame):
    """``DataFetcher.get_data_for_location`` on one location's points."""
    if points_location.empty:
        raise ValueError("no points")
    points_ = points_location.drop_duplicates(subset=["geometry", "dtm_version"])
    kpi = {d: get_slope_kpi(points_[points_["dtm_date"] == d]) for d in points_["dtm_date"].unique()}
    return kpi, create_lines_from_points(points_, min_points=5)


# ── preference ────────────────────────────────────────────────────────────────


def get_stats_from_line_wrt_centreline(line1, centreline) -> dict:
    distances, _ = calculate_perpendicular_distances(_centreline_points(centreline), line1)
    if len(distances) < 2:
        return {"num_points": len(distances), **dict.fromkeys(("dist_p10", "dist_p50", "dist_p90", "diff_p10", "diff_p50", "diff_p90"), np.nan)}
    dist_p10, dist_p50, dist_p90 = np.percentile(distances, [10, 50, 90])
    diff = (np.diff(distances) ** 2) ** 0.5
    diff_p10, diff_p50, diff_p90 = np.percentile(diff, [10, 50, 90])
    return {"num_points": len(distances), "dist_p10": dist_p10, "dist_p50": dist_p50, "dist_p90": dist_p90,
            "diff_p10": diff_p10, "diff_p50": diff_p50, "diff_p90": diff_p90}


def get_grouped_results(lines_height_gdf, lines_segmentation_gdf, centreline):
    all_results = []
    for gdf, kind in ((lines_height_gdf, "height"), (lines_segmentation_gdf, "segmentation")):
        for date_str in gdf["date"].unique():
            stats = get_stats_from_line_wrt_centreline(
                MultiLineString(list(gdf[gdf["date"] == date_str].geometry)), centreline
            )
            stats["date"], stats["line_type"] = date_str, kind
            all_results.append(stats)
    df = pd.DataFrame(all_results)
    grouped = df.groupby("line_type").agg(
        {"num_points": "sum", "dist_p10": "min", "dist_p50": "mean", "dist_p90": "max",
         "diff_p10": "max", "diff_p50": "max", "diff_p90": "max"}
    ) if len(df) else pd.DataFrame()
    return grouped, df


def _bucket(ratio, edges_points):
    """First (upper_bound, (h, s)) whose bound exceeds ratio; the last entry is the else."""
    for bound, pts in edges_points[:-1]:
        if ratio < bound:
            return pts
    return edges_points[-1][1]


def determine_model_preference(g, kpi_type_oever, years_h, years_s):
    """Luke's points count; ties go to height."""
    rationale = {"distinct_dates_height": years_h, "distinct_dates_segmentation": years_s}
    if years_h < 2:
        pref = "segmentation"
    elif years_s < 2:
        pref = "height"
    else:
        ph = ps = 0
        if years_s > years_h + 2:
            ps += 2
        elif years_s > years_h:
            ps += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            r_pts = g.loc["height", "num_points"] / g.loc["segmentation", "num_points"]
            r_p10 = g.loc["height", "dist_p10"] / g.loc["segmentation", "dist_p10"]
            r_p50 = g.loc["height", "dist_p50"] / g.loc["segmentation", "dist_p50"]
            r_p90 = g.loc["height", "dist_p90"] / g.loc["segmentation", "dist_p90"]
            r_n50 = g.loc["height", "diff_p50"] / g.loc["segmentation", "diff_p50"]
            r_n90 = g.loc["height", "diff_p90"] / g.loc["segmentation", "diff_p90"]
        # Written as >= thresholds in the source; equivalent to these < buckets, NaN falls to the else.
        if r_pts >= 2:
            ph += 2
        elif r_pts >= 1:
            ph += 1
        elif r_pts >= 0.7:
            pass
        elif r_pts >= 0.4:
            ps += 1
        else:
            ps += 2
        for ratio, table in (
            (r_p10, [(0.5, (0, 1)), (0.95, (1, 0)), (1.05, (1, 1)), (2, (0, 1)), (None, (1, 0))]),
            (r_p50, [(0.95, (1, 0)), (1.05, (1, 1)), (None, (0, 1))]),
            (r_p90, [(0.95, (1, 0)), (1.05, (1, 1)), (None, (0, 1))]),
            (r_n50, [(0.9, (1, 0)), (1.1, (1, 1)), (None, (0, 1))]),
            (r_n90, [(0.9, (1, 0)), (1.1, (1, 1)), (1.5, (0, 1)), (None, (0, 2))]),
        ):
            dh, ds = _bucket(ratio, table)
            ph, ps = ph + dh, ps + ds
        slope = np.mean(list(kpi_type_oever.values()))
        if slope >= 0.75:
            ph += 3
        elif slope >= 0.5:
            ph += 2
        elif slope >= 0.25:
            ph += 1
        pref = "height" if ph >= ps else "segmentation"
        for k, col in (("nr_points", "num_points"), ("p50_dist", "dist_p50"), ("p90_dist", "dist_p90"),
                       ("diff_p50", "diff_p50"), ("diff_p90", "diff_p90")):
            rationale[f"{k}_height"] = g.loc["height", col]
            rationale[f"{k}_segmentation"] = g.loc["segmentation", col]
        rationale["kpi_slope"] = slope
        rationale["points_height [0-10]"] = ph
        rationale["points_segmentation [0-10]"] = ps
    rationale["model_preference"] = pref
    return pref, rationale


def hybrid_location(location_id, points_location, lines_seg, centreline):
    """One location of ``create_hybrid_model``: rationale dict and both models' lines."""
    try:
        kpi, lines_h = height_data_for_location(points_location)
    except ValueError:
        kpi, lines_h = {}, gpd.GeoDataFrame(columns=[LOCATION_ID, "date", "model", "geometry"], geometry="geometry")
    grouped, _ = get_grouped_results(lines_h, lines_seg, centreline)
    pref, rationale = determine_model_preference(grouped, kpi, lines_h["date"].nunique(), lines_seg["date"].nunique())
    rationale[LOCATION_ID] = location_id
    return rationale, lines_h
