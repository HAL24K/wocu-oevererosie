"""Predicted bank lines from the segment forecast, with gaps where there is no data.

A region has ``R`` strips along its centreline; only strips with observations
get a forecast. Strips that follow each other form one line piece, from the
start edge of the first strip to the end edge of the last, through the strip
centres. A strip without data breaks the line, as the measured lines break
where nothing was detected: the prediction never bridges a gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString


def forecast_dist(g: pd.DataFrame, year) -> np.ndarray:
    """Forecast distance to the centreline per strip in ``year``.

    ``last_dist_m + v_pred * min(dt, short_years) + v_long * dt`` with
    ``dt = year - last_year``: the model's speed (``v_pred``) holds for at most
    ``short_years`` (a correction towards where the bank really is), ``v_long`` is
    the lasting drift. Without those two columns this is the plain
    ``last_dist_m + v_pred * dt``.
    """
    dt = np.clip(np.asarray(year, float) - g["last_year"].to_numpy(float), 0, None)
    cap = g["short_years"].to_numpy(float) if "short_years" in g else np.inf
    v_long = g["v_long"].to_numpy(float) if "v_long" in g else 0.0
    return (
        g["last_dist_m"].to_numpy(float)
        + g["v_pred"].to_numpy(float) * np.minimum(dt, cap)
        + v_long * dt
    )


def runs_of_consecutive(segs: np.ndarray) -> list[np.ndarray]:
    """Indices into ``segs`` (sorted) grouped into runs of consecutive values."""
    if len(segs) == 0:
        return []
    breaks = np.flatnonzero(np.diff(segs) != 1) + 1
    return np.split(np.arange(len(segs)), breaks)


def _offset_points(cline, stations, dists, side_xy):
    """Points at ``dists`` from the centreline, perpendicular, on the bank's side."""
    eps = 0.02
    p0 = np.array(
        [
            cline.interpolate(max(s - eps, 0.0), normalized=True).coords[0]
            for s in stations
        ]
    )
    p1 = np.array(
        [
            cline.interpolate(min(s + eps, 1.0), normalized=True).coords[0]
            for s in stations
        ]
    )
    base = np.array([cline.interpolate(s, normalized=True).coords[0] for s in stations])
    tx, ty = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
    norm = np.hypot(tx, ty)
    norm[norm == 0] = 1.0
    nx, ny = -ty / norm, tx / norm
    sign = np.sign((side_xy[0] - base[:, 0]) * nx + (side_xy[1] - base[:, 1]) * ny)
    sign[sign == 0] = 1.0
    return np.column_stack(
        [base[:, 0] + sign * nx * dists, base[:, 1] + sign * ny * dists]
    )


def region_lines(
    g: pd.DataFrame, cline, side_xy: tuple[float, float], R: int, year: int
) -> MultiLineString:
    """The predicted bank in ``year`` for one region's segment forecasts ``g``.

    ``g`` needs ``seg``, ``last_dist_m``, ``last_year`` and ``v_pred``.
    """
    g = g.sort_values("seg")
    seg = g["seg"].to_numpy()
    dist = forecast_dist(g, year)
    pieces = []
    for run in runs_of_consecutive(seg):
        s, d = seg[run], dist[run]
        stations = np.concatenate([[s[0] / R], (s + 0.5) / R, [(s[-1] + 1) / R]])
        dists = np.concatenate([[d[0]], d, [d[-1]]])
        stations = np.clip(stations, 0.0, 1.0)
        pieces.append(LineString(_offset_points(cline, stations, dists, side_xy)))
    return MultiLineString(pieces)
