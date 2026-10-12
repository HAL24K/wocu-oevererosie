"""The predicted bank line breaks where strips have no data."""

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src.analysis.predicted_lines import region_lines, runs_of_consecutive


def test_runs_split_at_missing_strips():
    runs = runs_of_consecutive(np.array([0, 1, 2, 5, 6, 9]))
    assert [r.tolist() for r in runs] == [[0, 1, 2], [3, 4], [5]]


def test_region_line_has_a_gap_where_a_strip_has_no_data():
    cline = LineString([(0, 0), (100, 0)])  # R=10 → 10 m strips
    g = pd.DataFrame(
        {
            "seg": [0, 1, 2, 5, 6, 9],
            "last_dist_m": [20.0] * 6,
            "last_year": [2024] * 6,
            "v_pred": [1.0] * 6,
        }
    )
    lines = region_lines(g, cline, side_xy=(50, 30), R=10, year=2026)
    pieces = list(lines.geoms)
    assert len(pieces) == 3
    xs = [(p.coords[0][0], p.coords[-1][0]) for p in pieces]
    assert xs == [(0, 30), (50, 70), (90, 100)]  # nothing drawn over strips 3-4, 7-8
    assert all(
        abs(y - 22.0) < 1e-9 for p in pieces for _, y in p.coords
    )  # 20 + 2 yr x 1
