"""Tests for the bank-observation IO layer.

Synthetic data only — no GeoPackages, no network — so these run in CI.
The end-to-end check that HeightModelPointSource reproduces
03_features/20260617a/dist_per_year.parquet needs the 848 MB raw delivery and
is therefore a manual check, not a test.
"""

import pandas as pd
import pytest

from src.sources.geometry import normalise_location_id
from src.sources.observations import (
    OBSERVATION_COLUMNS,
    BankObservations,
    combine,
)


def _obs(rows) -> BankObservations:
    """Build a BankObservations from (location_id, date, dist_m) triples."""
    frame = pd.DataFrame(
        [
            {
                "location_id": loc,
                "date": pd.Timestamp(date),
                "dist_m": dist,
                "n_candidates": 3.0,
                "n_selected": 3.0,
                "source": src,
            }
            for loc, date, dist, src in rows
        ]
    )
    return BankObservations(frame[OBSERVATION_COLUMNS])


def test_requires_the_full_schema():
    with pytest.raises(ValueError, match="missing columns"):
        BankObservations(pd.DataFrame({"location_id": ["a"], "dist_m": [1.0]}))


def test_one_observation_per_year_is_unaffected_by_the_collapse_rule():
    """The point cloud has a single survey per year, so every rule agrees.

    This is what lets the refactor reproduce the existing dist_per_year output
    regardless of which rule is configured.
    """
    obs = _obs(
        [
            ("rijn_l_0000_0010", "2020-01-01", 100.0, "hoogtemodel"),
            ("rijn_l_0000_0010", "2021-01-01", 102.0, "hoogtemodel"),
        ]
    )
    results = [
        obs.to_dist_per_year(within_year=r) for r in ("median", "mean", "max", "last")
    ]
    for other in results[1:]:
        pd.testing.assert_frame_equal(results[0], other)
    assert results[0]["dist_m"].tolist() == [100.0, 102.0]


def test_collapse_rules_differ_when_a_year_has_several_surveys():
    obs = _obs(
        [
            ("maas3_r_0100_0110", "2023-03-02", 10.0, "segmentation"),
            ("maas3_r_0100_0110", "2023-06-01", 20.0, "segmentation"),
            ("maas3_r_0100_0110", "2023-09-04", 60.0, "segmentation"),
        ]
    )
    by_rule = {
        r: obs.to_dist_per_year(within_year=r)["dist_m"].iloc[0]
        for r in ("median", "mean", "max", "last")
    }
    assert by_rule["median"] == 20.0
    assert by_rule["mean"] == 30.0
    assert by_rule["max"] == 60.0
    assert by_rule["last"] == 60.0


def test_collapse_emits_the_legacy_schema_and_dtypes():
    """build_region_split consumes this shape, so it must not drift."""
    out = _obs(
        [("ijssel1_l_0010_0020", "2024-01-01", 50.0, "hoogtemodel")]
    ).to_dist_per_year()
    assert list(out.columns) == [
        "location_id",
        "year",
        "dist_m",
        "n_ok_pts",
        "n_selected",
    ]
    assert out["year"].dtype == "int64"
    assert all(out[c].dtype == "float64" for c in ("dist_m", "n_ok_pts", "n_selected"))


def test_rejects_an_unknown_collapse_rule():
    obs = _obs([("a", "2020-01-01", 1.0, "hoogtemodel")])
    with pytest.raises(ValueError, match="within_year must be one of"):
        obs.to_dist_per_year(within_year="furthest")


def test_combine_lets_a_later_source_win_on_a_shared_date():
    height = _obs([("rijn_l_0000_0010", "2025-01-01", 100.0, "hoogtemodel")])
    hybrid = _obs([("rijn_l_0000_0010", "2025-01-01", 111.0, "segmentation")])
    merged = combine(height, hybrid).frame
    assert len(merged) == 1
    assert merged["dist_m"].iloc[0] == 111.0
    assert merged["source"].iloc[0] == "segmentation"


def test_combine_keeps_distinct_dates_from_both_sources():
    height = _obs([("a", "2022-01-01", 1.0, "hoogtemodel")])
    hybrid = _obs([("a", "2023-06-01", 2.0, "segmentation")])
    assert len(combine(height, hybrid)) == 2


def test_combine_of_nothing_is_empty_but_well_formed():
    empty = combine()
    assert len(empty) == 0
    assert list(empty.frame.columns) == OBSERVATION_COLUMNS


@pytest.mark.parametrize("alias", ["position_id", "scope_region_id"])
def test_location_id_aliases_are_normalised(alias):
    df = pd.DataFrame({alias: ["rijn_l_0000_0010"], "value": [1]})
    assert "location_id" in normalise_location_id(df).columns


def test_normalise_leaves_an_already_correct_frame_alone():
    df = pd.DataFrame({"location_id": ["a"], "position_id": ["b"]})
    out = normalise_location_id(df)
    assert out["location_id"].iloc[0] == "a"
