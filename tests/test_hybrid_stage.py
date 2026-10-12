"""Tests for the hybrid stage's configuration and the logic around Luke's steps.

Synthetic data only — these run in CI.
"""

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from src.hybrid.config import HybridConfig
from src.hybrid.run import apply_height_fallback, winner_lines
from src.hybrid.sam_cleaning import side_channel_mask
from src.pipeline.config import ExperimentConfig


def test_config_writes_under_03_hybrid(tmp_path):
    cfg = HybridConfig(name="h1", data_dir=tmp_path)
    assert cfg.hybrid_gpkg == tmp_path / "03_hybrid/h1/hybrid.gpkg"
    assert cfg.nearest_source == tmp_path / "03_hybrid/h1/nearest.gpkg"
    assert (
        cfg.sam_path == tmp_path / "01_raw/sam/oeverlijn_segmentatie_full_aoi.geojson"
    )
    assert not cfg.cleans_sam


def test_nearest_from_reuses_another_hybrids_step_1(tmp_path):
    cfg = HybridConfig(name="h2", data_dir=tmp_path, nearest_from="base")
    assert cfg.nearest_source == tmp_path / "03_hybrid/base/nearest.gpkg"
    assert "sam_path" not in cfg.inputs()  # step 1 is skipped, so SAM is not read
    assert cfg.inputs()["nearest_gpkg"] == cfg.nearest_source


def test_inputs_follow_the_cleaning_options(tmp_path):
    plain = HybridConfig(name="a", data_dir=tmp_path).inputs()
    assert set(plain) == {"height_gpkg", "sam_path"}
    full = HybridConfig(
        name="b", data_dir=tmp_path, sam_rules=True, side_channel_m=10, jetty_m=100
    )
    assert full.cleans_sam
    assert {"legger_gpkg", "structures_gpkg"} <= set(full.inputs())


def test_settings_exclude_paths(tmp_path):
    s = HybridConfig(name="c", data_dir=tmp_path, jetty_m=50).settings()
    assert s["jetty_m"] == 50
    assert "sam_path" not in s and "name" not in s


def test_height_fallback_keeps_regions_the_height_model_can_carry():
    rationale = pd.DataFrame(
        {
            "location_id": ["kept", "lost_ok", "lost_thin"],
            "model_preference": ["segmentation", "height", "segmentation"],
        }
    )
    out = apply_height_fallback(rationale, lost={"lost_ok", "lost_thin"})
    assert list(out["location_id"]) == ["kept", "lost_ok"]
    assert list(out["height_fallback"]) == [False, True]


def test_winner_lines_take_only_the_preferred_model():
    lines = gpd.GeoDataFrame(
        {
            "location_id": ["a", "a", "b", "b"],
            "model": ["hoogtemodel", "segmentation", "hoogtemodel", "segmentation"],
        },
        geometry=[LineString([(0, 0), (1, 1)])] * 4,
    )
    pref = pd.Series({"a": "height", "b": "segmentation"})
    out = winner_lines(lines, pref)
    assert list(zip(out["location_id"], out["model"], strict=True)) == [
        ("a", "hoogtemodel"),
        ("b", "segmentation"),
    ]


def test_side_channel_mask_without_any_hit_leaves_lines_alone():
    """Regression: an empty hit list used to be read as a column selection."""
    seg = gpd.GeoDataFrame(
        {"location_id": ["a"]}, geometry=[LineString([(0, 0), (10, 0)])], crs=28992
    )
    legger = gpd.GeoDataFrame(
        geometry=[LineString([(500, 500), (600, 500)])], crs=28992
    )
    centrelines = gpd.GeoSeries({"a": LineString([(0, 5), (10, 5)])}, crs=28992)
    out = side_channel_mask(seg, legger, centrelines, buffer_m=10)
    assert len(out) == 1 and out.geometry.iloc[0].length == 10


def test_model_stage_takes_a_hybrid_by_name(tmp_path):
    cfg = ExperimentConfig(experiment="e", data_dir=tmp_path, hybrid="h1")
    assert cfg.hybrid_gpkg == tmp_path / "03_hybrid/h1/hybrid.gpkg"
    with pytest.raises(ValueError, match="not both"):
        ExperimentConfig(
            experiment="e", data_dir=tmp_path, hybrid="h1", hybrid_gpkg=tmp_path / "x"
        )


def test_run_record_carries_the_hybrid_record(tmp_path):
    from src.pipeline.provenance import describe_inputs

    hdir = tmp_path / "03_hybrid/h1"
    hdir.mkdir(parents=True)
    (hdir / "hybrid.json").write_text(
        json.dumps({"name": "h1", "settings": {"sam_rules": True}, "extra": 1})
    )
    cfg = ExperimentConfig(experiment="e", data_dir=tmp_path, hybrid="h1")
    rec = describe_inputs(cfg)["hybrid_gpkg"]["hybrid_record"]
    assert rec["name"] == "h1" and rec["settings"] == {"sam_rules": True}
    assert "extra" not in rec
