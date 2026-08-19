"""Tests for the experiment config and the orchestration steps lifted from
the master notebook. Synthetic data only — these run in CI."""

import pandas as pd
import pytest

from src.pipeline.config import ExperimentConfig
from src.pipeline.run import build_start_points, combine_feature_tables


def test_config_resolves_default_paths_under_data_dir(tmp_path):
    cfg = ExperimentConfig(experiment="t1", data_dir=tmp_path)
    assert cfg.raw_gpkg == tmp_path / "01_raw/erosion/wocu_output_fase2_20260210.gpkg"
    assert cfg.features_dir == tmp_path / "03_features/t1"
    assert cfg.output_gpkg.name == "wocu_lgb_predictions_t1.gpkg"


def test_config_override_wins_over_default(tmp_path):
    cfg = ExperimentConfig(
        experiment="t2", data_dir=tmp_path, raw_gpkg=tmp_path / "other.gpkg"
    )
    assert cfg.raw_gpkg == tmp_path / "other.gpkg"


def test_config_reports_missing_inputs(tmp_path):
    cfg = ExperimentConfig(experiment="t3", data_dir=tmp_path)
    missing = cfg.missing_inputs()
    assert cfg.raw_gpkg in missing  # nothing exists under an empty tmp dir


def test_start_points_prefer_t3_and_fall_back_to_inference_t2():
    features = pd.DataFrame(
        {"dist_t3": [100.0], "v_train": [0.5]},
        index=pd.Index(["a"], name="location_id"),
    )
    split = pd.DataFrame({"t3": [2024]}, index=pd.Index(["a"], name="location_id"))
    inference = pd.DataFrame(
        {"dist_t2": [50.0], "v_train": [0.2]}, index=pd.Index(["b"], name="location_id")
    )
    meta = pd.DataFrame({"t2": [2021]}, index=pd.Index(["b"], name="location_id"))

    sp = build_start_points(features, split, inference, meta)
    assert sp["a"] == {"last_dist": 100.0, "last_year": 2024, "v_hist": 0.5}
    assert sp["b"] == {"last_dist": 50.0, "last_year": 2021, "v_hist": 0.2}


def test_start_points_never_overwrites_a_train_region_with_inference():
    features = pd.DataFrame(
        {"dist_t3": [100.0], "v_train": [0.5]},
        index=pd.Index(["a"], name="location_id"),
    )
    split = pd.DataFrame({"t3": [2024]}, index=features.index)
    inference = pd.DataFrame({"dist_t2": [1.0], "v_train": [9.9]}, index=features.index)
    meta = pd.DataFrame({"t2": [2020]}, index=features.index)
    sp = build_start_points(features, split, inference, meta)
    assert sp["a"]["last_dist"] == 100.0  # the t3 entry wins


def test_combine_backfills_test_span_and_keeps_shared_columns_only():
    features = pd.DataFrame(
        {"v_train": [1.0], "train_span_yr": [5], "test_span_yr": [3], "v_test": [0.9]},
        index=pd.Index(["a"], name="location_id"),
    )
    inference = pd.DataFrame(
        {"v_train": [2.0], "train_span_yr": [4]},
        index=pd.Index(["b"], name="location_id"),
    )
    combined = combine_feature_tables(features, inference)
    assert list(combined.index) == ["a", "b"]
    assert combined.loc["b", "test_span_yr"] == 4  # backfilled from train span
    assert "v_test" not in combined.columns  # not shared → dropped


@pytest.mark.parametrize(
    "key",
    ["experiment", "n_points", "seed", "raw_gpkg", "mask_buffer_m", "structures_gpkg"],
)
def test_config_table_is_complete(tmp_path, key):
    cfg = ExperimentConfig(experiment="t4", data_dir=tmp_path)
    assert key in cfg.to_table()
