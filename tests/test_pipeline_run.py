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
    [
        "experiment",
        "n_points",
        "seed",
        "raw_gpkg",
        "mask_buffer_m",
        "structures_gpkg",
        "source",
        "hybrid_gpkg",
        "water_mask",
        "temporal_max_dev",
        "segment_R",
        "val_frac",
    ],
)
def test_config_table_is_complete(tmp_path, key):
    cfg = ExperimentConfig(experiment="t4", data_dir=tmp_path)
    assert key in cfg.to_table()


def test_cleaning_rules_reflect_config(tmp_path):
    from src.pipeline.hybrid_prep import cleaning_rules

    cfg = ExperimentConfig(
        experiment="t5", data_dir=tmp_path, maze_min_iqr=33.0, temporal_max_dev=9.0
    )
    rules = dict(cleaning_rules(cfg))
    assert rules["maze_survey"]["min_iqr"] == 33.0
    assert rules["temporal_outlier_survey"]["max_dev"] == 9.0
    assert rules["temporal_outlier_survey"]["detrend"] is True


def test_trajectory_features_respect_the_origin():
    from src.pipeline.trajectory import trajectory_features

    obs = pd.DataFrame(
        {
            "location_id": ["a"] * 4,
            "date": pd.to_datetime(
                ["2022-06-01", "2023-06-01", "2024-06-01", "2026-06-01"]
            ),
            "dist_m": [10.0, 12.0, 14.0, 99.0],  # 2026 is a post-origin artefact
            "source": ["segmentation"] * 4,
        }
    )
    out = trajectory_features(obs, pd.Series({"a": 2024}))
    assert out.loc["a", "n_hist"] == 3  # the 2026 survey must not leak in
    assert abs(out.loc["a", "theil_v"] - 2.0) < 0.1


def test_train_honest_validation_never_touches_test(tmp_path):
    import numpy as np

    from src.pipeline.train import FEATS_LGB, train_and_save_models

    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {c: rng.normal(size=n) for c in FEATS_LGB},
        index=pd.Index([f"r{i}" for i in range(n)], name="location_id"),
    )
    df["is_nvo"] = rng.random(n) > 0.5
    df["v_test"] = df["v_train"] * 0.5 + rng.normal(scale=0.1, size=n)
    df["split"] = ["train"] * 90 + ["test"] * 30
    results = train_and_save_models(df, tmp_path / "out", seed=1, val_frac=0.2)
    assert "5 – LightGBM" in results
    assert results["5 – LightGBM"]["test_mae"] < results["0 – Naive mean"]["test_mae"]
