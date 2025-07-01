"""Tests for the ML predictive model"""

import numpy as np
import pickle
import pytest
import sklearn.linear_model as sklm

import src.data.data_handler as DH
import src.constants as CONST

import src.model.ml_predictive_model as MLPM

NUMBER_OF_FUTURES = 2


@pytest.fixture
def ml_data_configuration(default_data_configuration):
    """Create a default data configuration for the ML model."""
    data_configuration = default_data_configuration
    data_configuration.number_of_futures = NUMBER_OF_FUTURES
    data_configuration.known_categorical_columns = [
        "BrpGewas_majority_class_category",
        "rws_vegetatielegger:vegetatieklassen_majority_class_vlklasse",
    ]

    return data_configuration


@pytest.fixture
def ml_data(
    ml_data_configuration,
    prediction_regions_for_test,
    local_enrichment_geodata,
    erosion_data_for_test,
    real_erosion_border,
):
    data_handler = DH.DataHandler(
        config=ml_data_configuration,
        prediction_regions=prediction_regions_for_test,
        local_data_for_enrichment=local_enrichment_geodata,
        erosion_data=erosion_data_for_test,
        erosion_border=real_erosion_border,
    )

    data_handler.process_erosion_features()
    data_handler.create_data_from_remote()
    data_handler.add_remote_data_to_processed()
    data_handler.generate_erosion_features()

    return data_handler.erosion_features_complete


def test_columns(ml_data_configuration, ml_data):
    """Test that the target and input columns are correctly generated."""
    predictive_model = MLPM.PredictiveModel(
        config=ml_data_configuration,
        training_data=ml_data,
    )

    target_columns = predictive_model.target_columns

    assert np.all([c for c in target_columns if CONST.UPCOMING in c])
    assert (
        len(target_columns) == 2
    )  # Assuming we have two future steps as per the configuration

    input_columns = predictive_model.input_columns

    assert len(input_columns) == len(ml_data.columns) - len(target_columns)
    assert set(input_columns + target_columns) == set(ml_data.columns)


def test_train_save_load(tmp_path, ml_data_configuration, ml_data, caplog):
    """Test training, saving, and loading the predictive model."""
    empty_predictive_model = MLPM.PredictiveModel(
        config=ml_data_configuration,
        training_data=None,
    )
    assert not empty_predictive_model.model_is_trained
    empty_predictive_model.train()
    assert not empty_predictive_model.model_is_trained
    assert "No training data provided" in caplog.text
    caplog.clear()

    predictive_model = MLPM.PredictiveModel(
        config=ml_data_configuration,
        training_data=ml_data,
        model=sklm.LinearRegression(),
    )

    # Train the model
    assert not predictive_model.model_is_trained
    predictive_model.train()
    assert predictive_model.model_is_trained

    predictive_model.train()
    assert "already trained" in caplog.text
    caplog.clear()

    # Save the model
    path_to_save = tmp_path / "predictive_model.pkl"

    assert not path_to_save.exists()
    predictive_model.save(
        path_to_save, keep_training_data=False
    )  # be explicit about the data
    assert path_to_save.exists()

    # Load the whole class
    with open(path_to_save, "rb") as f:
        loaded_class_without_data = pickle.load(f)

    assert isinstance(loaded_class_without_data, MLPM.PredictiveModel)
    assert loaded_class_without_data.training_data is None

    # test preserving the data
    predictive_model.save(path_to_save, keep_training_data=True)

    with open(path_to_save, "rb") as f:
        loaded_class_with_data = pickle.load(f)

    assert isinstance(loaded_class_with_data, MLPM.PredictiveModel)
    assert loaded_class_with_data.training_data.equals(predictive_model.training_data)

    # Load just the model
    new_predictive_model = MLPM.PredictiveModel(config=ml_data_configuration)

    assert not new_predictive_model.model_is_trained
    assert new_predictive_model.training_data is None
    assert new_predictive_model.model is None

    new_predictive_model.load_model(path_to_save)

    assert new_predictive_model.model_is_trained
    assert new_predictive_model.training_data is None
    assert new_predictive_model.model is not None
