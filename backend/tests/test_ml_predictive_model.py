"""Tests for the ML predictive model"""

import pickle

import numpy as np
import pytest
import sklearn.linear_model as sklm

import src.constants as CONST
import src.data.data_handler as DH
import src.model.ml_predictive_model as MLPM

# Builds its inputs through DataHandler, which calls live WFS services, so these
# need network access. Excluded from CI via -m "not integration".
pytestmark = pytest.mark.integration

NUMBER_OF_FUTURES = 2


@pytest.fixture
def ml_data_configuration(default_data_configuration):
    """Create a default data configuration for the ML model."""
    data_configuration = default_data_configuration

    # TODO: update the test data to possibly use the default dtm_date here
    data_configuration.timestamp_column_name = "ahn_version"

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

    return (
        data_handler.columns_added_in_feature_creation,
        data_handler.erosion_features_complete,
    )


def test_columns(ml_data_configuration, ml_data):
    """Test that the target and input columns are correctly generated."""
    columns_in_data, model_data = ml_data

    predictive_model = MLPM.PredictiveModel(
        config=ml_data_configuration,
        training_data=model_data,
        column_kinds=columns_in_data,
    )

    target_columns = predictive_model.target_columns

    assert np.all([c for c in target_columns if CONST.UPCOMING in c])
    assert (
        len(target_columns) == 2
    )  # Assuming we have two future steps as per the configuration

    input_columns = predictive_model.input_columns

    assert len(input_columns) == len(model_data.columns) - len(target_columns)
    assert set(input_columns + target_columns) == set(model_data.columns)


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

    columns_in_data, model_data = ml_data

    predictive_model = MLPM.PredictiveModel(
        config=ml_data_configuration,
        training_data=model_data,
        column_kinds=columns_in_data,
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


def test_check_column_types(ml_data_configuration, ml_data, caplog):
    """Test that the model checks the column types correctly."""
    columns_in_data, model_data = ml_data

    # Forgot the column_kinds
    with pytest.raises(AssertionError) as exc_info:
        _ = MLPM.PredictiveModel(
            config=ml_data_configuration,
            training_data=model_data,
        )

    assert "You are providing" in str(exc_info)

    # unknown column_kind
    unknown_column_kind = "unknown_column_kind"
    columns_in_data_with_extra_key = columns_in_data.copy()
    columns_in_data_with_extra_key[unknown_column_kind] = ["colX", "colY"]

    with pytest.raises(AssertionError) as exc_info:
        _ = MLPM.PredictiveModel(
            config=ml_data_configuration,
            training_data=model_data,
            column_kinds=columns_in_data_with_extra_key,
        )

    assert unknown_column_kind in str(exc_info)

    # a column not sorted into any column kind
    extra_column = "extra_column"
    data_with_extra_column = model_data.copy()
    data_with_extra_column[extra_column] = np.random.rand(len(model_data))

    with pytest.raises(AssertionError) as exc_info:
        _ = MLPM.PredictiveModel(
            config=ml_data_configuration,
            training_data=data_with_extra_column,
            column_kinds=columns_in_data,
        )

    assert extra_column in str(exc_info)


def test_predict(ml_data_configuration, ml_data):
    NUMBER_OF_PREDICTION_STEPS = 10
    NUMBER_OF_TEST_SAMPLES = 4

    columns_in_data, model_data = ml_data

    predictive_model = MLPM.PredictiveModel(
        config=ml_data_configuration,
        training_data=model_data,
        column_kinds=columns_in_data,
        model=sklm.LinearRegression(),
    )

    predictive_model.train()

    input_data = model_data[predictive_model.input_columns].dropna()
    prediction = predictive_model.predict(
        input_data.iloc[:NUMBER_OF_TEST_SAMPLES],
        prediction_steps=NUMBER_OF_PREDICTION_STEPS,
    )
    assert prediction.shape == (NUMBER_OF_TEST_SAMPLES, NUMBER_OF_PREDICTION_STEPS)
