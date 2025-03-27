"""Test the features of the baseline model."""

import pytest
import pandas as pd

import src.constants as CONST
import src.model.baseline_model as BM


@pytest.fixture
def training_data():
    """Return training data in the required format, e.g.

                                     | distance_to_erosion_border
    prediction_region_id | timestamp |
    ---------------------+-----------+----------------------------
      1                  | 1         | 100
      1                  | 2         | 95
      2                  | 1         | 50
      2                  | 2         | 42
    """
    # region 1 is eroding across the line, region 2 is eroding without any crossing, region 3 on average does not move,
    # region 4 is adding mass, region 5 does not have all measurements
    training_data = pd.DataFrame(
        {
            CONST.PREDICTION_REGION_ID: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
            CONST.TIMESTAMP: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3],
            CONST.DISTANCE_TO_EROSION_BORDER: [
                50,
                10,
                -10,
                100,
                80,
                60,
                -10,
                10,
                -10,
                10,
                50,
                30,
                50,
                10,
            ],
        }
    )

    training_data = training_data.set_index(
        [CONST.PREDICTION_REGION_ID, CONST.TIMESTAMP]
    )

    return training_data


@pytest.fixture
def baseline_model(default_data_configuration, training_data):
    """Initialize a baseline model with the training data."""
    model = BM.BaselineErosionModel(
        config=default_data_configuration,
        training_data=training_data,
        verbose=True,
    )

    return model


@pytest.fixture
def prediction_data():
    pred_data = pd.DataFrame(
        {
            CONST.PREDICTION_REGION_ID: [
                1,
                1,
                4,
                4,
                4,
            ],
            CONST.TIMESTAMP: [
                1,
                4,
                1,
                3,
                5,
            ],
            CONST.DISTANCE_TO_EROSION_BORDER: [50, 10, -10, 10, 10],
        }
    )

    pred_data = pred_data.set_index([CONST.PREDICTION_REGION_ID, CONST.TIMESTAMP])

    return pred_data


def test_train_baseline_model(baseline_model, caplog):
    """Test training the baseline model."""
    assert not baseline_model.model_is_trained
    assert baseline_model.model is None

    baseline_model.train()

    assert baseline_model.model_is_trained
    assert baseline_model.model is not None
    assert isinstance(baseline_model.model, dict)

    assert set(baseline_model.model.keys()) == set(
        baseline_model.training_data.index.get_level_values(CONST.PREDICTION_REGION_ID)
    )

    # retraining model does nothing
    baseline_model.train()
    assert "already trained" in caplog.text


def test_predict_baseline_model(baseline_model, prediction_data):
    """Test predicting with the baseline model."""

    # CASE 1: no model trained
    assert not baseline_model.model_is_trained
    prediction = baseline_model.predict(prediction_data, prediction_length=1)
    assert prediction.empty

    # CASE 2: standard prediction
    prediction_length = 10  # time steps

    baseline_model.train()
    assert baseline_model.model_is_trained
    prediction = baseline_model.predict(
        prediction_data, prediction_length=prediction_length
    )

    assert not prediction.empty
    assert (
        len(prediction.columns) == prediction_length
    )  # TODO: fix this once we label the columns correctly

    # CASE 3: nonexistent prediction region returns empty
    extra_region = pd.DataFrame(
        {
            CONST.PREDICTION_REGION_ID: [42, 42],
            CONST.TIMESTAMP: [1, 2],
            CONST.DISTANCE_TO_EROSION_BORDER: [666, 667],
        }
    )
    extra_region = extra_region.set_index([CONST.PREDICTION_REGION_ID, CONST.TIMESTAMP])

    amended_prediction_data = pd.concat([prediction_data, extra_region])
    prediction = baseline_model.predict(
        amended_prediction_data, prediction_length=prediction_length
    )

    assert prediction.empty


def test_storing_baseline_model(baseline_model, tmp_path):
    """Test storing the baseline model."""
    baseline_model.train()

    # make sure we have training data
    assert baseline_model.training_data is not None

    path_to_model_with_data = tmp_path / "model_with_data.pkl"
    path_to_model_without_data = tmp_path / "model_without_data.pkl"

    baseline_model.save_model(path_to_model_with_data, keep_training_data=True)
    baseline_model.save_model(path_to_model_without_data, keep_training_data=False)

    # make sure that the training data is still there
    assert baseline_model.training_data is not None

    # load the models
    loaded_model_with_data = BM.BaselineErosionModel.load_model(path_to_model_with_data)
    loaded_model_without_data = BM.BaselineErosionModel.load_model(
        path_to_model_without_data
    )

    assert loaded_model_with_data.model == loaded_model_without_data.model
    assert loaded_model_with_data.training_data.equals(baseline_model.training_data)

    assert loaded_model_without_data.training_data is None
