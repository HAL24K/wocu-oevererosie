"""The baseline model for the river bank erosion as an average of the erosion recorded for each individual
scope region.
"""

import logging
import numpy as np
import pandas as pd
import pathlib
import pickle

import src.constants as CONST
import src.data.config as DATA_CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaselineErosionModel:
    """Baseline erosion model."""

    def __init__(
        self,
        config: DATA_CONFIG.DataConfiguration,
        training_data: pd.DataFrame,
        verbose: bool = False,
    ):
        """Initialise the object with the input data.

        :param config: the configuration for the model
        :param training_data: training data, in the format of DataHandler.processed_erosion_data

        # TODO: make a bespoke config perhaps?
        """
        self.configuration = config
        self.training_data = training_data
        self.verbose = verbose

        self.model = None

        self.model_is_trained = self.model is not None

    def train(self, retrain: bool = False):
        """Train the model.

        TODO: so far this is hardcoded to be a mean of the individual erosion steps. Make this way better, e.g.:
          - mean of various (all?) time step combinations
          - not just mean
          - ...
        """
        if self.model is not None and not retrain:
            logger.info("Model already trained, skipping training.")
            return

        trained_model = {}
        for region_id, region_data in self.training_data.groupby(
            CONST.PREDICTION_REGION_ID
        ):
            if self.verbose:
                logger.info(f"Training the baseline model for region {region_id}")

            region_data_simple_index = region_data.sort_index().reset_index()

            erosion_speeds = []
            for i in range(len(region_data_simple_index) - 1):
                # loop through the consecutive time steps and calculate erosion speeds
                time_step_size = (
                    region_data_simple_index.iloc[i + 1][
                        self.configuration.timestamp_column_name
                    ]
                    - region_data_simple_index.iloc[i][
                        self.configuration.timestamp_column_name
                    ]
                )
                erosion_step_size = (
                    region_data_simple_index.iloc[i + 1][
                        CONST.DISTANCE_TO_EROSION_BORDER
                    ]
                    - region_data_simple_index.iloc[i][CONST.DISTANCE_TO_EROSION_BORDER]
                )

                erosion_speeds.append(erosion_step_size / time_step_size)

            erosion_speed = np.mean(np.array(erosion_speeds))

            if self.verbose:
                logger.info(
                    f"Mean erosion speed for region {region_id} is {erosion_speed}."
                )

            trained_model[region_id] = erosion_speed

        self.model = trained_model
        self.model_is_trained = True

    def predict(self, data: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
        """Predict on the input data.

        :param data: The data to predict on.
        :param prediction_length: The number of time units (normally years) to predict for.

        NOTE: we assume the input data to have the same format as the DataHandler.processed_erosion_data, i.e. sth like

                                                   | distance_to_erosion_border
          prediction_region_id | timestamp |
          ---------------------+-----------+----------------------------
            1                  | 1         | 100
            1                  | 2         | 95
            2                  | 1         | 50
            2                  | 2         | 42

        That means that if we only want to predict for the future, we only provide the current timestamps
        TODO: deal with NaNs in the input data
        TODO: label the columns correctly, now it's future years from the current timestamp but they can overlap
        """
        # cannot predict without a model
        if not self.model_is_trained:
            logger.warning("The model has not been trained, no prediction possible.")
            return pd.DataFrame()

        # cannot predict on new regions
        # TODO: better dealing with unknown regions
        unknown_regions = set(
            data.index.get_level_values(CONST.PREDICTION_REGION_ID)
        ) - set(self.model.keys())
        if unknown_regions:
            logger.warning(
                f"The data contains unknown regions: {unknown_regions}, please first train the model on these. "
                f"Returning no prediction for now."
            )
            return pd.DataFrame()

        predicted_data = pd.DataFrame(index=data.index)

        for prediction_step in range(1, prediction_length + 1):
            predicted_data[
                f"future_{CONST.DISTANCE_TO_EROSION_BORDER}_{prediction_step}"
            ] = data[
                CONST.DISTANCE_TO_EROSION_BORDER
            ] + prediction_step * data.index.get_level_values(
                CONST.PREDICTION_REGION_ID
            ).map(
                self.model
            )

        return predicted_data

    def save_model(self, path: pathlib.Path, keep_training_data: bool = False):
        """Save the model for future use.

        :param path: The path to save the model to.
        :param keep_training_data: Whether to keep the training data in the model file.
        """
        if not keep_training_data:
            # put the training data to the side temporarily
            tmp_training_data = self.training_data.copy()
            self.training_data = None

        with open(path, "wb") as f:
            pickle.dump(self, f)

        if not keep_training_data:
            # restore the training data
            self.training_data = tmp_training_data
