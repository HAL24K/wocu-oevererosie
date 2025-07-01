"""A wrapper to various sklearn's predictive models.

TODO: at the moment we are only assuming unknown numerical columns to be present, update this if and when relevant.
TODO: deal with categorical columns, right now they are directly passed as numbers to the model.
TODO: can we predict more than one thing at a time? Right now no - complete that
TODO: figure out a better way of multi-timestep prediction. Right now we train it to predict one time step ahead
  and we can loop and predict more from there. BUT if the user prepares the sequences differently, we should be ablt
  to make use of that (teacher forcing and whatnot).
"""

import logging
import numpy as np
import pandas as pd
import pathlib
import pickle

import src.data.config as DATA_CONFIG
import src.constants as CONST
import src.model.utils as MODEL_UTILS

logger = logging.getLogger(__name__)


class PredictiveModel:
    """A wrapper to various sklearn's predictive models."""

    def __init__(
        self,
        config: DATA_CONFIG.DataConfiguration,
        model: object = None,  # This should be a scikit-learn model or similar
        training_data: pd.DataFrame = None,
        verbose: bool = False,
    ):
        """
        :param model: The model to use for predictions.
        :param verbose: Whether to print debug information.
        """
        self.configuration = config

        self.training_data = training_data
        self.model = model
        self.verbose = verbose

        self.model_is_trained = False

    @property
    def target_columns(self):
        """Get the target columns from the training data."""
        return [
            column
            for column in self.configuration.unknown_numerical_columns
            if CONST.UPCOMING in column
        ]

    @property
    def input_columns(self):
        """Get the model input columns from the training data.

        TODO: make this more robust to a change in order of the columns.
        """
        return sorted(list(set(self.training_data.columns) - set(self.target_columns)))

    def train(self, retrain: bool = False):
        """Train the data."""
        if self.model_is_trained and not retrain:
            logger.info("Model already trained, skipping training.")
            return

        if self.training_data is None:
            logger.warning("No training data provided. Cannot train the model.")
            return

        # only use the first time step of the future
        # TODO: improve the target column finding logic
        target_column = MODEL_UTILS.find_the_first_future_time_step(self.target_columns)

        self.model.fit(
            self.training_data[self.input_columns],
            self.training_data[target_column],
        )

        self.model_is_trained = True

    def predict(self, input_data, prediction_steps: int = 1):
        if self.model is None:
            logger.warning("No model has been provided.")
            return

        if not self.model_is_trained:
            logger.warning(
                "Model is not trained. Please train the model before predicting."
            )
            return

        predictions = []
        for time_step in range(prediction_steps):
            # Shift the input data to predict the next time step
            prediction = self.model.predict(input_data)
            input_data = self._shift_input_data(
                input_data[self.input_columns], prediction
            )

            predictions.append(prediction)

        predictions = pd.DataFrame(
            np.concat(predictions, axis=1), columns=self.target_columns
        )
        return predictions

    def _shift_input_data(self, data, prediction):
        """Shift the input columns so that they apply for the next time step.

        More specifically:
        * shift the numerical data one year further
        * replace the "current" data with the predicted ones
        * move forward the known data
        """

    def save_model(self, path: pathlib.Path, keep_training_data: bool = False):
        """Save the model for future use.

        :param path: The path to save the model to.
        :param keep_training_data: Whether to keep the training data in the model file.

        TODO: this is copied from BaseModel, refactor to use a common base class.
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

    def load_model(self, path_to_model: pathlib.Path):
        """Load the model from the file."""
        with open(path_to_model, "rb") as f:
            model = pickle.load(f)

        self.model_is_trained = True
        return model
