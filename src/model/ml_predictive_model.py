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
import src.utils as UTILS
import src.model.utils as MODEL_UTILS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PredictiveModel:
    """A wrapper to various sklearn's predictive models."""

    def __init__(
        self,
        config: DATA_CONFIG.DataConfiguration,
        model: object = None,  # This should be an untrained scikit-learn model or similar
        training_data: pd.DataFrame = None,
        column_kinds: dict = None,
        verbose: bool = False,
    ):
        """
        :param config: The data configuration to use for the model.
        :param model: The model to use for predictions.
        :param training_data: training data to use for the model.
        :param column_kinds: A dictionary mapping column names to their kinds (unknown numerical, categorical, etc.).
        :param verbose: Whether to print debug information.
        """
        self.configuration = config

        self.training_data = training_data
        self.column_kinds = column_kinds
        self.model = model
        self.verbose = verbose

        self.model_is_trained = False

        self._check_column_kinds()

    def _check_column_kinds(self):
        """Check if the column kinds match the training data.

        TODO: this will fall over at the first issue, consider changing it so that every issue is reported so that all can be fixed at once.
        """
        if self.training_data is None:
            # only run this if we have training data
            return

        allowed_column_kinds = [kind.value for kind in CONST.KnownColumnTypes]

        # column_kinds need to be provided if training_data is given
        if self.training_data is not None:
            assert self.column_kinds is not None, (
                f"You are providing the training data with columns {self.training_data.columns} but no column kinds. "
                f"Please provide the column kinds as a dictionary with keys {allowed_column_kinds}."
            )

        # the keys of the column kinds have to be of the allowed values
        for column_kind in self.column_kinds:
            assert (
                column_kind in allowed_column_kinds
            ), f"Column kind {column_kind} is not among the allowed ones: {allowed_column_kinds}."

        # the columns in the training data have to be in the column kinds
        # TODO: do we need to make sure that all columns are in the column kinds are unique?
        # TODO: do we care if column_kinds contains columns not in self.training_data?
        all_columns_in_kinds = []
        for column_kind in self.column_kinds:
            all_columns_in_kinds.extend(self.column_kinds[column_kind])

        for column in self.training_data.columns:
            assert column in all_columns_in_kinds, (
                f"Column {column} is not in the column kinds. "
                "Please make sure that all columns in the training data are in the column kinds."
            )

            all_columns_in_kinds.remove(column)

    @property
    def target_columns(self):
        """Get the target columns from the training data."""
        return [
            column
            for column in self.training_data.columns
            for column_root in self.configuration.unknown_numerical_columns
            if CONST.UPCOMING in column and column_root in column
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

        training_data = self.training_data.copy()
        training_data.dropna(subset=[target_column] + self.input_columns, inplace=True)

        self.model.fit(
            training_data[self.input_columns],
            training_data[target_column],
        )

        self.model_is_trained = True

    def predict(self, input_data, prediction_steps: int = 1):
        if self.model is None:
            logger.warning("No model has been provided.")
            return None

        if not self.model_is_trained:
            logger.warning(
                "Model is not trained. Please train the model before predicting."
            )
            return None

        predictions = []
        for time_step in range(prediction_steps):
            # Shift the input data to predict the next time step
            try:
                prediction = self.model.predict(input_data.values).reshape(-1, 1)
            except AttributeError as e:
                logger.error(f"Model prediction failed: {e}")
                return None

            predictions.append(prediction)

            # Only shift if we have more steps to predict
            if time_step < prediction_steps - 1:
                input_data = self._shift_input_data(
                    input_data[self.input_columns], prediction
                )
                if input_data is None:
                    logger.error("Failed to shift input data for next prediction step.")
                    return None

        predictions_df = pd.DataFrame(
            np.concatenate(predictions, axis=1),
            columns=[
                f"future_{time_step}" for time_step in range(1, prediction_steps + 1)
            ],
        )
        return predictions_df

    def _shift_input_data(self, data, prediction):
        """Shift the input columns so that they apply for the next time step.

        More specifically:
        * shift the numerical data one year further
        * replace the "current" data with the predicted ones
        * move forward the known data

        Returns:
            pd.DataFrame: The shifted input data for the next time step, or None if shifting fails.
        """
        if data is None or prediction is None:
            return None

        shifted_data = data.copy()

        for column_kind in self.column_kinds:
            if not self.column_kinds[column_kind]:
                # don't shift nonexistent columns
                continue
            match column_kind:
                case CONST.KnownColumnTypes.UNKNOWN_NUMERIC.value:
                    # TODO: finish this!
                    column_renaming = {
                        original_column: UTILS.get_temporally_previous_column_name(
                            original_column
                        )
                        for original_column in self.column_kinds[column_kind]
                    }
                    # NOTE: The shifting logic is incomplete - this is a placeholder
                    # Full implementation needed for proper multi-step prediction
                case CONST.KnownColumnTypes.UNKNOWN_CATEGORICAL.value:
                    # we cannot predict these yet
                    # TODO: implement this
                    raise NotImplementedError(
                        f"Shifting for {column_kind} not implemented yet."
                    )
                case CONST.KnownColumnTypes.KNOWN_NUMERIC.value:
                    # TODO: finish this!
                    pass
                case CONST.KnownColumnTypes.KNOWN_CATEGORICAL.value:
                    # known categoricals presumably don't change (a category remains a category)
                    pass

        # For now, return the data as-is since the shifting logic is incomplete
        # This prevents crashes but multi-step prediction may not work correctly
        # until the full implementation is completed
        return shifted_data

    def save(self, path: pathlib.Path, keep_training_data: bool = False):
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

    def load_model(self, path_to_stored_class: pathlib.Path):
        """Load the model from the file.

        TODO: this is quite a weird logic: we save the whole class (with or without data) but then loading it here
          we only want the model itself. Figure it out.
        """
        with open(path_to_stored_class, "rb") as f:
            model = pickle.load(f)

        self.model = model.model
        self.model_is_trained = True
