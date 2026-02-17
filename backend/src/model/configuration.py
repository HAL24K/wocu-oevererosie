"""Configuration for the deep learning model."""

from typing import Callable, List, Optional, Dict

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

import src.constants as CONST


@dataclass
class ModelConfiguration:
    """Configuration of the deep learning model."""

    unknown_continuous_columns: List[str] = Field(
        description=(
            "The names of the unknown continuous features, i.e. the defects, corresponding to the third "
            "dimension of the corresponding part of the dataset."
        )
    )

    unknown_categorical_columns: List[str] = Field(
        default=CONST.DEFAULT_COLUMN_TYPE_NOT_PRESENT,
        description=(
            "The categorical feature names, corresponding to the third "
            "dimension of the corresponding part of the dataset."
        ),
    )

    # TODO: specify default propagation function
    known_continuous_columns: List[str] = Field(
        default=CONST.DEFAULT_COLUMN_TYPE_NOT_PRESENT,
        description="The known continuous columns (we know thier future values) to use in the model.",
    )

    known_categorical_columns: List[str] = Field(
        default=CONST.DEFAULT_COLUMN_TYPE_NOT_PRESENT,
        description=(
            "The categorical feature names, corresponding to the third "
            "dimension of the corresponding part of the dataset."
        ),
    )

    embedding_dimension: int = Field(
        default=CONST.DEFAULT_EMBEDDING_DIMENSION,
        description="How many elements to use to represent a categorical column.",
        gt=0,
    )

    use_batch_normalization: bool = Field(
        default=CONST.DEFAULT_USE_BATCH_NORMALIZATION,
        description="Whether to use batch normalization in linear layers.",
    )

    dropout: float = Field(
        default=CONST.DEFAULT_DROPOUT,
        description="The dropout probability.",
        ge=0.0,
        lt=1.0,
    )

    nonlinearity_function: Callable = Field(
        default=CONST.DEFAULT_NONLINEARITY_FUNCTION,
        description="Pytorch non-linearity to use in the model.",
    )

    loss: Callable = Field(
        default=CONST.DEFAULT_LOSS,
        description="Pytorch loss function to use in the model.",
    )

    # TODO: whats the type hint for torch.float32 etc? torch.dtype gives an error...
    torch_float = Field(
        default=CONST.DEFAULT_TORCH_FLOAT_TYPE,
        description="The default torch float type to use in the model.",
    )

    number_of_recurrent_layers: int = Field(
        default=CONST.DEFAULT_NUMBER_OF_RECURRENT_LAYERS,
        description="The number of recurrent layers to use in the model.",
        gt=0,
    )

    hidden_mlp_layer_sizes: list[int] = Field(
        default=CONST.DEFAULT_HIDDEN_MLP_LAYER_SIZES,
        description="Neuron counts for the hidden layers of the MLP.",
    )

    optimizer: Callable = Field(
        default=CONST.DEFAULT_OPTIMIZER,
        description="The optimizer to use in the model.",
    )

    learning_rate: float = Field(
        default=CONST.DEFAULT_LEARNING_RATE,
        description="The learning rate to use in the model.",
    )

    teacher_forcing_probability: float = Field(
        default=CONST.DEFAULT_TEACHER_FORCING_PROBABILITY,
        description="The probability of using teacher forcing during an iteration in the training.",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def check_some_named_continuous_features_exist(self):
        """Check that the user explicitly provided something here."""
        if not self.unknown_continuous_columns:
            raise ValueError("There must be at least one continuous column.")

        return self
