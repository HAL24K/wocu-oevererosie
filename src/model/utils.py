"""Utilities for the model package."""

import logging
import re
import torch.nn as nn

import src.constants as CONST

logger = logging.getLogger(__name__)


class LinearModule(nn.Module):
    """
    A fully connected linear module that consists of multiple linear layers
    with optional batch normalization, activation, and dropout.

    :param in_features: Number of input features.
    :type in_features: int
    :param out_features: Number of output features.
    :type out_features: int
    :param hidden_layer_sizes: List of hidden layer sizes.
    :type hidden_layer_sizes: list[int]
    :param use_batch_norm: Whether to use batch normalization.
    :type use_batch_norm: bool
    :param dropout: Dropout probability.
    :type dropout: float
    :param nonlinearity_function: Nonlinearity function to use.
    :type nonlinearity_function: Callable
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layer_sizes: list[int],
        use_batch_norm: bool = CONST.DEFAULT_USE_BATCH_NORMALIZATION,
        dropout: float = CONST.DEFAULT_DROPOUT,
        nonlinearity_function: callable = CONST.DEFAULT_NONLINEARITY_FUNCTION,
    ):
        super().__init__()

        all_layer_sizes = [in_features] + hidden_layer_sizes + [out_features]

        self.layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.nonlinearity_function = nonlinearity_function

        # Input layer
        self.layers.append(nn.Linear(in_features, hidden_layer_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            self.layers.append(
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            )
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_layer_sizes[i + 1]))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            self.layers.append(nonlinearity_function())

        # Output layer
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def find_the_first_future_time_step(column_names: list[str]) -> str:
    """From a list of (future) column names, find the one the closest in the future."""
    future_column_pattern = re.compile(f"{CONST.UPCOMING}.*_1")

    candidate_columns = []
    for column in column_names:
        hits = re.findall(future_column_pattern, column)
        if hits:
            candidate_columns.append(column)

    if not candidate_columns:
        raise ValueError(
            "No future time step column found in the provided column names. "
            "Expected a column name with '_1' in it."
        )

    if len(candidate_columns) > 1:
        raise ValueError(
            f"Found more than one candidate column with '_1' in the name: {candidate_columns}"
        )

    return candidate_columns[0]
