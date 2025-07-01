"""Tests for the model_utils module."""

import pytest
import torch

import src.constants as CONST
import src.model.utils as U


@pytest.fixture
def default_linear_module_parameters():
    """Fixture for parameters of the LinearModule."""
    return {
        "in_features": 10,
        "out_features": 5,
        "hidden_layer_sizes": [20, 15],
        "use_batch_norm": True,
        "dropout": 0.3,
        "nonlinearity_function": CONST.DEFAULT_NONLINEARITY_FUNCTION,
    }


@pytest.fixture
def simple_linear_module(default_linear_module_parameters):
    """Fixture for a simple linear module."""
    return U.LinearModule(
        default_linear_module_parameters["in_features"],
        default_linear_module_parameters["out_features"],
        default_linear_module_parameters["hidden_layer_sizes"],
        default_linear_module_parameters["use_batch_norm"],
        default_linear_module_parameters["dropout"],
        default_linear_module_parameters["nonlinearity_function"],
    )


def test_linear_module_initialization(
    simple_linear_module, default_linear_module_parameters
):
    """Test the initialization of the LinearModule."""
    linear_module = simple_linear_module

    assert (
        len(linear_module.layers) == 6
    )  # 1 input + 2 hidden + 1 output + batch norm + dropout
    assert (
        linear_module.use_batch_norm
        is default_linear_module_parameters["use_batch_norm"]
    )
    assert linear_module.dropout == default_linear_module_parameters["dropout"]


def test_linear_module_forward_pass(
    simple_linear_module, default_linear_module_parameters
):
    """Test the forward pass of the LinearModule."""
    linear_module = simple_linear_module

    batch_size = 1
    # Create a random input tensor
    input_tensor = torch.randn(
        batch_size, default_linear_module_parameters["in_features"]
    )

    # Forward pass
    linear_module.eval()  # evaluation mode so that batch norm uses running stats
    output_tensor = linear_module(input_tensor)

    assert output_tensor.shape == (
        batch_size,
        default_linear_module_parameters["out_features"],
    )


def test_find_the_first_future_time_step():
    """Test that identifying the first future time step works correctly."""
    # Case 1: All fine
    columns = ["past_1", "past_2", "future_1", "future_2"]

    first_future = U.find_the_first_future_time_step(columns)
    assert first_future == "future_1"

    # Case 2: No future time steps
    columns = ["past_1", "past_2"]
    with pytest.raises(ValueError):
        U.find_the_first_future_time_step(columns)

    # Case 3: Multiple future columns
    columns = ["future1_1", "future2_1", "future_2", "future_3"]
    with pytest.raises(ValueError):
        U.find_the_first_future_time_step(columns)
