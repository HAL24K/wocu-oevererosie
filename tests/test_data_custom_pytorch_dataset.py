"""Test the custom_pytoch_dataset code."""

import pytest
import numpy as np

import src.constants as CONST
import src.data.custom_pytorch_dataset as CPD


@pytest.fixture
def dataset_configuration():
    """Create a dataset configuration for testing."""
    return {
        "number_of_samples": 10,
        "number_of_timesteps": 5,
        "max_number_of_features": 10,
    }


@pytest.fixture
def random_dataset(dataset_configuration):
    """Create a random dataset with consistent lengths."""
    dataset = {}
    for feature_type in CONST.KnownColumnTypes:
        number_of_features = (
            np.random.randint(dataset_configuration["max_number_of_features"]) + 1
        )  # to have at least one feature
        dataset[feature_type.value] = np.random.normal(
            size=(
                dataset_configuration["number_of_samples"],
                dataset_configuration["number_of_timesteps"],
                number_of_features,
            )
        )

    return dataset


def test_pytorch_dataset_initialization(random_dataset, dataset_configuration):
    """Test the initialization of the PytorchDataset."""
    dataset = CPD.PytorchDataset(random_dataset)

    assert len(dataset) == dataset_configuration["number_of_samples"]

    random_sample_index = np.random.randint(dataset_configuration["number_of_samples"])
    random_sample = dataset[random_sample_index]

    assert isinstance(random_sample, dict)
