"""Test the model configuration class."""

import pytest

import src.model.configuration as CONFIG
from pydantic import ValidationError


def test_invalid_model_configuration():
    """Test what all makes the model configuration fall over."""
    with pytest.raises(ValidationError):
        # not all parameters have defaults
        CONFIG.ModelConfiguration()

    with pytest.raises(ValidationError) as exc_info:
        CONFIG.ModelConfiguration(
            unknown_continuous_columns=[], known_categorical_columns=[]
        )

    assert "at least one" in str(exc_info.value)


def test_valid_model_configurations():
    """Can we run a normal configuration?"""
    CONFIG.ModelConfiguration(
        unknown_continuous_columns=["A", "B", "C"],
        known_categorical_columns=["D", "E", "F"],
    )
