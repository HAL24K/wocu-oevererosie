"""Test the feature generation config functionality"""

import pytest

from pydantic_core import ValidationError
import src.data.feature_generation_config_schema as FGCS


def test_correct_definitions():
    # These should simply just run
    _ = FGCS.FeatureGenerationConfiguration(
        majority_class=["colX", "colY"], area_fraction=True
    )


def test_incorrect_definitions():
    with pytest.raises(ValidationError):
        # fail on asking for an aggregation operation that is not defined
        _ = FGCS.FeatureGenerationConfiguration(nonexistent_aggregation=True)

    with pytest.raises(ValidationError):
        # fail on asking for a non-string column name
        _ = FGCS.FeatureGenerationConfiguration(majority_class=[42])

    with pytest.raises(ValidationError):
        # fail on asking for a non-boolean value
        # NOTE: This is a bit weird, as pydantic is happy to also interpret "true", "yes" etc and transform into
        #   boolean, which I don't like that much
        _ = FGCS.FeatureGenerationConfiguration(area_fraction="sure")
