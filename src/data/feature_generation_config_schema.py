"""Schema of the feature generation configuration for a particular data layer.

In practice, we have geospatial data provided by the user or downloaded from a WFS, which we need to aggregate into
features for a particular prediction region.
"""

from pydantic import Field, create_model
from typing import Optional

import src.constants as CONST

# TODO: without the default=None, the Optional does not seem to work and pydantic complains that they are missing
#  if not specified. Figure out how to make it work.
aggregation_operations = {
    CONST.AggregationOperations.NUM_DENSITY.value: (Optional[bool], Field(default=None, description="Whether to calculate the numerical density of the features within the prediction region.") ),
    CONST.AggregationOperations.COUNT.value: (Optional[bool], Field(default=None, description="Whether to count the number of features within the prediction region.") ),
    CONST.AggregationOperations.TOTAL_AREA.value: (Optional[bool], Field(default=None, description="Whether to calculate the total area of the features within the prediction region.") ),
    CONST.AggregationOperations.AREA_FRACTION.value: (Optional[bool], Field(default=None, description="Whether to calculate the fraction of the prediction region covered by the features.") ),
    CONST.AggregationOperations.MAJORITY_CLASS.value: (Optional[dict], Field(default=None, description="Which columns to create the majority class for.") ),
}

config_dict = {
    'extra': 'forbid',
    'validate_assignment': True,
}


FeatureGenerationConfiguration = create_model(
    "FeatureGenerationConfiguration",
    **aggregation_operations,
    __config__=config_dict,
)
