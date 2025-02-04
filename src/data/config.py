"""Configuration for various objects in the code."""

from typing import Callable, List, Optional, Dict

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

import src.constants as CONST
import src.config as CONFIG
import src.data.schema_wfs_service as SWS

@dataclass
class DataConfiguration:
    """Configuration for the data handler."""
    include_targets: bool = Field(
        default=CONST.DEFAULT_INCLUDE_TARGETS,
        description="Whether to include the targets with the features - used in model training.",
    )
    feature_creation_config: dict = Field(
        default=CONFIG.AGGREGATION_COLUMNS,
        description="The details for how to process the input data into features.",
    )
    prediction_region_buffer: float = Field(
        default=CONST.DEFAULT_PREDICTION_REGION_BUFFER,
        description="The buffer by which the prediction region is inflated to get more geospatial data.",
    )
    known_wfs_services: List[SWS.WfsService] = Field(
        default=CONFIG.KNOWN_WFS_SERVICES,
        description="The known WFS services to collect data from.",
    )
    no_of_points_for_distance_calculation: int = Field(
        default=CONST.DEFAULT_NO_OF_POINTS_FOR_DISTANCE_CALCULATION,
        description=(
            "The mean of this many closest points within the scope polygon is used to calculate the distance"
            "between the erosion limit line and the river bank."
        ),
        gt=0,
    )
    prediction_region_id_column_name: str = Field(
        default=CONST.PREDICTION_REGION_ID,
        description="The name of the column in the input dataframes that contains the ID of the region.",
    )
    timestamp_column_name: str = Field(
        default=CONST.TIMESTAMP,
        description="The name of the column in the input dataframes that contains the timestamp of the data.",
    )
    use_only_certain_river_bank_points: bool = Field(
        default=CONST.DEFAULT_USE_ONLY_CERTAIN_RIVER_BANK_POINTS,
        description="Whether to use only certain points (labelled OK or so) from the river bank data.",
    )

    @model_validator(mode="after")
    def check_feature_creation_config_nonempty(self):
        if len(self.feature_creation_config) < 1:
            raise ValueError(
                "The attribute 'feature_creation_config' must contain at least one entry, "
                "otherwise we create no features."
            )

        return self
