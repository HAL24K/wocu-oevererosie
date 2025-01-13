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