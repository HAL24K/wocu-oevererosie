"""Configuration for various objects in the code."""

from typing import Callable, List, Optional, Dict

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

import src.constants as CONST

@dataclass
class DataConfiguration:
    """Configuration for the data handler."""
    erosion_shape_type: CONST.ErosionShapeType = Field(
        default=CONST.DEFAULT_EROSION_SHAPE_TYPE,
        description="The shape type in the general erosion data to use for calculating the erosion features.",
    )
