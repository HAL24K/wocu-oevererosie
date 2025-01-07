"""Schema of the WFS service information."""

from pydantic.dataclasses import dataclass
from pydantic import Field

import src.constants as CONST

@dataclass
class WfsService:
    """Schema of the WFS service information."""

    name: str = Field(
        description="The human readable name of the WFS service."
    )
    url: str = Field(
        description="The URL of the WFS service."
    )
    relevant_layers: list[str] = Field(
        description="The relevant layers of the WFS service."
    )
    version: str = Field(
        default=CONST.DEFAULT_WFS_VERSION,
        description="The version of the WFS service."
    )
