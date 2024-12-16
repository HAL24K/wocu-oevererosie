"""Various configurations for the application."""

import src.data.schema_wfs_service as SWS
import src.constants as CONST


KNOWN_WFS_SERVICES = [
    SWS.WfsService(
        name=CONST.WFS_LAND_USE,
        url="https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0",
        relevant_layers=["BrpGewas"],
    ),
    SWS.WfsService(
        name=CONST.WFS_BUILDING_LOCATION,
        url="https://service.pdok.nl/lv/bag/wfs/v2_0",
        version="2.0.0",
        relevant_layers=["bag:pand"],
    ),
]
