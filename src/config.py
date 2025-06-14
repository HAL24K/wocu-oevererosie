"""Various configurations for the application.

TODO: These should technically be user-provided, so place them as such
"""

import src.data.schema_wfs_service as SWS
from src.data.feature_generation_config_schema import (
    FeatureGenerationConfiguration as FGC,
)
import src.constants as CONST

KNOWN_WFS_SERVICES = [
    SWS.WfsService(
        name=CONST.WFS_LAND_USE,
        url="https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0",
        relevant_layers=[CONST.BRPGEWAS],
    ),
    SWS.WfsService(
        name=CONST.WFS_BUILDING_LOCATION,
        url="https://service.pdok.nl/lv/bag/wfs/v2_0",
        version="2.0.0",
        relevant_layers=[CONST.BAG_PAND],
    ),
    SWS.WfsService(
        name=CONST.WFS_VEGETATION,
        url="https://geo.rijkswaterstaat.nl/services/ogc/gdr/rws_vegetatielegger/ows?version=2.0.0",
        version="2.0.0",
        relevant_layers=[
            CONST.RWS_WL_BOMEN,
            CONST.RWS_WL_HEGGEN,
            CONST.RWS_WL_VEGETATIEKLASSEN,
        ],
    ),
]

# for each WFS layer or a local dataset name, specify which column(s) to use to find the most relevant feature in the
# erosion area
# TODO: come up with more features
AGGREGATION_COLUMNS = {
    # for the plants get some majority classes and also the fraction of the area taken by plants
    CONST.BRPGEWAS: FGC(
        majority_class={"columns": ["category", "gewas"]}, area_fraction=True
    ),
    # similarly for buildings get the majority class for the usage and the fraction of the area taken by buildings
    CONST.BAG_PAND: FGC(
        majority_class={"columns": ["gebruiksdoel"]}, area_fraction=True
    ),
    # we count the tree density within our area
    CONST.RWS_WL_BOMEN: FGC(numerical_density=True),
    # (questionable) hedge density
    # TODO: hedges are linestrings so have length - would including that be helpful?
    # TODO: numerical density of lines is forbidden, as we require points. Figure out and uncomment
    # RWS_WL_HEGGEN: FGC(numerical_density=True),
    # we also get the majority class from the vegetatielegger
    CONST.RWS_WL_VEGETATIEKLASSEN: FGC(majority_class={"columns": ["vlklasse"]}),
    CONST.RIVER_CENTERLINE: FGC(centerline_shape={"neighbourhood_radius": 100}),
}
