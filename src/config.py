"""Various configurations for the application."""

import src.data.schema_wfs_service as SWS
import src.constants as CONST

BRPGEWAS = "BrpGewas"
BAG_PAND = "bag:pand"
RWS_WL_BOMEN = "rws_vegetatielegger:bomen"
RWS_WL_HEGGEN = "rws_vegetatielegger:heggen"
RWS_WL_VEGETATIEKLASSEN = "rws_vegetatielegger:vegetatieklassen"


KNOWN_WFS_SERVICES = [
    SWS.WfsService(
        name=CONST.WFS_LAND_USE,
        url="https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0",
        relevant_layers=[BRPGEWAS],
    ),
    SWS.WfsService(
        name=CONST.WFS_BUILDING_LOCATION,
        url="https://service.pdok.nl/lv/bag/wfs/v2_0",
        version="2.0.0",
        relevant_layers=[BAG_PAND],
    ),
    SWS.WfsService(
        name=CONST.WFS_VEGETATION,
        url="https://geo.rijkswaterstaat.nl/services/ogc/gdr/rws_vegetatielegger/ows?version=2.0.0",
        version="2.0.0",
        relevant_layers=[
            RWS_WL_BOMEN,
            RWS_WL_HEGGEN,
            RWS_WL_VEGETATIEKLASSEN,
        ],
    ),
]

# for each WFS layer or a local dataset name, specify which column(s) to use to find the most relevant feature in the
# erosion area
# TODO: come up with more features
AGGREGATION_COLUMNS = {
    # for the plants get some majority classes and also the fraction of the area taken by plants
    BRPGEWAS: {
        CONST.AggregationOperations.MAJORITY_CLASS.value: ["category", "gewas"],
        CONST.AggregationOperations.AREA_FRACTION.value: True,
    },
    # similarly for buildings get the majority class for the usage and the fraction of the area taken by buildings
    BAG_PAND: {
        CONST.AggregationOperations.MAJORITY_CLASS.value: ["gebruiksdoel"],
        CONST.AggregationOperations.AREA_FRACTION.value: True,
    },
    # we count the tree density within our area
    RWS_WL_BOMEN: {CONST.AggregationOperations.NUM_DENSITY.value: True},
    # (questionable) hedge density
    # TODO: hedges are linestrings so have length - would including that be helpful?
    RWS_WL_HEGGEN: {CONST.AggregationOperations.NUM_DENSITY.value: True},
    # we also get the majority class from the vegetatielegger
    RWS_WL_VEGETATIEKLASSEN: {
        CONST.AggregationOperations.MAJORITY_CLASS.value: ["vlklasse"]
    },
}
