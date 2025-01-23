"""All the hardcoded values live here. This is the place to change them if needed."""

from enum import Enum

# where the Netherlands is
CENTRE_NL_LON = 5.2913
CENTRE_NL_LAT = 52.1326

DEFAULT_NL_ZOOM = 8  # at this zoom most of the country is visible when centred of the centre of the country

EPSG_WGS84 = 4326
EPSG_RD = 28992

# constants for the DataCollector class
DEFAULT_COLLECTOR_BUFFER = 0.0  # metres around the input shape

# WFS sources
DEFAULT_WFS_VERSION = "1.0.0"

WFS_LAND_USE = "land_use"
WFS_BUILDING_LOCATION = "building_location"
WFS_VEGETATION = "vegetation"

WFS_JSON_OUTPUT_FORMAT = "json"
WFS_MAX_FEATURES_TO_REQUEST = 10_000  # a big number to get all the features available

EPSG_REGEX = r"(?<=EPSG\:\:)[0-9]{4,5}"


class ErosionShapeType(Enum):
    """The type of the erosion shape."""

    POINT = "Point"
    POLYGON = "Polygon"


DEFAULT_EROSION_SHAPE_TYPE = ErosionShapeType.POINT.value


class AggregationOperations(Enum):
    """Operations allowed to be performed on the geospatial data to get features."""

    NUM_DENSITY = "numerical_density"
    COUNT = "count"
    TOTAL_AREA = "total_area"
    AREA_FRACTION = "area_fraction"
    MAJORITY_CLASS = "majority_class"
    CENTERLINE_SHAPE = "centerline_shape"


DEFAULT_INCLUDE_TARGETS = False
DEFAULT_PREDICTION_REGION_BUFFER = 10  # metres

DEFAULT_NEIGHBOURHOOD_RADIUS = 100  # metres
