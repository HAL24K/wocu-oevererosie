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

# for generating the measure of the line shape at a given point, we need to have a neighbourhood in which to look,
# this defines its size
DEFAULT_NEIGHBOURHOOD_RADIUS = 100  # metres

DEFAULT_USE_ONLY_CERTAIN_RIVER_BANK_POINTS = (
    True  # only use the OK data points, not the outliers
)
OK_POINT_LABEL = "OK"  # how are the OK points labelled in the data

# when calculating the distance between the erosion limit line and the river bank, use the mean distance of this many
# closest points as the true value
DEFAULT_NO_OF_POINTS_FOR_DISTANCE_CALCULATION = 3

# the input data should contain some standard column names
PREDICTION_REGION_ID = "location_id"
# TODO: get a real timestamp and rename the timestamp column!
TIMESTAMP = "ahn_version"
RIVER_BANK_POINT_STATUS = "status"

# column names used in the processed data
DISTANCE_TO_EROSION_BORDER = "distance_to_erosion_border"
DIRECTION_FACTOR = "direction_factor"
