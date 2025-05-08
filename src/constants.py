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

# flag to whether to use absolute values as features (e.g. distance to the signalling line) or the differences between
# them (e.g. how much erosion has happened). We lean towards using the differences, as e.g. the signalling line can
# move arbitrarily but the erosion remains
DEFAULT_USE_DIFFERENCES_IN_FEATURES = True

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

DEFAULT_NUMBER_OF_LAGS = 1
DEFAULT_NUMBER_OF_FUTURES = 1

# some suffixes for creating temporary columns
FLOAT = "float"
AS_NUMBER = "as_number"


class KnownColumnTypes(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class KnownFutureFillOperations(Enum):
    """For "know future" columns, these are the possible operations how the features evolve."""

    FILL = "fill"  # propagate the last known value forever
    INCREMENT = "increment"  # every time step (year) increment the previous value by 1


KNOWN_CATEGORIES = {}

NUMERICAL_COLUMNS_KEY = "numerical"
CATEGORICAL_COLUMNS_KEY = "categorical"

PREVIOUS = "past"
UPCOMING = "future"

TIMESTEPS_SINCE_LAST_MEASUREMENT = "timesteps_since_last_measurement"
DEFAULT_LENGTH_OF_TIME_GAP_BETWEEN_MEASSUREMENTS = 1.0  # year

DEFAULT_KNOWN_NUMERICAL_COLUMNS = [TIMESTEPS_SINCE_LAST_MEASUREMENT]
DEFAULT_UNKNOWN_NUMERICAL_COLUMNS = [
    DISTANCE_TO_EROSION_BORDER,
]
DEFAULT_KNOWN_CATEGORICAL_COLUMNS = []
DEFAULT_UNKNOWN_CATEGORICAL_COLUMNS = []
