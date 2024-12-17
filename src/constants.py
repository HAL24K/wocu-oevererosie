"""All the hardcoded values live here. This is the place to change them if needed."""

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
# TODO: replace the below with an automated way of determining
# some of the WFS appear to have a limit of 1000 features that are to be returned
# here we download the features in a loop until we have them all
WFS_MAX_RETURNABLE_FEATURES = 1_000

EPSG_REGEX = r"(?<=EPSG\:\:)[0-9]{4,5}"
