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

WFS_JSON_OUTPUT_FORMAT = "json"

EPSG_REGEX = r"(?<=EPSG\:\:)[0-9]{4,5}"
