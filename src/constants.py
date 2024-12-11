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
WFS_LAND_USE = "https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0"
WFS_BUILDING_LOCATION = "https://service.pdok.nl/lv/bag/wfs/v2_0"

WFS_SERVICES = [WFS_LAND_USE, WFS_BUILDING_LOCATION]
