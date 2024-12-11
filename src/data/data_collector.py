"""A class for automatically collecting (geospatial) data from various sources."""

from owslib.wfs import WebFeatureService
import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry

import src.constants as CONST


class DataCollector:
    """A class for automatically collecting (geospatial) data from various sources."""

    def __init__(self, source_shape: BaseGeometry, source_epsg_crs: int = CONST.EPSG_RD, buffer: float = CONST.DEFAULT_COLLECTOR_BUFFER):
        """Initializes the data collector."""
        self.source_shape = source_shape
        self.source_epsg_crs = source_epsg_crs
        self.buffer = buffer

        self.initialize_services()


    def initialize_services(self):
        self.wfs_services = {}
        for service in CONST.WFS_SERVICES:
            # TODO: Assing a version here?
            self.wfs_services[service] = WebFeatureService(service)
