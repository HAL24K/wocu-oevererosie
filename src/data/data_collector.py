"""A class for automatically collecting (geospatial) data from various sources."""

from owslib.wfs import WebFeatureService
import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry

import src.constants as CONST
import src.utils as U


class DataCollector:
    """A class for automatically collecting (geospatial) data from various sources."""

    def __init__(
            self,
            source_shape: BaseGeometry,
            source_epsg_crs: int = CONST.EPSG_RD,
            buffer_in_metres: float = CONST.DEFAULT_COLLECTOR_BUFFER
    ):
        """Initializes the data collector.

        :param source_shape: The shape defining the area of interest.
        :param source_epsg_crs: The EPSG code of the source CRS, so that we know how to transform if needed.
        :param buffer_in_metres: The buffer in meters around the source shape from which to get the data.
        """
        self.source_shape_raw = source_shape
        self.source_epsg_crs = source_epsg_crs
        self.buffer = buffer_in_metres
        self.source_shape = self.define_source_shape()

        self.initialize_services()


    def initialize_services(self):
        self.wfs_services = {}
        for service in CONST.WFS_SERVICES:
            # TODO: Assing a version here?
            self.wfs_services[service] = WebFeatureService(service)

    def define_source_shape(self) -> BaseGeometry:
        """Defines the source shape with the buffer.

        NOTE: Here we are NL-centric and transform the shape to (meter-based) RD coordinates to inflate it.
           We then put it back into the original CRS.
        """
        source_shape_rd = U.trasnform_shape_crs(
            self.source_epsg_crs, CONST.EPSG_RD, self.source_shape_raw
        )
        beefed_up_source_shape = source_shape_rd.buffer(self.buffer)

        return U.trasnform_shape_crs(
            CONST.EPSG_RD, self.source_epsg_crs, beefed_up_source_shape
        )
