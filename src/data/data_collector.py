"""A class for automatically collecting (geospatial) data from various sources."""
import json
import logging

from owslib.wfs import WebFeatureService
import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry

import src.constants as CONST
import src.config as CONFIG
import src.utils as U
import src.data.schema_wfs_service as SWS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataCollector:
    """A class for automatically collecting (geospatial) data from various sources.

    TODO: Very Dutch-centric (assumes RD all over the place), make more universal - read the CRS from the WFS or so...
    """

    def __init__(
            self,
            source_shape: BaseGeometry,
            source_epsg_crs: int = CONST.EPSG_RD,
            buffer_in_metres: float = CONST.DEFAULT_COLLECTOR_BUFFER,
            wfs_services: list[SWS.WfsService] = CONFIG.KNOWN_WFS_SERVICES,
    ):
        """Initializes the data collector.

        :param source_shape: The shape defining the area of interest.
        :param source_epsg_crs: The EPSG code of the source CRS, so that we know how to transform if needed.
        :param buffer_in_metres: The buffer in meters around the source shape from which to get the data.
        """
        self.source_shape_raw = source_shape
        self.source_epsg_crs = source_epsg_crs
        self.buffer = buffer_in_metres
        self.wfs_services_raw = wfs_services
        self.wfs_services = None

        self.source_shape = self.define_source_shape()

        self.initialize_wfs_services()


    def initialize_wfs_services(self):
        self.wfs_services = {}
        for service in self.wfs_services_raw:
            self.wfs_services[service.name] = WebFeatureService(service.url, version=service.version)

    def define_source_shape(self) -> BaseGeometry:
        """Defines the source shape with the buffer.

        NOTE: Here we are NL-centric and transform the shape to (meter-based) RD coordinates to inflate it.
           We then put it back into the original CRS.
        """
        source_shape_rd = U.transform_shape_crs(
            self.source_epsg_crs, CONST.EPSG_RD, self.source_shape_raw
        )
        beefed_up_source_shape = source_shape_rd.buffer(self.buffer)

        return U.transform_shape_crs(
            CONST.EPSG_RD, self.source_epsg_crs, beefed_up_source_shape
        )

    def load_data_from_single_wfs(self, wfs_name: str) -> gpd.GeoDataFrame:
        """Get a geodataframe from the specified WFS service.

        :param wfs_name: The name of the WFS service known to the class to get the data from.
        """
        bounding_box = U.transform_shape_crs(self.source_epsg_crs, CONST.EPSG_RD, self.source_shape).bounds

        # TODO: horridly awkward for loop, connect self.wfs_services nad self.wfs_services_raw
        for raw_wfs in self.wfs_services_raw:
            if raw_wfs.name == wfs_name:
                relevant_layers = raw_wfs.relevant_layers
                break

        raw_data = self.wfs_services[wfs_name].getfeature(typename=relevant_layers, bbox=bounding_box, outputFormat=CONST.WFS_JSON_OUTPUT_FORMAT)
        raw_data = raw_data.read()
        data_as_json = json.loads(raw_data)

        geospatial_data = gpd.GeoDataFrame.from_features(data_as_json["features"])

        # TODO: make the reading from the dictionary safe
        epsg_code = U.get_epsg_from_urn(data_as_json["crs"]["properties"]["name"])
        if epsg_code is None:
            logger.warning(f"Could not extract EPSG code from the WFS data, assuming {CONST.EPSG_RD}")
            epsg_code = CONST.EPSG_RD

        geospatial_data.crs = epsg_code

        return geospatial_data
