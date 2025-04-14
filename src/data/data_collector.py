"""A class for automatically collecting (geospatial) data from various sources."""

import json
import logging

from owslib.wfs import WebFeatureService
import geopandas as gpd
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
        local_geospatial_data: dict[str, gpd.GeoDataFrame] = None,
    ):
        """Initializes the data collector.

        :param source_shape: The shape defining the area of interest.
        :param source_epsg_crs: The EPSG code of the source CRS, so that we know how to transform if needed.
        :param buffer_in_metres: The buffer in meters around the source shape from which to get the data.
        :param wfs_services: The list of WFS services to get the data from.
        :param local_geospatial_data: The local geospatial data to use (in addition to the WFS data).
        """
        self.source_shape_raw = source_shape
        self.source_epsg_crs = source_epsg_crs
        self.buffer = buffer_in_metres
        self.wfs_services_raw = wfs_services
        self.wfs_services = None

        # TODO: We only transform the CRS - if needed - when processing the data. Should we do it here?
        self.local_geospatial_data_raw = local_geospatial_data

        self.source_shape = self.define_source_shape()

        self.initialize_wfs_services()

        self.relevant_geospatial_data = {}

    def initialize_wfs_services(self):
        self.wfs_services = {}
        for service in self.wfs_services_raw:
            self.wfs_services[service.name] = WebFeatureService(
                service.url, version=service.version
            )

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

    def load_data_from_single_wfs(self, wfs_name: str) -> dict[str, gpd.GeoDataFrame]:
        """Get a geodataframe from the specified WFS service.

        :param wfs_name: The name of the WFS service known to the class to get the data from.

        NOTE: We have seen that some WFS-s appear to limit the number of features they return. To get past that,
          we first try to ask for a lot of features at once and use whatever number of features is returned as
          the max allowed, asking for more batches (and shifting the starting index), until we get nothing back.
        TODO: figure out an automated way to see the limits
        TODO: some WFS refuse to return data past a high starting index (we saw 50_000). Also keep that in mind
          and make the code robust against that.
        """
        bounding_box = U.transform_shape_crs(
            self.source_epsg_crs, CONST.EPSG_RD, self.source_shape
        ).bounds

        # TODO: horridly awkward for loop, connect self.wfs_services nad self.wfs_services_raw
        relevant_layers = None
        for raw_wfs in self.wfs_services_raw:
            if raw_wfs.name == wfs_name:
                relevant_layers = raw_wfs.relevant_layers
                break

        assert relevant_layers is not None, f"WFS {wfs_name} is not known."

        geospatial_data = {}
        for layer in relevant_layers:
            logger.info(f"Getting data from the layer {layer} in {wfs_name}")

            # first try getting a lot of data at once
            geo_features, crs_info = self.load_data_from_single_wfs_layer(
                wfs_name, layer, bounding_box, CONST.WFS_MAX_FEATURES_TO_REQUEST
            )

            starting_index = len(geo_features)
            max_features_to_be_returned = len(geo_features)

            while True:
                geo_features_batch, _ = self.load_data_from_single_wfs_layer(
                    wfs_name,
                    layer,
                    bounding_box,
                    max_features_to_be_returned,
                    starting_index,
                )

                if not geo_features_batch:
                    break

                starting_index += max_features_to_be_returned
                geo_features.extend(geo_features_batch)

            if not geo_features:
                # if we don't get any data back, just return an empty geodataframe
                # TODO: is this the best way to do it? Maybe return none?
                geospatial_data[layer] = gpd.GeoDataFrame()
                continue

            geospatial_data_single_layer = gpd.GeoDataFrame.from_features(geo_features)

            # TODO: make the reading from the dictionary safe
            epsg_code = U.get_epsg_from_urn(crs_info["properties"]["name"])
            if epsg_code is None:
                logger.warning(
                    f"Could not extract EPSG code from the WFS data, assuming {CONST.EPSG_RD}"
                )
                epsg_code = CONST.EPSG_RD

            geospatial_data_single_layer.crs = epsg_code

            geospatial_data[layer] = geospatial_data_single_layer

        return geospatial_data

    def load_data_from_single_wfs_layer(
        self,
        wfs_service_name: str,
        wfs_layer_name: str,
        bounding_box: tuple = None,
        number_of_requested_features: int = CONST.WFS_MAX_FEATURES_TO_REQUEST,
        starting_index: int = 0,
    ) -> (list[dict], dict):
        """Get data from the specified WFS service layer."""
        raw_data = self.wfs_services[wfs_service_name].getfeature(
            typename=[wfs_layer_name],
            bbox=bounding_box,
            outputFormat=CONST.WFS_JSON_OUTPUT_FORMAT,
            maxfeatures=number_of_requested_features,
            startindex=starting_index,
        )
        raw_data = raw_data.read()
        data_as_json = json.loads(raw_data)
        geodata = data_as_json["features"]

        if len(geodata) == 0:
            return [], {}

        logger.info(
            f"Getting features {starting_index} to {starting_index + len(geodata)}."
        )

        crs_info = data_as_json["crs"]

        return data_as_json["features"], crs_info

    def get_data_from_all_wfs(self):
        """Loop through all the know WFS services and get data overlapping with the source shape."""
        for wfs_service in self.wfs_services:
            logger.info(f"Getting data from the WFS service {wfs_service}.")
            self.relevant_geospatial_data[wfs_service] = self.load_data_from_single_wfs(
                wfs_service
            )

    def get_local_geospatial_data(self):
        """Process the local geospatial data to only include the relevant parts."""
        for data_label, data in self.local_geospatial_data_raw.items():
            logger.info(f"Processing the local geospatial data {data_label}.")
            data_with_correct_crs = data.to_crs(epsg=self.source_epsg_crs)
            mask = data_with_correct_crs.intersects(self.source_shape)
            self.relevant_geospatial_data[data_label] = data_with_correct_crs[
                mask
            ].copy()
