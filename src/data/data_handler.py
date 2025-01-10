"""This class takes in the shape data, enriches them and generates inputs for a machine learning model."""

import logging
import geopandas as gpd

import src.constants as CONST
import src.utils as UTILS
import src.config as CONFIG
import src.data.schema_wfs_service as SWS
import src.data.data_collector as DC

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataHandler():
    """A clas to enrich the shape data and generate inputs for a machine learning model."""

    def __init__(self, prediction_regions: gpd.GeoDataFrame, local_data_for_enrichment: dict[str, gpd.GeoDataFrame] = None, wfs_services: list[SWS.WfsService] = CONFIG.KNOWN_WFS_SERVICES, erosion_data: gpd.GeoDataFrame = None):
        """Initialise the object.

        :param prediction_regions: a geodataframe with the regions (polygons) to predict for.
        :param local_data_for_enrichment: a dictionary with geodataframes with the local data to enrich the erosion data with.
        :param wfs_services: a list of WFS services to collect remote data from
        :param erosion_data: a geodataframe with the erosion data to calculate targets
        """
        self.prediction_regions = prediction_regions
        self.raw_enrichment_geospatial_data = local_data_for_enrichment
        self.known_wfs_services = wfs_services
        self.raw_erosion_data = erosion_data


    def generate_prediction_region_features(self):
        """Generate features for the prediction regions."""


    def generate_erosion_features(self):
        """Using the prediction_regions and the raw erosion data, generate erosion-only features."""
        raise NotImplementedError