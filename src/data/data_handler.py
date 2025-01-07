"""This class takes in the shape data, enriches them and generates inputs for a machine learning model."""

import logging
import geopandas as gpd

import src.constants as CONST
import src.utils as UTILS
import src.config as CONFIG
import src.data.data_collector as DC

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataHandler():
    """A clas to enrich the shape data and generate inputs for a machine learning model."""

    def __init__(self, input_data: gpd.GeoDataFrame, local_data_for_enrichment: dict[str, gpd.GeoDataFrame] = None):
        """"""
        self.raw_input_data = input_data
        self.raw_enrichment_geospatial_data = local_data_for_enrichment
