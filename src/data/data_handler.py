"""This class takes in the shape data, enriches them and generates inputs for a machine learning model."""

import logging
import pandas as pd
import geopandas as gpd
import shapely.geometry.base

import src.constants as CONST
import src.utils as UTILS
import src.config as CONFIG
import src.data.schema_wfs_service as SWS
import src.data.config as DATA_CONFIG
import src.data.data_collector as DC

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataHandler():
    """A clas to enrich the shape data and generate inputs for a machine learning model."""

    def __init__(
        self,
        config: DATA_CONFIG.DataConfiguration,
        prediction_regions: gpd.GeoDataFrame,
        local_data_for_enrichment: dict[str, gpd.GeoDataFrame] = None,
        erosion_data: gpd.GeoDataFrame = None
    ):
        """Initialise the object.

        :param config: a DataConfig object with the configuration for the data handler.
        :param prediction_regions: a geodataframe with the regions (polygons) to predict for.
        :param local_data_for_enrichment: a dictionary with geodataframes with the local data to enrich the erosion data with.
        :param wfs_services: a list of WFS services to collect remote data from
        :param erosion_data: a geodataframe with the erosion data to calculate targets (if required)
        """
        self.config = config
        self.prediction_regions = prediction_regions
        self.raw_enrichment_geospatial_data = local_data_for_enrichment
        self.raw_erosion_data = erosion_data

        # TODO: we assume the features always start here - maybe there is a more elegant way of doing it
        self.model_features = self.prediction_regions.copy()


    def create_features_from_remote(self):
        """For each prediction region, get the WFS data and calculate the features."""
        wfs_features = []
        for region in self.prediction_regions.geometry.values:
            data_collector = DC.DataCollector(
                source_shape=region,
                source_epsg_crs=self.prediction_regions.crs.to_epsg(),
                buffer_in_metres=self.config.prediction_region_buffer,
                wfs_services=self.config.known_wfs_services,
            )

            data_collector.get_data_from_all_wfs()

            # reminder: relevant_geospatial_data is a nested dictionary with [wfs_service][layer_name] as keys,
            #    inside of which is a geodataframe
            single_region_features = {}
            for wfs_service in data_collector.relevant_geospatial_data:
                for layer_name in data_collector.relevant_geospatial_data[wfs_service]:
                    feature_config = self.config.feature_creation_config.get(layer_name)
                    if feature_config is None:
                        logger.warning(
                            f"No feature configuration found for {layer_name} of the WFS {wfs_service}, skipping."
                            f"Why are we downloading the data though?"
                        )
                        continue

                    single_layer_features = self._generate_region_features(
                        region, data_collector.relevant_geospatial_data[wfs_service][layer_name], feature_config
                    )
                    # TODO: generate the feature name with a function so that it's callable from elsewhere
                    #   and easy to change
                    single_layer_features = {
                        f"{layer_name}_{agg_function}": feature_value
                        for agg_function, feature_value in single_layer_features.items()
                    }

                    single_region_features.update(single_layer_features)

            # TODO: flatten the dictionary first? so that it can be transformed in a nice dataframe easily
            single_region_features = UTILS.flatten_dictionary(single_region_features)
            wfs_features.append(single_region_features)
        wfs_features = pd.DataFrame(wfs_features)

        self.model_features = pd.concat([self.model_features, wfs_features], axis=1)


    @staticmethod
    def _generate_region_features(
        region: shapely.geometry.base.BaseGeometry,
        geospatial_data: gpd.GeoDataFrame,
        feature_config: dict
    ) -> dict:
        """Generate features for a single region and a single geospatial dataset."""
        single_region_features = {}
        # if we have a feature configuration for this layer, add the data to the features
        # TODO: is dict(feature_config) the right way to get data out of a pydantic BaseModel?
        for agg_operation, agg_params in dict(feature_config).items():
            if agg_params is None:
                # no data aggregation takes place
                continue

            # TODO: this could be done better, using sth like a dict of agg_operation -> function mapping
            match agg_operation:
                case CONST.AggregationOperations.NUM_DENSITY.value:
                    single_region_features[agg_operation] = UTILS.get_object_density(region, geospatial_data)
                case CONST.AggregationOperations.COUNT.value:
                    single_region_features[agg_operation] = UTILS.get_count_object_intersects(region, geospatial_data)
                case CONST.AggregationOperations.TOTAL_AREA.value:
                    single_region_features[agg_operation] = UTILS.get_total_area(region, geospatial_data)
                case CONST.AggregationOperations.AREA_FRACTION.value:
                    single_region_features[agg_operation] = UTILS.get_area_fraction(region, geospatial_data)
                case CONST.AggregationOperations.MAJORITY_CLASS.value:
                    single_region_features[agg_operation] = UTILS.get_majority_class(region, geospatial_data, agg_params)
                case _:
                    # this should not happen, since the config dict would not have been valid.
                    raise NotImplementedError(
                        f"Unknown aggregation operation {agg_operation}, "
                        f"pick one of {[agg.value for agg in CONST.AggregationOperations]}. "
                        f"You might add the operation to the schema and add the implementation to utils."
                    )

        return single_region_features


    def generate_prediction_region_features(self):
        """Generate features for the prediction regions."""


    def generate_erosion_features(self):
        """Using the prediction_regions and the raw erosion data, generate erosion-only features."""

    def generate_targets(self):
        """If needed (for training), add targets to the existing features."""
        raise NotImplementedError
