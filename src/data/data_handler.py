"""This class takes in the shape data, enriches them and generates inputs for a machine learning model."""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry.base
from shapely.geometry import LineString

import src.constants as CONST
import src.utils as UTILS
import src.data.config as DATA_CONFIG
import src.data.data_collector as DC
import src.data.custom_pytorch_dataset as CPD

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataHandler:
    """A clas to enrich the shape data and generate inputs for a machine learning model."""

    def __init__(
        self,
        config: DATA_CONFIG.DataConfiguration,
        prediction_regions: gpd.GeoDataFrame,
        local_data_for_enrichment: dict[str, gpd.GeoDataFrame] = None,
        erosion_data: gpd.GeoDataFrame = None,
        erosion_border: LineString = None,
        scaler_parameters: dict = None,
    ):
        """Initialise the object.

        :param config: a DataConfig object with the configuration for the data handler.
        :param prediction_regions: a geodataframe with the regions (polygons) to predict for.
        :param local_data_for_enrichment: a dictionary with geodataframes with the local data to enrich the erosion data with.
        :param erosion_data: a geodataframe with the erosion data to calculate targets (if required)
        :param erosion_border: a linestring with how far the erosion can proceed
        :param scaler_parameters: a dictionary with the parameters for the scaler, if any. If None, gets calculated from the data.
        """
        self.config = config
        self.prediction_regions = prediction_regions
        self.raw_enrichment_geospatial_data = local_data_for_enrichment
        self.raw_erosion_data = erosion_data
        self.erosion_border = erosion_border
        self.scaler_parameters = scaler_parameters

        # TODO: we assume the features always start here - maybe there is a more elegant way of doing it
        self.scope_region_features = self.prediction_regions.copy()

        self.processed_erosion_data = None
        self.erosion_features_complete = None
        self.erosion_features = None

        self.columns_added_in_feature_creation = {
            column_type.value: [] for column_type in CONST.KnownColumnTypes
        }

        # TODO: define this correctly
        logger.warning(
            f"Setting the number extra features to 0, even though it should be automatically calculated."
        )
        self.number_of_extra_futures = 0

        # track the remote status
        self.remote_data_downloaded = False
        self.remote_features_added = False

    def process_erosion_features(self, reload=False):
        """Process the erosion features into a usable format.

        :param reload: whether to recalculate if self.processed_erosion_data already exists

        NOTES: the goal here is to turn the erosion data, provided as a geodataframe with one river bank location per
          a time stamp per row into a dataframe with multiindex and distances to the erosion border:

                                           | distance_to_erosion_border
          prediction_region_id | timestamp |
          ---------------------+-----------+----------------------------
            1                  | 1         | 100
            1                  | 2         | 95
            2                  | 1         | 50
            2                  | 2         | 42
            ...

               we assume that both the erosion data and the prediction region data contain an ID column to be matched on

        TODO: make this less point-centered
        TODO: this assumes that the erosion border never changes - is this a valid assumption?
        """

        if self.processed_erosion_data is not None and not reload:
            logger.warning("Erosion data already processed, skipping.")
            return

        processed_erosion_data = list()

        available_region_ids = self.prediction_regions[
            self.config.prediction_region_id_column_name
        ].unique()
        internal_erosion_data = self.raw_erosion_data[
            self.raw_erosion_data[self.config.prediction_region_id_column_name].isin(
                available_region_ids
            )
        ].copy()
        logger.info(
            f"{len(internal_erosion_data)} lie within the specified prediction regions, "
            f"hence we drop the rest {len(self.raw_erosion_data) - len(internal_erosion_data)} from further analysis."
        )

        if self.config.use_only_certain_river_bank_points:
            ok_mask = (
                internal_erosion_data[CONST.RIVER_BANK_POINT_STATUS]
                == CONST.OK_POINT_LABEL
            )
            logger.info(
                f"Further, we only use the robustly detected river bank points (labeled {CONST.OK_POINT_LABEL}) "
                f"and drop {~ok_mask.sum()} points from the further analysis."
            )
            internal_erosion_data = internal_erosion_data[ok_mask]

        for (
            prediction_region_id,
            timestamp,
        ), local_erosion_data in internal_erosion_data.groupby(
            [
                self.config.prediction_region_id_column_name,
                self.config.timestamp_column_name,
            ]
        ):
            # TODO: distance is a metric that is always positive, so if we are a distance X from the line one year
            #   and then cross and end up Y<X on the other side, the speed of erosion will be wrong!
            # local_distances_bank_to_border = local_erosion_data.distance(self.erosion_border)
            local_distances_bank_to_border = (
                self.calculate_river_bank_distances_to_erosion_border(
                    local_erosion_data
                )
            )

            # Calculate the mean distance of the self.config.no_of_points_for_distance_calculation closest points
            # to the erosion border
            # I know that pandas has a .nsmallest() method, but it handles duplicates in a way that is not right for us
            # (it either drops them, or keeps more than N points, in any case messing up the average)
            # TODO: make the below more configurable
            local_distance_bank_to_border = (
                local_distances_bank_to_border.sort_values()
                .iloc[: self.config.no_of_points_for_distance_calculation]
                .mean()
            )

            processed_erosion_data.append(
                {
                    self.config.prediction_region_id_column_name: prediction_region_id,
                    self.config.timestamp_column_name: timestamp,
                    CONST.DISTANCE_TO_EROSION_BORDER: local_distance_bank_to_border,
                }
            )

        processed_erosion_data = pd.DataFrame(processed_erosion_data)
        processed_erosion_data.set_index(
            [
                self.config.prediction_region_id_column_name,
                self.config.timestamp_column_name,
            ],
            inplace=True,
        )

        self.processed_erosion_data = processed_erosion_data

        # TODO: decide whether this should come BEFORE we assign self.processed_erosion_data - perhaps more elegant,
        #   but we'd need to pass the dataframe
        self._enrich_processed_data_with_automated_features()

    def _enrich_processed_data_with_automated_features(self):
        """Add columns to the processed data that can be automatically calculated.

        This includes:
            - time since the last measurement
        """
        self._add_time_since_last_measurement()

    def _add_time_since_last_measurement(self):
        """Calculate the time since the last measurement and add it to the processed data.

        TODO: refactor TIMESTEPS_... into YEARS_... or something like that, once we get rid of AHN
        """
        self.processed_erosion_data[CONST.TIMESTEPS_SINCE_LAST_MEASUREMENT] = (
            self.processed_erosion_data.reset_index(level=CONST.TIMESTAMP)
            .groupby(CONST.PREDICTION_REGION_ID)[CONST.TIMESTAMP]
            .diff()
            .fillna(
                CONST.DEFAULT_LENGTH_OF_TIME_GAP_BETWEEN_MEASSUREMENTS
            )  # for unknown assume a single year
            .values
        )

    def calculate_river_bank_distances_to_erosion_border(
        self, erosion_data: gpd.GeoDataFrame
    ) -> pd.Series:
        """Calculate the distances between the existing river bank and the erosion border.

        :param erosion_data: locations of the river bank (points
        :return: a series with the distances between the individual points of the river bank and the erosion border

        NOTE: the distances returned are positive if the river bank lies inside the erosion border (we want this),
            negative if it lies outside (we don't want this).
        """
        assert (
            self.raw_enrichment_geospatial_data is not None
            and self.raw_enrichment_geospatial_data.get(
                CONST.AggregationOperations.CENTERLINE_SHAPE.value
            )
            is not None
        ), (
            f"Calculation of the distances of the river bank distances from the erosion border requires the "
            f"centerline shape data, which is not available. Please provide the river centreline as a geodataframe "
            f"inside the local_data_for_enrichment parameter "
            f"with the key {CONST.AggregationOperations.CENTERLINE_SHAPE.value}."
        )

        direction_factors = list()
        for _, row in erosion_data.iterrows():
            # TODO: can this be vectorized for the love of god?!
            local_point = row["geometry"]
            local_centerline = UTILS.get_relevant_centerline(
                local_point,
                self.raw_enrichment_geospatial_data[
                    CONST.AggregationOperations.CENTERLINE_SHAPE.value
                ],
            )

            local_direction_factor = (
                1
                if UTILS.is_point_between_two_lines(
                    local_point, local_centerline, self.erosion_border
                )
                else -1
            )
            direction_factors.append(local_direction_factor)

        local_distances = pd.DataFrame(erosion_data.distance(self.erosion_border))
        local_distances.columns = [CONST.DISTANCE_TO_EROSION_BORDER]

        local_distances[CONST.DIRECTION_FACTOR] = direction_factors
        local_distances[CONST.DISTANCE_TO_EROSION_BORDER] = (
            local_distances[CONST.DISTANCE_TO_EROSION_BORDER]
            * local_distances[CONST.DIRECTION_FACTOR]
        )

        return local_distances[CONST.DISTANCE_TO_EROSION_BORDER]

    def create_data_from_remote(self):
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
                        region,
                        data_collector.relevant_geospatial_data[wfs_service][
                            layer_name
                        ],
                        feature_config,
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

        # TODO: here we glue the old and the new features next to each other, ignoring the order. It **should** work
        #   due to the for loop but should replace this with a proper merge
        self.scope_region_features = pd.concat(
            [self.scope_region_features, wfs_features], axis=1
        )

        self.remote_data_downloaded = True

    def add_remote_data_to_processed(self):
        """Add the downloaded data to the processed data."""
        if not self.remote_data_downloaded:
            logger.warning(
                f"The remote data has not been downloaded yet, please do so first."
            )
            return

        if self.processed_erosion_data is None:
            logger.warning(
                "No processed data present, please process the inspection data first."
            )
            return

        if self.remote_features_added:
            logger.warning("Remote features already added, nothing is changing.")
            return

        # TODO: is this too cautious with dropping the geometry?
        internal_df = self.scope_region_features.drop("geometry", axis=1).copy()

        processed_data_index_names = self.processed_erosion_data.index.names
        self.processed_erosion_data = self.processed_erosion_data.reset_index().merge(
            internal_df, on=self.config.prediction_region_id_column_name, how="left"
        )
        self.processed_erosion_data.set_index(processed_data_index_names, inplace=True)

        self.remote_features_added = True

    @staticmethod
    def _generate_region_features(
        region: shapely.geometry.base.BaseGeometry,
        geospatial_data: gpd.GeoDataFrame,
        feature_config: dict,
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
                    single_region_features[agg_operation] = UTILS.get_object_density(
                        region, geospatial_data
                    )
                case CONST.AggregationOperations.COUNT.value:
                    single_region_features[agg_operation] = (
                        UTILS.get_count_object_intersects(region, geospatial_data)
                    )
                case CONST.AggregationOperations.TOTAL_AREA.value:
                    single_region_features[agg_operation] = UTILS.get_total_area(
                        region, geospatial_data
                    )
                case CONST.AggregationOperations.AREA_FRACTION.value:
                    single_region_features[agg_operation] = UTILS.get_area_fraction(
                        region, geospatial_data
                    )
                case CONST.AggregationOperations.MAJORITY_CLASS.value:
                    single_region_features[agg_operation] = UTILS.get_majority_class(
                        region, geospatial_data, **agg_params
                    )
                case CONST.AggregationOperations.CENTERLINE_SHAPE.value:
                    relevant_centerline = UTILS.get_relevant_centerline(
                        region, geospatial_data
                    )
                    single_region_features[agg_operation] = (
                        UTILS.get_nearby_linestring_shape(
                            region, relevant_centerline, **agg_params
                        )
                    )
                case _:
                    # this should not happen, since the config dict would not have been valid.
                    raise NotImplementedError(
                        f"Unknown aggregation operation {agg_operation}, "
                        f"pick one of {[agg.value for agg in CONST.AggregationOperations]}. "
                        f"You might add the operation to the schema and add the implementation to utils."
                    )

        return single_region_features

    def generate_erosion_features(self, reload: bool = False):
        """Generate features for the prediction regions."""
        if self.processed_erosion_data is None:
            logger.warning(
                "No processed data present, please process the inspection data first."
            )
            return

        if self.erosion_features is not None and not reload:
            logger.warning("Erosion features already present, nothing is changing.")
            return

        for parameter in (
            self.config.known_numerical_columns
            + self.config.unknown_numerical_columns
            + self.config.known_categorical_columns
            + self.config.unknown_categorical_columns
        ):
            missing_parameters = []
            if parameter not in self.processed_erosion_data.columns:
                missing_parameters.append(parameter)

            if missing_parameters:
                warning_message = (
                    f"Column{'s' if len(missing_parameters) > 1 else ''} {parameter} not "
                    f"present in the processed erosion data. Please add "
                    f"{'them' if len(missing_parameters) > 1 else 'it'} or change the configuration."
                )

                logger.warning(warning_message)

        self.erosion_features = pd.DataFrame(index=self.processed_erosion_data.index)

        for column in self.config.known_numerical_columns:
            # TODO: define how to pass future operation
            # TODO: things like the time since last operation is already a diff in the processed data, how to handle this here?
            self._prepare_single_feature(
                column,
                CONST.KnownColumnTypes.KNOWN_NUMERIC.value,
                use_differences=False,
                known_future=True,
            )

        for column in self.config.unknown_numerical_columns:
            self._prepare_single_feature(
                column,
                CONST.KnownColumnTypes.UNKNOWN_NUMERIC.value,
                use_differences=self.config.use_differences_in_features,
                known_future=False,
            )

        for column in self.config.known_categorical_columns:
            # TODO: define how to pass the future operation
            self._prepare_single_feature(
                column,
                CONST.KnownColumnTypes.KNOWN_CATEGORICAL.value,
                known_future=True,
            )

        for column in self.config.unknown_categorical_columns:
            self._prepare_single_feature(
                column,
                CONST.KnownColumnTypes.UNKNOWN_CATEGORICAL.value,
                known_future=False,
            )

        self.erosion_features_complete = self.erosion_features.copy()
        self.erosion_features = self.erosion_features.dropna()

        number_of_complete_features = len(self.erosion_features_complete)
        number_of_nonna_features = len(self.erosion_features)

        logger.info(
            f"Dropped {number_of_complete_features - number_of_nonna_features} samples "
            f"({100 * (number_of_complete_features - number_of_nonna_features)/number_of_complete_features:.2f}% "
            f"of the original) that contained missing values."
        )
        logger.info(
            f"Lagged features dataframe contains {len(self.erosion_features)} samples."
        )

        if len(self.erosion_features) == 0:
            logger.warning("No data left after dropping NaNs.")

    @staticmethod
    def _prepare_feature_column_name_root(column: str, column_type: str) -> str:
        """Prepare the root name for the feature column.

        :param column: the name of the column in the processed data
        :return: a string with the root name of the feature column
        """
        return f"{column}_{CONST.FLOAT if CONST.NUMERIC in column_type else CONST.AS_NUMBER}"

    def _prepare_single_feature(
        self,
        column: str,
        column_type: str,
        use_differences: bool = False,
        known_future: bool = False,
        future_operation: str = CONST.KnownFutureFillOperations.FILL.value,
    ):
        """Add one feature column to the prepared features.

        :param column: the name of the column in the processed data
        :param column_type: the type of column defined in CONST.KnownColumnTypes
        :param use_differences: if True, the feature is calculated as a difference between the current and the previous value
        :param known_future: a flag to see whether the future values are known (easily calculable)
        :param future_operation: if the future values are known, how to fill in the features

        TODO: for the known futures, implement the future filling in - probably has to be in the model!
        """
        assert column_type in [
            col_type.value for col_type in CONST.KnownColumnTypes
        ], f"Unknown column type {column_type} for column {column}."

        if use_differences and CONST.NUMERIC not in column_type:
            logger.warning(
                f"The `use_differences` feature only works for numeric columns. "
                f"You are trying to use it for {column_type} column {column}, where it will have no effect."
            )

        if known_future:
            assert future_operation in [
                op.value for op in CONST.KnownFutureFillOperations
            ], f"Unknown future operation {future_operation} for column {column}."

        # make sure that you don't overwrite the existing column
        temporary_column_name = self._prepare_feature_column_name_root(
            column, column_type
        )
        if use_differences:
            temporary_column_name = f"{CONST.DIFFERENCE}_{temporary_column_name}"

        if CONST.NUMERIC in column_type:
            self.processed_erosion_data[temporary_column_name] = (
                self.processed_erosion_data[column].astype(float)
            )
            if use_differences:
                self.processed_erosion_data[temporary_column_name] = (
                    self.processed_erosion_data.groupby(
                        self.config.prediction_region_id_column_name
                    )[temporary_column_name].diff()
                )
        elif CONST.CATEGORICAL in column_type:
            try:
                ordered_categories = CONST.KNOWN_CATEGORIES[column]
                ordered_categories = {
                    element: i for i, element in enumerate(ordered_categories)
                }
            except KeyError:
                raise KeyError(
                    f"Unknown categories for {column}, please define them first."
                )

            self.processed_erosion_data[
                temporary_column_name
            ] = self.processed_erosion_data[column].map(
                lambda x: ordered_categories.get(
                    x, CONST.DEFAULT_UNKNOWN_CATEGORY_LABEL
                )
            )

            # bookkeep on unknown categories
            entries_with_unknown_codes = list(
                set(self.processed_erosion_data[column].unique())
                - set(ordered_categories.keys())
            )

            if entries_with_unknown_codes:
                logger.warning(
                    f"Entries {list(entries_with_unknown_codes)} in column {column} "
                    f"have an unknown mapping to numerical categories, please define."
                )

        else:
            # this will only be raised if we introduced a new column type and forgot to account for that
            raise NotImplementedError(
                f"Unknown processing for type {column_type} (for column {column}), please define."
            )

        self._add_past_and_future_columns_for_feature(
            unique_grouping_id=self.config.prediction_region_id_column_name,
            source_column=temporary_column_name,
            column_type=column_type,
            additional_futures=self.number_of_extra_futures,
        )

        self.processed_erosion_data.drop(temporary_column_name, axis=1, inplace=True)

    def _add_past_and_future_columns_for_feature(
        self,
        unique_grouping_id: str,
        source_column: str,
        column_type: str,
        additional_futures: int,
    ):
        """Add all the lag and future columns for a particular feature by shifting the existing ones.

        :param unique_grouping_id: the name of the region
        :param source_column: the dataframe column to shift
        :param column_type: the type of the column, defined in CONST.KnownColumnTypes
        :param additional_futures: the number of additional future shifts (w.r.t. to the number from the config) that
           are to be added to the lagged features
        """
        for lag in range(self.config.number_of_lags):
            self._add_shifted_column(
                unique_grouping_id,
                source_column,
                column_type,
                lag,
                past_shift=True,
            )

        for future in range(1, self.config.number_of_futures + 1 + additional_futures):
            self._add_shifted_column(
                unique_grouping_id,
                source_column,
                column_type,
                future,
                past_shift=False,
            )

    def _add_shifted_column(
        self,
        grouping: str,
        original_column: str,
        column_type: str,
        shift: int,
        past_shift=True,
    ):
        """Add a shifted version of a column from processed data to the features.

        :param grouping: what to group the data by
        :param original_column: the column from processed_data to shift
        :param column_type: the type of the column, defined in CONST.KnownColumnTypes
        :param shift: the number of steps to shift the column by
        :param past_shift: if True, shift the column to the past
                           (downwards, otherwise to the future (upwards)
        """
        new_column_name = UTILS.generate_shifted_column_name(
            original_column, shift, past_shift
        )

        self.erosion_features[new_column_name] = np.nan
        self.erosion_features.loc[:, new_column_name] = (
            self.processed_erosion_data.groupby(level=grouping)[original_column].shift(
                shift if past_shift else -shift
            )
        )
        self.columns_added_in_feature_creation[column_type].append(new_column_name)

    def generate_pytorch_features(self):
        """Transform the existing feature dataframe into a pytorch dataset."""
        if self.erosion_features is None:
            logger.warning(
                "No erosion features present, please generate them first using generate_erosion_features()."
            )
            return

        data = {}

        for column_type in CONST.KnownColumnTypes:
            data[column_type.value] = self.prepare_feature_array(
                column_type=column_type.value
            )

        self.pytorch_dataset = CPD.PytorchDataset(data)

    def prepare_feature_array(self, column_type: str):
        """Prepare the feature array for the pytorch dataset."""
        known_column_types = [col_type.value for col_type in CONST.KnownColumnTypes]
        assert (
            column_type in known_column_types
        ), f"Unknown column type {column_type}, pick one of {known_column_types}."

        feature_values = []
        for column in getattr(self.config, f"{column_type}_columns"):
            feature_column_root = self._prepare_feature_column_name_root(column)

            single_feature_values = []
            for lag in range(self.config.number_of_lags - 1, 0, -1):
                # we start with the last lag and go backwards, so that the first lag is the most recent one
                feature_column_name = UTILS.generate_shifted_column_name(
                    feature_column_root, lag, past_shift=True
                )
                single_feature_values.append(
                    self.erosion_features[feature_column_name].values[:, np.newaxis]
                )

            for future in range(
                1, self.config.number_of_futures + 1 + self.number_of_extra_futures
            ):
                # we start with the first future and go forwards, so that the first future is the most recent one
                feature_column_name = UTILS.generate_shifted_column_name(
                    feature_column_root, future, past_shift=False
                )
                single_feature_values.append(
                    self.erosion_features[feature_column_name].values[:, np.newaxis]
                )

            single_feature_values = np.concatenate(single_feature_values, axis=1)[
                :, :, np.newaxis
            ]

            if CONST.NUMERIC in column_type:
                # scale the values
                if column not in self.scaler_parameters.keys():
                    self.scaler_parameters[column] = self.calculate_scaler_parameters(
                        column
                    )

                lo_scaling, hi_scaling = self.scaler_parameters[column]
                single_feature_values = self.scale_values(
                    single_feature_values, lo_scaling, hi_scaling, inverse=False
                )

            feature_values.append(single_feature_values)

        feature_values = np.concatenate(feature_values, axis=2)

        return feature_values

    def calculate_scaler_parameters(self, column: str) -> tuple:
        """Calculate the scaler parameters for a given column.

        Specifically, at the moment we calculate the min and max values of the column, for the minmax scaling.
        """
        # the internal column name is the name of the "present" feature
        internal_column_name = self._prepare_feature_column_name_root(
            column, CONST.FLOAT
        )
        internal_column_name = UTILS.generate_shifted_column_name(
            internal_column_name, 0, past_shift=True
        )

        low = self.erosion_features[internal_column_name].min()
        high = self.erosion_features[internal_column_name].max()

        return (low, high)

    @staticmethod
    def scale_values(
        input_array: np.array, lower_bound, upper_bound, inverse: bool = False
    ) -> np.array:
        """Scale the values in the input array using the scaler parameters.

        :param input_array: the input array to scale
        :param lower_bound: the lower bound of the scaling
        :param upper_bound: the upper bound of the scaling
        :param inverse: if True, scale the values back to the original range
        :return: the scaled values
        """
        if inverse:
            return input_array * (upper_bound - lower_bound) + lower_bound
        else:
            return (input_array - lower_bound) / (upper_bound - lower_bound)
