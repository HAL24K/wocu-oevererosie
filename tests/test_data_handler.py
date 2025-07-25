"""Test the data handler functionality

TODO: test for when there is no remote or other data nearby
"""

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import LineString, Point

import src.data.data_handler as DH
import src.paths as PATHS
import src.data.config as DATA_CONFIG
import src.constants as CONST
from conftest import real_erosion_border, default_data_configuration


@pytest.fixture
def prediction_regions_for_test():
    """Fixture for the prediction regions for testing."""
    return gpd.read_file(
        PATHS.TEST_DIR / "assets" / "prediction_regions_for_tests.geojson"
    )


@pytest.fixture
def erosion_data_for_test():
    """Fixture for the erosion data for testing."""
    return gpd.read_file(
        PATHS.TEST_DIR / "assets" / "river_bank_points_for_tests.geojson"
    )


@pytest.fixture
def local_enrichment_geodata():
    """Local geodata for data enrichment."""
    geodata = {}

    geodata[CONST.AggregationOperations.CENTERLINE_SHAPE.value] = gpd.read_file(
        PATHS.TEST_DIR / "assets" / "river_centerline.geojson"
    )

    return geodata


@pytest.fixture
def basic_beefed_up_data_handler(
    prediction_regions_for_test,
    erosion_data_for_test,
    local_enrichment_geodata,
    real_erosion_border,
):
    """Fixture for the basic data handler with defaults except for the big region buffer to have data."""
    big_buffer = 1_000  # metres, known to contain data
    data_configuration = DATA_CONFIG.DataConfiguration(
        prediction_region_buffer=big_buffer,
    )

    return DH.DataHandler(
        config=data_configuration,
        prediction_regions=prediction_regions_for_test,
        local_data_for_enrichment=local_enrichment_geodata,
        erosion_data=erosion_data_for_test,
        erosion_border=real_erosion_border,
    )


def test_generate_region_features(basic_beefed_up_data_handler, caplog):
    """Test the generation of features for a region."""
    # at first, the features are just the prediction regions
    # TODO: make this just the geometry, so that other potential columns are not taken into account?
    assert basic_beefed_up_data_handler.prediction_regions.equals(
        basic_beefed_up_data_handler.scope_region_features
    )

    basic_beefed_up_data_handler.add_remote_data_to_processed()
    assert "not been downloaded" in caplog.text

    basic_beefed_up_data_handler.create_data_from_remote()

    # make sure that we don't increase the number of regions in the feature creation
    assert len(basic_beefed_up_data_handler.scope_region_features) == len(
        basic_beefed_up_data_handler.prediction_regions
    )

    columns_added_in_feature_creation = list(
        set(basic_beefed_up_data_handler.scope_region_features.columns)
        - set(basic_beefed_up_data_handler.prediction_regions.columns)
    )

    # check that there is at least one feature created from each data layer
    for wfs_layer in basic_beefed_up_data_handler.config.feature_creation_config:
        # TODO: improve this so that it only checks the remote layers
        if wfs_layer == "river_centerline":
            continue
        feature_present = False
        for feature_column in columns_added_in_feature_creation:
            if wfs_layer in feature_column:
                feature_present = True
                break

        assert feature_present

    basic_beefed_up_data_handler.add_remote_data_to_processed()
    assert "process the inspection" in caplog.text

    basic_beefed_up_data_handler.process_erosion_features()

    original_processed_columns = (
        basic_beefed_up_data_handler.processed_erosion_data.columns
    )
    original_processed_data = basic_beefed_up_data_handler.processed_erosion_data.copy()

    basic_beefed_up_data_handler.add_remote_data_to_processed()
    assert basic_beefed_up_data_handler.processed_erosion_data.index.equals(
        original_processed_data.index
    )

    # make sure we added SOME columns
    assert len(original_processed_columns) < len(
        basic_beefed_up_data_handler.processed_erosion_data.columns
    )

    # check that repeated addition does not change the data
    original_processed_data = basic_beefed_up_data_handler.processed_erosion_data.copy()
    basic_beefed_up_data_handler.add_remote_data_to_processed()

    assert basic_beefed_up_data_handler.processed_erosion_data.equals(
        original_processed_data
    )
    assert "already added" in caplog.text


def test_erosion_data_processing(
    prediction_regions_for_test,
    erosion_data_for_test,
    local_enrichment_geodata,
    caplog,
):
    """Test the processing of the erosion data."""

    # this erosion border is stupidly far but formally works
    test_longitude_rd = 100_000
    test_latitude_rd = 400_000
    test_erosion_border_length = 100_000  # 100 km
    test_erosion_border = LineString(
        [
            (test_longitude_rd, test_latitude_rd),
            (test_longitude_rd + test_erosion_border_length, test_latitude_rd),
        ]
    )

    # we also update the erosion data with nonrobust points that line on the line and in the nonfiltering case
    # should influence us a lot
    internal_erosion_data = erosion_data_for_test.copy()
    additional_nonrobust_points = CONST.DEFAULT_NO_OF_POINTS_FOR_DISTANCE_CALCULATION * [
        {
            CONST.RIVER_BANK_POINT_STATUS: "NON OK POINT",
            CONST.PREDICTION_REGION_ID: prediction_regions_for_test[
                CONST.PREDICTION_REGION_ID
            ].iloc[0],
            CONST.TIMESTAMP: 3,  # we know this is in the data
            "geometry": Point(
                test_longitude_rd, test_latitude_rd, 0
            ),  # we want the point to lie exactly on the erosion border, that's how we know they are (not) there
        }
    ]
    additional_nonrobust_points_gdf = gpd.GeoDataFrame(
        additional_nonrobust_points, crs=internal_erosion_data.crs
    )

    # add points that have negative distance to the erosion border, i.e. they lie over the erosion border
    # note that this creation is a bit dodgy - we assign the prediction_region_id explicitly. If we did it by checking
    # which point lies inside which scope polygon this may filter out these points as they lie far from everything
    additional_robust_points = CONST.DEFAULT_NO_OF_POINTS_FOR_DISTANCE_CALCULATION * [
        {
            CONST.RIVER_BANK_POINT_STATUS: CONST.OK_POINT_LABEL,
            CONST.PREDICTION_REGION_ID: prediction_regions_for_test[
                CONST.PREDICTION_REGION_ID
            ].iloc[0],
            CONST.TIMESTAMP: 5,  # we know this is in the data
            # the points lie a bit randomly below the erosion border
            "geometry": Point(
                test_longitude_rd,
                test_latitude_rd - np.random.randint(1, 10),
                0,
            ),
        }
    ]
    additional_robust_points_gdf = gpd.GeoDataFrame(
        additional_robust_points, crs=internal_erosion_data.crs
    )

    internal_erosion_data = pd.concat(
        [
            internal_erosion_data,
            additional_nonrobust_points_gdf,
            additional_robust_points_gdf,
        ],
        ignore_index=True,
    )

    for filter_out_bad_points in [True, False]:
        # test for both filtering and not filtering the river bank points
        data_configuration = DATA_CONFIG.DataConfiguration(
            use_only_certain_river_bank_points=filter_out_bad_points
        )

        data_handler = DH.DataHandler(
            config=data_configuration,
            prediction_regions=prediction_regions_for_test,
            local_data_for_enrichment=local_enrichment_geodata,
            erosion_data=internal_erosion_data,
            erosion_border=test_erosion_border,
        )

        assert data_handler.processed_erosion_data is None

        data_handler.process_erosion_features()

        assert data_handler.processed_erosion_data is not None
        assert isinstance(data_handler.processed_erosion_data, pd.DataFrame)
        assert data_handler.processed_erosion_data.index.get_level_values(
            CONST.PREDICTION_REGION_ID
        ).nunique() == len(data_handler.prediction_regions)

        assert (
            data_handler.processed_erosion_data[CONST.DISTANCE_TO_EROSION_BORDER] < 0
        ).any()

        assert (
            CONST.TIMESTEPS_SINCE_LAST_MEASUREMENT
            in data_handler.processed_erosion_data.columns
        )
        assert (
            data_handler.processed_erosion_data[CONST.TIMESTEPS_SINCE_LAST_MEASUREMENT]
            .notna()
            .all()
        )

        # the first measurement for sure does not hape a previous one, so the time sine previous is the default one
        assert (
            data_handler.processed_erosion_data[
                CONST.TIMESTEPS_SINCE_LAST_MEASUREMENT
            ].values[0]
            == CONST.DEFAULT_LENGTH_OF_TIME_GAP_BETWEEN_MEASSUREMENTS
        )

        # again, bad points are the only ones directly at the erosion border:
        # * if present, there are points with zero distance
        # * if they are absent - not filtered out, we have data with zero distance
        if filter_out_bad_points:
            assert (
                data_handler.processed_erosion_data[CONST.DISTANCE_TO_EROSION_BORDER]
                != 0
            ).all()
        else:
            assert (
                data_handler.processed_erosion_data[CONST.DISTANCE_TO_EROSION_BORDER]
                == 0
            ).any()

    # check that we don't recompute if we don't need to
    data_handler.process_erosion_features()
    assert "Erosion data already processed" in caplog.text


def test_generate_model_features_local(
    default_data_configuration,
    prediction_regions_for_test,
    erosion_data_for_test,
    local_enrichment_geodata,
    real_erosion_border,
    caplog,
):
    """Test the feature creation from the processed data using only the local data.

    TODO: would it make more sense to also include the remotes here?
    """

    for use_differences in [True, False]:
        data_configuration = default_data_configuration
        data_configuration.use_differences_in_features = use_differences

        data_handler = DH.DataHandler(
            config=data_configuration,
            prediction_regions=prediction_regions_for_test,
            local_data_for_enrichment=local_enrichment_geodata,
            erosion_data=erosion_data_for_test,
            erosion_border=real_erosion_border,
        )

        data_handler.process_erosion_features()

        assert data_handler.erosion_features is None

        data_handler.generate_erosion_features()

        assert data_handler.erosion_features is not None

        total_no_of_expected_features = (
            default_data_configuration.number_of_lags
            + default_data_configuration.number_of_futures
        ) * (
            len(default_data_configuration.unknown_numerical_columns)
            + len(default_data_configuration.known_numerical_columns)
            + len(default_data_configuration.unknown_categorical_columns)
            + len(default_data_configuration.known_categorical_columns)
        )
        assert (
            len(data_handler.erosion_features.columns) == total_no_of_expected_features
        )

        # the ends of the groupings get chopped off as there are no differences
        expected_extra_nans = (
            2
            * data_handler.processed_erosion_data.index.get_level_values(
                data_configuration.prediction_region_id_column_name
            ).nunique()
            if use_differences
            else data_configuration.number_of_futures
            * data_handler.processed_erosion_data.index.get_level_values(
                data_configuration.prediction_region_id_column_name
            ).nunique()
        )

        assert (
            len(data_handler.erosion_features)
            == len(data_handler.processed_erosion_data) - expected_extra_nans
        )
        assert len(data_handler.erosion_features_complete) == len(
            data_handler.processed_erosion_data
        )

        # test column naming
        if use_differences:
            assert np.any(
                [
                    CONST.DIFFERENCE in col
                    for col in data_handler.erosion_features.columns
                ]
            )
        else:
            assert np.all(
                [
                    CONST.DIFFERENCE not in col
                    for col in data_handler.erosion_features.columns
                ]
            )


def test_generate_features_with_remote_data(
    default_data_configuration,
    prediction_regions_for_test,
    erosion_data_for_test,
    local_enrichment_geodata,
    real_erosion_border,
    caplog,
):
    """Test the enrichment of the data with remote data."""
    data_configuration = default_data_configuration
    data_configuration.known_categorical_columns = [
        "BrpGewas_majority_class_category",
        "rws_vegetatielegger:vegetatieklassen_majority_class_vlklasse",
    ]

    data_handler = DH.DataHandler(
        config=data_configuration,
        prediction_regions=prediction_regions_for_test,
        local_data_for_enrichment=local_enrichment_geodata,
        erosion_data=erosion_data_for_test,
        erosion_border=real_erosion_border,
    )

    data_handler.process_erosion_features()

    assert data_handler.erosion_features is None

    data_handler.create_data_from_remote()

    number_of_original_processed_columns = len(
        data_handler.processed_erosion_data.columns
    )

    data_handler.add_remote_data_to_processed()
    number_of_processed_columns_with_remote_data = len(
        data_handler.processed_erosion_data.columns
    )

    assert (
        number_of_processed_columns_with_remote_data
        > number_of_original_processed_columns
    )

    data_handler.generate_erosion_features()

    # make sure that all column types are included in this test data
    for column_type in CONST.KnownColumnTypes:
        assert (
            column_type.value in data_handler.columns_added_in_feature_creation.keys()
        )

    # make sure that the columns we have logged as added realyl exist
    assert len(data_handler.erosion_features.columns) == sum(
        [len(cols) for cols in data_handler.columns_added_in_feature_creation.values()]
    )
