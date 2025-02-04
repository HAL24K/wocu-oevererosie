"""Test the data handler functionality

TODO: test for when there is no remote or other data nearby
"""

import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import LineString, Point

import src.data.data_handler as DH
import src.paths as PATHS
import src.data.config as DATA_CONFIG
import src.constants as CONST


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
def basic_beefed_up_data_handler(prediction_regions_for_test):
    """Fixture for the basic data handler with defaults except for the big region buffer to have data."""
    big_buffer = 1_000  # metres, known to contain data
    data_configuration = DATA_CONFIG.DataConfiguration(
        prediction_region_buffer=big_buffer
    )

    return DH.DataHandler(
        config=data_configuration,
        prediction_regions=prediction_regions_for_test,
    )


def test_generate_region_features(basic_beefed_up_data_handler):
    """Test the generation of features for a region."""
    # at first, the features are just the prediction regions
    # TODO: make this just the geometry, so that other potential columns are not taken into account?
    assert basic_beefed_up_data_handler.prediction_regions.equals(
        basic_beefed_up_data_handler.model_features
    )

    basic_beefed_up_data_handler.create_features_from_remote()

    # make sure that we don't increase the number of regions in the feature creation
    assert len(basic_beefed_up_data_handler.model_features) == len(
        basic_beefed_up_data_handler.prediction_regions
    )

    columns_added_in_feature_creation = list(
        set(basic_beefed_up_data_handler.model_features.columns)
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


def test_erosion_data_processing(
    prediction_regions_for_test, erosion_data_for_test, caplog
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
    additional_nonrobust_points = (
        CONST.DEFAULT_NO_OF_POINTS_FOR_DISTANCE_CALCULATION
        * [
            {
                CONST.RIVER_BANK_POINT_STATUS: "NON OK POINT",
                CONST.PREDICTION_REGION_ID: prediction_regions_for_test[
                    CONST.PREDICTION_REGION_ID
                ].iloc[0],
                CONST.TIMESTAMP: 3,  # we know this is in the data
                "geometry": Point(test_longitude_rd, test_latitude_rd, 0),
            }
        ]
    )
    additional_nonrobust_points_df = gpd.GeoDataFrame(
        additional_nonrobust_points, crs=internal_erosion_data.crs
    )
    internal_erosion_data = pd.concat(
        [internal_erosion_data, additional_nonrobust_points_df], ignore_index=True
    )

    for filter_out_bad_points in [True, False]:
        # test for both filtering and not filtering the river bank points
        data_configuration = DATA_CONFIG.DataConfiguration(
            use_only_certain_river_bank_points=filter_out_bad_points
        )

        data_handler = DH.DataHandler(
            config=data_configuration,
            prediction_regions=prediction_regions_for_test,
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

        if filter_out_bad_points:
            assert (
                data_handler.processed_erosion_data[
                    CONST.DISTANCE_TO_EROSION_BORDER
                ].min()
                > 0
            )
        else:
            assert (
                data_handler.processed_erosion_data[
                    CONST.DISTANCE_TO_EROSION_BORDER
                ].min()
                == 0
            )

    # check that we don't recompute if we don't need to
    data_handler.process_erosion_features()
    assert "Erosion data already processed" in caplog.text


def test_enrichment_with_remote_data():
    """Test the enrichment of the data with remote data."""
    # TODO: create this test
    pass
