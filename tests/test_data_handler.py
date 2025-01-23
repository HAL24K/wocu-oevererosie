"""Test the data handler functionality

TODO: test for when there is no remote or other data nearby
"""

import pytest
import geopandas as gpd

import src.data.data_handler as DH
import src.paths as PATHS
import src.data.config as DATA_CONFIG


@pytest.fixture
def basic_beefed_up_data_handler():
    """Fixture for the basic data handler with defaults except for the big region buffer to have data."""
    test_prediction_regions = gpd.read_file(
        PATHS.TEST_DIR / "assets" / "prediction_regions_for_tests.geojson"
    )
    big_buffer = 1_000  # metres, known to contain data
    data_configuration = DATA_CONFIG.DataConfiguration(
        prediction_region_buffer=big_buffer
    )

    return DH.DataHandler(
        config=data_configuration,
        prediction_regions=test_prediction_regions,
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
        # TODO: improve this so that it uses what's needed
        if wfs_layer == "river_centerline":
            continue
        feature_present = False
        for feature_column in columns_added_in_feature_creation:
            if wfs_layer in feature_column:
                feature_present = True
                break

        assert feature_present


def test_enrichment_with_remote_data():
    """Test the enrichment of the data with remote data."""
    # TODO: create this test
    pass
