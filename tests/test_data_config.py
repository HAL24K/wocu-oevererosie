"""Test the config functionality"""

import pytest

import src.data.config as DATA_CONFIG


def test_valid_setup():
    # empty works as at the moment everything has defaults
    _ = DATA_CONFIG.DataConfiguration()

    # empty WFS services should also work (we would simply not use that data
    _ = DATA_CONFIG.DataConfiguration(known_wfs_services=[])


def test_invalid_setup():
    # we need at least one feature creation config
    with pytest.raises(ValueError):
        _ = DATA_CONFIG.DataConfiguration(feature_creation_config={})

    with pytest.raises(ValueError):
        _ = DATA_CONFIG.DataConfiguration(no_of_points_for_distance_calculation=0)
