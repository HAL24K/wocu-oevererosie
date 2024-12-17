"""Tests for the DataCollector class.

TODO: explicitly test various WFS versions whether the API is the same
"""

import geopandas as gpd
import pytest

import src.data.data_collector as DC
import src.constants as CONST


@pytest.fixture
def data_collector_around_fake_eroded_bank(fake_eroded_bank_wgs84):
    return DC.DataCollector(
        source_shape=fake_eroded_bank_wgs84,
        source_epsg_crs=CONST.EPSG_WGS84,
        buffer_in_metres=1_000,  # metres, known to contain data
    )


def test_connect(data_collector_around_fake_eroded_bank):

    assert data_collector_around_fake_eroded_bank.source_shape_raw.within(
        data_collector_around_fake_eroded_bank.source_shape
    )

    for wfs_service in data_collector_around_fake_eroded_bank.wfs_services_raw:
        known_layer_names = list(
            data_collector_around_fake_eroded_bank.wfs_services[
                wfs_service.name
            ].contents.keys()
        )

        assert known_layer_names  # make sure that we can connect somewhere

        # make sure that the layers we care about actually exist
        assert set(wfs_service.relevant_layers).issubset(known_layer_names)


def test_wfs_geodata(data_collector_around_fake_eroded_bank):
    # make sure that we can get some data
    for wfs_service in data_collector_around_fake_eroded_bank.wfs_services_raw:
        geospatial_data = (
            data_collector_around_fake_eroded_bank.load_data_from_single_wfs(
                wfs_service.name
            )
        )
        assert isinstance(geospatial_data, gpd.GeoDataFrame)

        # TODO: this is a weird NL-centric assert
        # also, comparing the .crs object (complicated) with just a string works for some reason?!
        assert geospatial_data.crs == CONST.EPSG_RD
