"""Tests for the DataCollector class.

TODO: explicitly test various WFS versions whether the API is the same
"""

import geopandas as gpd
import pytest

import src.data.data_collector as DC
import src.constants as CONST
import src.config as CONFIG


@pytest.fixture
def data_collector_around_fake_eroded_bank(fake_eroded_bank_wgs84):
    return DC.DataCollector(
        source_shape=fake_eroded_bank_wgs84,
        source_epsg_crs=CONST.EPSG_WGS84,
        buffer_in_metres=1_000,  # metres, known to contain data
    )


def test_connect(data_collector_around_fake_eroded_bank):
    # check that the buffer makes the shape bigger
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
        # TODO: this calls an actual webservice, consider mocking it
        geospatial_data = (
            data_collector_around_fake_eroded_bank.load_data_from_single_wfs(
                wfs_service.name
            )
        )
        assert isinstance(geospatial_data, dict)

        for layer in geospatial_data:
            assert isinstance(geospatial_data[layer], gpd.GeoDataFrame)

            # TODO: this is a weird NL-centric assert
            # also, comparing the .crs object (complicated) with just a string works for some reason?!
            assert geospatial_data[layer].crs == CONST.EPSG_RD


def test_getting_all_wfs_data(fake_eroded_bank_wgs84):
    # this is almost the same data collector as the one from the fixture, but this one knows fewer services
    abridged_known_wfs_services = CONFIG.KNOWN_WFS_SERVICES[:1]
    data_collector_single_wfs = DC.DataCollector(
        source_shape=fake_eroded_bank_wgs84,
        source_epsg_crs=CONST.EPSG_WGS84,
        buffer_in_metres=1_000,
        wfs_services=abridged_known_wfs_services,  # use only one service
    )

    # make sure we have no data ahead of time
    assert not data_collector_single_wfs.relevant_geospatial_data

    data_collector_single_wfs.get_data_from_all_wfs()

    # check the structure of the data, which is a nested dictionary
    assert isinstance(data_collector_single_wfs.relevant_geospatial_data, dict)
    assert (
        len(data_collector_single_wfs.relevant_geospatial_data) == 1
    )  # we only use 1 WFS in this test

    known_wfs_name = data_collector_single_wfs.wfs_services_raw[0].name
    wfs_data_structure = data_collector_single_wfs.relevant_geospatial_data[
        known_wfs_name
    ]
    assert isinstance(wfs_data_structure, dict)  # the dictionary is nested

    assert len(wfs_data_structure) == len(
        abridged_known_wfs_services[0].relevant_layers
    )

    layer_name = list(wfs_data_structure.keys())[0]
    assert isinstance(wfs_data_structure[layer_name], gpd.GeoDataFrame)
