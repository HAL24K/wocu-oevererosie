"""Tests for the DataCollector class."""

import pytest

import src.data.data_collector as DC
import src.constants as CONST


def test_connect(fake_eroded_bank_wgs84):
    data_collector = DC.DataCollector(
        source_shape=fake_eroded_bank_wgs84,
        source_epsg_crs=CONST.EPSG_WGS84,
        buffer_in_metres=100,  # metres
    )

    assert data_collector.source_shape_raw.within(data_collector.source_shape)

    # wfs_land = data_collector.wfs_services[CONST.WFS_LAND_USE]
    #
    # layer_name = list(wfs_land.contents.keys())[0]
    # wfs_land.getfeature([layer_name])
    #
    # pass
