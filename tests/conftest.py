"""Tools for all the tests."""

import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import pytest

import src.paths as PATHS


def point_near_zaltbommel_wgs84():
    # WTF, copilot suggested these values on its own, and they look correct...
    return Point(5.243, 51.812)


def point_near_zaltbommel_rd():
    # transformed using https://epsg.io/transform#s_srs=4326&t_srs=28992&x=5.2430000&y=51.8120000
    return Point(145055.84668598988, 424829.3675167101)


def line_near_zaltbommel_wgs84():
    # slightly shuffled values of the point above
    return LineString([(5.243, 51.812), (5.253, 51.802)])


def line_near_zaltbommel_rd():
    return LineString(
        [
            (145055.84668598988, 424829.3675167101),
            (145743.39332654528, 423715.50267567066),
        ]
    )


def polygon_near_zaltbommel_wgs84():
    return Polygon([(5.243, 51.812), (5.253, 51.802), (5.253, 51.812)])


def polygon_near_zaltbommel_rd():
    return Polygon(
        [
            (145055.84668598988, 424829.3675167101),
            (145743.39332654528, 423715.50267567066),
            (145745.4437876893, 424828.0491105658),
        ]
    )


@pytest.fixture
def shapes_near_zaltbommel():
    # using fixtures in pytest.mark.parametrize is a bit tricky, so here is a workaround
    # TODO: make more elegant
    return {
        "point_wgs84": point_near_zaltbommel_wgs84(),
        "point_rd": point_near_zaltbommel_rd(),
        "line_wgs84": line_near_zaltbommel_wgs84(),
        "line_rd": line_near_zaltbommel_rd(),
        "polygon_wgs84": polygon_near_zaltbommel_wgs84(),
        "polygon_rd": polygon_near_zaltbommel_rd(),
    }


@pytest.fixture
def fake_eroded_bank_wgs84():
    return Polygon(
        [(5.4639, 51.8875), (5.4664, 51.8879), (5.4665, 51.8872), (5.4634, 51.8869)]
    )


@pytest.fixture
def sample_assets():
    assets = {}
    assets["sample_assets_rd"] = gpd.read_file(
        PATHS.TEST_DIR / "assets" / "sample_data_rd.geojson"
    )
    assets["sample_assets_wgs84"] = gpd.read_file(
        PATHS.TEST_DIR / "assets" / "sample_data_wgs84.geojson"
    )

    return assets
