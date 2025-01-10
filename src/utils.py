"""Utilities that are not a part of any of the classes."""

import geopandas as gpd
import logging
import pyproj
import re
from shapely.ops import transform
from shapely.geometry.base import BaseGeometry
from typing import Union

import src.constants as CONST


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def transform_shape_crs(
    from_epsg: Union[int, str], to_epsg: Union[int, str], input_shape: BaseGeometry
) -> BaseGeometry:
    """Transforms the shape from one CRS to another.

    Args:
        from_epsg (Union[int,str]): The EPSG code of the source CRS.
        to_epsg (Union[int, str]): The EPSG code of the destination CRS.
        input_shape (BaseGeometry): The shape to transform.

    Returns:
        BaseGeometry: The transformed shape.
    """
    project = pyproj.Transformer.from_proj(
        pyproj.CRS.from_epsg(int(from_epsg)),  # source coordinate system
        pyproj.CRS.from_epsg(int(to_epsg)),  # destination coordinate system
        always_xy=True,
    )

    return transform(project.transform, input_shape)  # apply projection


def get_epsg_from_urn(urn_string: str) -> str:
    """Extract the EPSG code from a URN string.

    Example of a URN string: 'urn:ogc:def:crs:EPSG::28992'
    """
    epsg_regex = re.compile(CONST.EPSG_REGEX)

    found_epsgs = re.findall(epsg_regex, urn_string)

    if found_epsgs:
        assert (
            len(found_epsgs) == 1
        ), f"More than one EPSG found in URN string {urn_string}: {found_epsgs}"

        return found_epsgs[0]

    else:
        logger.warning(f"No EPSG found in URN string: {urn_string}")
        return None


def get_object_density(base_shape: BaseGeometry, geo_data: gpd.GeoDataFrame) -> float:
    """Calculate the density of objects in the target shape.

    :param base_shape: The shape of the area for which we are calculating the density.
    :param geo_data: geospatial data with Points in geometry

    """
    assert geo_data.geometry.geom_type.nunique() == 1, "Only one geometry type allowed"
    assert (
        geo_data.geometry.geom_type.unique()[0] == "Point"
    ), "Only Point geometry allowed"

    return len(geo_data.within(base_shape)) / base_shape.area


def get_count_inside_object(
    base_shape: BaseGeometry, geo_data: gpd.GeoDataFrame
) -> int:
    """Get the count of objects inside the base_shape.

    :param base_shape: The shape of the area for which we are calculating the count.
    :param geo_data: geospatial data
    """
    return len(geo_data[geo_data.intersects(base_shape)])


def get_total_area(base_shape: BaseGeometry, geo_data: gpd.GeoDataFrame) -> float:
    """Get the total area of the objects inside the base_shape.

    :param base_shape: The shape of the area for which we are calculating the total area.
    :param geo_data: geospatial data
    """
    return geo_data[geo_data.intersects(base_shape)].area.sum()


def get_area_fraction(base_shape: BaseGeometry, geo_data: gpd.GeoDataFrame) -> float:
    """Get the fraction of the base_shape that is covered by the objects.

    :param base_shape: The shape of the area for which we are calculating the area fraction.
    :param geo_data: geospatial data
    """
    return geo_data[geo_data.intersects(base_shape)].area.sum() / base_shape.area


def get_majority_class(
    base_shape: BaseGeometry, geo_data: gpd.GeoDataFrame, columns: list[str]
) -> dict[str, str]:
    """Get the majority class from each required column in the geo_data.

    :param base_shape: The shape of the area for which we are calculating the majority class.
    :param geo_data: geospatial data
    :param columns: The columns to get the majority class from
    """
    assert set(geo_data.columns).issuperset(
        columns
    ), f"Columns {set(columns) - set(geo_data.columns)} not found in geo_data."

    majority_classes = {}
    for column in columns:
        majority_classes[column] = (
            geo_data[geo_data.intersects(base_shape)][column].mode().values[0]
        )

    return majority_classes
