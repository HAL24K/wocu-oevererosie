"""Utilities that are not a part of any of the classes."""

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
