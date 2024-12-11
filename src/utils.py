"""Utilities that are not a part of any of the classes."""

import pyproj
from shapely.ops import transform
from shapely.geometry.base import BaseGeometry
from typing import Union


def trasnform_shape_crs(
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
