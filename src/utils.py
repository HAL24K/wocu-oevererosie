"""Utilities that are not a part of any of the classes."""

import geopandas as gpd
import logging
import pyproj
import numpy as np
import re

from shapely.ops import transform
from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString, Point
from typing import Union

import src.constants as CONST


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def transform_shape_crs(
    from_epsg: Union[int, str],
    to_epsg: Union[int, str],
    input_shape: BaseGeometry,
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


def flatten_dictionary(input_dictionary: dict, key_separator: str = "_") -> dict:
    """Flatten a nested dictionary.

    :param input_dictionary: The dictionary to flatten.
    :param key_separator: The separator to use between the flattened keys.

    NOTE: In the feature creation, some features return dictionaries - e.g. the majority class can be calculated
       for multiple columns. This makes for annoying feature dataframes, hence we flatten the dictionary here to end
       up with single-element values. The keys of the new dictionary are a combination of the various level keys
       joined by key_separator.
    """

    output_dict = {}
    for key, value in input_dictionary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in flatten_dictionary(value).items():
                output_dict[key + key_separator + sub_key] = sub_value
        elif isinstance(value, list):
            raise NotImplementedError("Flattening of lists is not implemented.")
        else:
            output_dict[key] = value

    return output_dict


def get_object_density(base_shape: BaseGeometry, geo_data: gpd.GeoDataFrame) -> float:
    """Calculate the density of objects in the target shape.

    :param base_shape: The shape of the area for which we are calculating the density.
    :param geo_data: geospatial data with Points in geometry

    """
    if geo_data.empty:
        logger.warning(
            "The geodataframe is empty and thus the density cannot be calculated. It is assumed it would be 0."
        )
        return 0.0

    assert geo_data.geometry.geom_type.nunique() == 1, "Only one geometry type allowed"
    assert (
        geo_data.geometry.geom_type.unique()[0] == "Point"
    ), "Only Point geometry allowed"

    return len(geo_data[geo_data.within(base_shape)]) / base_shape.area


def get_count_object_intersects(
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
    if geo_data.empty:
        logger.warning(
            "The geodataframe is empty and thus the area fraction cannot be calculated. It is assumed it would be 0."
        )
        return 0.0

    return geo_data.union_all().intersection(base_shape).area / base_shape.area


def get_majority_class(
    base_shape: BaseGeometry,
    geo_data: gpd.GeoDataFrame,
    columns: Union[list[str], str],
) -> dict[str, str | None]:
    """Get the majority class from each required column in the geo_data.

    :param base_shape: The shape of the area for which we are calculating the majority class.
    :param geo_data: geospatial data
    :param columns: The columns to get the majority class from
    """
    if isinstance(columns, str):
        columns = [columns]

    if geo_data.empty:
        logger.warning(
            f"The geodataframe is empty and the majority class cannot be calculated for {columns}."
        )
        return {col: None for col in columns}

    assert set(geo_data.columns).issuperset(
        columns
    ), f"Columns {set(columns) - set(geo_data.columns)} not found in geo_data (it contains {geo_data.columns})."

    majority_classes = geo_data.loc[geo_data.intersects(base_shape), columns].mode()

    if len(majority_classes) == 0:
        # no data intersects the region
        # TODO: do we return None here or some "none string" so that we don't have missing features?
        return {col: None for col in columns}
    elif len(majority_classes) > 1:
        # TODO: improve the log, so that it tells which columns and which values were dropped.
        #   note: mode returns a new DF with multiple rows if at least one col has multiple modes; other cols have NaNs
        logger.warning(
            f"Multiple majority classes found in some of the columns, we only take the first one."
        )

    majority_classes = majority_classes.iloc[0].to_dict()
    return majority_classes


def get_nearby_linestring_shape(
    base_shape: BaseGeometry,
    line: LineString,
    neighbourhood_radius: float = CONST.DEFAULT_NEIGHBOURHOOD_RADIUS,
) -> float:
    """Get a measure of the shape of the linestring near another object.

    :param base_shape: The shape of the area that is next to the line.
    :param line: The line to calculate the shape of
    :param neighbourhood_radius: How far along the line to consider the neighbourhood shape

    NOTES: The main application here is to provide a measure of the shape of the river near the center of the prediction
       region. To account for the sides etc we:
       * find the point along the line (river centerline) closest to the center of the region
       * get the direction from that point to the center of the region (vector 1)
       * get the direction from that point to the point the neighbourhood_radius up- and downstream (vectors 2 and 3)
       * we calculate the cosine products of vecs 1 and 2 and vecs 1 and 3
       * we take the mean of the two
    # TODO: would it be better - and more elegant - to define this as a second derivative instead of what it is now?
    """
    base_shape_centroid = base_shape.centroid
    distance_along_line = line.project(base_shape_centroid)
    central_point_for_all_vectors = line.interpolate(distance_along_line)
    vector_end_downstream = line.interpolate(distance_along_line - neighbourhood_radius)
    vector_end_upstream = line.interpolate(distance_along_line + neighbourhood_radius)

    direction_to_centroid = np.array(base_shape_centroid.coords[0]) - np.array(
        central_point_for_all_vectors.coords[0]
    )
    direction_to_downstream = np.array(vector_end_downstream.coords[0]) - np.array(
        central_point_for_all_vectors.coords[0]
    )
    direction_to_upstream = np.array(vector_end_upstream.coords[0]) - np.array(
        central_point_for_all_vectors.coords[0]
    )

    cosine_similarity_downstream = cosine_similarity(
        direction_to_centroid, direction_to_downstream
    )
    cosine_similarity_upstream = cosine_similarity(
        direction_to_centroid, direction_to_upstream
    )

    return 0.5 * (cosine_similarity_downstream + cosine_similarity_upstream)


def cosine_similarity(vector1: np.array, vector2: np.array) -> float:
    """Calculate the cosine similarity between two vectors."""
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def get_relevant_centerline(
    base_shape: BaseGeometry, geo_data: gpd.GeoDataFrame
) -> LineString:
    """If the centerline is provided as a GeoDataFrame with several lines, get the closest one.

    :param base_shape: The shape of the area for which we are calculating the centerline.
    :param geo_data: geospatial data with LineStrings in geometry
    """
    assert geo_data.geometry.geom_type.nunique() == 1, (
        f"River centerline geodataframe must only have one geometery type, "
        f"yours has {geo_data.geometry.geom_type.nunique()}"
    )
    assert (
        geo_data.geometry.geom_type.unique()[0] == "LineString"
    ), f"The river centerline must be a LineString, yours is {geo_data.geometry.geom_type.unique()[0]}"

    # get the closest centerline
    closest_centerline = geo_data.loc[
        geo_data.distance(base_shape).idxmin(), "geometry"
    ]

    return closest_centerline


def is_point_between_two_lines(
    point: Point, first_line: LineString, second_line: LineString
) -> bool:
    """Check that a point lies between two lines that don't cross.

    :param point: the point in question
    :param first_line: the first line (e.g. the river centerline)
    :param second_line: the second line (e.g. the erosion border)
    :return: a boolean flag

    NOTE: this function projects the point onto the two lines and checks whether the angle projection1-point-projection2
       is acute (point does not lie between the lines) or not (point lies between the lines).
    TODO: check that the lines don't cross
    """

    first_projected_point = first_line.interpolate(first_line.project(point))
    second_projected_point = second_line.interpolate(second_line.project(point))

    first_vector = np.array(
        [first_projected_point.x - point.x, first_projected_point.y - point.y]
    )
    second_vector = np.array(
        [second_projected_point.x - point.x, second_projected_point.y - point.y]
    )

    return bool(np.dot(first_vector, second_vector) < 0)


def generate_shifted_column_name(
    original_column: str, shift: int, past_shift: bool = True
) -> str:
    """Define the name of the shifted column.

    :param original_column: the original column name
    :param shift: the number of time steps to shift by
    :param past_shift: whether the shift is in the past or future
    """
    return (
        f"{CONST.PREVIOUS if past_shift else CONST.UPCOMING}_{original_column}_{shift}"
    )
