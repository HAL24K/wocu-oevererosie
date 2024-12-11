"""Test the utilites."""

import pytest

import src.utils as U


def test_transform_shape_crs(shapes_near_zaltbommel):
    """Test the coordinate transformation."""
    crs_wgs84 = 4326
    crs_rd = "28992"

    for shape_type in ["point", "line", "polygon"]:
        output_rd = U.trasnform_shape_crs(
            crs_wgs84, crs_rd, shapes_near_zaltbommel[f"{shape_type}_wgs84"]
        )
        output_wgs = U.trasnform_shape_crs(
            crs_rd, crs_wgs84, shapes_near_zaltbommel[f"{shape_type}_rd"]
        )

        assert output_rd.equals_exact(
            shapes_near_zaltbommel[f"{shape_type}_rd"], tolerance=1
        )  # tolerance of 1 meter
        assert output_wgs.equals_exact(
            shapes_near_zaltbommel[f"{shape_type}_wgs84"], tolerance=1e-6
        )
