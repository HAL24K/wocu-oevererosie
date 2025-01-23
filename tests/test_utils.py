"""Test the utilites."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, LineString

import src.utils as U


def test_transform_shape_crs(shapes_near_zaltbommel):
    """Test the coordinate transformation."""
    crs_wgs84 = 4326
    crs_rd = "28992"

    for shape_type in ["point", "line", "polygon"]:
        output_rd = U.transform_shape_crs(
            crs_wgs84, crs_rd, shapes_near_zaltbommel[f"{shape_type}_wgs84"]
        )
        output_wgs = U.transform_shape_crs(
            crs_rd, crs_wgs84, shapes_near_zaltbommel[f"{shape_type}_rd"]
        )

        assert output_rd.equals_exact(
            shapes_near_zaltbommel[f"{shape_type}_rd"], tolerance=1
        )  # tolerance of 1 meter
        assert output_wgs.equals_exact(
            shapes_near_zaltbommel[f"{shape_type}_wgs84"], tolerance=1e-6
        )


def test_get_epsg_from_urn():
    """Test the extraction of EPSG from URN."""
    # Case 1: URN strings with EPSG as expected
    urn = "urn:ogc:def:crs:EPSG::28992"
    assert U.get_epsg_from_urn(urn) == "28992"

    urn = "urn:ogc:def:crs:EPSG::4326"
    assert U.get_epsg_from_urn(urn) == "4326"

    # Case 2: a string with no EPSG
    urn = "no_epsg_here"
    assert U.get_epsg_from_urn(urn) is None

    # Case 3: a string with multiple EPSGs, which is forbidden
    urn = "urn:ogc:def:crs:EPSG::28992,urn:ogc:def:crs:EPSG::4326"
    with pytest.raises(AssertionError) as e:
        U.get_epsg_from_urn(urn)

    assert "More than one" in str(e.value)


def test_get_object_density(fake_eroded_bank_rd, sample_assets):
    """Check that the calculated object density is right."""
    with pytest.raises(AssertionError):
        # expecting this to fail as the raw sample assets are polygons
        U.get_object_density(fake_eroded_bank_rd, sample_assets["sample_assets_rd"])

    # make points out of the polygons, use the RC coords
    sample_assets_as_points = sample_assets["sample_assets_rd"].copy()
    sample_assets_as_points["geometry"] = sample_assets_as_points["geometry"].centroid

    # TODO: turns out the sample data does not overlap/itersect much with the sample polygon, so we beef it up a bit
    #   so that it works. This could be fixed and make less awkward by using better test data.
    chubby_fake_eroded_bank = fake_eroded_bank_rd.buffer(500)

    object_density = U.get_object_density(
        chubby_fake_eroded_bank, sample_assets_as_points
    )

    assert object_density > 0  # make sure that SOME objects are inside the shape

    # only include the points inside the source shape - our sample data is from a larger area
    assert object_density < len(sample_assets_as_points) / chubby_fake_eroded_bank.area


def test_get_object_intersects(fake_eroded_bank_rd, sample_assets):
    """Check that we correctly get the intersected objects."""
    no_of_true_intersects = U.get_count_object_intersects(
        fake_eroded_bank_rd, sample_assets["sample_assets_rd"]
    )

    assert no_of_true_intersects > 0  # make sure that SOME objects are inside the shape

    huge_buffer = 10_000  # get a buffer that for sure contains all the shapes (calculated with a buffer of 5_000)
    no_of_all_objects = U.get_count_object_intersects(
        fake_eroded_bank_rd.buffer(huge_buffer), sample_assets["sample_assets_rd"]
    )

    assert no_of_all_objects == len(
        sample_assets["sample_assets_rd"]
    )  # make sure that the buffer is huge enough

    assert no_of_true_intersects < no_of_all_objects


def test_get_total_area(fake_eroded_bank_rd, sample_assets):
    """Check that we correctly get the total area of the intersected objects."""
    total_area = U.get_total_area(
        fake_eroded_bank_rd, sample_assets["sample_assets_rd"]
    )

    assert total_area > 0  # make sure that SOME objects are inside the shape

    huge_buffer = 10_000  # get a buffer that for sure contains all the shapes (calculated with a buffer of 5_000)
    total_area_all_objects = U.get_total_area(
        fake_eroded_bank_rd.buffer(huge_buffer), sample_assets["sample_assets_rd"]
    )

    assert (
        total_area_all_objects == sample_assets["sample_assets_rd"].area.sum()
    )  # make sure that the buffer is huge enough

    assert total_area < total_area_all_objects


def test_get_area_fraction(fake_eroded_bank_rd, sample_assets):
    """Check that we correctly get the area fraction of the intersected objects."""
    area_fraction = U.get_area_fraction(
        fake_eroded_bank_rd, sample_assets["sample_assets_rd"]
    )

    assert 0 < area_fraction < 1  # make sure that SOME objects are inside the shape

    huge_buffer = 10_000  # get a buffer that for sure contains all the shapes (calculated with a buffer of 5_000)
    area_fraction_all_objects = U.get_area_fraction(
        fake_eroded_bank_rd, sample_assets["sample_assets_rd"].buffer(huge_buffer)
    )

    assert area_fraction_all_objects == pytest.approx(
        1
    )  # make sure that the buffer is huge enough

    assert area_fraction < area_fraction_all_objects


def test_get_majority_class(fake_eroded_bank_rd, sample_assets):
    """Check that we get the majority class right."""
    columns_to_use = ["category", "gewas"]

    with pytest.raises(AssertionError):
        # if we include some columns that don't exist, we want the thing to fail
        columns_to_make_it_fail = columns_to_use + ["non_existent_column"]
        U.get_majority_class(
            fake_eroded_bank_rd,
            sample_assets["sample_assets_rd"],
            columns_to_make_it_fail,
        )

    majority_classes_simple = U.get_majority_class(
        fake_eroded_bank_rd, sample_assets["sample_assets_rd"], columns_to_use
    )

    assert len(majority_classes_simple) == len(columns_to_use)

    huge_buffer = 10_000  # get a buffer that for sure contains all the shapes (calculated with a buffer of 5_000)
    majority_classes_all_objects = U.get_majority_class(
        fake_eroded_bank_rd.buffer(huge_buffer),
        sample_assets["sample_assets_rd"],
        columns_to_use,
    )

    assert majority_classes_all_objects != majority_classes_simple

    # test that it also works for only one column
    _ = U.get_majority_class(
        fake_eroded_bank_rd, sample_assets["sample_assets_rd"], columns_to_use[0]
    )


def test_flatten_dictionary():
    """Test that dictionaries flatten well."""
    # CASE 1: a simple dictionary, no change
    example_dictionary = {"a": 1, "b": 2, "c": True}
    flattened_dictionary = U.flatten_dictionary(example_dictionary)
    assert flattened_dictionary == example_dictionary

    # CASE 2: an empty dictionary, no change
    example_dictionary = {}
    flattened_dictionary = U.flatten_dictionary(example_dictionary)
    assert flattened_dictionary == example_dictionary

    # CASE 3: an easily nested dictionary
    example_dictionary = {"a": False, "b": {"c": 2, "d": 3}}
    flattened_dictionary = U.flatten_dictionary(example_dictionary)
    assert flattened_dictionary == {"a": False, "b_c": 2, "b_d": 3}

    # CASE 4: multiple levels of nesting
    example_dictionary = {"a": 1, "b": {"c": True, "d": {"e": 3, "f": 4}}}
    flattened_dictionary = U.flatten_dictionary(example_dictionary)
    assert flattened_dictionary == {"a": 1, "b_c": True, "b_d_e": 3, "b_d_f": 4}

    # CASE 5: a list makes it fall over
    example_dictionary = {"a": 1, "b": [1, 2, 3]}
    with pytest.raises(NotImplementedError):
        U.flatten_dictionary(example_dictionary)

    # CASE 6: different separator
    example_dictionary = {"mac": {"cheese": 42}}
    flattened_dictionary = U.flatten_dictionary(example_dictionary, key_separator="&")
    assert flattened_dictionary == {"mac&cheese": 42}


def test_cosine_similarity():
    """Test that cosine similarity works"""
    number_of_random_trials = 42  # more than 1 but not too many

    for _ in range(number_of_random_trials):
        multiplier1 = np.random.randint(1, 1000)  # enable coordinates to be big
        multiplier2 = np.random.randint(1, 1000)
        number_of_dimensions = np.random.randint(1, 42)

        vector1 = np.random.rand(number_of_dimensions) * multiplier1
        vector2 = np.random.rand(number_of_dimensions) * multiplier2

        assert U.cosine_similarity(vector1, vector1) == pytest.approx(1, abs=1e-6)
        assert U.cosine_similarity(vector2, vector2) == pytest.approx(1, abs=1e-6)
        assert U.cosine_similarity(vector1, vector2) == U.cosine_similarity(
            vector2, vector1
        )
        assert U.cosine_similarity(vector1, vector2) <= 1

    example_vector = np.array([1, 0, 0])
    orthogonal_vector = np.array([0, 1, 0])

    assert U.cosine_similarity(example_vector, orthogonal_vector) == pytest.approx(
        0, abs=1e-6
    )


def test_get_nearby_line_shape():
    """Test that we get the nearby line shape correctly."""
    # a square whose centroid lies on the X axis
    example_shape = Polygon([(0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5)])

    # a vertical line
    vertical_line = LineString([(0.0, -1.0), (0.0, 1.0)])

    for example_radius in np.linspace(0.1, 0.9, 5):
        line_shape_metric = U.get_nearby_linestring_shape(
            base_shape=example_shape,
            line=vertical_line,
            neighbourhood_radius=example_radius,
        )
        assert line_shape_metric == 0.0  # straight line


def test_get_relevant_centerline():
    """Test that we get the closest line"""
    top_line = LineString([(-1, 1), (1, 1)])
    bottom_line = LineString([(-1, -1), (1, -1)])
    shape_closer_to_top_line = Polygon([(-1, 2), (1, 2), (0, 3)])

    example_centerlines = gpd.GeoDataFrame(
        [{"geometry": bottom_line}, {"geometry": top_line}]
    )

    closer_line = U.get_relevant_centerline(
        shape_closer_to_top_line, example_centerlines
    )

    assert closer_line == top_line
