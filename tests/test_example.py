"""A stupid example of a unit test."""

import pytest


def test_addition():
    assert 1 + 1 == 2

    with pytest.raises(TypeError):
        1 + "1"
