"""Making sure that the important locations exist"""

import src.paths as PATHS


def test_important_locations_exist():
    assert PATHS.SRC_DIR.exists()
    assert PATHS.TEST_DIR.exists()
