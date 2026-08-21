"""Loop batch 3 — recover coverage with the IQR-refined maze rule.

Batch 2's best (c2-dev20) sits at 0.878 CORE coverage, below the 90% floor;
the eye-check attributed the loss to the maze rule killing legitimate surveys
with duplicated on-bank geometry. This batch sweeps the new min_iqr condition,
probes the temporal-repair strictness around dev=20, extends repair to
3-survey regions, and re-tests dropping the legacy region filter on the best
recipe.
"""

import logging
import warnings

import geopandas as gpd

from src.loop.harness import load_caches, run_variant
from src.loop.rules import make_structure_geom

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

caches = load_caches()
kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
KRIB_GEOM = make_structure_geom(kribs, 10.0)


def stack(iqr=10.0, dev=20.0, min_surveys=4):
    return [
        ("structure_mask", {}),
        ("max_tortuosity_line", {"max_tort": 3.0}),
        ("near_bank_line", {"frac": 0.25, "min_ref": 30.0}),
        ("maze_survey", {"max_ratio": 1.8, "min_iqr": iqr}),
        ("fragment_survey", {"min_cov": 0.3}),
        ("min_samples_survey", {"min_n": 12}),
        ("temporal_outlier_survey", {"max_dev": dev, "min_surveys": min_surveys}),
    ]


VARIANTS = [
    ("d1-iqr5", stack(iqr=5.0), 50.0, "maze needs IQR > 5 m"),
    ("d2-iqr10", stack(iqr=10.0), 50.0, "maze needs IQR > 10 m"),
    ("d3-iqr20", stack(iqr=20.0), 50.0, "maze needs IQR > 20 m"),
    ("d4-iqr10-dev15", stack(iqr=10.0, dev=15.0), 50.0, "stricter repair"),
    ("d5-iqr10-dev30", stack(iqr=10.0, dev=30.0), 50.0, "looser repair"),
    ("d6-minsurveys3", stack(min_surveys=3), 50.0, "repair 3-survey regions too"),
    ("d7-noregionfilter", stack(), None, "best recipe, no |v| exclusion"),
]

for name, rules, v_limit, note in VARIANTS:
    run_variant(name, caches, rules, structures=KRIB_GEOM, v_limit=v_limit, notes=note)
