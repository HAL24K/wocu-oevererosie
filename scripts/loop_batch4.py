"""Loop batch 4 — detrended repair vs median repair; coverage recovery.

Batch 3 showed strict median-referenced repair (dev15, min_surveys 3) posts
the best numbers, but the synthetic test proved that mode clips genuine fast
eroders. If the detrended (Theil–Sen) mode reproduces the gains, they are
real artefact removal; if they vanish, batch 3 was partly erasing signal.

Second question: coverage. min_samples (600 regions) and fragment (200) are
now the big region killers; with artefact surveys handled upstream, both may
be relaxable.
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


def stack(dev=15.0, min_surveys=4, detrend=True, min_n=12, frag=0.3, iqr=20.0):
    return [
        ("structure_mask", {}),
        ("max_tortuosity_line", {"max_tort": 3.0}),
        ("near_bank_line", {"frac": 0.25, "min_ref": 30.0}),
        ("maze_survey", {"max_ratio": 1.8, "min_iqr": iqr}),
        ("fragment_survey", {"min_cov": frag}),
        ("min_samples_survey", {"min_n": min_n}),
        (
            "temporal_outlier_survey",
            {"max_dev": dev, "min_surveys": min_surveys, "detrend": detrend},
        ),
    ]


VARIANTS = [
    ("e1-detrend-dev20", stack(dev=20.0), "detrended repair, dev 20"),
    ("e2-detrend-dev15", stack(dev=15.0), "detrended repair, dev 15"),
    ("e3-detrend-dev15-ms3", stack(dev=15.0, min_surveys=3), "e2 + 3-survey regions"),
    ("e4-e2-minn8", stack(dev=15.0, min_n=8), "e2 + relaxed survival rule"),
    ("e5-e2-frag02", stack(dev=15.0, frag=0.2), "e2 + relaxed fragment"),
    (
        "e6-coverage-max",
        stack(dev=15.0, min_surveys=3, min_n=8, frag=0.25),
        "coverage kitchen sink",
    ),
    ("e7-detrend-dev10", stack(dev=10.0), "how strict can detrended go"),
]

for name, rules, note in VARIANTS:
    run_variant(name, caches, rules, structures=KRIB_GEOM, v_limit=50.0, notes=note)
