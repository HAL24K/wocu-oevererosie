"""Loop batch 2 — compositions + parameter sensitivity.

Batch-1 marginals: temporal repair and maze dominate; fragment and
tortuosity help; multiline/near-bank are metric-neutral. This batch composes
them (line-level rules before survey-level, temporal repair last as the
safety net) and probes the two sensitive parameters, plus whether the legacy
region-level |v| exclusion is still needed once repair is in place.
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


def stack(maze=1.8, dev=30.0, lines=False, frag=True):
    rules = [("structure_mask", {})]
    if lines:
        rules += [
            ("max_tortuosity_line", {"max_tort": 3.0}),
            ("near_bank_line", {"frac": 0.25, "min_ref": 30.0}),
        ]
    rules += [("maze_survey", {"max_ratio": maze})]
    if frag:
        rules += [("fragment_survey", {"min_cov": 0.3})]
    rules += [
        ("min_samples_survey", {"min_n": 12}),
        ("temporal_outlier_survey", {"max_dev": dev, "min_surveys": 4}),
    ]
    return rules


VARIANTS = [
    ("c1-temporal-maze", stack(frag=False), 50.0, "temporal repair + maze"),
    ("c2-plus-fragment", stack(), 50.0, "c1 + fragment"),
    ("c3-plus-linerules", stack(lines=True), 50.0, "c2 + tortuosity/near-bank"),
    ("c2-maze2.2", stack(maze=2.2), 50.0, "softer maze for coverage"),
    ("c2-dev20", stack(dev=20.0), 50.0, "stricter temporal repair"),
    ("c2-dev50", stack(dev=50.0), 50.0, "looser temporal repair"),
    ("c2-noregionfilter", stack(), None, "repair only, no |v|>50 exclusion"),
]

for name, rules, v_limit, note in VARIANTS:
    run_variant(name, caches, rules, structures=KRIB_GEOM, v_limit=v_limit, notes=note)
