"""Loop batch 6 — track 2 phase 2: survey-level increments.

Batch 5's insight: year collapse caps pairwise training at +22% rows because
it destroys the sub-year richness first. These variants train on
survey-level increments (43k cleaned surveys), with span filters/weights to
keep noisy short-span targets from dominating, plus a Huber-objective probe
on the heavy-tailed target. Test rows stay the standard year-target t2→t3
increments throughout — same ruler as g1–g3.
"""

import logging
import warnings

import geopandas as gpd

from experiments.loop.harness.harness import load_caches
from experiments.loop.harness.multi_t import load_obs_e8, run_t2_variant
from src.cleaning.rules import make_structure_geom

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

caches = load_caches()
kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
obs = load_obs_e8(caches, make_structure_geom(kribs, 10.0))

VARIANTS: list[dict] = [  # noqa: C408 — kwargs-style rows read better here
    dict(  # noqa: C408
        name="h1-survey",
        training="survey",
        min_span=0.4,
        notes="consecutive survey increments, span >= 0.4 yr",
    ),
    dict(  # noqa: C408
        name="h2-survey-weighted",
        training="survey",
        min_span=0.4,
        weight=True,
        notes="h1 + span-proportional sample weight",
    ),
    dict(  # noqa: C408
        name="h3-survey-allpairs",
        training="survey",
        min_span=0.8,
        all_pairs=True,
        weight=True,
        notes="all forward pairs, span >= 0.8 yr, weighted",
    ),
    dict(  # noqa: C408
        name="h4-survey-huber",
        training="survey",
        min_span=0.4,
        weight=True,
        objective="huber",
        notes="h2 + huber objective",
    ),
    dict(  # noqa: C408
        name="h5-yearpair-huber",
        training="pairwise",
        objective="huber",
        notes="g3 + huber objective (isolate objective effect)",
    ),
]

for v in VARIANTS:
    v = dict(v)
    name = v.pop("name")
    notes = v.pop("notes")
    run_t2_variant(name, caches, obs, features="traj", target="year", notes=notes, **v)
