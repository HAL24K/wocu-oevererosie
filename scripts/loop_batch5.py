"""Loop batch 5 — track 2 phase 1: the three multi-t levers, factorial-lite.

g1 re-derives e8 through the track-2 code path (sanity: must match e8).
g2/g3 isolate trajectory features and pairwise training on the comparable
year target; g4–g6 repeat under the date-true target with its own paired
baseline (a new ruler — only g4 vs g5 vs g6 are comparable to each other).
"""

import logging
import warnings

import geopandas as gpd

from src.loop.harness import load_caches
from src.loop.multi_t import load_obs_e8, run_t2_variant
from src.cleaning.rules import make_structure_geom

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

caches = load_caches()
kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
obs = load_obs_e8(caches, make_structure_geom(kribs, 10.0))
print(f"e8 observations: {len(obs):,} surveys, {obs.location_id.nunique():,} regions")

VARIANTS = [
    ("g1-sanity-e8path", "base", "standard", "year", "must reproduce e8"),
    ("g2-trajfeats", "traj", "standard", "year", "Theil-Sen history features"),
    ("g3-pairwise", "traj", "pairwise", "year", "all increments as training rows"),
    ("g4-datetrue-base", "base", "standard", "date", "date-true target, baseline"),
    ("g5-datetrue-traj", "traj", "standard", "date", "date-true + traj features"),
    ("g6-datetrue-pairwise", "traj", "pairwise", "date", "everything on"),
]

for name, features, training, target, note in VARIANTS:
    run_t2_variant(
        name,
        caches,
        obs,
        features=features,
        training=training,
        target=target,
        notes=note,
    )
