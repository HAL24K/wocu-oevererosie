"""Loop batch 5b — resume the track-2 lever matrix from g3 (after the
dict(groupby) fix; g1/g2 completed and are in the ledger)."""

import logging
import warnings

import geopandas as gpd

from src.loop.harness import load_caches
from src.loop.multi_t import load_obs_e8, run_t2_variant
from src.loop.rules import make_structure_geom

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

caches = load_caches()
kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
obs = load_obs_e8(caches, make_structure_geom(kribs, 10.0))

VARIANTS = [
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
