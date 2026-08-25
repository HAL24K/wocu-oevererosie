"""Loop batch 7 — track 3 phase 1: R sweep, base vs traj2 segment features.

Questions: (1) how does segment-level predictability decay with R,
(2) does re-aggregating segment predictions match or beat the R=1 region
ruler (i1-traj2: core MAE 2.13, tail_frozen 3.62), (3) how much of the
region-level error is the aggregation operator itself (oracle floor).
"""

import logging
import warnings

import geopandas as gpd

from experiments.loop.harness.harness import load_caches
from experiments.loop.harness.multi_t import load_obs_e8
from experiments.loop.harness.resolution import load_dense, run_t3_variant
from src.cleaning.rules import make_structure_geom

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

caches = load_caches()
kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
obs = load_obs_e8(caches, make_structure_geom(kribs, 10.0))
dense = load_dense(caches)
print(f"dense: {len(dense):,} samples · {dense.location_id.nunique():,} regions")

VARIANTS = [
    ("j1-R2", 2, "base", "two scalars per region"),
    ("j2-R5", 5, "base", "five scalars per region"),
    ("j3-R10", 10, "base", "ten scalars per region"),
    ("j4-R2-traj2", 2, "traj2", "R2 + segment trajectory features"),
    ("j5-R5-traj2", 5, "traj2", "R5 + segment trajectory features"),
    ("j6-R10-traj2", 10, "traj2", "R10 + segment trajectory features"),
]

for name, R, features, note in VARIANTS:
    run_t3_variant(name, caches, dense, obs, R=R, features=features, notes=note)
