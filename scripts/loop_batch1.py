"""Loop batch 1 — v0 parity + first cleaning-rule sweep.

v0 reproduces the 20260820-hybrid-masked recipe (kribben mask 10 m, >=12
samples per survey, |v|>50 region exclusion) through the fast harness. Its
job is parity: LGB test MAE should land on ~4.01 / R² ~0.33. It then freezes
CORE (surviving frozen-test regions) and the frozen tail so every later
variant reports fixed-denominator views.

The sweep after it: each round-2 family rule alone on top of v0, to measure
marginal effect before composing.
"""

import logging
import sys
import warnings

import geopandas as gpd

from src.loop.harness import freeze_core, load_caches, run_variant
from src.cleaning.rules import make_structure_geom

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

caches = load_caches()
cfg_gpkg = caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg"
kribs = gpd.read_file(cfg_gpkg, layer="Kribben_BKN").to_crs(28992)
KRIB_GEOM = make_structure_geom(kribs, 10.0)

BASE = [
    ("structure_mask", {}),
    ("min_samples_survey", {"min_n": 12}),
]

# ── v0: parity with 20260820-hybrid-masked ────────────────────────────────────
r0 = run_variant(
    "v0-baseline",
    caches,
    BASE,
    structures=KRIB_GEOM,
    v_limit=50.0,
    notes="parity with 20260820-hybrid-masked",
)
freeze_core(r0["preds"], caches)

if "--v0-only" in sys.argv:
    sys.exit(0)

# reload so CORE views apply to everything after v0
caches_post = load_caches()

# ── single-rule marginals on top of v0 ────────────────────────────────────────
VARIANTS = [
    (
        "v1-maze-survey",
        BASE + [("maze_survey", {"max_ratio": 1.8})],
        "drop surveys with line length > 1.8x region",
    ),
    (
        "v2-tortuosity",
        BASE + [("max_tortuosity_line", {"max_tort": 3.0})],
        "drop wandering lines (tort > 3)",
    ),
    (
        "v3-multiline",
        BASE + [("multiline_far_line", {"gap": 25.0, "ratio": 1.5})],
        "two-banks surveys: keep history-consistent group",
    ),
    (
        "v4-temporal",
        BASE + [("temporal_outlier_survey", {"max_dev": 30.0, "min_surveys": 4})],
        "drop surveys deviating > 30 m from region history",
    ),
    (
        "v5-fragment",
        BASE + [("fragment_survey", {"min_cov": 0.3})],
        "drop surveys covering < 30% of stations",
    ),
    (
        "v6-nearbank",
        BASE + [("near_bank_line", {"frac": 0.25, "min_ref": 30.0})],
        "drop mid-channel lines",
    ),
]

for name, rules, note in VARIANTS:
    run_variant(
        name, caches_post, rules, structures=KRIB_GEOM, v_limit=50.0, notes=note
    )
