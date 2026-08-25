"""Deck experiment A — structures ablation on the e8 recipe (water mask OFF).

(a) no structure mask · (b) old kribben (Waal/Nederrijn) · (c) new kribben +
kunstwerken (brug, kade_damwand, steiger_afmeer, sluis_stuw) · (d) v0
baseline rules without any mask.
"""

import logging
import warnings

import geopandas as gpd
import shapely

from experiments.loop.harness.harness import load_caches, run_variant
from experiments.loop.harness.multi_t import E8_RULES
from src.cleaning.rules import make_structure_geom

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

caches = load_caches()
data = caches.cache_dir.parents[1]
BUF = 10.0

old = gpd.read_file(
    data / "01_raw/scope/Levering_erosie_data.gpkg", layer="Kribben_BKN"
).to_crs(28992)
OLD = make_structure_geom(old, BUF)

sg = data / "02_processed/structures/structures.gpkg"
kribs = gpd.read_file(sg, layer="kribben").to_crs(28992)
kw = gpd.read_file(sg, layer="kunstwerken").to_crs(28992)
kw = kw[kw["categorie"].isin(["brug", "kade_damwand", "steiger_afmeer", "sluis_stuw"])]
NEW = shapely.union_all(
    [
        shapely.union_all(kribs.geometry.buffer(BUF).values),
        shapely.union_all(kw.geometry.buffer(BUF).values),
    ]
)
shapely.prepare(NEW)
print(f"old kribben {len(old)} · new kribben {len(kribs)} · kunstwerken {len(kw)}")

E8_NOMASK = [r for r in E8_RULES if r[0] != "structure_mask"]
V0_NOMASK = [("min_samples_survey", {"min_n": 12})]

run_variant(
    "s0-v0-nomask",
    caches,
    V0_NOMASK,
    structures=None,
    notes="deck: v0 rules, no structure mask",
)
run_variant(
    "s1-e8-nomask",
    caches,
    E8_NOMASK,
    structures=None,
    notes="deck: e8, no structure mask",
)
run_variant(
    "s2-e8-oldkribben",
    caches,
    E8_RULES,
    structures=OLD,
    notes="deck: e8, old kribben Waal/Nederrijn (=e8-final-protected)",
)
run_variant(
    "s3-e8-newstructures",
    caches,
    E8_RULES,
    structures=NEW,
    notes="deck: e8, new kribben + kunstwerken (brug/kade/steiger/sluis)",
)
