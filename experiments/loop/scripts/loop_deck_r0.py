"""Deck: the new structures layer alone on the 19-aug (v0) recipe → r0-newstructures."""

import logging
import warnings

import geopandas as gpd
import shapely

from experiments.loop.harness.harness import load_caches, run_variant

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
caches = load_caches()
data = caches.cache_dir.parents[1]
sg = data / "02_processed/structures/structures.gpkg"
kribs = gpd.read_file(sg, layer="kribben").to_crs(28992)
kw = gpd.read_file(sg, layer="kunstwerken").to_crs(28992)
kw = kw[kw["categorie"].isin(["brug", "kade_damwand", "steiger_afmeer", "sluis_stuw"])]
NEW = shapely.union_all(
    [shapely.union_all(kribs.geometry.buffer(10.0).values), shapely.union_all(kw.geometry.buffer(10.0).values)]
)
shapely.prepare(NEW)
run_variant(
    "r0-newstructures",
    caches,
    [("structure_mask", {}), ("min_samples_survey", {"min_n": 12})],
    structures=NEW,
    notes="deck: v0 rules with new kribben + kunstwerken mask",
)
