"""Derive the lean structures GeoPackage from the BKN deliveries.

Input (as delivered by the GIS engineers, 2026-08-25; ``data/01_raw/structures``):
  - ``BKN_kribben/BKN_kribben.shp``            5,307 groyne polygons, nationwide
  - ``kunstwerk_vlakken/kunstwerk_vlakkenPolygon.shp``  20,630 structure polygons
    (culverts, jetties, locks, weirs, bridges, quay walls, ...), 40+ columns each

Output ``data/02_processed/structures/structures.gpkg`` with two layers, both
clipped to the scope (500 m) and reduced to the columns the pipeline and QGIS
labelling actually use:

  - ``kribben``      id, objectnaam, watersysteem, zijde, geobron, geometry
  - ``kunstwerken``  id, objecttype, categorie, watersysteem, zijde, geobron, geometry

``categorie`` groups the ~80 raw ``objecttype`` values into the handful that
matter for a bank mask (see CATEGORIES); the pipeline masks the categories in
``ExperimentConfig.kunstwerk_categories``. Everything near the scope is kept in
the file so the choice can be revisited in QGIS without going back to the shp.

Run: ``uv run python scripts/prep_structures.py``
"""

from __future__ import annotations

import geopandas as gpd
import shapely

import src.paths as PATHS

RAW = PATHS.DATA_DIR / "01_raw/structures"
OUT = PATHS.DATA_DIR / "02_processed/structures/structures.gpkg"
SCOPE = PATHS.DATA_DIR / "01_raw/scope/scope_fase2.gpkg"
SCOPE_BUFFER_M = 500.0

# objecttype substring (lower-case) → category. First match wins.
CATEGORIES = [
    ("brug", "brug"),
    ("viaduct", "brug"),
    ("kade", "kade_damwand"),
    ("damwand", "kade_damwand"),
    ("muur", "kade_damwand"),
    ("beschoeiing", "kade_damwand"),
    ("steiger", "steiger_afmeer"),
    ("meerstoel", "steiger_afmeer"),
    ("meerpaal", "steiger_afmeer"),
    ("afmeer", "steiger_afmeer"),
    ("ligplaats", "steiger_afmeer"),
    ("wachtplaats", "steiger_afmeer"),
    ("meerplaats", "steiger_afmeer"),
    ("dukdalf", "steiger_afmeer"),
    ("sluis", "sluis_stuw"),
    ("stuw", "sluis_stuw"),
    ("overlaat", "sluis_stuw"),
    ("kering", "sluis_stuw"),
    ("vistrap", "sluis_stuw"),
    ("vispassage", "sluis_stuw"),
    ("gemaal", "sluis_stuw"),
    ("duiker", "duiker"),
]


def categorise(objecttype: str) -> str:
    t = (objecttype or "").lower()
    for needle, cat in CATEGORIES:
        if needle in t:
            return cat
    return "overig"


def main() -> None:
    scope = gpd.read_file(SCOPE).to_crs(28992)
    near = shapely.union_all(scope.geometry.buffer(SCOPE_BUFFER_M).values)
    shapely.prepare(near)

    kribben = gpd.read_file(RAW / "BKN_kribben/BKN_kribben.shp").to_crs(28992)
    kribben = kribben[kribben.intersects(near)]
    kribben = gpd.GeoDataFrame(
        {
            "id": kribben["globalid"].values,
            "objectnaam": kribben["objectnaam"].values,
            "watersysteem": kribben["watersyste"].values,
            "zijde": kribben["zijde"].values,
            "geobron": kribben["geobron"].values,
        },
        geometry=shapely.force_2d(kribben.geometry.values),
        crs=28992,
    )

    kw = gpd.read_file(RAW / "kunstwerk_vlakken/kunstwerk_vlakkenPolygon.shp").to_crs(
        28992
    )
    kw = kw[kw.intersects(near)]
    kw = gpd.GeoDataFrame(
        {
            "id": kw["globalid"].values,
            "objecttype": kw["objecttype"].values,
            "categorie": [categorise(t) for t in kw["objecttype"].values],
            "watersysteem": kw["watersyste"].values,
            "zijde": kw["zijde"].values,
            "geobron": kw["geobron"].values,
        },
        geometry=shapely.force_2d(kw.geometry.values),
        crs=28992,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    kribben.to_file(OUT, layer="kribben", driver="GPKG")
    kw.to_file(OUT, layer="kunstwerken", driver="GPKG")

    print(
        f"kribben     : {len(kribben):>6}  ",
        kribben.watersysteem.value_counts().head(6).to_dict(),
    )
    print(f"kunstwerken : {len(kw):>6}  ", kw.categorie.value_counts().to_dict())
    print("→", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
