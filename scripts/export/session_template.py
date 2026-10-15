"""Template GeoPackage for a session on one river, for ``session_geopackage.py``.

Etienne's session GeoPackage (14–15 Oct) covered the stretches Pam and Rik chose. For a
session without chosen locations (IJssel, Joost), this builds the same layers from the
scope and the VVR register:

- ``scoperegios_kruisingsjaar``: every scope area whose id starts with ``--prefix``;
- ``vvr_vlak``, ``signaleringslijn``: the VVR areas and signal lines along them;
- ``locaties``: one stretch per VVR area (the scope areas it touches), named after the
  site and the river-km of its scope areas, so the QGIS project gets a bookmark each;
- ``gemeten_oeverlijnen``: only the ids, as ``session_geopackage.py`` expects.

Run: ``PYTHONPATH=. uv run python scripts/export/session_template.py --run 20261014-stable
        --prefix ijssel --sessie IJssel --out <template.gpkg>``
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--prefix", required=True, help="scope id prefix, e.g. ijssel")
ap.add_argument(
    "--stretch",
    nargs=3,
    metavar=("NAME", "FIRST", "LAST"),
    help="one named stretch from scope id FIRST to LAST (both banks), instead of one per VVR area",
)
ap.add_argument("--sessie", required=True, help="session name for the bookmarks")
ap.add_argument("--waterlichaam", default=None)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
cfg = ExperimentConfig.from_run(args.run)

scope = (
    pyogrio.read_dataframe(cfg.raw_gpkg, layer="vlakken_scope")
    .rename(columns={"position_id": LOCATION_ID})
    .to_crs(28992)
)
scope = scope[scope[LOCATION_ID].str.startswith(args.prefix)][[LOCATION_ID, "geometry"]]
if args.stretch:
    _, first, last = args.stretch
    km = scope[LOCATION_ID].str.split("_").str[2].astype(int)
    scope = scope[km.between(int(first), int(last))]
scope["waterlichaam"] = args.waterlichaam or args.sessie
area = scope.union_all()
vvr = gpd.read_file(
    cfg.signalering_gpkg, layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl"
).to_crs(28992)
vvr = vvr[vvr.intersects(area)].reset_index(drop=True)
sig = gpd.read_file(cfg.signalering_gpkg, layer=cfg.signalering_layer).to_crs(28992)
sig = sig[sig.intersects(area.buffer(50))]

# One stretch per VVR area; a scope area touching two goes with the first.
hits = gpd.sjoin(scope, vvr[["objectnaam", "geometry"]], predicate="intersects")
hits = hits.sort_values("index_right").drop_duplicates(LOCATION_ID)
parts = hits[LOCATION_ID].str.split("_", expand=True)
hits["km0"] = parts[2].astype(int)
hits["km1"] = parts[3].astype(int)
# Scope ids count in units of 10 m along the river section (``ijssel1_l_2470_2480`` is
# km 24.7–24.8 of section ijssel1, left bank).
names = {}
for idx, g in hits.groupby("index_right"):
    section, bank = g[LOCATION_ID].iloc[0].split("_")[:2]
    km = f"km {g['km0'].min() / 100:.1f}–{g['km1'].max() / 100:.1f}"
    site = g["objectnaam"].iloc[0] or "VVR zonder naam"
    names[idx] = f"{site} ({section} {bank}, {km})"
if args.stretch:
    hits = scope.assign(index_right=0)
    names = {0: args.stretch[0]}
locaties = hits.assign(
    sessie=args.sessie,
    locatie=hits["index_right"].map(names),
    aangedragen="VVR-register",
)[[LOCATION_ID, "waterlichaam", "sessie", "locatie", "aangedragen", "geometry"]]

args.out.unlink(missing_ok=True)
locaties.to_file(args.out, layer="locaties")
vvr.to_file(args.out, layer="vvr_vlak")
sig.to_file(args.out, layer="signaleringslijn")
scope.to_file(args.out, layer="scoperegios_kruisingsjaar")
gpd.GeoDataFrame(
    {
        LOCATION_ID: scope[LOCATION_ID],
        "date": pd.Timestamp("2026-01-01"),
        "model": "",
        "jaar": 2026,
    },
    geometry=[None] * len(scope),
    crs=28992,
).to_file(args.out, layer="gemeten_oeverlijnen")
print(
    f"{len(scope):,} scope areas · {len(vvr)} VVR areas · {locaties['locatie'].nunique()} "
    f"stretches ({len(locaties)} scope areas) → {args.out}"
)
