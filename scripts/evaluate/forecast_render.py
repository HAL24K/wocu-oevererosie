"""Draw a strip forecast on the session stretches, for checking by eye.

Takes a forecast table (``location_id, seg, last_year, start, v``, optional ``cap`` and
``v2``, from ``forecast_lab``: position = start + v * min(dt, cap) + v2 * dt)
and the session GeoPackage of the run, swaps in predicted lines from that forecast
(2026–2031, gaps where a strip has no data), and draws every session stretch with QGIS
through ``scripts/export/session_project_v2.py`` into ``<out>/``.

Run: ``PYTHONPATH=. uv run python scripts/evaluate/forecast_render.py --run 20261013-preclean
        --forecast <fc.parquet> --out <dir>``
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.analysis.predicted_lines import region_lines
from src.analysis.region_inspector import RegionInspector
from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
SESSION = Path("data/05_deliverables/sessie_20261014/van_sharepoint")
QGIS_PY = "/Applications/QGIS-final-4_2_3.app/Contents/MacOS/python"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--forecast", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--years", nargs=2, type=int, default=[2026, 2031])
ap.add_argument("--closeup", nargs="*", default=[], help="extra name=id1,id2 groups")
args = ap.parse_args()
cfg = ExperimentConfig.from_run(args.run)
R = cfg.segment_R

src_gpkg = SESSION / f"wocu_sessies_14_15okt_{args.run}.gpkg"
args.out.mkdir(parents=True, exist_ok=True)
work = args.out / "_qgis"
work.mkdir(exist_ok=True)
gpkg = work / "sessie.gpkg"
shutil.copy(src_gpkg, gpkg)
shutil.copy(
    SESSION / "Presentatie_Oktober2026.qgz", work / "Presentatie_Oktober2026.qgz"
)

regions = set(gpd.read_file(gpkg, layer="scoperegios_kruisingsjaar")[LOCATION_ID])
fc = pd.read_parquet(args.forecast)
fc = fc[fc[LOCATION_ID].isin(regions)].copy()
for c, default in (("cap", float("inf")), ("v2", 0.0)):
    if c not in fc:
        fc[c] = default
fc["cap"] = fc["cap"].fillna(float("inf"))
fc["v2"] = fc["v2"].fillna(0.0)
ri = RegionInspector(hybrid_gpkg=cfg.hybrid_gpkg)
samples = pd.read_parquet(
    cfg.features_dir / "samples.parquet", columns=[LOCATION_ID, "x", "y"]
)
side = (
    samples[samples[LOCATION_ID].isin(regions)].groupby(LOCATION_ID)[["x", "y"]].mean()
)

rows = []
for loc, g in fc.groupby(LOCATION_ID):
    cl = ri.geometry.centrelines.get(loc)
    if cl is None or loc not in side.index:
        continue
    for year in range(args.years[0], args.years[1] + 1):
        dt = (year - g["last_year"]).clip(lower=0)
        pos = g["start"] + g["v"] * dt.clip(upper=g["cap"]) + g["v2"] * dt
        # region_lines draws last_dist_m + v_pred * (year - last_year): give it the
        # position directly with a zero speed.
        gg = g.assign(last_dist_m=pos, v_pred=0.0)
        geom = region_lines(gg, cl, tuple(side.loc[loc]), R, year)
        for part in geom.geoms:
            rows.append({LOCATION_ID: loc, "year": year, "geometry": part})
pred = gpd.GeoDataFrame(rows, crs=28992)
pred.to_file(gpkg, layer="voorspelde_oeverlijnen", driver="GPKG")
print(f"predicted line pieces: {len(pred):,}")

env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen")
subprocess.run(
    [
        QGIS_PY,
        "scripts/export/session_project_v2.py",
        str(work),
        "--data",
        gpkg.name,
        "--render",
        str(args.out),
    ],
    env=env,
    check=True,
    capture_output=True,
)
# Close-ups of the spots that went wrong on 13 Oct, plus a few calm ones.
CLOSEUPS = [
    "zuilichem_oost=rijn_l_8350_8360,rijn_l_8360_8370,rijn_l_8370_8380,rijn_l_8380_8390",
    "zuilichem_west=rijn_l_8500_8510,rijn_l_8510_8520,rijn_l_8520_8530,rijn_l_8530_8540",
    "druten=rijn_r_4460_4470,rijn_r_4470_4480,rijn_r_4480_4490",
    "elst=nederrijn_r_4740_4750,nederrijn_r_4750_4760,nederrijn_r_4760_4770,nederrijn_r_4770_4780",
    "wageningen_zuid=nederrijn_l_3310_3320,nederrijn_l_3320_3330,nederrijn_l_3330_3340",
]
subprocess.run(
    [
        QGIS_PY,
        "scripts/evaluate/qgis_render_regions.py",
        str(work / "Presentatie_Oktober2026_v2.qgz"),
        str(args.out / "closeup"),
        *CLOSEUPS,
        *args.closeup,
    ],
    env=env,
    check=True,
    capture_output=True,
)
print(f"→ {args.out}")
