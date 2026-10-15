"""Pictures for the "how the warning light works" slide, built up in five steps.

Per example region: (1) the scope region on the aerial photo, (2) + the VVR with its
signal line, (3) + the measured bank lines (latest thick yellow, older faint),
(4) + the predicted bank line, (5) + the region's colour for its crossing class.
Reads a session GeoPackage (``session_geopackage.py``); writes
``<out>/<key>_<step>.jpg``.

Run: ``PYTHONPATH=. uv run python scripts/slides/warning_light_steps.py
        --gpkg <session.gpkg> --out <dir>``
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
# 2024: the current PDOK photo shows high water along the Maas and Neder-Rijn
AERIAL = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/2024_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"
PURPLE, YELLOW, BLUE = "#8c0fd5", "#ffff3c", "#1f6fe0"
CLASS_COLOUR = {
    "reeds gekruist (2026)": "#8b0000",
    "2027": "#d7301f",
    "2028-2029": "#ef6548",
    "2030-2031": "#fdbb47",
    "2032-2035": "#fee08b",
    "2036-2040": "#a6d96a",
    "2041-2050": "#4daf4a",
    "2036-2042": "#a6d96a",  # classes since 15 Oct (programmeerlijn 2042)
    "2043-2050": "#4daf4a",
    "geen kruising voor 2050": "#1a7837",
}

ap = argparse.ArgumentParser()
ap.add_argument("--gpkg", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--margin", type=float, default=18.0, help="metres around the bank")
ap.add_argument("--only-steps", nargs="*", type=int, default=[1, 2, 3, 4, 5])
ap.add_argument(
    "--examples",
    nargs="*",
    default=[
        "kruist=maas3_l_8840_8850",
        "binnen=maas3_l_8800_8810",
        "voorbij=maas3_l_8820_8830",
    ],
    help="key=region or key=region@year (year of the predicted line to draw)",
)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
L = lambda n: gpd.read_file(args.gpkg, layer=n).to_crs(28992)  # noqa: E731
regions, vvr, measured, pred = (
    L("scoperegios_kruisingsjaar"),
    L("vvr_vlak"),
    L("gemeten_oeverlijnen"),
    L("voorspelde_oeverlijnen"),
)
measured = measured[measured["gebruikt"] == 1]

for item in args.examples:
    key, loc = item.split("=")
    loc, _, year_override = loc.partition("@")  # key=region@2031 draws that year
    reg = regions[regions["location_id"] == loc].iloc[0]
    m = measured[measured["location_id"] == loc]
    # Zoom on the bank: the latest line and the VVR inside the region, plus a margin.
    focus = m[m["date"] == m["date"].max()].geometry.union_all()
    vr = vvr[vvr.intersects(reg.geometry)].geometry.intersection(reg.geometry)
    if len(vr):
        focus = focus.union(vr.union_all())
    x0, y0, x1, y1 = focus.buffer(args.margin).bounds
    side = max(x1 - x0, y1 - y0)
    cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
    ext = (cxm - side / 2, cxm + side / 2, cym - side / 2, cym + side / 2)
    clip = gpd.GeoSeries.from_xy([cxm], [cym], crs=28992).buffer(side).iloc[0]
    latest = m[m["date"] == m["date"].max()]
    older = m[m["date"] < m["date"].max()]
    p = pred[pred["location_id"] == loc]
    year = reg["kruisingsjaar"]
    pred_year = int(year) if year == year and year > 2026 else 2031
    if reg["kruising_klasse"] == "reeds gekruist (2026)":
        pred_year = 2027
    if year_override:
        pred_year = int(year_override)
    p = p[p["year"] == pred_year]
    v = vvr[vvr.intersects(clip)].copy()
    v["geometry"] = v.geometry.intersection(clip)

    for step in args.only_steps:
        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_axis_off()
        cx.add_basemap(ax, crs="EPSG:28992", source=AERIAL, attribution=False, zoom=19)
        if step >= 5:
            gpd.GeoSeries([reg.geometry]).plot(
                ax=ax,
                color=CLASS_COLOUR.get(reg["kruising_klasse"], "#999999"),
                alpha=0.5,
            )
        if step >= 2 and len(v):
            v.plot(ax=ax, color=PURPLE, alpha=0.18)
            v.boundary.explode().plot(ax=ax, color=PURPLE, lw=2.4, linestyle="dashdot")
        if step >= 3:
            if len(older):
                older.plot(ax=ax, color="#f0a030", lw=1.2, alpha=0.55)
            latest.plot(ax=ax, color=YELLOW, lw=4.2)
        if step >= 4 and len(p):
            p.explode().plot(ax=ax, color=BLUE, lw=3.6, linestyle="dashed")
        gpd.GeoSeries([reg.geometry]).boundary.plot(ax=ax, color="white", lw=2.6)
        fig.savefig(args.out / f"{key}_{step}.jpg", pil_kwargs={"quality": 82})
        plt.close(fig)
    print(key, loc, reg["kruising_klasse"], "predicted line year", pred_year)
