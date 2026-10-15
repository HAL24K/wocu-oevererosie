"""Picture for the "what is a scope region" slide: one stretch on the aerial photo with
one region highlighted and its size along and across the bank.

Run: ``PYTHONPATH=. uv run python scripts/slides/scope_region_figure.py --gpkg <session.gpkg>
        --stretch Broekhuizerweerd --region maas3_l_8800_8810 --out <file.jpg>``
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
AERIAL = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/2024_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"

ap = argparse.ArgumentParser()
ap.add_argument("--gpkg", type=Path, required=True)
ap.add_argument("--stretch", required=True)
ap.add_argument("--region", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument(
    "--half-width", type=float, default=190.0, help="metres from centre to edge"
)
args = ap.parse_args()
regions = gpd.read_file(args.gpkg, layer="scoperegios_kruisingsjaar").to_crs(28992)
loc = gpd.read_file(args.gpkg, layer="locaties").to_crs(28992)
ids = set(loc.loc[loc["locatie"] == args.stretch, "location_id"])
stretch = regions[regions["location_id"].isin(ids)]
reg = regions[regions["location_id"] == args.region].geometry.iloc[0]

# Square window on the highlighted region, with its neighbours around it.
cxm, cym = reg.centroid.x, reg.centroid.y
half = args.half_width
fig = plt.figure(figsize=(7, 7), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(cxm - half, cxm + half)
ax.set_ylim(cym - half, cym + half)
ax.set_axis_off()
cx.add_basemap(ax, crs="EPSG:28992", source=AERIAL, attribution=False)
stretch.boundary.plot(ax=ax, color="white", lw=1.2, alpha=0.9)
gpd.GeoSeries([reg]).plot(ax=ax, color="#ffff3c", alpha=0.25)
gpd.GeoSeries([reg]).boundary.plot(ax=ax, color="#ffff3c", lw=3.2)

# Size along and across the bank from the region's minimum rotated rectangle.
rect = np.asarray(reg.minimum_rotated_rectangle.exterior.coords)[:4]
edges = [(rect[i], rect[(i + 1) % 4]) for i in range(4)]
lengths = [np.hypot(*(b - a)) for a, b in edges]
short = int(np.argmin(lengths))
for idx, label in (
    (short, "100 m langs de oever"),
    ((short + 1) % 4, "±175 m landinwaarts"),
):
    a, b = edges[idx]
    mid = (a + b) / 2
    centre = rect.mean(axis=0)
    out = mid + (mid - centre) / np.linalg.norm(mid - centre) * 38
    ang = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
    ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
    ax.text(
        *out,
        label,
        color="white",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=ang,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#00000099", "ec": "none"},
    )
fig.savefig(args.out, pil_kwargs={"quality": 85})
print(args.out, [round(x) for x in lengths])
