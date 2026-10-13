"""Before / after: how one scope region is processed (slide "Hoe een regio verwerkt wordt").

Left: the scope region with every delivered bank line (Luke's hybrid), coloured
by survey date. Right: what the model works with: the lines that survive
cleaning, the region cut into R strips along the centreline, and per strip and
survey one bank position (mean of the three samples furthest from the river).
Writes ``<run>/figures/region_processing_<region>.png``.

Run: ``PYTHONPATH=. uv run python scripts/slides/region_processing.py --run 20261013-ref
        --region maas3_l_16480_16490``
"""

from __future__ import annotations

import argparse
import warnings

import contextily as cx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from src.analysis.predicted_lines import _offset_points  # noqa: E402
from src.pipeline.config import ExperimentConfig  # noqa: E402
from src.pipeline.hybrid_prep import load_lines  # noqa: E402
from src.sources.geometry import ScopeGeometry  # noqa: E402

warnings.filterwarnings("ignore")
SAM, HEIGHT, REMOVED, INK = "#2e7d5b", "#496a8e", "#d0442a", "#18252b"
AERIAL = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--region", required=True)
ap.add_argument(
    "--pad", type=float, default=35.0, help="map margin around the region (m)"
)
args = ap.parse_args()

cfg = ExperimentConfig.from_run(args.run)
R = cfg.segment_R
geo = ScopeGeometry(centreline_gpkg=cfg.raw_gpkg)
cl = geo.centrelines[args.region]
scope = pyogrio.read_dataframe(
    cfg.raw_gpkg, layer="vlakken_scope", where=f"position_id = '{args.region}'"
)
poly = scope.geometry.iloc[0]
lines = load_lines(cfg, geo)
lines = lines[lines["location_id"] == args.region]
samples = pd.read_parquet(cfg.features_dir / "samples.parquet")
samples = samples[samples["location_id"] == args.region]
kept = set(samples["line_idx"])
lines = lines.assign(kept=lines.index.isin(kept), date=pd.to_datetime(lines["date"]))
dates = sorted(lines["date"].unique())
side_xy = (samples["x"].mean(), samples["y"].mean())


def date_colour(d, model):
    base = np.array(to_rgb(SAM if model == "segmentation" else HEIGHT))
    k = dates.index(d) / max(len(dates) - 1, 1)
    return tuple(base * (0.45 + 0.55 * k) + np.ones(3) * (1 - (0.45 + 0.55 * k)) * 0.9)


x0, y0, x1, y1 = poly.buffer(args.pad).bounds
fig, axes = plt.subplots(1, 2, figsize=(13, 13 * (y1 - y0) / (x1 - x0) / 2 + 0.9))
for ax in axes:
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    cx.add_basemap(ax, crs="EPSG:28992", source=AERIAL, attribution=False, zoom=18)
    ax.add_patch(
        MplPolygon(
            np.asarray(poly.exterior.coords),
            closed=True,
            fill=False,
            ec="white",
            lw=1.6,
            ls=(0, (5, 3)),
        )
    )

# ── before: every delivered line ─────────────────────────────────────────────
ax = axes[0]
for _, r in lines.sort_values("date").iterrows():
    for g in getattr(r.geometry, "geoms", [r.geometry]):
        xy = np.asarray(g.coords)
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=date_colour(r.date, r.model),
            lw=1.6,
            solid_capstyle="round",
        )
ax.set_title(
    f"Voor: alle aangeleverde oeverlijnen\n{len(lines)} lijnen · {len(dates)} metingen · {dates[0]:%Y}–{dates[-1]:%Y}",
    loc="left",
    fontsize=12,
    color=INK,
)

# ── after: cleaned lines, strips, one position per strip and survey ──────────
ax = axes[1]
dmax = samples["dist"].max() + 12
for k in range(R):
    st = np.linspace(k / R, (k + 1) / R, 6)
    inner = _offset_points(cl, st, np.zeros_like(st), side_xy)
    outer = _offset_points(cl, st, np.full_like(st, dmax), side_xy)
    ax.add_patch(
        MplPolygon(
            np.vstack([inner, outer[::-1]]),
            closed=True,
            fc="white",
            alpha=0.22 if k % 2 else 0.08,
            ec="white",
            lw=0.6,
        )
    )
for _, r in lines[~lines["kept"]].iterrows():
    for g in getattr(r.geometry, "geoms", [r.geometry]):
        xy = np.asarray(g.coords)
        ax.plot(xy[:, 0], xy[:, 1], color=REMOVED, lw=1.0, ls=(0, (3, 2)), alpha=0.7)
for _, r in lines[lines["kept"]].iterrows():
    for g in getattr(r.geometry, "geoms", [r.geometry]):
        xy = np.asarray(g.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=0.8, alpha=0.7)
samples = samples.assign(
    strip=np.minimum((samples["station"] * R).astype(int), R - 1),
    date=pd.to_datetime(samples["date"]),
)
top = samples.sort_values("dist", ascending=False).groupby(["date", "strip"]).head(3)
pos = top.groupby(["date", "strip"], as_index=False).agg(
    dist=("dist", "mean"), model=("model", "first")
)
for d, g in pos.groupby("date"):
    xy = _offset_points(
        cl, (g["strip"].to_numpy() + 0.5) / R, g["dist"].to_numpy(), side_xy
    )
    c = date_colour(d, g["model"].iloc[0])
    ax.scatter(
        xy[:, 0], xy[:, 1], s=22, color=c, edgecolor="white", linewidth=0.6, zorder=5
    )
n_removed = int((~lines["kept"]).sum())
ax.set_title(
    f"Na: opgeschoond, in {R} strookjes van ±5 m\n{n_removed} lijnen gefilterd (rood) · één oeverpositie per strookje per meting",
    loc="left",
    fontsize=12,
    color=INK,
)

fig.text(
    0.01,
    0.01,
    f"{args.region} · lichter = oudere meting, donkerder = recenter · luchtfoto PDOK",
    fontsize=8.5,
    color="#5b6a70",
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
cfg.figures_dir.mkdir(parents=True, exist_ok=True)
out = cfg.figures_dir / f"region_processing_{args.region}.png"
fig.savefig(out, dpi=200, facecolor="white")
print(
    f"{len(lines)} lines ({n_removed} removed), {len(dates)} surveys, {len(pos)} strip positions → {out}"
)
