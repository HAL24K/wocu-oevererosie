"""Track-3 showcase: predicted 2027 bank as a stitched segment polyline.

Picks frozen-test regions where the R=5 model predicts genuinely different
velocities per segment (the 'single scalar misleads' cases), and draws:
measured lines (year greens), the observed 2026 segment polyline, the
predicted 2027 segment polyline (browns), and the R=1 prediction as a dashed
offset for contrast. OSM underneath.
"""

import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from experiments.loop.harness.harness import load_caches
from experiments.loop.harness.multi_t import load_obs_e8, prepare_standard
from experiments.loop.harness.resolution import build_segment_frame, load_dense
from src.cleaning.rules import make_structure_geom
from src.erosion.region_inspector import (
    MODEL_FILL,
    RegionInspector,
    _line_parts,
    measured_color,
    predicted_color,
)
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = 5
PRED_YEAR = 2027.5
GREY = "#444444"
plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]

caches = load_caches()
kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
obs = load_obs_e8(caches, make_structure_geom(kribs, 10.0))
dense = load_dense(caches)
_, reg_split, _ = prepare_standard(caches, obs)

frame, _ = build_segment_frame(dense, caches, R)
seg_pred = pd.read_parquet(
    caches.cache_dir / "variants/j5-R5-traj2/test_preds_segments.parquet"
)
f = frame.merge(
    seg_pred[[LOCATION_ID, "seg", "pred_lgb"]], on=[LOCATION_ID, "seg"], how="inner"
)
f["pos_2027"] = f["dist_t3"] + f["pred_lgb"] * (PRED_YEAR - f["t3"])

# region-level (R=1) prediction for contrast
i1 = pd.read_parquet(caches.cache_dir / "variants/i1-traj2/test_preds.parquet")

# candidates: full segment coverage, big predicted spread across segments
g = f.groupby(LOCATION_ID)
cand = pd.DataFrame(
    {"n_seg": g.size(), "spread": g["pred_lgb"].max() - g["pred_lgb"].min()}
)
cand = cand[(cand["n_seg"] == R) & cand.index.isin(i1.index)]
targets = cand.sort_values("spread", ascending=False).head(8).index
print("showcase regions (pred-velocity spread m/yr):")
print(cand.loc[targets, "spread"].round(2).to_string())

ri = RegionInspector()
fig, axes = plt.subplots(2, 4, figsize=(13.6, 8.0), constrained_layout=True)


def anchor(cline, station, dist, side_pt):
    base = cline.interpolate(station, normalized=True)
    eps = 0.02
    p0 = cline.interpolate(max(station - eps, 0), normalized=True)
    p1 = cline.interpolate(min(station + eps, 1), normalized=True)
    tx, ty = p1.x - p0.x, p1.y - p0.y
    norm = np.hypot(tx, ty) or 1.0
    nx, ny = -ty / norm, tx / norm
    sign = np.sign((side_pt[0] - base.x) * nx + (side_pt[1] - base.y) * ny) or 1.0
    return base.x + sign * nx * dist, base.y + sign * ny * dist


for ax, loc in zip(axes.flatten(), targets, strict=False):
    rows = f[f[LOCATION_ID] == loc].sort_values("seg")
    stats_r = ri.line_stats[ri.line_stats[LOCATION_ID] == loc]
    sgeom = ri.geometry.polygons.get(loc)
    cline = ri.geometry.centrelines.get(loc)
    model = stats_r["model"].iloc[0] if len(stats_r) else "?"
    samp = caches.samples[caches.samples[LOCATION_ID] == loc]
    side_pt = (samp["x"].mean(), samp["y"].mean())

    ax.set_aspect("equal")
    ax.tick_params(labelsize=4)
    v1 = i1.loc[loc, "pred_lgb"] if loc in i1.index else np.nan
    ax.set_title(
        f"{loc} · R=1: {v1:+.1f} m/jr · segmenten:"
        f" {rows['pred_lgb'].min():+.1f}…{rows['pred_lgb'].max():+.1f}",
        fontsize=6.2,
        color=GREY,
    )
    if sgeom is not None:
        ax.fill(
            *sgeom.exterior.xy,
            fc=MODEL_FILL.get(model, "#eee"),
            ec="#aaa",
            lw=0.8,
            alpha=0.40,
            zorder=1,
        )
        minx, miny, maxx, maxy = sgeom.bounds
        ax.set_xlim(minx - 40, maxx + 40)
        ax.set_ylim(miny - 40, maxy + 40)
    ri._add_basemap(ax)
    if cline is None:
        continue
    ax.plot(*cline.xy, color="black", lw=1.3, zorder=5)
    for _, lrow in ri.lines.loc[stats_r.index].iterrows():
        for part in _line_parts(lrow.geometry):
            ax.plot(
                *part.xy,
                color=measured_color(lrow["year"]),
                lw=1.0,
                alpha=0.75,
                zorder=3,
            )

    # observed 2026 and predicted 2027 segment polylines
    for col, color, lw, _label_year in [
        ("dist_t3", "#1f7d2f", 1.6, None),
        ("pos_2027", predicted_color(2027), 2.2, 2027),
    ]:
        pts = [
            anchor(cline, (s + 0.5) / R, d, side_pt)
            for s, d in zip(rows["seg"], rows[col], strict=True)
        ]
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, color=color, lw=lw, marker="o", ms=2.5, zorder=6)

    # R=1 2027 prediction: constant offset (dashed)
    if loc in i1.index and loc in reg_split.index:
        d_2027 = reg_split.loc[loc, "dist_t3"] + v1 * (
            PRED_YEAR - reg_split.loc[loc, "t3"]
        )
        line_pts = [
            anchor(cline, s, d_2027, side_pt) for s in np.linspace(0.05, 0.95, 12)
        ]
        xs, ys = zip(*line_pts, strict=True)
        ax.plot(
            xs,
            ys,
            color=predicted_color(2027),
            lw=1.4,
            ls=(0, (4, 3)),
            alpha=0.85,
            zorder=5,
        )

fig.suptitle(
    "Resolutie R=5 · voorspelde oever 2027 per segment (bruin, gestippeld = R=1)"
    " · donkergroen = gemeten 2026-lijn per segment",
    fontsize=11,
)
out = caches.cache_dir / "variants/j5-R5-traj2/showcase_2027.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"→ {out}")
