"""Build the outlier-triage set: ranked list, QGIS labelling gpkg, top-40 grid.

Three families feed the ranking, with provenance kept:
  - temporal_jump   biggest shifts between consolidated per-year scalars
  - multiline_gap   wildest within-survey disagreement between lines
  - wild_prediction largest |predicted velocity| in the experiment output
"""

import logging
import math
import warnings

import geopandas as gpd
import matplotlib
import numpy as np

from src.erosion.region_inspector import (
    MODEL_FILL,
    RegionInspector,
    _line_parts,
    measured_color,
)
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — after backend selection

ri = RegionInspector()
out_dir = ri.cfg.data_dir / "02_processed/triage"
out_dir.mkdir(parents=True, exist_ok=True)

# ── metrics for every measurable region (thresholds off) ────────────────────
metrics = ri.farbank_candidates(min_gap_m=-1.0, min_ratio=-1.0, v_limit=-1.0)

pred_v = (
    ri.predictions.groupby(LOCATION_ID)["velocity_m_per_yr"]
    .apply(lambda s: float(s.abs().max()))
    .rename("abs_pred_v")
)
metrics = metrics.join(pred_v, how="left")

TOP_JUMP, TOP_GAP, TOP_PRED = 25, 25, 15
fam_jump = set(metrics["max_abs_v"].nlargest(TOP_JUMP).index)
fam_gap = set(metrics["max_multiline_gap_m"].nlargest(TOP_GAP).index)
fam_pred = set(metrics["abs_pred_v"].nlargest(TOP_PRED).index)

sel = sorted(fam_jump | fam_gap | fam_pred)
tri = metrics.loc[sel].copy()
tri["why"] = [
    "+".join(
        w
        for w, fam in (
            ("temporal_jump", fam_jump),
            ("multiline_gap", fam_gap),
            ("wild_prediction", fam_pred),
        )
        if loc in fam
    )
    for loc in tri.index
]
tri["triage_score"] = (
    tri["max_abs_v"].rank(pct=True).fillna(0)
    + tri["max_multiline_gap_m"].rank(pct=True).fillna(0)
    + tri["abs_pred_v"].rank(pct=True).fillna(0)
)
tri = tri.sort_values("triage_score", ascending=False)
tri["rank"] = range(1, len(tri) + 1)

print(
    f"{len(tri)} triage regions ({len(fam_jump)}/{len(fam_gap)}/{len(fam_pred)} per family)"
)
print(
    tri[
        [
            "rank",
            "model",
            "n_dates",
            "max_abs_v",
            "max_multiline_gap_m",
            "abs_pred_v",
            "why",
        ]
    ]
    .head(15)
    .round(1)
    .to_string()
)

# ── QGIS labelling GeoPackage ────────────────────────────────────────────────
cols = [
    "rank",
    "why",
    "model",
    "n_dates",
    "n_multiline_dates",
    "max_abs_v",
    "max_multiline_gap_m",
    "abs_pred_v",
]
regions = (
    gpd.GeoDataFrame(
        tri[cols].round(1).assign(verdict="", note=""),
        geometry=ri.geometry.polygons.reindex(tri.index).values,
        crs=28992,
    )
    .dropna(subset=["geometry"])
    .reset_index()
)
gpkg = out_dir / "triage_20260819.gpkg"
regions.to_file(gpkg, layer="triage_regions", driver="GPKG")

lines = ri.lines[ri.lines[LOCATION_ID].isin(tri.index)].copy()
stats = ri.line_stats
lines["dist_p50"] = stats["dist_p50"].reindex(lines.index).round(1)
lines["date"] = lines["date"].dt.strftime("%Y-%m-%d")
lines[[LOCATION_ID, "date", "year", "model", "dist_p50", "geometry"]].to_file(
    gpkg, layer="triage_lines", driver="GPKG"
)
print(f"wrote {len(regions)} regions + {len(lines)} lines to {gpkg}")

tri.round(2).to_csv(out_dir / "triage_ranked.csv")


# ── top-40 grid ──────────────────────────────────────────────────────────────
def draw_panel(ax, loc_id, why, rank):
    stats_r = stats[stats[LOCATION_ID] == loc_id]
    sgeom = ri.geometry.polygons.get(loc_id)
    cline = ri.geometry.centrelines.get(loc_id)
    model = stats_r["model"].iloc[0] if len(stats_r) else "?"
    ax.set_aspect("equal")
    ax.tick_params(labelsize=4)
    ax.set_title(f"#{rank} {loc_id}\n{why}", fontsize=6)
    if sgeom is not None:
        bx, by = sgeom.exterior.xy
        ax.fill(
            bx, by, fc=MODEL_FILL.get(model, "#eeeeee"), ec="#aaaaaa", lw=0.8, zorder=1
        )
        minx, miny, maxx, maxy = sgeom.bounds
        ax.set_xlim(minx - 40, maxx + 40)
        ax.set_ylim(miny - 40, maxy + 40)
    if cline is not None:
        ax.plot(*cline.xy, color="black", lw=1.4, zorder=5)
    for _, row in ri.lines.loc[stats_r.index].iterrows():
        for part in _line_parts(row.geometry):
            ax.plot(
                *part.xy, color=measured_color(row["year"]), lw=1.2, alpha=0.9, zorder=4
            )


top40 = list(tri.index[:40])
n_cols, n_rows = 5, math.ceil(len(top40) / 5)
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(3.1 * n_cols, 3.6 * n_rows), constrained_layout=True
)
for ax, loc in zip(np.array(axes).flatten(), top40, strict=False):
    draw_panel(ax, loc, tri.loc[loc, "why"], tri.loc[loc, "rank"])
for ax in np.array(axes).flatten()[len(top40) :]:
    ax.set_visible(False)
fig.suptitle(
    "Triage top 40 — greens = measured lines (darker = newer), fill = preferred model",
    fontsize=12,
)
fig.savefig(out_dir / "triage_top40_grid.png", dpi=130, bbox_inches="tight")
print(f"grid → {out_dir / 'triage_top40_grid.png'}")
