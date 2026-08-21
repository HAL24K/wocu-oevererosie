"""Eye-validation grids for a loop variant (contract rule 1).

For a given variant, finds the regions whose target moved most against v0 —
plus the regions that left the frozen tail — re-applies the variant's rules to
recover exactly which lines/samples were removed, and renders a grid: kept
lines in the year greens, removed lines dashed red, OSM underneath, with the
v0 → variant v_test in each panel title.

Usage: uv run python scripts/loop_visual_check.py <variant> [n_panels]
"""

import json
import sys
import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from src.erosion.region_inspector import (
    MODEL_FILL,
    RegionInspector,
    _line_parts,
    measured_color,
)
from src.loop.harness import load_caches
from src.loop.rules import RuleContext, apply_rules, make_structure_geom
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v1-maze-survey"
N_PANELS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
GREY = "#444444"
REMOVED = "#d93025"
plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]

caches = load_caches()
ledger = pd.read_csv(caches.exp_dir / "ledger.csv")
row = ledger[ledger["variant"] == VARIANT].iloc[-1]
rules = [(n, p) for n, p in json.loads(row["rules"])]

kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
ctx = RuleContext(
    line_metrics=caches.line_metrics,
    cl_len=caches.static["cl_len"],
    structures=make_structure_geom(kribs, 10.0),
)
survivors = apply_rules(caches.samples, ctx, rules)
kept_lines = set(survivors["line_idx"].unique())
kept_samples_per_line = survivors.groupby("line_idx").size()

# what changed vs v0
var_dir = caches.cache_dir / "variants"
d0 = pd.read_parquet(var_dir / "v0-baseline/dist_per_year.parquet")
d1 = pd.read_parquet(var_dir / VARIANT / "dist_per_year.parquet")


def last_v(d):
    d = d.sort_values([LOCATION_ID, "year"])
    g = d.groupby(LOCATION_ID)
    v = g["dist_m"].diff() / g["year"].diff()
    return v.groupby(d[LOCATION_ID]).last()


v0, v1 = last_v(d0), last_v(d1)
both = v0.index.intersection(v1.index)
delta = (v1[both] - v0[both]).abs().sort_values(ascending=False)
revived = d1[~d1[LOCATION_ID].isin(d0[LOCATION_ID])][LOCATION_ID].unique()
targets = list(delta.head(N_PANELS - min(4, len(revived))).index) + list(revived[:4])
targets = targets[:N_PANELS]

print(
    f"{VARIANT}: {len(delta[delta > 0.5])} regions moved > 0.5 m/yr, "
    f"{len(revived)} revived; showing {len(targets)}"
)

ri = RegionInspector()
ncol = 4
nrow = int(np.ceil(len(targets) / ncol))
fig, axes = plt.subplots(
    nrow, ncol, figsize=(13.0, 3.5 * nrow), constrained_layout=True
)
for ax, loc in zip(np.array(axes).flatten(), targets, strict=False):
    stats_r = ri.line_stats[ri.line_stats[LOCATION_ID] == loc]
    sgeom = ri.geometry.polygons.get(loc)
    cline = ri.geometry.centrelines.get(loc)
    model = stats_r["model"].iloc[0] if len(stats_r) else "?"
    tag = (
        f"{v0.get(loc, np.nan):+.1f} → {v1.get(loc, np.nan):+.1f} m/yr"
        if loc not in revived
        else "revived"
    )
    ax.set_aspect("equal")
    ax.tick_params(labelsize=4)
    ax.set_title(f"{loc} · {tag}", fontsize=6.5, color=GREY)
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
    if cline is not None:
        ax.plot(*cline.xy, color="black", lw=1.4, zorder=5)
    for line_idx, lrow in ri.lines.loc[stats_r.index].iterrows():
        kept = line_idx in kept_lines
        for part in _line_parts(lrow.geometry):
            ax.plot(
                *part.xy,
                color=measured_color(lrow["year"]) if kept else REMOVED,
                lw=1.2 if kept else 1.0,
                ls="-" if kept else (0, (2, 2)),
                alpha=0.9 if kept else 0.8,
                zorder=4,
            )
for ax in np.array(axes).flatten()[len(targets) :]:
    ax.set_visible(False)

fig.suptitle(
    f"{VARIANT} · rood gestreept = verwijderd door de regel · "
    "titel: v_test v0 → variant",
    fontsize=11,
)
out = caches.cache_dir / "variants" / VARIANT / "eye_check.png"
fig.savefig(out, dpi=135, bbox_inches="tight")
print(f"→ {out}")
