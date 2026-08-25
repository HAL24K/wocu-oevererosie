"""Coverage autopsy for a loop variant: who lost CORE regions, and why.

Leave-one-out attribution: re-runs the variant's rule stack with each rule
removed and checks which lost CORE regions come back. Then renders the lost
regions (all lines, kept vs removed) for eye validation, so the coverage gap
is documented rather than assumed.

Usage: uv run python scripts/loop_coverage_autopsy.py <variant> [n_panels]
"""

import json
import sys
import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from experiments.loop.harness.harness import (
    aggregate_observations,
    farbank_region_filter,
    load_caches,
    to_dist_per_year,
)
from src.cleaning.rules import RuleContext, apply_rules, make_structure_geom
from src.erosion.region_inspector import (
    MODEL_FILL,
    RegionInspector,
    _line_parts,
    measured_color,
)
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "e2-detrend-dev15"
N_PANELS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
GREY, REMOVED = "#444444", "#d93025"
plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]

caches = load_caches()
ledger = pd.read_csv(caches.exp_dir / "ledger.csv")
row = ledger[ledger["variant"] == VARIANT].iloc[-1]
rules = [(n, p) for n, p in json.loads(row["rules"])]
v_limit = None if pd.isna(row["v_limit"]) else float(row["v_limit"])

kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
GEOM = make_structure_geom(kribs, 10.0)
ok_ids = set(caches.static.index[caches.static["quality"] == "OK"])


def survivors(rule_list):
    ctx = RuleContext(
        line_metrics=caches.line_metrics,
        cl_len=caches.static["cl_len"],
        structures=GEOM,
    )
    s = apply_rules(caches.samples, ctx, rule_list)
    dpy = to_dist_per_year(aggregate_observations(s))
    if v_limit is not None:
        dpy, _ = farbank_region_filter(dpy, v_limit)
    counts = dpy.groupby(LOCATION_ID)["year"].count()
    return set(counts[counts >= 3].index) & ok_ids, s


core = caches.core
full, s_full = survivors(rules)
lost = sorted(core - full)
print(
    f"{VARIANT}: {len(lost)} of {len(core)} CORE regions lost "
    f"(coverage {1 - len(lost) / len(core):.4f})"
)

attribution: dict[str, list] = {}
still_lost = set(lost)
for i, (name, _params) in enumerate(rules):
    without = rules[:i] + rules[i + 1 :]
    recovered, _ = survivors(without)
    saved = sorted(still_lost & (recovered & set(lost)))
    attribution[name] = saved
print("\nleave-one-out attribution (regions recovered when rule removed):")
for name, saved in attribution.items():
    print(f"  {name:<26} {len(saved):>4}")
joint = set(lost) - set().union(*attribution.values())
print(f"  {'(joint/none alone)':<26} {len(joint):>4}")

# eye grid over a spread of lost regions, labelled with the culprit rule
culprit = {}
for name, saved in attribution.items():
    for loc in saved:
        culprit.setdefault(loc, name)
targets = []
for name in list(attribution) + [None]:
    pool = attribution.get(name, []) if name else sorted(joint)
    targets += [(loc, name or "joint") for loc in pool[: max(2, N_PANELS // 6)]]
targets = targets[:N_PANELS]

kept_lines = set(s_full["line_idx"].unique())
ri = RegionInspector()
ncol = 4
nrow = int(np.ceil(len(targets) / ncol)) if targets else 1
fig, axes = plt.subplots(
    nrow, ncol, figsize=(13.0, 3.5 * nrow), constrained_layout=True
)
for ax, (loc, why) in zip(np.array(axes).flatten(), targets, strict=False):
    stats_r = ri.line_stats[ri.line_stats[LOCATION_ID] == loc]
    sgeom = ri.geometry.polygons.get(loc)
    cline = ri.geometry.centrelines.get(loc)
    model = stats_r["model"].iloc[0] if len(stats_r) else "?"
    ax.set_aspect("equal")
    ax.tick_params(labelsize=4)
    ax.set_title(f"{loc} · door: {why}", fontsize=6.5, color=GREY)
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
    f"{VARIANT} · verloren CORE-regio's ({len(lost)}) · rood gestreept = "
    "verwijderd · titel noemt de verantwoordelijke regel",
    fontsize=11,
)
out = caches.cache_dir / "variants" / VARIANT / "coverage_autopsy.png"
fig.savefig(out, dpi=135, bbox_inches="tight")
print(f"→ {out}")
