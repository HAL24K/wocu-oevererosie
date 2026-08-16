"""
Visualise "3 closest" vs "3 furthest" point selection for several scope regions.
Saves to scripts/point_selection_viz.png
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RAW_GPKG = "data/01_raw/erosion/wocu_output_fase2_20260210.gpkg"

# ── load layers ──────────────────────────────────────────────────────────────
pts = gpd.read_file(RAW_GPKG, layer="punten_oever")
cl  = gpd.read_file(RAW_GPKG, layer="centrelines")

# Build a quick lookup: position_id → centerline geometry
cl_lookup = cl.set_index("position_id")["geometry"].to_dict()

# ── pick representative regions with 3 timestamps ─────────────────────────
ts_counts = pts.groupby("location_id")["dtm_date"].nunique()
three_ts   = ts_counts[ts_counts == 3].index.tolist()

# Pick one from each cluster (grab first match)
CLUSTERS = ["rijn", "ijssel1", "maas1", "maas2", "maas3", "neder"]
selected = []
for cluster in CLUSTERS:
    matches = [lid for lid in three_ts if cluster in lid]
    if matches:
        selected.append(matches[0])

selected = selected[:4]   # keep 4 panels

# ── colour helpers ────────────────────────────────────────────────────────
YEAR_COLORS  = ["#4393c3", "#f4a582", "#a50026"]  # blue → orange → red
SEL_CLOSEST  = "#e41a1c"   # vivid red
SEL_FURTHEST = "#4daf4a"   # vivid green
STATUS_ALPHA = {"OK": 0.75, "OUTLIER": 0.35, "UNCERTAIN": 0.50}

fig, axes = plt.subplots(2, len(selected), figsize=(5.5 * len(selected), 11),
                         constrained_layout=True)

for col, loc_id in enumerate(selected):
    grp     = pts[pts["location_id"] == loc_id].copy()
    dates   = sorted(grp["dtm_date"].unique())
    cline   = cl_lookup.get(loc_id)

    for row, approach in enumerate(["3 closest (current)", "3 furthest (proposed)"]):
        ax = axes[row][col]
        ax.set_aspect("equal")
        ax.set_title(f"{loc_id}\n{approach}", fontsize=8.5, pad=4)

        # draw centerline
        if cline is not None:
            xs, ys = cline.xy
            ax.plot(xs, ys, color="black", linewidth=1.5, zorder=5,
                    label="centreline")
            # label the centerline end
            ax.text(xs[0], ys[0], "CL", fontsize=6, color="black",
                    ha="center", va="bottom", zorder=6)

        selected_pts_x, selected_pts_y = [], []

        for d_idx, date in enumerate(dates):
            year_grp = grp[grp["dtm_date"] == date]
            color    = YEAR_COLORS[d_idx % len(YEAR_COLORS)]

            for _, pt in year_grp.iterrows():
                alpha = STATUS_ALPHA.get(pt["status"], 0.4)
                ax.scatter(pt.geometry.x, pt.geometry.y,
                           c=color, s=12, alpha=alpha,
                           linewidths=0, zorder=3)

        # Overlay selection for the LAST date (t3) to show current choice
        last_date = dates[-1]
        last_grp  = grp[grp["dtm_date"] == last_date]

        if row == 0:   # 3 closest
            chosen = last_grp.nsmallest(3, "dist")
            sel_color = SEL_CLOSEST
        else:          # 3 furthest
            chosen = last_grp.nlargest(3, "dist")
            sel_color = SEL_FURTHEST

        ax.scatter(chosen.geometry.x, chosen.geometry.y,
                   c=sel_color, s=90, edgecolors="black", linewidths=0.6,
                   zorder=7, label=f"selected ({last_date})")

        mean_dist = chosen["dist"].mean()
        statuses  = ", ".join(sorted(chosen["status"].unique()))
        ax.set_xlabel(
            f"Selected mean dist = {mean_dist:.1f} m  |  status: {statuses}",
            fontsize=7.5
        )
        ax.tick_params(labelsize=6)

        if col == 0:
            ax.set_ylabel("Northing (m RD)", fontsize=8)

# ── shared legend ─────────────────────────────────────────────────────────
year_patches = [mpatches.Patch(color=YEAR_COLORS[i], label=f"year t{i+1}",
                                alpha=0.8)
                for i in range(3)]
red_patch   = mpatches.Patch(color=SEL_CLOSEST,  label="3 closest (row 1)")
green_patch = mpatches.Patch(color=SEL_FURTHEST, label="3 furthest (row 2)")
cl_line     = mpatches.Patch(color="black",       label="centreline")

fig.legend(handles=year_patches + [red_patch, green_patch, cl_line],
           loc="lower center", ncol=6, fontsize=8, frameon=True,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Bank-point selection: 3 closest vs 3 furthest to centreline\n"
    "(large circles = the 3 selected points at t3; "
    "faded small circles = status OUTLIER/UNCERTAIN)",
    fontsize=11, y=1.01
)

out = "scripts/point_selection_viz.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
