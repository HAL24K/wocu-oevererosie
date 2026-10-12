"""Opener figure: worst-100 map by hybrid preference + share bars + CV-error split.

Reads ``worst_predicted_v2_<run>.gpkg`` and the run's hybrid delivery
(``model_preference`` layer, for the contest-margin annotation).

Run: ``uv run python scripts/hybrid_pref_figure.py --run 20260907-near-fz --hybrid-gpkg …``
"""

from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import matplotlib
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.pipeline.config import ExperimentConfig  # noqa: E402
from src.sources.geometry import LOCATION_ID, normalise_location_id  # noqa: E402

TEAL, DARK, GREY, ORANGE = "#2BB5A6", "#1F3A3D", "#B9C4C4", "#E07A3F"
plt.rcParams.update({"font.family": ["Verdana", "DejaVu Sans"], "font.size": 9, "text.color": DARK,
                     "axes.edgecolor": DARK, "axes.labelcolor": DARK,
                     "xtick.color": DARK, "ytick.color": DARK})
NB = chr(8239)  # narrow no-break space (Verdana lacks U+2009)

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--hybrid-gpkg", default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()
cfg = ExperimentConfig(experiment=args.run, hybrid_gpkg=args.hybrid_gpkg)

wp = normalise_location_id(
    gpd.read_file(cfg.data_dir / "02_processed/triage" / f"worst_predicted_v2_{args.run}.gpkg", layer="vlakken")
).to_crs(28992)
mp = normalise_location_id(gpd.read_file(cfg.hybrid_gpkg, layer="model_preference")).set_index(LOCATION_ID)
H, S = "points_height [0-10]", "points_segmentation [0-10]"

cent = wp.geometry.centroid
wp["px"], wp["py"] = cent.x, cent.y
top = wp[wp["rank_horizon"] <= 100]
n_all = len(wp)
share_all = (wp["model_preference"] == "segmentation").mean()
n_seg, n_h = int((top["model_preference"] == "segmentation").sum()), int((top["model_preference"] == "height").sum())
seg_top = top[top["model_preference"] == "segmentation"].set_index(LOCATION_ID).join(mp[[H, S, "distinct_dates_height"]])
via_gate = int((seg_top["distinct_dates_height"] < 2).sum())
near_flip = int(((seg_top[H] - seg_top[S]) >= -2).sum() - via_gate) if len(seg_top) else 0
stats = {}
for pref in ["height", "segmentation"]:
    e = wp.loc[wp["model_preference"] == pref, "err_1jr_cv"].dropna()
    stats[pref] = (e.mean(), e.median(), (e > 2).mean())
print(f"worst-100: {n_seg} SAM / {n_h} height · baseline SAM share {share_all:.0%} · via gate {via_gate} · near-flip {near_flip}")
print("CV error by preference:", {k: tuple(round(x, 3) for x in v) for k, v in stats.items()})

fig = plt.figure(figsize=(12.5, 7.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1.15], wspace=0.16, hspace=0.55)
ax = fig.add_subplot(gs[:, 0])
ax.scatter(wp["px"], wp["py"], s=3, c=GREY, alpha=0.5, lw=0)
for pref, col in [("segmentation", ORANGE), ("height", TEAL)]:
    g = top[top["model_preference"] == pref]
    ax.scatter(g["px"], g["py"], s=46, c=col, edgecolors=DARK, linewidths=0.5,
               label=f"worst-100 · {'SAM' if pref == 'segmentation' else 'hoogte'}-voorkeur ({len(g)})", zorder=3)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("100 slechtst voorspelde regio's — welk model koos het hybride model?", fontsize=11, color=DARK, pad=10)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.annotate(f"grijs = alle {n_all:,} gemodelleerde regio's".replace(",", NB), (0.02, 0.02), xycoords="axes fraction", fontsize=8, color="#6b7b7b")

ax1 = fig.add_subplot(gs[0, 1])
rows = [(f"alle regio's\n({n_all:,})".replace(",", NB), share_all, 1 - share_all), ("100 slechtst\nvoorspeld", n_seg / 100, n_h / 100)]
for (lab, s, h), yy in zip(rows, [1.0, 0.35]):
    ax1.barh(yy, s, height=0.42, color=ORANGE)
    ax1.barh(yy, h, left=s, height=0.42, color=TEAL)
    ax1.text(-0.02, yy, lab, ha="right", va="center", fontsize=9)
    ax1.text(s / 2, yy, f"{s:.0%} SAM", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax1.text(s + h / 2, yy, f"{h:.0%}", ha="center", va="center", color="white", fontsize=8)
ax1.text(0, -0.12, f"{n_seg - via_gate} van de {n_seg} via de puntentelling ({via_gate} via de datagate);\n{near_flip} daarvan verloren met maar 1–2 punten",
         fontsize=8, color="#6b7b7b", va="top")
ax1.set_xlim(0, 1); ax1.set_ylim(-0.55, 1.45); ax1.axis("off")
ax1.set_title("SAM-voorkeur oververtegenwoordigd in de staart", fontsize=10, pad=8)

ax2 = fig.add_subplot(gs[1, 1])
for x, (lab, pref, col) in enumerate([("hoogte-voorkeur", "height", TEAL), ("SAM-voorkeur", "segmentation", ORANGE)]):
    mean, med, tail = stats[pref]
    ax2.bar(x, mean, width=0.55, color=col)
    ax2.text(x, mean + 0.08, f"{mean:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax2.text(x, -0.40, lab, ha="center", fontsize=9)
    ax2.text(x, -0.72, f"mediaan {med:.2f}", ha="center", fontsize=7.5, color="#6b7b7b")
    ax2.text(x, -0.98, f"{tail:.0%} > 2{NB}m/jr", ha="center", fontsize=7.5, color="#6b7b7b")
ax2.set_xlim(-0.6, 1.6); ax2.set_ylim(0, max(v[0] for v in stats.values()) * 1.15)
ax2.set_xticks([]); ax2.spines[["top", "right", "bottom"]].set_visible(False)
ax2.set_ylabel("CV-fout (m/jr, 1-jaars, 5-fold)", fontsize=8.5)
ratio = stats["segmentation"][0] / stats["height"][0]
ax2.set_title(f"Alle regio's: {ratio:.1f}× hogere fout waar SAM gekozen is", fontsize=10, pad=8)

fig.suptitle(f"Hybride-modelvoorkeur × voorspelfout — grouped 5-fold CV · run {args.run}", fontsize=12, y=0.99, color=DARK)
out = args.out or str(cfg.data_dir.parent / "docs/presentations/fig" / f"hybrid_pref_worst100_{args.run}.png")
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print("→", out)
