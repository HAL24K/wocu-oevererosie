"""Funnel slide: scope regions → delivered → ≥3 years → after cleaning → modelled.

Run: ``uv run python scripts/slides/region_funnel.py --run 20260907-near-fz --hybrid-gpkg …``
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.pipeline.config import ExperimentConfig  # noqa: E402
from src.sources.geometry import LOCATION_ID, normalise_location_id  # noqa: E402

TEAL, DARK, GREY, ORANGE = "#2BB5A6", "#1F3A3D", "#B9C4C4", "#E07A3F"
plt.rcParams.update({"font.family": ["Verdana", "DejaVu Sans"], "font.size": 10, "text.color": DARK})
NB = chr(8239)  # narrow no-break space (Verdana lacks U+2009)

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--hybrid-gpkg", default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()
cfg = ExperimentConfig.from_run(args.run, hybrid_gpkg=args.hybrid_gpkg)

scope = set(normalise_location_id(gpd.read_file(cfg.scope_gpkg))[LOCATION_ID])
lines = normalise_location_id(gpd.read_file(cfg.hybrid_gpkg, layer="lines"))
lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
lines["year"] = pd.to_datetime(lines["date"]).dt.year
delivered = lines.groupby(LOCATION_ID)["year"].nunique()
obs = pd.read_parquet(cfg.features_dir / "observations.parquet")
obs["year"] = pd.to_datetime(obs["date"]).dt.year
kept_years = obs.groupby(LOCATION_ID)["year"].nunique()
modelled = set(pd.read_parquet(cfg.features_dir / "region_features.parquet").index)
pref = normalise_location_id(gpd.read_file(cfg.hybrid_gpkg, layer="model_preference")).set_index(LOCATION_ID)["model_preference"]

cat = {}
for loc in scope:
    dy, ky = delivered.get(loc, 0), kept_years.get(loc, 0)
    if loc in modelled:
        cat[loc] = "modelled"
    elif dy == 0:
        cat[loc] = "no_lines"
    elif dy < 3:
        cat[loc] = "thin_delivery"
    elif ky == 0:
        cat[loc] = "clean_all"
    elif ky < 3:
        cat[loc] = "clean_partial"
    else:
        cat[loc] = "guards"
c = pd.Series(cat).value_counts()
thin_seg = int((pref.reindex([k for k, v in cat.items() if v == "thin_delivery"]) == "segmentation").sum())
n = {k: int(c.get(k, 0)) for k in ["no_lines", "thin_delivery", "clean_all", "clean_partial", "guards", "modelled"]}
N = len(scope)
print(n, "thin & SAM-pref:", thin_seg)


def fmt(v):
    return f"{v:,}".replace(",", NB)


stages = [
    ("Scope-regio's", N, None),
    ("Met geleverde lijnen", N - n["no_lines"], f"− {fmt(n['no_lines'])} · geen lijnen in de levering"),
    ("≥ 3 meetjaren in de levering", N - n["no_lines"] - n["thin_delivery"],
     f"− {fmt(n['thin_delivery'])} · levering zelf te dun — {fmt(thin_seg)} daarvan SAM-voorkeur:\n   jaren van het verliezende model weggegooid (winner-takes-all)"),
    ("Na cleaning nog ≥ 3 jaren", n["guards"] + n["modelled"],
     f"− {fmt(n['clean_all'] + n['clean_partial'])} · onze filtering ({fmt(n['clean_all'])} alles weg · {fmt(n['clean_partial'])} onder de 3 jaren)"),
    ("Gemodelleerd (geldige t1/t2/t3)", n["modelled"],
     f"− {fmt(n['guards'])} · sanity-guards + laatste 3 surveys overspannen\n   geen 3 bruikbare jaren"),
]

fig, ax = plt.subplots(figsize=(11.69, 8.27))
ax.set_xlim(0, 100); ax.set_ylim(-2.3, len(stages) + 1.1); ax.axis("off")
LBL_X, BAR_X, BAR_MAX = 30, 32, 36
for i, (lab, v, loss) in enumerate(stages):
    y = len(stages) - 1 - i
    w = BAR_MAX * v / N
    color = TEAL if i < len(stages) - 1 else DARK
    ax.text(LBL_X, y, lab, ha="right", va="center", fontsize=10.5, fontweight="bold" if i == len(stages) - 1 else "normal")
    ax.add_patch(FancyBboxPatch((BAR_X, y - 0.28), w, 0.56, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="none"))
    ax.text(BAR_X + w / 2, y, fmt(v), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.text(BAR_X + BAR_MAX + 1.5, y, f"{v / N:.0%}", ha="left", va="center", fontsize=10, color="#6b7b7b")
    if loss:
        ax.text(BAR_X + BAR_MAX + 7.5, y + 0.5, loss, ha="left", va="center", fontsize=8.4, color=DARK)
        ax.annotate("", xy=(BAR_X + BAR_MAX + 6.5, y + 0.28), xytext=(BAR_X + BAR_MAX + 6.5, y + 0.72),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
lost_data = n["no_lines"] + n["thin_delivery"]
filt = n["clean_all"] + n["clean_partial"]
ax.add_patch(FancyBboxPatch((3, -2.05), 45, 1.4, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#eef6f5", ec=TEAL, lw=1.2))
ax.text(5, -1.35, f"Grootste verlies = data die er nooit was ({fmt(lost_data)} regio's),\nniet onze filtering ({fmt(filt)} = {filt / N:.0%} van de scope).", fontsize=10, va="center", color=DARK)
ax.add_patch(FancyBboxPatch((51, -2.05), 46, 1.4, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#fdf1ea", ec=ORANGE, lw=1.2))
ax.text(53, -1.35, "Both-lines export: de te dunne SAM-regio's hebben vaak wél\nhoogtejaren — samen halen velen alsnog ≥ 3 jaren (bovengrens).", fontsize=10, va="center", color=DARK)
ax.text(3, len(stages) + 0.75, "Van scope naar model — waar blijven de regio's?", fontsize=15, fontweight="bold", color=DARK)
ax.text(3, len(stages) + 0.32, f"run {args.run} · alle foutcijfers gaan over de {fmt(n['modelled'])} gemodelleerde regio's", fontsize=9.5, color="#6b7b7b")
out = args.out or str(cfg.figures_dir / "funnel_regions.png")
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print("→", out)
