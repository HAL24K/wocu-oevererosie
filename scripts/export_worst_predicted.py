"""Worst-predicted regions package for Luke / Etienne / Joost.

Goal: surface the 100 test regions the honest run 20260904-r20 predicts worst,
cross-referenced with the existing cleaning flags, so reviewers can hunt for
common patterns the current filters do NOT catch.

Outputs (data/02_processed/triage/):
  worst100_20260904.gpkg
    vlakken_worst100  scope polygon per region: rank, v_train/v_pred/v_test,
                      abs_err, spans, n survey years, river, is_nvo, the five
                      cleaning-flag counts from sam_review, ``geen_flags``.
    lijnen            every delivered line of those 100 regions (full history).
  worst100_20260904.pdf
    cover (error distribution + flag/river/preference split + the ask), then
    6 OSM panels per page sorted by rank: measured lines year-shaded (SAM
    green, hoogtemodel blue-grey), VVR purple, centreline black.

Run: ``uv run python scripts/export_worst_predicted.py``
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import joblib
import matplotlib
import numpy as np
import pandas as pd

from src.erosion.region_inspector import (
    VVR_PURPLE,
    RegionInspector,
    _line_parts,
    measured_color,
)
from src.sources.geometry import LOCATION_ID, normalise_location_id

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]
GREY = "#444444"
RUN = "20260904-r20"
STAMP = "20260904"
MODES = ["doolhof", "kronkelend", "tijd_uitschieter", "fragment", "verkeerde_oever"]

from src.pipeline.config import ExperimentConfig  # noqa: E402

ri = RegionInspector()
cfg = ExperimentConfig(experiment=RUN)
OUT_DIR = cfg.data_dir / "02_processed/triage"

# ── reproduce the honest test predictions ────────────────────────────────────
bundle = joblib.load(cfg.model_out_dir / "bundle.joblib")
feats = bundle["config"]["FEATS_LGB"]
model = joblib.load(cfg.model_out_dir / "model_lgb.joblib")
rf = pd.read_parquet(cfg.features_dir / "region_features.parquet")
test = rf[rf["split"] == "test"].copy()
test["v_pred"] = model.predict(test[feats].astype(float))
test["abs_err"] = (test["v_pred"] - test["v_test"]).abs()
mae = test["abs_err"].mean()
assert abs(mae - 2.473) < 0.01, f"MAE {mae:.3f} does not reproduce the bundle"
print(f"test regions {len(test):,} · MAE {mae:.4f} (bundle 2.473) — reproduced")

worst = test.sort_values("abs_err", ascending=False).head(100).copy()
worst["rank"] = np.arange(1, len(worst) + 1)
worst["river"] = worst.index.str.extract(r"^([a-z]+\d*)", expand=False)

# ── cleaning-flag counts from the sam_review package ─────────────────────────
flags = gpd.read_file(OUT_DIR / f"sam_review_{STAMP}.gpkg", layer="vlakken")
flags = flags.set_index(LOCATION_ID)[[f"n_{m}" for m in MODES]]
worst = worst.join(flags, how="left")
worst[[f"n_{m}" for m in MODES]] = (
    worst[[f"n_{m}" for m in MODES]].fillna(0).astype(int)
)
worst["geen_flags"] = (worst[[f"n_{m}" for m in MODES]].sum(axis=1) == 0).astype(int)

# ── model preference (context on the cover) ──────────────────────────────────
pref = normalise_location_id(
    gpd.read_file(cfg.data_dir / "02_processed/hybrid/model_preference_20260710.gpkg")
).set_index(LOCATION_ID)["model_preference"]
worst["model_preference"] = pref.reindex(worst.index).fillna("onbekend")

# ── gpkg ─────────────────────────────────────────────────────────────────────
scope = ri.geometry.polygons
cols = [
    "rank",
    "v_train",
    "v_pred",
    "v_test",
    "abs_err",
    "train_span_yr",
    "test_span_yr",
    "nyears_hist",
    "river",
    "is_nvo",
    "model_preference",
    *[f"n_{m}" for m in MODES],
    "geen_flags",
]
vl = worst[cols].round({"v_train": 2, "v_pred": 2, "v_test": 2, "abs_err": 2})
vlakken = gpd.GeoDataFrame(
    vl.reset_index(names=LOCATION_ID),
    geometry=scope.reindex(worst.index).values,
    crs=28992,
)
lijnen = ri.lines[ri.lines[LOCATION_ID].isin(worst.index)][
    [LOCATION_ID, "date", "year", "model", "geometry"]
].copy()
lijnen["rank"] = lijnen[LOCATION_ID].map(worst["rank"])
lijnen["date"] = pd.to_datetime(lijnen["date"]).dt.strftime("%Y-%m-%d")

gpkg = OUT_DIR / f"worst100_{STAMP}.gpkg"
if gpkg.exists():
    gpkg.unlink()
vlakken.to_file(gpkg, layer="vlakken_worst100", driver="GPKG")
lijnen.to_file(gpkg, layer="lijnen", driver="GPKG")
print(f"gpkg: {len(vlakken)} vlakken · {len(lijnen):,} lijnen → {gpkg}")

# ── PDF ──────────────────────────────────────────────────────────────────────
vvr = gpd.read_file(
    cfg.data_dir / "01_raw/scope/20260205_signaleringslijn.gpkg",
    layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl",
).to_crs(28992)
vvr_sidx = vvr.sindex

HM_GREYS = {2015: "#cfd8dc", 2017: "#cfd8dc", 2021: "#90a4ae", 2025: "#455a64"}


def hm_color(year: int) -> str:
    ks = sorted(HM_GREYS)
    key = min(ks, key=lambda k: abs(k - year))
    return HM_GREYS[key]


def flag_label(row) -> str:
    parts = [f"{m}×{int(row[f'n_{m}'])}" for m in MODES if row[f"n_{m}"] > 0]
    return ", ".join(parts) if parts else "geen"


def panel(ax, loc, row):
    sgeom = scope.get(loc)
    cline = ri.geometry.centrelines.get(loc)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = (
        f"#{int(row['rank'])} {loc} · v_pred {row['v_pred']:+.1f} vs "
        f"v_test {row['v_test']:+.1f} (err {row['abs_err']:.1f})\n"
        f"flags: {flag_label(row)}"
    )
    ax.set_title(title, fontsize=6.5, color=GREY)
    if sgeom is None:
        return
    minx, miny, maxx, maxy = sgeom.bounds
    ax.set_xlim(minx - 40, maxx + 40)
    ax.set_ylim(miny - 40, maxy + 40)
    ax.fill(*sgeom.exterior.xy, fc="#dddddd", ec="#aaaaaa", lw=0.8, alpha=0.4, zorder=1)
    cand = vvr.iloc[list(vvr_sidx.intersection(sgeom.bounds))]
    for geom in cand[cand.intersects(sgeom)].geometry:
        for poly in getattr(geom, "geoms", [geom]):
            ax.fill(
                *poly.exterior.xy,
                fc=VVR_PURPLE,
                ec=VVR_PURPLE,
                lw=1.2,
                alpha=0.18,
                zorder=2,
            )
    ri._add_basemap(ax)
    if cline is not None:
        ax.plot(*cline.xy, color="black", lw=1.2, zorder=5)
    sub = ri.lines[ri.lines[LOCATION_ID] == loc].sort_values("year")
    for _, lrow in sub.iterrows():
        col = (
            hm_color(int(lrow["year"]))
            if lrow["model"] == "hoogtemodel"
            else measured_color(int(lrow["year"]))
        )
        for part in _line_parts(lrow.geometry):
            ax.plot(*part.xy, color=col, lw=1.2, alpha=0.85, zorder=4)


pdf_path = OUT_DIR / f"worst100_{STAMP}.pdf"
with PdfPages(pdf_path) as pdf:
    # cover
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(
        0.06,
        0.92,
        "100 slechtst voorspelde vlakken · eerlijke run 20260904-r20",
        fontsize=17,
        weight="bold",
    )
    n_clean = int(worst["geen_flags"].sum())
    riv = worst["river"].value_counts()
    prefc = worst["model_preference"].value_counts()
    fig.text(
        0.06,
        0.87,
        f"Testset: {len(test):,} vlakken, MAE {mae:.2f} m/jr · de 100 hierin zijn fout ≥ "
        f"{worst['abs_err'].min():.1f} m/jr (max {worst['abs_err'].max():.1f}).\n"
        f"Bestaande filters: {100 - n_clean} van de 100 dragen ≥ 1 cleaning-flag · "
        f"{n_clean} zijn 'schoon' volgens de huidige regels — daar zit de winst.\n"
        f"Rivieren: {' · '.join(f'{k} {v}' for k, v in riv.items())}\n"
        f"Hybride voorkeur: {' · '.join(f'{k} {v}' for k, v in prefc.items())}",
        fontsize=10,
        va="top",
    )
    fig.text(
        0.06,
        0.70,
        "De vraag: welke patronen zien jullie in deze vlakken die onze filters missen?",
        fontsize=12,
        weight="bold",
    )
    ax = fig.add_axes((0.10, 0.10, 0.83, 0.52))
    ax.hist(test["abs_err"], bins=80, color="#2BB5A6")
    thr = worst["abs_err"].min()
    ax.axvline(thr, color="#E07A3F", lw=2)
    ax.text(
        thr + 0.3,
        ax.get_ylim()[1] * 0.85,
        f"top-100 grens: {thr:.1f} m/jr",
        color="#E07A3F",
        fontsize=10,
    )
    ax.set_yscale("log")
    ax.set_xlabel("|v_pred − v_test| (m/jr) · log-schaal op de y-as")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)

    order = worst.sort_values("rank")
    for start in range(0, len(order), 6):
        chunk = order.iloc[start : start + 6]
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
        fig.suptitle(
            f"slechtst voorspeld · #{int(chunk['rank'].min())}–#{int(chunk['rank'].max())} · "
            "groen = SAM (donker = recenter) · blauwgrijs = hoogtemodel · paars = VVR",
            fontsize=9.5,
        )
        for ax, (loc, row) in zip(axes.flatten(), chunk.iterrows(), strict=False):
            panel(ax, loc, row)
        for ax in axes.flatten()[len(chunk) :]:
            ax.set_axis_off()
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)
        print(f"page #{int(chunk['rank'].min())}–#{int(chunk['rank'].max())}")

print(f"pdf: {pdf_path}")
print(
    f"geen_flags: {n_clean}/100 · rivers: {riv.to_dict()} · pref: {prefc.to_dict()}"
)
