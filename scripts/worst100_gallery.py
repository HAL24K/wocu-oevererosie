"""Gallery PDF (+ gpkg) of the worst-predicted regions of a run, 6 panels/page.

Panels: measured lines (SAM greens, height model grey-blues; solid = at least
one sample survived cleaning and fed the model, dashed/faded = fully filtered
out), predicted R=20 bank in orange, VVR purple, panel border = hybrid
preference (orange SAM / teal height). Reads ``worst_predicted_v2_<run>.gpkg``
(``scripts/export_worst_predicted_v2.py --run …``).

With ``--causes-gpkg`` (Etienne's review, field ``hoofdoorzaak``) the panels
carry the root cause, are sorted by cause, and lead pages summarise the cause
distribution and the fix per cause. ``--mark-outlier`` draws in red the part of
each kept survey that is off the region's agreeing majority of surveys
(``src.cleaning.consensus``), bin by bin along the bank.

Run: ``uv run python scripts/worst100_gallery.py --run 20260907-near-fz
        --hybrid-gpkg …_nearest_20260710.gpkg --sample-spacing-m 5
        [--preference segmentation] [--n 100] [--out …] [--cover-text …]
        [--causes-gpkg …Reactie_Etienne.gpkg] [--mark-outlier]``
"""

from __future__ import annotations

import argparse
import textwrap
import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src.cleaning.consensus import survey_consensus  # noqa: E402
from src.cleaning.rules import N_STATION_BINS  # noqa: E402
from src.erosion.region_inspector import (  # noqa: E402
    VVR_PURPLE,
    RegionInspector,
    _line_parts,
    measured_color,
)
from src.pipeline.config import ExperimentConfig  # noqa: E402
from src.sources.geometry import LOCATION_ID, normalise_location_id  # noqa: E402

plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]
GREY = "#444444"
ORANGE = "#E07A3F"
TEAL = "#2BB5A6"
DARK = "#1F3A3D"
RED = "#D62828"
MODES = ["doolhof", "kronkelend", "tijd_uitschieter", "fragment", "verkeerde_oever"]
MODE_EN = {
    "doolhof": "maze",
    "kronkelend": "meander",
    "tijd_uitschieter": "time outlier",
    "fragment": "fragment",
    "verkeerde_oever": "wrong bank",
}
HM_GREYS = {2015: "#cfd8dc", 2017: "#cfd8dc", 2021: "#90a4ae", 2025: "#455a64"}

# Etienne's labels, in the order the lead chart and the gallery sections use.
CAUSE_EN = {
    "Bewolking": "Clouds",
    "Nevengeul": "Side channel",
    "Schaduw": "Shadow",
    "Schip": "Ship (at a lock)",
    "Overig (sneeuw/donker/droogte)": "Other: snow / dark / drought",
    "Overig (krib/akker/vegetatie)": "Other: groyne / field / vegetation",
    "Werkzaamheden": "Works",
    "Hoogwater": "High water",
    "Erosie (echt)": "Real erosion",
    "Niet gecontroleerd": "Not reviewed",
}
NOTE_EN = {
    "hoogwater": "high water",
    "bewolking": "clouds",
    "wolken": "clouds",
    "nevengeul": "side channel",
    "schaduw": "shadow",
    "schip": "ship",
    "akker": "field",
    "donker": "dark",
    "sneeuw": "snow",
    "droogte": "drought",
    "krib": "groyne",
    "vegetatie": "vegetation",
    "werkzaamheden": "works",
    "erosie": "erosion",
    "en": "and",
}
# From the 14 Sep internal session, slide 7.
FIX_ROWS = [
    ("Clouds (20 as main cause, 23 mentioned)", "cloud threshold 10 % → 0–5 % in SAM processing; reprocess IJssel", "before hybrid", "Lars / Sytze"),
    ("Side channel (12)", "mask: official layer + vegetatielegger 'Water'", "before hybrid", "us"),
    ("Ships at locks (5)", "wider mask around sluis_stuw (now 10 m)", "before hybrid", "us"),
    ("Shadow (8)", "per image; no geometric rule — model check on the newest measurement?", "after hybrid", "us"),
    ("Works (2) / interventions", "sand on the image; satellite data portal as a check on height-model outliers", "after hybrid", "us + van Oord"),
    ("Wrong source chosen (85 of 100)", "both-lines export; our masks before the points count", "hybrid", "Luke + us"),
    ("Too few measurement years", "more DTM years? SAM years via both-lines", "source", "Luke / Joost"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--hybrid-gpkg", default=None)
ap.add_argument("--sample-spacing-m", type=float, default=None)
ap.add_argument("--preference", choices=["all", "segmentation", "height"], default="all")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", default=None, help="output stem (default triage/worst<n>_gallery_<run>)")
ap.add_argument("--cover-text", default=None, help="text file rendered as page 1")
ap.add_argument("--gpkg", action="store_true", help="also write vlakken + lijnen gpkg")
ap.add_argument("--causes-gpkg", default=None, help="review gpkg with hoofdoorzaak / Opmerking_Et per location")
ap.add_argument("--mark-outlier", action="store_true", help="draw the survey furthest off trend in red")
args = ap.parse_args()

cfg = ExperimentConfig(
    experiment=args.run, hybrid_gpkg=args.hybrid_gpkg, sample_spacing_m=args.sample_spacing_m
)
TRIAGE = cfg.data_dir / "02_processed/triage"
stem = args.out or str(TRIAGE / f"worst{args.n}_gallery_{args.run}")
if args.preference != "all":
    stem = args.out or str(TRIAGE / f"worst{args.n}_{args.preference}_{args.run}")

vl = normalise_location_id(
    gpd.read_file(TRIAGE / f"worst_predicted_v2_{args.run}.gpkg", layer="vlakken")
).set_index(LOCATION_ID)
voorspelde = normalise_location_id(
    gpd.read_file(TRIAGE / f"worst_predicted_v2_{args.run}.gpkg", layer="voorspelde_lijn")
)

ri = RegionInspector(hybrid_gpkg=cfg.hybrid_gpkg)
samples = pd.read_parquet(
    cfg.features_dir / "samples.parquet", columns=["line_idx", LOCATION_ID, "date", "station", "dist", "x", "y"]
)
kept_line_idx = set(samples["line_idx"].unique())
gemeten = ri.lines.copy()
gemeten["kept"] = gemeten.index.isin(kept_line_idx)
scope = ri.geometry.polygons
vvr = gpd.read_file(
    cfg.data_dir / "01_raw/scope/20260205_signaleringslijn.gpkg",
    layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl",
).to_crs(28992)
vvr_sidx = vvr.sindex

ranked = vl[vl["rank_horizon"].notna()].sort_values("rank_horizon")
if args.preference != "all":
    ranked = ranked[ranked["model_preference"] == args.preference]
top = ranked.head(args.n).copy()
top["rank_in_set"] = range(1, len(top) + 1)
print(f"{len(top)} regions · preference {top['model_preference'].value_counts().to_dict()}")

if args.causes_gpkg:
    causes = normalise_location_id(gpd.read_file(args.causes_gpkg)).set_index(LOCATION_ID)
    top["cause"] = causes["hoofdoorzaak"].reindex(top.index).fillna("Niet gecontroleerd")
    top["note"] = causes["Opmerking_Et"].reindex(top.index)
    order = {c: i for i, c in enumerate(CAUSE_EN)}
    top["_cause_order"] = top["cause"].map(order).fillna(len(order))
    top = top.sort_values(["_cause_order", "rank_in_set"])
    print(f"causes joined · {top['cause'].value_counts().to_dict()}")


_consensus: dict = {}


def consensus_for(loc: str):
    if loc not in _consensus:
        _consensus[loc] = survey_consensus(samples[samples[LOCATION_ID] == loc])
    return _consensus[loc]


def outlier_text(cons) -> str:
    if cons.reason != "ok":
        return f"no red line: {cons.reason} among {len(cons.profiles)} surveys"
    if cons.off.empty:
        return f"no red line: all {len(cons.profiles)} surveys agree"
    parts = []
    for d, r in cons.off.sort_index().iterrows():
        side = "landward" if r["dev_m"] > 0 else "riverward"
        last = " (latest)" if d == cons.profiles.index.max() else ""
        parts.append(f"{d:%Y-%m-%d} {r['dev_m']:+.0f} m {side}, {r['share_off']:.0%}{last}")
    if len(parts) > 2:
        parts = [*parts[:2], f"+{len(parts) - 2} more"]
    return "red: " + " · ".join(parts)


def hm_color(year):
    ks = sorted(HM_GREYS)
    return HM_GREYS[min(ks, key=lambda k: abs(k - year))]


def flag_label(row):
    parts = [f"{MODE_EN[m]}×{int(row[f'n_{m}'])}" for m in MODES if row[f"n_{m}"] > 0]
    return ", ".join(parts) if parts else "none"


def note_en(note) -> str:
    if not isinstance(note, str) or not note.strip():
        return ""
    return " ".join(NOTE_EN.get(w.lower(), w) for w in note.split())


def panel(ax, loc, row):
    sgeom = scope.get(loc)
    pref = row["model_preference"]
    pref_en = "SAM" if pref == "segmentation" else ("height" if pref == "height" else pref)
    cons = consensus_for(loc) if args.mark_outlier else None
    outlier_txt = outlier_text(cons) if cons is not None else ""
    cause_txt = ""
    if "cause" in row:
        cause_txt = CAUSE_EN.get(row["cause"], row["cause"])
        extra = note_en(row.get("note"))
        if extra and extra.lower() != cause_txt.lower():
            cause_txt += f" ({extra})"
        cause_txt += " — "
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{cause_txt}#{int(row['rank_in_set'])} {loc}\n"
        f"{pref_en} · horizon error {row['err_horizon_mean']:.1f} m (max {row['err_horizon_max']:.1f}) · flags: {flag_label(row)}\n"
        f"{outlier_txt.lstrip(' ·')}",
        fontsize=5.9,
        color=GREY,
        loc="left",
    )
    for side in ax.spines.values():
        side.set_edgecolor(ORANGE if pref == "segmentation" else TEAL)
        side.set_linewidth(1.6)
    if sgeom is None:
        return
    minx, miny, maxx, maxy = sgeom.bounds
    ax.set_xlim(minx - 40, maxx + 40)
    ax.set_ylim(miny - 40, maxy + 40)
    ax.fill(*sgeom.exterior.xy, fc="#dddddd", ec="#aaaaaa", lw=0.8, alpha=0.4, zorder=1)
    for geom in vvr.iloc[list(vvr_sidx.intersection(sgeom.bounds))][lambda d: d.intersects(sgeom)].geometry:
        for poly in getattr(geom, "geoms", [geom]):
            ax.fill(*poly.exterior.xy, fc=VVR_PURPLE, ec=VVR_PURPLE, lw=1.0, alpha=0.18, zorder=2)
    ri._add_basemap(ax)
    cline = ri.geometry.centrelines.get(loc)
    if cline is not None:
        ax.plot(*cline.xy, color="black", lw=1.1, zorder=5)
    sub = gemeten[gemeten[LOCATION_ID] == loc].sort_values("year")
    for _, lrow in sub.iterrows():
        col = hm_color(int(lrow["year"])) if lrow["model"] == "hoogtemodel" else measured_color(int(lrow["year"]))
        style = (
            {"lw": 1.1, "alpha": 0.85, "ls": "-", "zorder": 4}
            if lrow["kept"]
            else {"lw": 0.9, "alpha": 0.45, "ls": (0, (2, 2)), "zorder": 3}
        )
        for part in _line_parts(lrow.geometry):
            ax.plot(*part.xy, color=col, **style)
    if cons is not None and not cons.off.empty:
        draw_off_parts(ax, loc, cons)
    for geom in voorspelde[voorspelde[LOCATION_ID] == loc].sort_values("seg").geometry:
        ax.plot(*geom.xy, color=ORANGE, lw=2.0, zorder=6)


def draw_off_parts(ax, loc, cons):
    """Red over the stretches of a flagged survey whose bins are off the consensus."""
    s = samples[samples[LOCATION_ID] == loc].copy()
    s["date"] = pd.to_datetime(s["date"])
    s["bin"] = np.minimum((s["station"] * N_STATION_BINS).astype(int), N_STATION_BINS - 1)
    for d, r in cons.off.iterrows():
        pts = s[(s["date"] == d) & s["bin"].isin(r["bins_off"])]
        for _, line in pts.groupby("line_idx"):
            line = line.sort_values("station")
            run = (line["bin"].diff().fillna(1) > 1).cumsum()
            for _, seg in line.groupby(run):
                if len(seg) == 1:
                    ax.plot(seg["x"], seg["y"], "o", color=RED, ms=2.5, zorder=8)
                else:
                    ax.plot(seg["x"], seg["y"], color=RED, lw=2.0, zorder=8, solid_capstyle="round")


def lead_pages(pdf):
    """Cause distribution + what we saw (slides 4–5) and the fix per cause (slide 7), 14 Sep."""
    counts = top["cause"].map(lambda c: CAUSE_EN.get(c, c)).value_counts()
    labels = [CAUSE_EN[c] for c in CAUSE_EN if CAUSE_EN[c] in counts.index]
    vals = [int(counts[lab]) for lab in labels]
    reviewed = sum(v for lab, v in zip(labels, vals, strict=True) if lab != "Not reviewed")

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.93, "What Etienne saw on the satellite images", fontsize=17, color=DARK, weight="bold")
    fig.text(
        0.05, 0.895,
        f"The {len(top)} worst-predicted SAM regions of run {args.run}, after our cleaning — "
        f"he reviewed {reviewed}, the other {len(top) - reviewed} are not reviewed",
        fontsize=10, color=GREY,
    )
    ax = fig.add_axes((0.25, 0.12, 0.40, 0.72))
    ypos = np.arange(len(labels))[::-1]
    colors = ["#B0B7BB" if lab == "Not reviewed" else TEAL for lab in labels]
    ax.barh(ypos, vals, height=0.62, color=colors, edgecolor="white", linewidth=2)
    ax.set_yticks(ypos, labels, fontsize=9.5, color=DARK)
    for y, v in zip(ypos, vals, strict=True):
        ax.text(v + 0.6, y, str(v), va="center", fontsize=9, color=DARK)
    ax.set_xlabel("regions", fontsize=9, color=GREY)
    ax.tick_params(axis="x", colors=GREY, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#CCCCCC")
    ax.xaxis.grid(True, color="#EEEEEE")
    ax.set_axisbelow(True)
    bullets = [
        "These are the worst regions after cleaning: the measurement that causes the error got through all of our rules.",
        "Clouds mostly on the IJssel, shadow mostly on the Maas; SAM accepted images with up to 10 % cloud cover.",
        "Shadow was on a single image each time; ships wait at locks; sand on the image = works.",
        "Two regions are real, large erosion that the model missed.",
        "85 of the 100 worst regions overall have SAM preference (61 % of all scored regions); all 85 won on the points count, 44 by only 1–2 points.",
        "The temporal rule (outliers against each region's trend) was left as is on 13 Sep: tightening it gave nothing measurable.",
    ]
    if args.mark_outlier:
        bullets.append(
            "In the gallery, red = the stretch of a kept survey that is off the region's agreeing majority of surveys "
            "by 8 m or more, over at least 30 % of its line. It is our candidate, not Etienne's marking."
        )
    wrapped = "\n\n".join(textwrap.fill(f"• {b}", 48, subsequent_indent="  ") for b in bullets)
    fig.text(0.69, 0.84, wrapped, fontsize=9, color=DARK, va="top", linespacing=1.3)
    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.93, "Per cause: possible fix, and where in the chain", fontsize=17, color=DARK, weight="bold")
    fig.text(0.05, 0.895, "What can happen before the hybrid model does not have to be learned by it (14 Sep internal session)", fontsize=10, color=GREY)
    ax = fig.add_axes((0.05, 0.15, 0.9, 0.7))
    ax.set_axis_off()
    tbl = ax.table(
        cellText=[list(r) for r in FIX_ROWS],
        colLabels=["cause", "possible fix", "where", "who"],
        colWidths=[0.24, 0.52, 0.11, 0.13],
        loc="upper left",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.1)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_text_props(weight="bold", color=DARK)
            cell.set_facecolor("#F2F4F4")
    pdf.savefig(fig)
    plt.close(fig)

    if not args.mark_outlier:
        return
    rows = {}
    for loc in top.index:
        cons = consensus_for(loc)
        off = cons.off
        latest = cons.profiles.index.max() if len(cons.profiles) else None
        rows[loc] = {
            "cause": CAUSE_EN.get(top.at[loc, "cause"], top.at[loc, "cause"]),
            "flagged": int(not off.empty),
            "flags": len(off),
            "landward": int((off["dev_m"] > 0).sum()) if not off.empty else 0,
            "riverward": int((off["dev_m"] < 0).sum()) if not off.empty else 0,
            "latest": int(latest in off.index) if not off.empty else 0,
            "no_majority": int(cons.reason == "no majority"),
            "surveys": len(cons.profiles),
        }
    stats = pd.DataFrame.from_dict(rows, orient="index")
    table = (
        stats.groupby("cause", sort=False)
        .agg(
            regions=("flagged", "size"),
            flagged=("flagged", "sum"),
            flags=("flags", "sum"),
            landward=("landward", "sum"),
            riverward=("riverward", "sum"),
            latest=("latest", "sum"),
            no_majority=("no_majority", "sum"),
            surveys=("surveys", "median"),
        )
        .reindex([CAUSE_EN[c] for c in CAUSE_EN if CAUSE_EN[c] in set(stats["cause"])])
    )
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.93, "Which surveys disagree with the rest of their region?", fontsize=17, color=DARK, weight="bold")
    fig.text(
        0.05, 0.895,
        "Reference per region = the largest group of surveys that agree within 6 m along the bank, if it holds at least half of them.\n"
        "A survey is flagged when it is 8 m or more off that group over at least 30 % of its line (the red stretches in the gallery).",
        fontsize=9.5, color=GREY,
    )
    ax = fig.add_axes((0.05, 0.35, 0.9, 0.5))
    ax.set_axis_off()
    cells = [
        [c, *(int(r[k]) for k in ("regions", "flagged", "flags", "landward", "riverward", "latest", "no_majority")),
         f"{r['surveys']:.0f}"]
        for c, r in table.iterrows()
    ]
    tbl = ax.table(
        cellText=cells,
        colLabels=["cause", "regions", "with a flag", "flagged surveys", "landward", "riverward",
                   "latest survey", "no majority", "median surveys"],
        colWidths=[0.24, 0.08, 0.09, 0.1, 0.09, 0.09, 0.1, 0.1, 0.11],
        loc="upper left",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for (r, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if col == 0:
            cell.set_text_props(ha="left")
        if r == 0:
            cell.set_text_props(weight="bold", color=DARK)
            cell.set_facecolor("#F2F4F4")
    pdf.savefig(fig)
    plt.close(fig)


pdf_path = f"{stem}.pdf"
with PdfPages(pdf_path) as pdf:
    if args.cover_text:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.06, 0.93, open(args.cover_text).read(), fontsize=10.5, va="top", color=DARK, linespacing=1.6)
        pdf.savefig(fig)
        plt.close(fig)
    if args.causes_gpkg:
        lead_pages(pdf)
    groups = [(c, g) for c, g in top.groupby("cause", sort=False)] if args.causes_gpkg else [(None, top)]
    page = 0
    for cause, group in groups:
        for start in range(0, len(group), 6):
            chunk = group.iloc[start : start + 6]
            page += 1
            heading = (
                f"{CAUSE_EN.get(cause, cause)} — {len(group)} region{'s' if len(group) != 1 else ''}"
                if cause is not None
                else f"#{start + 1}–#{start + len(chunk)}"
            )
            fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
            fig.suptitle(
                f"worst-predicted regions · run {args.run}"
                + (f" · {args.preference} preference only" if args.preference != "all" else "")
                + f" · {heading}\n"
                "green = SAM (darker = more recent) · blue-grey = height model · orange = predicted bank (R = 20) · purple = VVR"
                + (" · red = stretch off the agreeing surveys" if args.mark_outlier else "")
                + "\nsolid = used by the model · dashed/faded = removed by cleaning · "
                "frame: orange = SAM preference, teal = height preference",
                fontsize=8.5,
            )
            for ax, (loc, row) in zip(axes.flatten(), chunk.iterrows(), strict=False):
                panel(ax, loc, row)
            for ax in axes.flatten()[len(chunk) :]:
                ax.set_axis_off()
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"page {page}", flush=True)
print("→", pdf_path)

if args.gpkg:
    gpkg = f"{stem}.gpkg"
    cols = ["rank_in_set", "rank_horizon", "err_horizon_mean", "err_horizon_max", "err_1jr_cv",
            "horizon_in_train", "model_preference", "geen_flags", *[f"n_{m}" for m in MODES], "geometry"]
    if args.causes_gpkg:
        cols = ["cause", "note", *cols]
    gpd.GeoDataFrame(top.reset_index()[[LOCATION_ID, *cols]], geometry="geometry", crs=28992).to_file(
        gpkg, layer="vlakken", driver="GPKG"
    )
    lines = gemeten[gemeten[LOCATION_ID].isin(top.index)].copy()
    lines["date"] = pd.to_datetime(lines["date"]).dt.strftime("%Y-%m-%d")
    lines["kept"] = lines["kept"].astype(int)
    lines = lines.join(top[["rank_in_set"]], on=LOCATION_ID)
    lines[[LOCATION_ID, "rank_in_set", "date", "year", "model", "kept", "geometry"]].to_file(
        gpkg, layer="lijnen", driver="GPKG"
    )
    voorspelde[voorspelde[LOCATION_ID].isin(top.index)].to_file(gpkg, layer="voorspelde_lijn", driver="GPKG")
    print("→", gpkg, f"({len(lines)} lijnen)")
