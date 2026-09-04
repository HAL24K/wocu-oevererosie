"""Review package for Etienne: every SAM line the cleaning stack removes.

Outputs (data/02_processed/triage/):
  sam_review_<date>.gpkg
    lijnen   every line flagged by the five quality modes; one prioritised
             ``mode`` class (doolhof > kronkelend > tijd_uitschieter >
             fragment > verkeerde_oever), per-mode booleans, ``severity_raw``
             (the mode's own metric) and ``severity`` (0-1 rank percentile
             within mode, for one graduated QGIS style).
    vlakken  scope polygons with per-mode survey counts.
    masker   the dissolved kribben + kunstwerken 10 m mask (area exclusion,
             not a quality verdict), exploded to parts.
  sam_review_<date>.pdf
    cover with the summary table, one page per quality mode (worst-3 +
    median-3 examples on OSM), and a masker page.

Run: ``uv run python scripts/export_etienne_review.py``
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely

from experiments.loop.harness.harness import load_caches
from src.cleaning.rules import (
    N_STATION_BINS,
    RuleContext,
    _line_p50,
    _survey_p50,
    apply_rules,
    make_structure_geom,
)
from src.erosion.region_inspector import (
    VVR_PURPLE,
    RegionInspector,
    _line_parts,
    measured_color,
)
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]
SURVEY = [LOCATION_ID, "date"]
RED = "#c62828"
GREY = "#444444"
STAMP = "20260904"

caches = load_caches()
DATA = caches.cache_dir.parents[1]
OUT_DIR = DATA / "02_processed/triage"
lm = caches.line_metrics
s = caches.samples
cl = caches.static["cl_len"]
ri = RegionInspector()

# ── per-mode flag sets + severity_raw ────────────────────────────────────────
# kronkelend (line level): severity = tortuosity
kronk = lm[(lm["tortuosity"] > 3.0) & (lm["length"] > 30.0)]
sev_kronk = kronk["tortuosity"]

# verkeerde_oever (line level): severity = 1 - p50/ref
p50 = _line_p50(s)
ref = _survey_p50(s).groupby(LOCATION_ID).median()
line_loc = s.groupby("line_idx")[LOCATION_ID].first()
ref_per_line = line_loc.map(ref)
nb_idx = p50.index[(p50 < 0.25 * ref_per_line) & (ref_per_line > 30.0)]
sev_nb = (1 - p50 / ref_per_line).loc[nb_idx]

# doolhof (survey level): severity = length ratio
tot_len = lm.groupby(SURVEY)["length"].sum()
ratio = tot_len / tot_len.index.get_level_values(0).map(cl)
q = s.groupby(SURVEY)["dist"].quantile([0.25, 0.75]).unstack()
iqr = q[0.75] - q[0.25]
maze_idx = ratio.index[(ratio > 1.8) & (iqr.reindex(ratio.index) > 20.0)]
sev_maze = ratio.loc[maze_idx]

# fragment (survey level): severity = 1 - coverage (station-bin replication)
bins = np.minimum((s["station"] * N_STATION_BINS).astype(int), N_STATION_BINS - 1)
cov = bins.groupby([s[LOCATION_ID], s["date"]]).nunique() / N_STATION_BINS
frag_idx = cov.index[cov < 0.25]
sev_frag = (1 - cov).loc[frag_idx]

# tijd_uitschieter (survey level): severity = |detrended Theil-Sen residual| m
TEMP_PARAMS = {
    "max_dev": 15.0,
    "min_surveys": 3,
    "detrend": True,
    "protect_min_years": 3,
}
ctx = RuleContext(line_metrics=lm, cl_len=cl)
after = apply_rules(s, ctx, [("temporal_outlier_survey", TEMP_PARAMS)])
all_sv = set(map(tuple, s[SURVEY].drop_duplicates().itertuples(index=False)))
kept_sv = set(map(tuple, after[SURVEY].drop_duplicates().itertuples(index=False)))
temp_idx = pd.MultiIndex.from_tuples(sorted(all_sv - kept_sv), names=SURVEY)
pos = _survey_p50(s)
t = pos.index.get_level_values("date").map(pd.Timestamp.toordinal)
frame = pd.DataFrame({"y": pos.values, "t": np.asarray(t) / 365.25}, index=pos.index)


def _resid(g: pd.DataFrame) -> pd.Series:
    if len(g) < 3:
        return pd.Series(0.0, index=g.index)
    tv, yv = g["t"].values, g["y"].values
    i, j = np.triu_indices(len(g), k=1)
    dt = tv[j] - tv[i]
    ok = dt != 0
    if not ok.any():
        return pd.Series(g["y"] - g["y"].median(), index=g.index)
    slope = np.median((yv[j] - yv[i])[ok] / dt[ok])
    intercept = np.median(yv - slope * tv)
    return pd.Series(yv - (slope * tv + intercept), index=g.index)


dev = (
    frame.groupby(LOCATION_ID, group_keys=False)
    .apply(_resid, include_groups=False)
    .abs()
)
sev_temp = dev.reindex(temp_idx)

MODES = {  # priority order; (kind, survey_index or line_index, severity, label, cutoff)
    "doolhof": (
        "survey",
        maze_idx,
        sev_maze,
        "lijnlengte ÷ hartlijn",
        "> 1.8 én IQR > 20 m",
    ),
    "kronkelend": (
        "line",
        kronk.index,
        sev_kronk,
        "lengte ÷ koorde",
        "> 3 (en > 30 m)",
    ),
    "tijd_uitschieter": (
        "survey",
        temp_idx,
        sev_temp,
        "afwijking van trend (m)",
        "> 15 m (jongste 3 jaar beschermd)",
    ),
    "fragment": (
        "survey",
        frag_idx,
        sev_frag,
        "1 − dekking van het vlak",
        "dekking < 25 %",
    ),
    "verkeerde_oever": (
        "line",
        nb_idx,
        sev_nb,
        "1 − afstand ÷ regioreferentie",
        "< 0.25 × referentie (ref ≥ 30 m)",
    ),
}

# ── lijnen layer ─────────────────────────────────────────────────────────────
lm_sv = pd.MultiIndex.from_frame(lm[SURVEY])
flag_lines: dict[str, pd.Series] = {}
for mode, (kind, idx, sev, _, _) in MODES.items():
    if kind == "line":
        mask = lm.index.isin(idx)
        sev_line = sev.reindex(lm.index[mask])
    else:
        mask = lm_sv.isin(idx)
        sev_line = pd.Series(sev.reindex(lm_sv[mask]).values, index=lm.index[mask])
    flag_lines[mode] = sev_line
    n_sv = (
        lm_sv[mask].unique().size
        if kind == "survey"
        else lm.loc[mask, SURVEY].drop_duplicates().shape[0]
    )
    print(f"{mode:16s} lines {int(mask.sum()):6,} · surveys {n_sv:6,}")

flagged_idx = sorted(set().union(*[v.index for v in flag_lines.values()]))
rows = lm.loc[
    flagged_idx, [LOCATION_ID, "date", "year", "model", "length", "tortuosity"]
].copy()
for mode, sev_line in flag_lines.items():
    rows[mode] = rows.index.isin(sev_line.index).astype(int)
rows["mode"] = ""
rows["severity_raw"] = np.nan
for mode in MODES:  # priority: first mode that flags a line wins
    take = (rows["mode"] == "") & (rows[mode] == 1)
    rows.loc[take, "mode"] = mode
    rows.loc[take, "severity_raw"] = flag_lines[mode].reindex(rows.index[take]).values
rows["severity"] = rows.groupby("mode")["severity_raw"].rank(pct=True).round(3)
geoms = ri.lines.geometry.reindex(rows.index)
lijnen = gpd.GeoDataFrame(rows, geometry=geoms.values, crs=28992)
lijnen["date"] = pd.to_datetime(lijnen["date"]).dt.strftime("%Y-%m-%d")

# ── vlakken layer ────────────────────────────────────────────────────────────
scope = ri.geometry.polygons
per_reg = rows.groupby(LOCATION_ID).agg(
    n_lijnen=("mode", "size"), **{f"n_{m}": (m, "sum") for m in MODES}
)
sv_per_reg = (
    rows[rows["mode"] != ""]
    .drop_duplicates([LOCATION_ID, "date"])
    .groupby(LOCATION_ID)
    .size()
    .rename("n_surveys")
)
per_reg = per_reg.join(sv_per_reg)
vlakken = gpd.GeoDataFrame(
    per_reg.reset_index(),
    geometry=scope.reindex(per_reg.index).values,
    crs=28992,
)

# ── masker layer ─────────────────────────────────────────────────────────────
sg = DATA / "02_processed/structures/structures.gpkg"
kribs = gpd.read_file(sg, layer="kribben")
kw = gpd.read_file(sg, layer="kunstwerken")
kw = kw[kw["categorie"].isin(["brug", "kade_damwand", "steiger_afmeer", "sluis_stuw"])]
mask_geom = make_structure_geom(
    pd.concat([kribs[["geometry"]], kw[["geometry"]]]), 10.0
)
masker = gpd.GeoDataFrame(
    geometry=list(getattr(mask_geom, "geoms", [mask_geom])), crs=28992
)

gpkg = OUT_DIR / f"sam_review_{STAMP}.gpkg"
if gpkg.exists():
    gpkg.unlink()
lijnen.to_file(gpkg, layer="lijnen", driver="GPKG")
vlakken.to_file(gpkg, layer="vlakken", driver="GPKG")
masker.to_file(gpkg, layer="masker", driver="GPKG")
print(
    f"lijnen totaal {len(lijnen):,} · vlakken {len(vlakken):,} · maskerdelen {len(masker):,}"
)
print("mode klassen:", rows["mode"].value_counts().to_dict())

# ── PDF ──────────────────────────────────────────────────────────────────────
vvr = gpd.read_file(
    DATA / "01_raw/scope/20260205_signaleringslijn.gpkg",
    layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl",
).to_crs(28992)
vvr_sidx = vvr.sindex


def panel(ax, loc, date, line_ids, title):
    sgeom = scope.get(loc)
    cline = ri.geometry.centrelines.get(loc)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7, color=GREY)
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
    others = lm[(lm[LOCATION_ID] == loc)].index.difference(line_ids)
    for li in others:
        lrow = ri.lines.loc[li]
        for part in _line_parts(lrow.geometry):
            ax.plot(
                *part.xy,
                color=measured_color(lrow["year"]),
                lw=1.0,
                alpha=0.75,
                zorder=3,
            )
    for li in line_ids:
        for part in _line_parts(ri.lines.loc[li].geometry):
            ax.plot(*part.xy, color=RED, lw=1.8, zorder=4)


def pick_examples(mode):
    """Worst-3 + median-3 (distinct regions) as (loc, date, line_ids, value)."""
    kind, idx, sev, _, _ = MODES[mode]
    sub = rows[rows[mode] == 1]
    if kind == "line":
        cases = sub.assign(val=flag_lines[mode].reindex(sub.index)).reset_index(
            names="line_idx"
        )
        cases = cases.groupby([LOCATION_ID, "date"], as_index=False).agg(
            val=("val", "max"), line_ids=("line_idx", list)
        )
    else:
        cases = (
            sub.reset_index(names="line_idx")
            .groupby([LOCATION_ID, "date"], as_index=False)
            .agg(line_ids=("line_idx", list))
        )
        cases["val"] = sev.reindex(pd.MultiIndex.from_frame(cases[SURVEY])).values
    cases = cases.sort_values("val", ascending=False).drop_duplicates(LOCATION_ID)
    worst = cases.head(3)
    rest = cases.iloc[3:]
    mid = len(rest) // 2
    median3 = rest.iloc[max(mid - 1, 0) : mid + 2]
    return pd.concat([worst, median3]).head(6)


table_rows = [
    ["mode", "niveau", "cut-off", "lijnen", "severity-metric"],
    [
        "doolhof",
        "meting",
        MODES["doolhof"][4],
        f"{int(rows['doolhof'].sum()):,}",
        MODES["doolhof"][3],
    ],
    [
        "kronkelend",
        "lijn",
        MODES["kronkelend"][4],
        f"{int(rows['kronkelend'].sum()):,}",
        MODES["kronkelend"][3],
    ],
    [
        "tijd_uitschieter",
        "meting",
        MODES["tijd_uitschieter"][4],
        f"{int(rows['tijd_uitschieter'].sum()):,}",
        MODES["tijd_uitschieter"][3],
    ],
    [
        "fragment",
        "meting",
        MODES["fragment"][4],
        f"{int(rows['fragment'].sum()):,}",
        MODES["fragment"][3],
    ],
    [
        "verkeerde_oever",
        "lijn",
        MODES["verkeerde_oever"][4],
        f"{int(rows['verkeerde_oever'].sum()):,}",
        MODES["verkeerde_oever"][3],
    ],
    [
        "masker (krib + kunstwerk)",
        "gebied",
        "binnen 10 m van constructie",
        "4 094",
        "17.3 % samples · ≈1 263 van 6 798 km",
    ],
]

pdf_path = OUT_DIR / f"sam_review_{STAMP}.pdf"
with PdfPages(pdf_path) as pdf:
    # cover
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.set_axis_off()
    fig.text(
        0.06,
        0.90,
        "SAM-review — uitgefilterde metingen · 4 sep 2026",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.06,
        0.83,
        f"Basis: {len(lm):,} meetbare lijnen ({(lm.model == 'segmentation').sum():,} SAM · "
        f"{(lm.model == 'hoogtemodel').sum():,} hoogtemodel) in {lm.groupby(SURVEY).ngroups:,} metingen.\n"
        f"Geflagde lijnen in dit pakket: {len(lijnen):,}. Eén klasse per lijn (prioriteit doolhof > kronkelend >\n"
        "tijd_uitschieter > fragment > verkeerde_oever); booleans per mode houden overlap zichtbaar.",
        fontsize=10,
        va="top",
    )
    tbl = ax.table(
        cellText=table_rows[1:],
        colLabels=table_rows[0],
        loc="center",
        cellLoc="left",
        colWidths=[0.19, 0.07, 0.27, 0.08, 0.33],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)
    fig.text(
        0.06,
        0.16,
        "Het masker (kribben + kunstwerken, 10 m) is géén kwaliteitsoordeel over SAM: het sluit gebied uit\n"
        "waar de waterlijn een constructieflank is. Lijnen daarbinnen zijn correct gedetecteerde waterranden.",
        fontsize=10,
        style="italic",
    )
    pdf.savefig(fig)
    plt.close(fig)

    for mode, (_kind, _idx, _sev, metric, cutoff) in MODES.items():
        ex = pick_examples(mode)
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
        n_lines = int(rows[mode].sum())
        hdr = f"{mode} · cut-off: {metric} {cutoff} · {n_lines:,} lijnen · links: 3 zwaarste · rechts van het midden: 3 mediane"
        if mode == "tijd_uitschieter":
            hdr += "\nNB: de geometrie ziet er goed uit — alleen de tijdreeks wijkt af; controleer via datum-vergelijking (zelfde vlak, andere datums)."
        fig.suptitle(hdr, fontsize=9.5)
        for ax, (_, case) in zip(axes.flatten(), ex.iterrows(), strict=False):
            d = pd.to_datetime(case["date"]).strftime("%Y-%m-%d")
            panel(
                ax,
                case[LOCATION_ID],
                case["date"],
                pd.Index(case["line_ids"]),
                f"{case[LOCATION_ID]} · {d} · {metric} = {case['val']:.2f}",
            )
        for ax in axes.flatten()[len(ex) :]:
            ax.set_axis_off()
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)
        print(f"page {mode}: {len(ex)} examples")

    # masker page
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.suptitle(
        "masker · kribben + kunstwerken (brug/kade/steiger/sluis), 10 m buffer · gebied uitgesloten vóór de verste-3-selectie",
        fontsize=9.5,
    )
    allscope = gpd.GeoSeries(scope.values, crs=28992)
    allscope.plot(ax=ax1, color="#a9c4ca", linewidth=0.4, edgecolor="#a9c4ca")
    masker.plot(ax=ax1, color=RED, linewidth=0)
    ax1.set_axis_off()
    ax1.set_title("overzicht: masker (rood) over de scope", fontsize=8)
    loc = "nederrijn_r_4470_4480"
    sl = s[s[LOCATION_ID] == loc]
    inside = shapely.contains_xy(mask_geom, sl["x"].values, sl["y"].values)
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    sgeom = scope.get(loc)
    minx, miny, maxx, maxy = sgeom.bounds
    ax2.set_xlim(minx - 60, maxx + 60)
    ax2.set_ylim(miny - 60, maxy + 60)
    masker.clip((minx - 60, miny - 60, maxx + 60, maxy + 60)).plot(
        ax=ax2, fc="#f2c4c0", ec=RED, lw=0.8, zorder=2
    )
    ax2.fill(*sgeom.exterior.xy, fc="none", ec="#888888", lw=1.0, zorder=3)
    ri._add_basemap(ax2)
    ax2.scatter(
        sl["x"][~inside],
        sl["y"][~inside],
        s=4,
        c="#1b7837",
        zorder=4,
        label="sample telt mee",
    )
    ax2.scatter(
        sl["x"][inside],
        sl["y"][inside],
        s=4,
        c=RED,
        zorder=4,
        label="sample binnen masker",
    )
    ax2.legend(fontsize=7, loc="lower right")
    ax2.set_title(f"voorbeeld kribvak: {loc}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)

print("→", gpkg)
print("→", pdf_path)
