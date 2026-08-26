"""Candidate galleries for the 2026-08-26 deck (pick-one contact sheets).

Outputs under docs/presentations/fig/candidates/:
  single_line/  height-model regions with VVR and wavy banks: measured lines
                vs the single R=1 scalar per year (straight dashed offsets)
  resolution/   R=5 segment prediction vs R=1 scalar, with VVR polygon
  r_sweep/      observed latest-year segment polyline for R = 1..100

Run: uv run python scripts/deck_candidates.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely

from experiments.loop.harness.harness import load_caches
from experiments.loop.harness.multi_t import prepare_standard
from experiments.loop.harness.resolution import build_segment_frame, load_dense
from src.erosion.region_inspector import (
    MODEL_FILL,
    VVR_PURPLE,
    RegionInspector,
    _line_parts,
    measured_color,
    predicted_color,
)
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/presentations/fig/candidates"
GREY = "#444444"
PRED_YEAR = 2027.5
N = 20
plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]

caches = load_caches()
DATA = caches.cache_dir.parents[1]
ri = RegionInspector()
kribs = gpd.read_file(DATA / "02_processed/structures/structures.gpkg", layer="kribben")
vvr = gpd.read_file(
    DATA / "01_raw/scope/20260205_signaleringslijn.gpkg",
    layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl",
).to_crs(28992)
vvr_sidx = vvr.sindex
pref = gpd.read_file(DATA / "02_processed/hybrid/model_preference_20260710.gpkg")
pref = pref.set_index(LOCATION_ID)["model_preference"]
static = (
    caches.static.set_index(LOCATION_ID)
    if LOCATION_ID in caches.static
    else caches.static
)


# ── helpers ──────────────────────────────────────────────────────────────────
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


def side_point(loc):
    s = caches.samples[caches.samples[LOCATION_ID] == loc]
    return (s["x"].mean(), s["y"].mean())


def vvr_for(sgeom):
    cand = vvr.iloc[list(vvr_sidx.intersection(sgeom.bounds))]
    return cand[cand.intersects(sgeom)]


def sig_for(sgeom):
    return ri.signalering[ri.signalering.intersects(sgeom.buffer(10))]


def has_vvr(loc):
    if loc not in ri.geometry.polygons.index:
        return False
    g = ri.geometry.polygons[loc]
    return len(vvr_for(g)) > 0 or len(sig_for(g)) > 0


def n_kribben(sgeom):
    return int(kribs.intersects(sgeom.buffer(20)).sum())


def base_axes(ax, loc, pad=40, title=None):
    sgeom = ri.geometry.polygons.get(loc)
    cline = ri.geometry.centrelines.get(loc)
    stats_r = ri.line_stats[ri.line_stats[LOCATION_ID] == loc]
    model = stats_r["model"].iloc[0] if len(stats_r) else "?"
    ax.set_aspect("equal")
    ax.tick_params(labelsize=4)
    if title:
        ax.set_title(title, fontsize=6.5, color=GREY)
    if sgeom is not None:
        ax.fill(
            *sgeom.exterior.xy,
            fc=MODEL_FILL.get(model, "#eee"),
            ec="#aaa",
            lw=0.8,
            alpha=0.4,
            zorder=1,
        )
        minx, miny, maxx, maxy = sgeom.bounds
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        for geom in vvr_for(sgeom).geometry:
            for poly in getattr(geom, "geoms", [geom]):
                ax.fill(
                    *poly.exterior.xy,
                    fc=VVR_PURPLE,
                    ec=VVR_PURPLE,
                    lw=1.4,
                    alpha=0.18,
                    zorder=2,
                )
        for geom in sig_for(sgeom).geometry:
            for part in _line_parts(geom):
                ax.plot(*part.xy, color=VVR_PURPLE, lw=1.8, zorder=6)
    ri._add_basemap(ax)
    if cline is not None:
        ax.plot(*cline.xy, color="black", lw=1.3, zorder=5)
    return sgeom, cline, stats_r


def draw_measured(ax, stats_r, lw=1.0):
    for _, lrow in ri.lines.loc[stats_r.index].iterrows():
        for part in _line_parts(lrow.geometry):
            ax.plot(
                *part.xy, color=measured_color(lrow["year"]), lw=lw, alpha=0.8, zorder=3
            )


def contact_sheet(paths, titles, out, ncol=5, suptitle=""):
    n = len(paths)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.2, nrow * 3.3))
    for ax in axes.flatten():
        ax.set_axis_off()
    for ax, p, t in zip(axes.flatten(), paths, titles, strict=False):
        ax.imshow(plt.imread(p))
        ax.set_title(t, fontsize=6.5, color=GREY)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ── 1 · single line ──────────────────────────────────────────────────────────
def gallery_single_line():
    out = OUT / "single_line"
    out.mkdir(parents=True, exist_ok=True)
    lm = caches.line_metrics
    per_reg = lm.groupby(LOCATION_ID).agg(
        tort=("tortuosity", "median"),
        n_years=("year", "nunique"),
        n_lines=("year", "size"),
    )
    st = static if static.index.name == LOCATION_ID else static.set_index(LOCATION_ID)
    per_reg["is_nvo"] = st["is_nvo"].reindex(per_reg.index)
    per_reg["pref"] = pref.reindex(per_reg.index)
    # R=1 scalar per (region, date): furthest-3 mean; spread vs the line median
    s = caches.samples
    g = s.groupby([LOCATION_ID, "date"], sort=False)["dist"]
    far3 = g.nlargest(3).groupby(level=[0, 1]).mean()
    med = g.median()
    spread = (far3 - med).groupby(level=0).mean().rename("spread")
    per_reg = per_reg.join(spread)
    ok = per_reg[
        (per_reg["is_nvo"] == 1)
        & (per_reg["pref"] == "height")
        & per_reg["tort"].between(1.15, 2.5)
        & (per_reg["n_years"] >= 3)
        & (per_reg["n_lines"] <= 2 * per_reg["n_years"])
    ].copy()
    ok = ok[[has_vvr(l) for l in ok.index]]
    ok["score"] = ok["tort"] * ok["spread"] * np.minimum(ok["n_years"], 4)
    print(f"single_line: {len(ok)} regions satisfy filters")
    picks = ok.sort_values("score", ascending=False).head(N)
    paths, titles = [], []
    for loc, row in picks.iterrows():
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        title = f"{loc} · tort {row.tort:.2f} · {int(row.n_years)} jr · R=1 t.o.v. lijn {row.spread:.0f} m"
        sgeom, cline, stats_r = base_axes(ax, loc, title=title)
        draw_measured(ax, stats_r, lw=1.4)
        if cline is not None:
            sp = side_point(loc)
            reg = far3.loc[loc]
            yearly = reg.groupby(pd.to_datetime(reg.index).year).median()
            for yr, d in yearly.items():
                pts = [anchor(cline, st_, d, sp) for st_ in np.linspace(0.03, 0.97, 14)]
                xs, ys = zip(*pts, strict=True)
                ax.plot(
                    xs,
                    ys,
                    color=measured_color(int(yr)),
                    lw=1.8,
                    ls=(0, (4, 3)),
                    zorder=6,
                )
        p = out / f"{loc}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
        titles.append(title)
    contact_sheet(
        paths,
        titles,
        out / "_contact_sheet.png",
        suptitle="Één afstand per vlak (gestippeld) vs gemeten oeverlijn (groen) · paars = VVR",
    )
    return picks


# ── 2 · resolution ───────────────────────────────────────────────────────────
def gallery_resolution():
    out = OUT / "resolution"
    out.mkdir(parents=True, exist_ok=True)
    R = 5
    obs = pd.read_parquet(caches.cache_dir / "obs_e8.parquet")
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
    i1 = pd.read_parquet(caches.cache_dir / "variants/i1-traj2/test_preds.parquet")
    st = static if static.index.name == LOCATION_ID else static.set_index(LOCATION_ID)
    g = f.groupby(LOCATION_ID)
    cand = pd.DataFrame(
        {"n_seg": g.size(), "spread": g["pred_lgb"].max() - g["pred_lgb"].min()}
    )
    cand = cand[(cand["n_seg"] == R) & cand.index.isin(i1.index)]
    cand["is_nvo"] = st["is_nvo"].reindex(cand.index)
    cand = cand[cand["is_nvo"] == 1]
    lm = caches.line_metrics
    lines_per_date = lm.groupby([LOCATION_ID, "date"]).size().groupby(level=0).mean()
    cand["lines_per_date"] = lines_per_date.reindex(cand.index)
    cand["n_krib"] = [
        n_kribben(ri.geometry.polygons[l]) if l in ri.geometry.polygons.index else 9
        for l in cand.index
    ]
    cand["has_vvr"] = [has_vvr(l) for l in cand.index]
    print(
        f"resolution: {len(cand)} NVO candidates, {(cand.has_vvr).sum()} with VVR polygon, {(cand.n_krib == 0).sum()} without kribben"
    )
    clean = cand[cand["has_vvr"] & (cand["lines_per_date"] <= 1.5)].copy()
    clean["score"] = clean["spread"] / (1 + clean["n_krib"])
    picks = clean.sort_values("score", ascending=False).head(N)
    paths, titles = [], []
    for loc, row in picks.iterrows():
        rows = f[f[LOCATION_ID] == loc].sort_values("seg")
        v1 = i1.loc[loc, "pred_lgb"]
        title = f"{loc} · R=1 {v1:+.1f} m/jr · segm {rows.pred_lgb.min():+.1f}…{rows.pred_lgb.max():+.1f} · kribben {int(row.n_krib)}"
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        sgeom, cline, stats_r = base_axes(ax, loc, title=title)
        draw_measured(ax, stats_r)
        if cline is None:
            continue
        sp = side_point(loc)
        for col, color, lw in [
            ("dist_t3", "#1f7d2f", 1.6),
            ("pos_2027", predicted_color(2027), 2.2),
        ]:
            pts = [
                anchor(cline, (s_ + 0.5) / R, d, sp)
                for s_, d in zip(rows["seg"], rows[col], strict=True)
            ]
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, color=color, lw=lw, marker="o", ms=2.5, zorder=6)
        if loc in reg_split.index:
            d_2027 = reg_split.loc[loc, "dist_t3"] + v1 * (
                PRED_YEAR - reg_split.loc[loc, "t3"]
            )
            pts = [anchor(cline, s_, d_2027, sp) for s_ in np.linspace(0.05, 0.95, 12)]
            xs, ys = zip(*pts, strict=True)
            ax.plot(
                xs,
                ys,
                color=predicted_color(2027),
                lw=1.4,
                ls=(0, (4, 3)),
                alpha=0.85,
                zorder=5,
            )
        p = out / f"{loc}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
        titles.append(title)
    contact_sheet(
        paths,
        titles,
        out / "_contact_sheet.png",
        suptitle="R=5: gemeten 2026 (donkergroen) · voorspeld 2027 (bruin) · R=1 gestippeld · paars = VVR",
    )
    return picks


# ── 3 · R sweep ──────────────────────────────────────────────────────────────
def resample_lines(loc, date, n=300):
    """Dense (station, dist) samples for the region's lines on one date."""
    cline = ri.geometry.centrelines[loc]
    sel = ri.lines[(ri.lines[LOCATION_ID] == loc) & (ri.lines["date"] == date)]
    st, di = [], []
    for geom in sel.geometry:
        for part in _line_parts(geom):
            for t in np.linspace(0, 1, n):
                pt = part.interpolate(t, normalized=True)
                st.append(cline.project(pt, normalized=True))
                di.append(pt.distance(cline))
    return pd.DataFrame({"station": st, "dist": di})


def _sweep_panel(ax, loc, R, latest, samp, sp, cline, stats_r, seg_len):
    latest_stats = stats_r[stats_r["date"] == latest]
    base_axes(ax, loc, pad=25, title=f"R = {R} · {seg_len / R:.0f} m per segment")
    draw_measured(ax, latest_stats, lw=1.8)
    lb = ri.lines.loc[latest_stats.index].total_bounds
    ax.set_xlim(lb[0] - 20, lb[2] + 20)
    ax.set_ylim(lb[1] - 20, lb[3] + 20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    seg = np.minimum((samp["station"] * R).astype(int), R - 1)
    per = samp.groupby(seg)["dist"].apply(
        lambda x: x.nlargest(min(3, len(x))).mean() if len(x) >= 2 else np.nan
    )
    per = per.reindex(range(R))
    pts = [
        anchor(cline, (s_ + 0.5) / R, dd, sp) if np.isfinite(dd) else (np.nan, np.nan)
        for s_, dd in per.items()
    ]
    xs, ys = zip(*pts, strict=True)
    ax.plot(
        xs, ys, color="#8B4513", lw=2.0, marker="o", ms=3 if R <= 20 else 1.5, zorder=7
    )
    starved = int(per.isna().sum())
    if starved:
        ax.text(
            0.02,
            0.02,
            f"{starved} segm. zonder samples",
            transform=ax.transAxes,
            fontsize=7,
            color="#b00",
            zorder=9,
        )
    ax.title.set_fontsize(9)


def gallery_r_sweep(locs):
    out = OUT / "r_sweep"
    out.mkdir(parents=True, exist_ok=True)
    dense = load_dense(caches)
    Rs = [1, 2, 5, 10, 20, 50, 100]
    for loc in locs:
        d = dense[dense[LOCATION_ID] == loc]
        if d.empty:
            continue
        latest = d["date"].max()
        samp = resample_lines(loc, latest)
        sp = side_point(loc)
        cline = ri.geometry.centrelines[loc]
        stats_r = ri.line_stats[ri.line_stats[LOCATION_ID] == loc]
        # bank length ≈ length of the latest survey line(s)
        seg_len = float(
            ri.lines.loc[stats_r[stats_r["date"] == latest].index].length.sum()
        )
        # 2-row figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
        for ax in axes.flatten():
            ax.set_axis_off()
        slots = [
            axes[0, 0],
            axes[0, 1],
            axes[0, 2],
            axes[1, 0],
            axes[1, 1],
            axes[1, 2],
            axes[1, 3],
        ]
        for ax, R in zip(slots, Rs, strict=True):
            ax.set_axis_on()
            _sweep_panel(ax, loc, R, latest, samp, sp, cline, stats_r, seg_len)
        lax = axes[0, 3]
        lax.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    color=measured_color(int(pd.Timestamp(latest).year)),
                    lw=1.8,
                    label=f"gemeten oeverlijn {pd.Timestamp(latest).date()}",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="#8B4513",
                    lw=2,
                    marker="o",
                    ms=3,
                    label="segment-representatie (verste 3 punten per segment)",
                ),
                plt.Line2D([0], [0], color="black", lw=1.3, label="hartlijn"),
                plt.Line2D(
                    [0], [0], color=VVR_PURPLE, lw=1.8, label="VVR / signaleringslijn"
                ),
            ],
            loc="center",
            fontsize=9,
            frameon=False,
        )
        fig.suptitle(f"{loc} · oeverlengte {seg_len:.0f} m", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"{loc}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        # separate panels
        for R in Rs:
            fig, ax = plt.subplots(figsize=(4.4, 4.4))
            _sweep_panel(ax, loc, R, latest, samp, sp, cline, stats_r, seg_len)
            fig.savefig(out / f"{loc}_R{R}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
        print("r_sweep →", loc)


def gallery_single_line_wavy():
    """Contrast wavy measured bank vs one straight distance; any preference."""
    out = OUT / "single_line_wavy"
    out.mkdir(parents=True, exist_ok=True)
    lm = caches.line_metrics
    per_reg = lm.groupby(LOCATION_ID).agg(
        tort=("tortuosity", "median"),
        n_years=("year", "nunique"),
        n_lines=("year", "size"),
    )
    lines_per_date = lm.groupby([LOCATION_ID, "date"]).size().groupby(level=0).max()
    per_reg["lines_per_date"] = lines_per_date.reindex(per_reg.index)
    s = caches.samples
    g = s.groupby([LOCATION_ID, "date"], sort=False)["dist"]
    far3 = g.nlargest(3).groupby(level=[0, 1]).mean().rename("far3")
    # mean perpendicular deviation of the line's samples from the straight scalar line
    dev = (
        (s.set_index([LOCATION_ID, "date"])["dist"] - far3)
        .abs()
        .groupby(level=0)
        .mean()
        .rename("dev_m")
    )
    per_reg = per_reg.join(dev)
    ok = per_reg[
        per_reg["tort"].between(1.4, 3.0)
        & (per_reg["n_years"] >= 3)
        & (per_reg["lines_per_date"] <= 2)
    ].copy()
    ok = ok[[has_vvr(l) for l in ok.index]]
    ok["river"] = [l.split("_")[0] for l in ok.index]
    print(
        f"single_line_wavy: {len(ok)} regions satisfy filters; rivers {ok.river.value_counts().to_dict()}"
    )
    # mix of rivers: round-robin over rivers sorted by dev
    ok = ok.sort_values("dev_m", ascending=False)
    ok["rank_in_river"] = ok.groupby("river").cumcount()
    picks = ok.sort_values(["rank_in_river", "dev_m"], ascending=[True, False]).head(N)
    paths, titles = [], []
    for loc, row in picks.iterrows():
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        title = f"{loc} · afwijking {row.dev_m:.0f} m · tort {row.tort:.2f} · {int(row.n_years)} jr"
        sgeom, cline, stats_r = base_axes(ax, loc, title=title)
        draw_measured(ax, stats_r, lw=1.4)
        if cline is not None:
            sp = side_point(loc)
            reg = far3.loc[loc]
            yearly = reg.groupby(pd.to_datetime(reg.index).year).median()
            for yr, dd in yearly.items():
                pts = [
                    anchor(cline, st_, dd, sp) for st_ in np.linspace(0.03, 0.97, 14)
                ]
                xs, ys = zip(*pts, strict=True)
                ax.plot(
                    xs,
                    ys,
                    color=measured_color(int(yr)),
                    lw=1.8,
                    ls=(0, (4, 3)),
                    zorder=6,
                )
        p = out / f"{loc}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
        titles.append(title)
    contact_sheet(
        paths,
        titles,
        out / "_contact_sheet.png",
        suptitle="Één afstand per vlak (gestippeld) vs gemeten oeverlijn (groen) · paars = VVR",
    )
    return picks


def gallery_single_line_height():
    """Height-model-preference regions (the March population), no tortuosity
    floor, ranked by deviation of the measured line from the R=1 scalar;
    regions whose deviation comes from a secondary water body are excluded."""
    out = OUT / "single_line_height"
    out.mkdir(parents=True, exist_ok=True)
    lm = caches.line_metrics
    per_reg = lm.groupby(LOCATION_ID).agg(
        tort=("tortuosity", "median"), n_years=("year", "nunique")
    )
    s = caches.samples
    g = s.groupby([LOCATION_ID, "date"], sort=False)["dist"]
    far3 = g.nlargest(3).groupby(level=[0, 1]).mean().rename("far3")
    dev = (
        (s.set_index([LOCATION_ID, "date"])["dist"] - far3)
        .abs()
        .groupby(level=0)
        .mean()
        .rename("dev_m")
    )
    # share of samples inside the secondary-water mask (10 m buffered parts)
    water = gpd.read_file(DATA / "02_processed/triage/secondary_water_mask.gpkg")
    wgeom = shapely.union_all(water.geometry.buffer(10).values)
    shapely.prepare(wgeom)
    pts = gpd.GeoSeries(gpd.points_from_xy(s.x, s.y), index=s.index)
    in_water = pd.Series(shapely.contains(wgeom, pts.values), index=s.index)
    water_frac = in_water.groupby(s[LOCATION_ID]).mean().rename("water_frac")
    per_reg = per_reg.join(dev).join(water_frac)
    per_reg["pref"] = pref.reindex(per_reg.index)
    hm = per_reg[(per_reg["pref"] == "height") & (per_reg["n_years"] >= 3)].copy()
    hm = hm[[has_vvr(l) for l in hm.index]]
    print(
        f"single_line_height: {len(hm)} height-model regions with VVR and ≥3 years; "
        f"dev_m p50 {hm.dev_m.median():.1f} m, p90 {hm.dev_m.quantile(0.9):.1f} m; "
        f"{int((hm.water_frac > 0.2).sum())} excluded for secondary water"
    )
    ok = hm[hm["water_frac"].fillna(0) <= 0.2].copy()
    ok["river"] = [l.split("_")[0] for l in ok.index]
    ok = ok.sort_values("dev_m", ascending=False)
    ok["rank_in_river"] = ok.groupby("river").cumcount()
    picks = ok.sort_values(["rank_in_river", "dev_m"], ascending=[True, False]).head(N)
    paths, titles = [], []
    for loc, row in picks.iterrows():
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        title = f"{loc} · afwijking {row.dev_m:.0f} m · tort {row.tort:.2f} · {int(row.n_years)} jr"
        sgeom, cline, stats_r = base_axes(ax, loc, title=title)
        draw_measured(ax, stats_r, lw=1.4)
        if cline is not None:
            sp = side_point(loc)
            reg = far3.loc[loc]
            yearly = reg.groupby(pd.to_datetime(reg.index).year).median()
            for yr, dd in yearly.items():
                pp = [anchor(cline, st_, dd, sp) for st_ in np.linspace(0.03, 0.97, 14)]
                xs, ys = zip(*pp, strict=True)
                ax.plot(xs, ys, color=measured_color(int(yr)), lw=1.8, ls=(0, (4, 3)), zorder=6)
        pth = out / f"{loc}.png"
        fig.savefig(pth, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(pth)
        titles.append(title)
    contact_sheet(
        paths,
        titles,
        out / "_contact_sheet.png",
        suptitle="Hoogtemodel-voorkeur · één afstand per vlak (gestippeld) vs gemeten oeverlijn (groen) · paars = VVR",
    )
    return picks


# ── 5 · structures cleanup (zonder / met masker) ─────────────────────────────
def gallery_structures_cleanup():
    import shapely

    out = OUT / "structures_cleanup"
    out.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(OUT / "structures_cleanup_candidates.csv")
    kw = gpd.read_file(
        DATA / "02_processed/structures/structures.gpkg", layer="kunstwerken"
    )
    kw = kw[
        kw["categorie"].isin(["brug", "kade_damwand", "steiger_afmeer", "sluis_stuw"])
    ]
    structs = gpd.GeoDataFrame(
        pd.concat([kribs[["geometry"]], kw[["geometry"]]], ignore_index=True), crs=28992
    )
    struct_sidx = structs.sindex
    dpy = {
        k: pd.read_parquet(
            DATA / f"03_features/loop/variants/{v}/dist_per_year.parquet"
        )
        for k, v in (("zonder", "s1-e8-nomask"), ("met", "s3-e8-newstructures"))
    }
    samples = caches.samples
    paths, titles = [], []
    for _, row in cand.iterrows():
        loc = row["location_id"]
        if loc not in ri.geometry.polygons.index:
            print("skip (no geometry):", loc)
            continue
        sgeom = ri.geometry.polygons[loc]
        near = structs.iloc[list(struct_sidx.intersection(sgeom.buffer(60).bounds))]
        near = near[near.intersects(sgeom.buffer(60))]
        buf = shapely.union_all(near.geometry.buffer(10).values) if len(near) else None
        s = samples[samples[LOCATION_ID] == loc]
        years = sorted(s["year"].unique())[-3:]
        s = s[s["year"].isin(years)]
        if buf is not None:
            inside = shapely.contains_xy(buf, s["x"].values, s["y"].values)
        else:
            inside = np.zeros(len(s), bool)
        sp = side_point(loc)
        title = (
            f"{loc} · v zonder {row.v_no_mask:+.1f} → v met {row.v_new_mask:+.1f} m/jr"
        )
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
        for ax, key, keep in (
            (axes[0], "zonder", np.ones(len(s), bool)),
            (axes[1], "met", ~inside),
        ):
            sub = (
                "zonder masker"
                if key == "zonder"
                else "met kribben + kunstwerken (10 m)"
            )
            _, cline, stats_r = base_axes(ax, loc, title=sub)
            draw_measured(ax, stats_r[stats_r["year"].isin(years)], lw=0.8)
            ss = s[keep]
            for yr in years:
                q = ss[ss["year"] == yr]
                ax.scatter(
                    q["x"],
                    q["y"],
                    s=4,
                    color=measured_color(int(yr)),
                    zorder=4,
                    linewidths=0,
                )
            if key == "met":
                for g in near.geometry:
                    for poly in getattr(g, "geoms", [g]):
                        ax.fill(
                            *poly.exterior.xy,
                            fc="#9a9a9a",
                            ec="#666",
                            lw=0.6,
                            alpha=0.75,
                            zorder=5,
                        )
                    for poly in getattr(g.buffer(10), "geoms", [g.buffer(10)]):
                        ax.plot(
                            *poly.exterior.xy,
                            color="#333",
                            lw=0.8,
                            ls=(0, (3, 2)),
                            zorder=5,
                        )
            d = dpy[key]
            d = d[(d[LOCATION_ID] == loc) & d["year"].isin(years)]
            if cline is not None:
                for _, r in d.iterrows():
                    pts = [
                        anchor(cline, st_, r["dist_m"], sp)
                        for st_ in np.linspace(0.03, 0.97, 14)
                    ]
                    xs, ys = zip(*pts, strict=True)
                    ax.plot(
                        xs,
                        ys,
                        color=measured_color(int(r["year"])),
                        lw=1.8,
                        ls=(0, (4, 3)),
                        zorder=6,
                    )
        fig.suptitle(title, fontsize=9, color=GREY)
        fig.tight_layout()
        p = out / f"{loc}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
        titles.append(title)
    contact_sheet(
        paths,
        titles,
        out / "_contact_sheet.png",
        ncol=4,
        suptitle="Structurenmasker · links zonder, rechts met (grijs = krib/kunstwerk, gestippeld = 10 m) · gestippelde kleurlijn = R=1 afstand",
    )
    return cand


if __name__ == "__main__":
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        gallery_r_sweep(sys.argv[2:])
        raise SystemExit
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        gallery_structures_cleanup()
        raise SystemExit
    if len(sys.argv) > 1 and sys.argv[1] == "single_line_height":
        print(gallery_single_line_height().round(2).to_string())
        raise SystemExit
    if len(sys.argv) > 1 and sys.argv[1] == "wavy":
        print(gallery_single_line_wavy().round(2).to_string())
        raise SystemExit
    p1 = gallery_single_line()
    print(p1.round(2).to_string())
    p2 = gallery_resolution()
    print(p2.round(2).to_string())
    gallery_r_sweep(list(p2.index[:3]))
    print("→", OUT)
