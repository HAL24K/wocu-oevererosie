"""Visualization helpers for scope region plots.

All drawing functions accept pre-built lookups and data rather than global
state, making them importable and testable outside notebooks.

Typical usage in a notebook::

    from src.erosion.plot_utils import plot_regions, make_legend_handles

    plot_regions(
        loc_ids=selected,
        title='Scope regions: historical + predicted',
        cl_lookup=cl_lookup,
        scope_lookup=scope_lookup,
        bank_points=bank_points,
        legend_handles=make_legend_handles(
            pred_colors=pred_colors, pred_years=PRED_YEARS,
            show_vvr=True, show_signaleringslijn=True,
        ),
        draw_kwargs=dict(
            show_predictions=True,
            show_vvr=True,
            show_signaleringslijn=True,
            vvr=signalering,
            predicted_bank_positions=predicted_bank_positions,
        ),
    )
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.erosion.centerline_utils import (
    ensure_axes_list,
    flatten_geom_to_lines,
    offset_line_toward,
    parallel_line_from_vvr,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

YEAR_COLORS: list[str] = ["#f4d03f", "#a50026", "#4393c3"]
VVR_COLOR: str = "#7b2d8b"
SIGNALERINGSLIJN_COLOR: str = "#7b2d8b"  # purple, continuous line at back of VVR
STATUS_ALPHA: dict[str, float] = {"OK": 0.80, "UNCERTAIN": 0.40, "OUTLIER": 0.20}


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------

def make_legend_handles(
    year_colors: list[str] = YEAR_COLORS,
    pred_colors: Optional[list] = None,
    pred_years: Optional[list[int]] = None,
    n_points: int = 3,
    show_vvr: bool = False,
    show_signaleringslijn: bool = False,
) -> list:
    """Build standard legend handles for scope region plots."""
    handles = [
        plt.Line2D([0], [0], color="black", lw=2.5, label="centreline"),
        mpatches.Patch(fc="#f0f0f0", ec="#aaaaaa", label="scope boundary"),
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="grey", markeredgecolor="black",
            markersize=8, label=f"{n_points} furthest OK (selected)",
        ),
    ]
    for i, color in enumerate(year_colors):
        handles.append(mpatches.Patch(color=color, alpha=0.80, label=f"t{i+1} (hist)"))
    if pred_colors and pred_years:
        for idx in [0, len(pred_years) // 2, -1]:
            handles.append(
                mpatches.Patch(color=pred_colors[idx], label=f"{pred_years[idx]} (pred)")
            )
        handles.append(
            plt.Line2D(
                [0], [0], marker="s", color="w",
                markerfacecolor="gray", markeredgecolor="black",
                markersize=8, label="predicted",
            )
        )
    if show_vvr:
        handles.append(
            plt.Line2D([0], [0], color=VVR_COLOR, lw=2.5, label="VVR")
        )
    if show_signaleringslijn:
        handles.append(
            plt.Line2D(
                [0], [0], color=SIGNALERINGSLIJN_COLOR, lw=2,
                label="signaleringslijn",
            )
        )
    return handles


# ---------------------------------------------------------------------------
# Core draw function
# ---------------------------------------------------------------------------

def draw_region(
    ax,
    loc_id: str,
    *,
    cl_lookup: dict,
    scope_lookup: dict,
    bank_points: gpd.GeoDataFrame,
    col_idx: int = 0,
    n_points: int = 3,
    year_colors: list[str] = YEAR_COLORS,
    pred_years: Optional[list[int]] = None,
    pred_colors: Optional[list] = None,
    predicted_bank_positions: Optional[gpd.GeoDataFrame] = None,
    vvr: Optional[gpd.GeoDataFrame] = None,
    show_t1_to_t2_arrow: bool = True,
    show_predictions: bool = False,
    show_vvr: bool = False,
    show_bank_points: bool = True,
    show_signaleringslijn: bool = False,
    compact: bool = False,
) -> None:
    """Draw one scope region panel onto ``ax``."""
    cline = cl_lookup.get(loc_id)
    sgeom = scope_lookup.get(loc_id)

    _setup_ax(ax, loc_id, col_idx, compact)
    _draw_scope_and_cl(ax, sgeom, cline, compact)

    offset_lines = []
    if show_bank_points:
        offset_lines = _draw_bank_points(
            ax, loc_id, cline, bank_points, n_points, year_colors
        )
    if show_t1_to_t2_arrow:
        _draw_arrows(ax, offset_lines)
    if show_predictions and predicted_bank_positions is not None:
        _draw_predictions(ax, loc_id, predicted_bank_positions, pred_years, pred_colors)
    if (show_vvr or show_signaleringslijn) and vvr is not None:
        _draw_vvr(ax, sgeom, cline, vvr, show_signaleringslijn, compact)


def plot_regions(
    loc_ids: list[str],
    title: str,
    legend_handles: list,
    *,
    cl_lookup: dict,
    scope_lookup: dict,
    bank_points: gpd.GeoDataFrame,
    draw_kwargs: Optional[dict] = None,
    per_region_xlabel: Optional[Callable[[str], str]] = None,
    n_cols: Optional[int] = None,
    fig_w: float = 5.5,
    fig_h: float = 8.0,
    compact: bool = False,
) -> None:
    """Plot multiple scope regions in a grid.

    Args:
        loc_ids:           Location IDs to plot (one panel each).
        title:             Figure title.
        legend_handles:    List of legend artists (from ``make_legend_handles``).
        cl_lookup:         ``{location_id: LineString}`` centerline lookup.
        scope_lookup:      ``{location_id: Polygon}`` scope geometry lookup.
        bank_points:       Full bank points GeoDataFrame.
        draw_kwargs:       Extra kwargs forwarded to ``draw_region``.
        per_region_xlabel: Optional callable returning x-axis label per location.
        n_cols:            Number of columns (default: all in one row).
        fig_w / fig_h:     Panel width/height in inches (non-compact mode).
        compact:           Use small dense grid layout.
    """
    draw_kw = draw_kwargs or {}
    n = len(loc_ids)

    if compact:
        n_cols = n_cols or 10
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.0 * n_cols, 2.0 * n_rows),
            constrained_layout=True,
        )
    else:
        n_cols = n_cols or n
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w * n_cols, fig_h * n_rows),
            constrained_layout=True,
        )

    axes_flat = np.array(axes).flatten()

    for col, loc_id in enumerate(loc_ids):
        draw_region(
            axes_flat[col], loc_id,
            cl_lookup=cl_lookup,
            scope_lookup=scope_lookup,
            bank_points=bank_points,
            col_idx=col % n_cols,
            compact=compact,
            **draw_kw,
        )
        if per_region_xlabel:
            axes_flat[col].set_xlabel(per_region_xlabel(loc_id), fontsize=7.5)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 9),
        fontsize=8 if not compact else 9,
        bbox_to_anchor=(0.5, -0.02 if compact else -0.06),
        frameon=True,
    )
    fig.suptitle(title, fontsize=11, y=1.01 if compact else 1.02)
    plt.show()


# ---------------------------------------------------------------------------
# Segmented bank line drawing
# ---------------------------------------------------------------------------

def draw_region_segmented(
    ax,
    loc_id: str,
    *,
    cl_lookup: dict,
    scope_lookup: dict,
    bank_points: gpd.GeoDataFrame,
    n_segments: int = 10,
    n_points: int = 1,
    year_colors: list[str] = YEAR_COLORS,
    col_idx: int = 0,
    compact: bool = False,
) -> None:
    """Draw one scope region with bank positions split into N along-centerline segments.

    Instead of one mean offset line per year, the centerline is divided into
    ``n_segments`` equal-length sub-segments.  Bank points are assigned to a
    sub-segment based on where they project onto the centerline.  For each
    sub-segment × year the furthest ``n_points`` OK points are used to draw
    an independent short offset line, revealing along-channel variation in
    bank position that a single global mean would obscure.

    Args:
        ax:          Matplotlib axes to draw into.
        loc_id:      Location identifier.
        cl_lookup:   ``{location_id: LineString}`` centerline lookup.
        scope_lookup: ``{location_id: Polygon}`` scope geometry lookup.
        bank_points: Full bank points GeoDataFrame (all locations).
        n_segments:  Number of equal-length sub-segments to divide into.
        n_points:    Furthest N OK points per sub-segment used to compute mean dist.
        year_colors: Colors per survey year.
        col_idx:     Column index (used to label y-axis only on leftmost panel).
        compact:     Use compact font/line sizes.
    """
    from shapely.ops import substring

    cline = cl_lookup.get(loc_id)
    sgeom = scope_lookup.get(loc_id)

    _setup_ax(ax, loc_id, col_idx, compact)
    _draw_scope_and_cl(ax, sgeom, cline, compact)

    if cline is None:
        return

    grp = bank_points[bank_points["location_id"] == loc_id].copy()
    if grp.empty:
        return

    dates = sorted(grp["dtm_date"].unique())
    cl_len = cline.length
    seg_len = cl_len / n_segments

    for d_idx, date in enumerate(dates):
        color = year_colors[d_idx % len(year_colors)]
        yr_grp = grp[grp["dtm_date"] == date]

        # Draw all points (faded background)
        for status, sub in yr_grp.groupby("status"):
            ax.scatter(
                sub.geometry.x, sub.geometry.y,
                c=color, s=6,
                alpha=STATUS_ALPHA.get(status, 0.3),
                linewidths=0, zorder=3,
            )

        ok = yr_grp[yr_grp["status"] == "OK"].copy()
        if ok.empty:
            continue

        # Assign each OK point to a segment by projecting onto the centerline
        ok["_proj_frac"] = ok.geometry.apply(
            lambda pt: cline.project(pt, normalized=True)
        )
        ok["_seg"] = (ok["_proj_frac"] * n_segments).astype(int).clip(0, n_segments - 1)

        for seg_idx in range(n_segments):
            seg_pts = ok[ok["_seg"] == seg_idx]
            if seg_pts.empty:
                continue

            chosen = seg_pts.nlargest(n_points, "dist")
            mean_dist = chosen["dist"].mean()

            # Extract the sub-segment of the centerline
            start_m = seg_idx * seg_len
            end_m   = min((seg_idx + 1) * seg_len, cl_len)
            sub_cline = substring(cline, start_m, end_m)
            if sub_cline is None or sub_cline.is_empty or sub_cline.length < 0.1:
                continue

            # Highlight the selected furthest points
            ax.scatter(
                chosen.geometry.x, chosen.geometry.y,
                c=color, s=80, edgecolors="black", lw=0.6, zorder=8,
            )

            # Offset sub-segment toward the bank side
            offset_seg = offset_line_toward(sub_cline, mean_dist, chosen.geometry)
            if offset_seg is None or offset_seg.is_empty:
                continue

            ox, oy = offset_seg.xy
            ax.plot(ox, oy, color=color, lw=2.5, ls="-", zorder=6, alpha=0.90,
                    solid_capstyle="butt")

    # Year labels at top-right
    for d_idx, date in enumerate(dates):
        color = year_colors[d_idx % len(year_colors)]
        ax.plot([], [], color=color, lw=2.5,
                label=f"t{d_idx+1} ({date})")
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.7)


# ---------------------------------------------------------------------------
# Private drawing helpers
# ---------------------------------------------------------------------------

def _setup_ax(ax, loc_id: str, col_idx: int, compact: bool) -> None:
    fs, fst = (7, 5) if compact else (8.5, 6)
    ax.set_aspect("equal")
    ax.set_title(loc_id, fontsize=fs, pad=3 if compact else 5)
    ax.tick_params(labelsize=fst)
    if col_idx == 0:
        ax.set_ylabel("Northing (m RD)", fontsize=8)
    ax.set_xlabel("Easting (m RD)", fontsize=8)


def _draw_scope_and_cl(ax, sgeom, cline, compact: bool, padding: float = 50.0) -> None:
    lw_cl = 1.5 if compact else 2.5
    if sgeom is not None:
        bx, by = sgeom.exterior.xy
        ax.fill(bx, by, fc="#f0f0f0", ec="#aaaaaa",
                lw=0.8 if compact else 1.2, zorder=1)
        minx, miny, maxx, maxy = sgeom.bounds
        ax.set_xlim(minx - padding, maxx + padding)
        ax.set_ylim(miny - padding, maxy + padding)
    if cline is not None:
        xs, ys = cline.xy
        ax.plot(xs, ys, color="black", lw=lw_cl, zorder=5, solid_capstyle="round")
        ax.text(xs[-1], ys[-1], " CL", fontsize=7, color="black", va="center", zorder=6)


def _draw_bank_points(
    ax, loc_id: str, cline,
    bank_points: gpd.GeoDataFrame,
    n_points: int,
    year_colors: list[str],
) -> list:
    """Returns list of offset lines (one per date, None if missing)."""
    grp = bank_points[bank_points["location_id"] == loc_id].copy()
    offset_lines = []
    for d_idx, date in enumerate(sorted(grp["dtm_date"].unique())):
        color = year_colors[d_idx % len(year_colors)]
        yr_grp = grp[grp["dtm_date"] == date]
        for status, sub in yr_grp.groupby("status"):
            ax.scatter(
                sub.geometry.x, sub.geometry.y,
                c=color, s=8,
                alpha=STATUS_ALPHA.get(status, 0.3),
                linewidths=0, zorder=3,
            )
        ok_grp = yr_grp[yr_grp["status"] == "OK"]
        chosen = ok_grp.nlargest(n_points, "dist")
        if chosen.empty or cline is None:
            offset_lines.append(None)
            if cline is not None:
                mid = cline.interpolate(0.5, normalized=True)
                ax.text(mid.x, mid.y, f"{date}\nno OK pts", fontsize=5.5,
                        color=color, ha="center", va="bottom", zorder=9,
                        bbox=dict(fc="white", ec=color, alpha=0.7, pad=1, lw=0.8))
            continue
        mean_dist = chosen["dist"].mean()
        ax.scatter(chosen.geometry.x, chosen.geometry.y, c=color, s=110,
                   edgecolors="black", lw=0.8, zorder=8)
        offset_line = offset_line_toward(cline, mean_dist, chosen.geometry)
        if offset_line is not None and not offset_line.is_empty:
            ox, oy = offset_line.xy
            ax.plot(ox, oy, color=color, lw=2.2, ls="--", zorder=6, alpha=0.95)
            mid = offset_line.interpolate(0.5, normalized=True)
            ax.text(
                mid.x, mid.y,
                f"{date}\n{mean_dist:.1f} m  ({len(ok_grp)}/{len(yr_grp)} OK)",
                fontsize=5.5, color=color, ha="center", va="bottom", zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.65, pad=1),
            )
        offset_lines.append(
            offset_line if offset_line and not offset_line.is_empty else None
        )
    return offset_lines


def _draw_predictions(
    ax, loc_id: str,
    predicted_bank_positions: gpd.GeoDataFrame,
    pred_years: Optional[list[int]],
    pred_colors: Optional[list],
) -> None:
    if pred_years is None or pred_colors is None:
        return
    pred = predicted_bank_positions[
        (predicted_bank_positions["location_id"] == loc_id)
        & predicted_bank_positions.geometry.notna()
    ].sort_values("year")
    for _, row in pred.iterrows():
        if row["year"] in pred_years:
            idx = pred_years.index(row["year"])
            ax.scatter(
                row.geometry.x, row.geometry.y,
                c=[pred_colors[idx]], s=80,
                edgecolors="black", lw=0.6, zorder=9, marker="s",
            )


def _draw_arrows(ax, offset_lines: list, arrow_offset: float = 10.0) -> None:
    pairs = [(0, 1, "red", arrow_offset, 1), (1, 2, "#4393c3", arrow_offset * 2, -1)]
    for i, j, color, offset, side in pairs:
        if len(offset_lines) <= j or offset_lines[i] is None or offset_lines[j] is None:
            continue
        p1 = offset_lines[i].interpolate(0.5, normalized=True)
        p2 = offset_lines[j].interpolate(0.5, normalized=True)
        dx, dy = p2.x - p1.x, p2.y - p1.y
        dn = np.hypot(dx, dy)
        ux, uy = (-dy / dn * side, dx / dn * side) if dn > 1e-6 else (0, 0)
        ax.annotate(
            "", xy=(p2.x + offset * ux, p2.y + offset * uy),
            xytext=(p1.x + offset * ux, p1.y + offset * uy),
            arrowprops=dict(
                arrowstyle="->", color=color, lw=2,
                linestyle=":", mutation_scale=25,
            ),
        )


def _draw_vvr(
    ax, sgeom, cline,
    vvr: gpd.GeoDataFrame,
    show_signaleringslijn: bool,
    compact: bool,
) -> None:
    if sgeom is None:
        return
    buf = sgeom.buffer(10) if show_signaleringslijn else sgeom
    local_vvr = vvr[vvr.intersects(buf)]
    for _, row in local_vvr.iterrows():
        for part in flatten_geom_to_lines(row.geometry):
            try:
                ax.plot(
                    *part.xy, color=VVR_COLOR,
                    lw=1.5 if compact else 2.5,
                    zorder=7, solid_capstyle="round",
                )
            except (NotImplementedError, AttributeError):
                pass
    if show_signaleringslijn and not local_vvr.empty and cline is not None:
        vvr_clipped = local_vvr.geometry.union_all().intersection(sgeom)
        if vvr_clipped and not vvr_clipped.is_empty:
            pline = parallel_line_from_vvr(cline, vvr_clipped)
            if pline and not pline.is_empty:
                for part in flatten_geom_to_lines(pline):
                    try:
                        ax.plot(
                            *part.xy, color=SIGNALERINGSLIJN_COLOR,
                            lw=1.5, zorder=6, solid_capstyle="round",
                        )
                    except (NotImplementedError, AttributeError):
                        pass