"""On-demand inspection of a single scope region in the hybrid delivery.

One figure per region, two panels:

* **map** — the scope polygon (tinted by which model the delivery preferred),
  the centreline, every delivered bank line coloured by the QGIS year ramp
  (darker green = newer), the signaleringslijn in purple, and predicted bank
  positions as brown squares (darker brown = later);
* **time series** — per-line median distance to the centreline over time.
  Multiple markers on one date are the delivery's multiple lines for that
  survey; a black dashed line shows the single value the pipeline collapsed
  them to, so a far-bank artefact is visible both as a bimodal gap and as the
  jump it caused in the modelled series.

The colours follow the QGIS styling contract: within any family darker means
later, greens are measured history, browns are predictions, purple is the VVR
family. Same year, same colour, everywhere.

Unlike ``src.sources.observations.HybridLineSource`` — which pools all lines of
a (region, date) before selecting the furthest samples — everything here works
per line, which is what makes the far-bank artefact visible at all.

Typical usage in a notebook::

    from src.erosion.region_inspector import RegionInspector

    ri = RegionInspector()                 # defaults to the 20260819-hybrid run
    ri.farbank_candidates().head(20)       # ranked suspect regions
    ri.inspect("rijn_l_5440_5450")         # one figure
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from src.pipeline.config import ExperimentConfig
from src.sources.geometry import (
    DEFAULT_CRS,
    LOCATION_ID,
    ScopeGeometry,
    normalise_location_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling contract (mirrors the QGIS projects)
# ---------------------------------------------------------------------------

#: Measured bank lines: shared green time ramp, darker = newer.
YEAR_CLASS_GREENS: dict[str, str] = {
    "2015–2017": "#a8d5a0",
    "2020–2022": "#79bd74",
    "2023": "#6cbb68",
    "2024": "#41a047",
    "2025": "#1f7d2f",
    "2026": "#0a5c20",
}

#: Predicted bank positions: brown ramp, darker = later.
PREDICTION_BROWNS: dict[int, str] = {
    2026: "#d9b28c",
    2027: "#c08a5a",
    2028: "#a3663a",
    2029: "#7e4622",
    2030: "#582c10",
}

#: Scope-polygon fill per preferred model (pale tints of the report colours).
MODEL_FILL: dict[str, str] = {
    "hoogtemodel": "#d6e5f7",
    "segmentation": "#fbe0d4",
}

VVR_PURPLE = "#7b2d8b"
PIPELINE_BLACK = "#333333"


def year_class(year: int) -> str:
    """Map a survey year onto the QGIS legend class."""
    if year <= 2017:
        return "2015–2017"
    if year <= 2022:
        return "2020–2022"
    return str(year)


def measured_color(year: int) -> str:
    """Green for a measured survey year (darker = newer)."""
    return YEAR_CLASS_GREENS.get(year_class(year), "#0a5c20")


def predicted_color(year: int) -> str:
    """Brown for a predicted year (darker = later, clamped past 2030)."""
    if year in PREDICTION_BROWNS:
        return PREDICTION_BROWNS[year]
    return "#582c10" if year > 2030 else "#d9b28c"


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


class RegionInspector:
    """Per-region view of the hybrid delivery and one experiment's predictions.

    Args:
        cfg: Experiment whose predictions and collapsed distances are shown.
            Defaults to ``ExperimentConfig(experiment="20260819-hybrid")``.
        hybrid_gpkg: The hybrid delivery. Defaults to the 20260710 file under
            the experiment's data directory.
        n_samples: Points sampled along each line when measuring its distance
            to the centreline (matches ``HybridLineSource``).

    Notes:
        Everything loads lazily; building the object is cheap. The first call
        that needs line distances samples every line in the delivery once
        (~2M point distances) and caches the result.
    """

    def __init__(
        self,
        cfg: ExperimentConfig | None = None,
        hybrid_gpkg: Path | None = None,
        n_samples: int = 20,
    ) -> None:
        self.cfg = cfg or ExperimentConfig(experiment="20260819-hybrid")
        self.hybrid_gpkg = Path(
            hybrid_gpkg
            or self.cfg.data_dir
            / "02_processed/hybrid/hybrid_model_results_20260710.gpkg"
        )
        self.n_samples = n_samples
        self.geometry = ScopeGeometry(centreline_gpkg=self.cfg.raw_gpkg)
        self._lines: gpd.GeoDataFrame | None = None
        self._line_stats: pd.DataFrame | None = None
        self._predictions: gpd.GeoDataFrame | None = None
        self._signalering: gpd.GeoDataFrame | None = None
        self._dist_per_year: pd.DataFrame | None = None

    # ── lazy inputs ─────────────────────────────────────────────────────────

    @property
    def lines(self) -> gpd.GeoDataFrame:
        """Every bank line in the delivery, one row per line."""
        if self._lines is None:
            gdf = normalise_location_id(gpd.read_file(self.hybrid_gpkg, layer="lines"))
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
            gdf["date"] = pd.to_datetime(gdf["date"])
            gdf["year"] = gdf["date"].dt.year
            self._lines = gdf
            logger.info("Loaded %d bank lines", len(gdf))
        return self._lines

    @property
    def line_stats(self) -> pd.DataFrame:
        """Per-line distance summary: location_id, date, year, model, p50, max.

        The per-line median (not the furthest sample) is the honest "where is
        this line" number; the far-bank artefact separates cleanly on it.
        """
        if self._line_stats is None:
            usable = self.lines[
                self.lines[LOCATION_ID].isin(self.geometry.centrelines.index)
            ]
            self._line_stats = self._sample_line_distances(usable)
            logger.info(
                "Measured %d lines across %d regions",
                len(self._line_stats),
                self._line_stats[LOCATION_ID].nunique(),
            )
        return self._line_stats

    @property
    def predictions(self) -> gpd.GeoDataFrame:
        """Predicted bank positions of the experiment (point per region-year)."""
        if self._predictions is None:
            self._predictions = gpd.read_file(
                self.cfg.output_gpkg, layer="predicted_bank_positions"
            )
        return self._predictions

    @property
    def signalering(self) -> gpd.GeoDataFrame:
        if self._signalering is None:
            self._signalering = gpd.read_file(self.cfg.signalering_gpkg)
        return self._signalering

    @property
    def dist_per_year(self) -> pd.DataFrame:
        """The collapsed per-year series the pipeline actually modelled on."""
        if self._dist_per_year is None:
            self._dist_per_year = pd.read_parquet(
                self.cfg.features_dir / "dist_per_year.parquet"
            )
        return self._dist_per_year

    # ── distance sampling ───────────────────────────────────────────────────

    def _sample_line_distances(self, lines: gpd.GeoDataFrame) -> pd.DataFrame:
        """Median/max distance to the centreline for each individual line."""
        fractions = np.linspace(0.0, 1.0, self.n_samples)
        line_geoms = np.repeat(lines.geometry.values, self.n_samples)
        cline_geoms = np.repeat(
            self.geometry.centrelines.reindex(lines[LOCATION_ID]).values,
            self.n_samples,
        )
        pts = shapely.line_interpolate_point(
            line_geoms, np.tile(fractions, len(lines)), normalized=True
        )
        dist = shapely.distance(pts, cline_geoms).reshape(len(lines), self.n_samples)

        return pd.DataFrame(
            {
                LOCATION_ID: lines[LOCATION_ID].values,
                "date": lines["date"].values,
                "year": lines["year"].values,
                "model": lines["model"].values,
                "dist_p50": np.nanmedian(dist, axis=1),
                "dist_max": np.nanmax(dist, axis=1),
            },
            index=lines.index,
        ).sort_values([LOCATION_ID, "date"])

    # ── far-bank screening ──────────────────────────────────────────────────

    def farbank_candidates(
        self, min_gap_m: float = 25.0, min_ratio: float = 1.5, v_limit: float = 50.0
    ) -> pd.DataFrame:
        """Rank regions by evidence of the far-bank artefact.

        Two independent signatures are scored:

        * ``max_multiline_gap_m`` — on a single date with several lines, the
          spread between the nearest and furthest line's median distance. A
          large gap with a large ratio means two different banks were drawn
          for one survey.
        * ``max_abs_v`` — the largest implied year-on-year velocity of the
          per-date series (nearest line per date), which catches the
          intermittent case where a whole survey is on the wrong bank.

        Returns one row per flagged region, worst first. ``flag`` says which
        signature(s) fired.
        """
        stats = self.line_stats

        per_date = stats.groupby([LOCATION_ID, "date"])["dist_p50"].agg(
            ["min", "max", "count"]
        )
        per_date["gap"] = per_date["max"] - per_date["min"]
        per_date["ratio"] = per_date["max"] / per_date["min"].clip(lower=0.1)
        gap = per_date.groupby(LOCATION_ID).agg(
            n_dates=("gap", "size"),
            n_multiline_dates=("count", lambda c: int((c > 1).sum())),
            max_multiline_gap_m=("gap", "max"),
            max_multiline_ratio=("ratio", "max"),
        )

        # Collapse to one value per year before differencing: surveys days
        # apart would otherwise turn metres of noise into thousands of m/yr.
        series = per_date["min"].reset_index()
        series["year"] = series["date"].dt.year
        yearly = (
            series.groupby([LOCATION_ID, "year"])["min"]
            .median()
            .reset_index()
            .sort_values([LOCATION_ID, "year"])
        )
        g = yearly.groupby(LOCATION_ID)
        v = g["min"].diff() / g["year"].diff()
        gap["max_abs_v"] = v.abs().groupby(yearly[LOCATION_ID]).max()
        gap["model"] = stats.groupby(LOCATION_ID)["model"].first()

        multi = (gap["max_multiline_gap_m"] > min_gap_m) & (
            gap["max_multiline_ratio"] > min_ratio
        )
        jump = gap["max_abs_v"] > v_limit
        gap["flag"] = ""
        gap.loc[multi & ~jump, "flag"] = "multiline_gap"
        gap.loc[~multi & jump, "flag"] = "temporal_jump"
        gap.loc[multi & jump, "flag"] = "multiline_gap+temporal_jump"

        out = gap[gap["flag"] != ""].copy()
        out["score"] = out["max_multiline_gap_m"].fillna(0) + out["max_abs_v"].fillna(0)
        return out.sort_values("score", ascending=False)

    # ── the figure ──────────────────────────────────────────────────────────

    def inspect(
        self,
        loc_id: str,
        pred_years: tuple[int, ...] = (2026, 2027, 2028, 2029, 2030),
        figsize: tuple[float, float] = (13.5, 8.0),
        save: Path | None = None,
        basemap: bool = True,
    ) -> plt.Figure:
        """Draw the two-panel inspection figure for one region.

        ``basemap=True`` puts OpenStreetMap tiles behind the map panel
        (needs network the first time; tiles are cached). If the tiles
        cannot be fetched the figure is drawn without them.
        """
        stats = self.line_stats[self.line_stats[LOCATION_ID] == loc_id]
        if stats.empty:
            raise KeyError(f"{loc_id}: no measurable lines in the hybrid delivery")

        fig, (ax_map, ax_ts) = plt.subplots(
            1, 2, figsize=figsize, width_ratios=[1.0, 1.15], constrained_layout=True
        )
        model = stats["model"].iloc[0]
        fig.suptitle(f"{loc_id}   ·   preferred model: {model}", fontsize=12)

        self._draw_map(ax_map, loc_id, stats, model, pred_years, basemap)
        self._draw_timeseries(ax_ts, loc_id, stats)

        if save is not None:
            fig.savefig(save, dpi=140, bbox_inches="tight")
        return fig

    def _draw_map(self, ax, loc_id, stats, model, pred_years, basemap) -> None:
        sgeom = self.geometry.polygons.get(loc_id)
        cline = self.geometry.centrelines.get(loc_id)

        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Easting (m RD)", fontsize=8)
        ax.set_ylabel("Northing (m RD)", fontsize=8)

        if sgeom is not None:
            bx, by = sgeom.exterior.xy
            # translucent over a basemap (the QGIS opacity trick), solid without
            ax.fill(
                bx,
                by,
                fc=MODEL_FILL.get(model, "#eeeeee"),
                ec="#aaaaaa",
                lw=1.0,
                alpha=0.45 if basemap else 1.0,
                zorder=1,
            )
            minx, miny, maxx, maxy = sgeom.bounds
            ax.set_xlim(minx - 50, maxx + 50)
            ax.set_ylim(miny - 50, maxy + 50)
        if basemap:
            self._add_basemap(ax)
        if cline is not None:
            xs, ys = cline.xy
            ax.plot(xs, ys, color="black", lw=2.2, zorder=5)
            ax.text(xs[-1], ys[-1], " CL", fontsize=7, va="center", zorder=6)

        # measured lines, green ramp; labels spread along the channel so
        # overlapping lines don't stack their labels on one spot
        region_lines = self.lines.loc[stats.index]
        n = len(region_lines)
        year_median = stats.groupby("year")["dist_p50"].transform("median")
        labelled_years: set[int] = set()
        for i, (idx, row) in enumerate(region_lines.iterrows()):
            color = measured_color(row["year"])
            for part in _line_parts(row.geometry):
                ax.plot(*part.xy, color=color, lw=2.0, alpha=0.9, zorder=4)
            # label one line per year, plus any line far from its year's median
            deviant = abs(stats.loc[idx, "dist_p50"] - year_median.loc[idx]) > 15
            if row["year"] in labelled_years and not deviant:
                continue
            labelled_years.add(row["year"])
            frac = 0.5 if n == 1 else 0.15 + 0.7 * i / (n - 1)
            mid = row.geometry.interpolate(frac, normalized=True)
            ax.annotate(
                f"{row['year']} · {stats.loc[idx, 'dist_p50']:.0f} m",
                (mid.x, mid.y),
                fontsize=5.5,
                color=color,
                ha="center",
                va="bottom",
                zorder=9,
                bbox={"fc": "white", "ec": "none", "alpha": 0.65, "pad": 1},
            )

        # signaleringslijn, purple
        if sgeom is not None:
            local = self.signalering[self.signalering.intersects(sgeom.buffer(10))]
            for geom in local.geometry:
                for part in _line_parts(geom):
                    ax.plot(*part.xy, color=VVR_PURPLE, lw=2.0, zorder=6)

        # predictions, brown squares
        pred = self.predictions[
            (self.predictions[LOCATION_ID] == loc_id)
            & self.predictions["year"].isin(pred_years)
            & self.predictions.geometry.notna()
        ].sort_values("year")
        for _, row in pred.iterrows():
            ax.scatter(
                row.geometry.x,
                row.geometry.y,
                c=[predicted_color(int(row["year"]))],
                marker="s",
                s=55,
                edgecolors="black",
                lw=0.5,
                zorder=8,
            )

        ax.legend(
            handles=self._map_legend_handles(
                model, stats["year"].unique(), pred["year"].astype(int).unique()
            ),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.09),
            ncol=3,
            fontsize=6,
            framealpha=0.9,
            handlelength=1.6,
            columnspacing=1.0,
        )

    @staticmethod
    def _map_legend_handles(model, measured_years, pred_years) -> list:
        """QGIS-style legend: one swatch per year class actually drawn."""
        import matplotlib.patches as mpatches

        handles = [
            plt.Line2D([0], [0], color="black", lw=2.2, label="centreline"),
            mpatches.Patch(
                fc=MODEL_FILL.get(model, "#eeeeee"),
                ec="#aaaaaa",
                label=f"scope region ({model})",
            ),
        ]
        seen: list[str] = []
        for year in sorted(measured_years):
            cls = year_class(year)
            if cls not in seen:
                seen.append(cls)
                handles.append(
                    plt.Line2D([0], [0], color=measured_color(year), lw=2.0, label=cls)
                )
        for year in sorted(pred_years):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    ls="none",
                    markerfacecolor=predicted_color(year),
                    markeredgecolor="black",
                    markersize=6,
                    label=f"{year} (pred)",
                )
            )
        handles.append(
            plt.Line2D([0], [0], color=VVR_PURPLE, lw=2.0, label="signaleringslijn")
        )
        return handles

    @staticmethod
    def _add_basemap(ax) -> None:
        """OpenStreetMap tiles behind the map panel; skipped when offline.

        OSM's tile policy requires an identifying user agent (contextily's
        random default gets a 403). If OSM still refuses, fall back to the
        visually similar CartoDB Voyager tiles.
        """
        try:
            import contextily as ctx
            import contextily.tile

            contextily.tile.USER_AGENT = "wocu-oevererosie/1.0 (erosion POC)"
        except ImportError as exc:
            logger.warning("basemap skipped: %s", exc)
            return
        for provider in (
            ctx.providers.OpenStreetMap.Mapnik,
            ctx.providers.CartoDB.Voyager,
        ):
            try:
                ctx.add_basemap(
                    ax,
                    crs=f"EPSG:{DEFAULT_CRS}",
                    source=provider,
                    attribution_size=4,
                    zorder=0,
                )
                return
            except Exception as exc:  # no network or tile server refused
                logger.warning("basemap %s failed: %s", provider.get("name"), exc)

    def _draw_timeseries(self, ax, loc_id, stats) -> None:
        ax.set_xlabel("survey date", fontsize=8)
        ax.set_ylabel("distance to centreline (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)

        # vertical connectors where one date carries several lines
        connector_labelled = False
        for date, grp in stats.groupby("date"):
            if len(grp) > 1:
                ax.plot(
                    [date, date],
                    [grp["dist_p50"].min(), grp["dist_p50"].max()],
                    color="#999999",
                    lw=1.0,
                    zorder=2,
                    label=None if connector_labelled else "several lines, one survey",
                )
                connector_labelled = True
        ax.scatter(
            stats["date"],
            stats["dist_p50"],
            c=[measured_color(y) for y in stats["year"]],
            s=45,
            edgecolors="black",
            lw=0.4,
            zorder=4,
            label="delivered line (p50)",
        )

        # what the pipeline modelled on
        dpy = self.dist_per_year
        dpy = dpy[dpy[LOCATION_ID] == loc_id].sort_values("year")
        if not dpy.empty:
            dates = pd.to_datetime(dpy["year"].astype(str) + "-07-01")
            ax.plot(
                dates,
                dpy["dist_m"],
                color=PIPELINE_BLACK,
                ls="--",
                lw=1.4,
                marker="o",
                ms=3.5,
                zorder=3,
                label="pipeline series (furthest-3 pooled)",
            )
        else:
            ax.set_title(
                "not in the pipeline run (far-bank velocity filter)",
                fontsize=8,
                color="#a33",
            )

        # predictions
        pred = self.predictions[self.predictions[LOCATION_ID] == loc_id].sort_values(
            "year"
        )
        if not pred.empty:
            dates = pd.to_datetime(pred["year"].astype(str) + "-07-01")
            ax.scatter(
                dates,
                pred["predicted_dist_m"],
                c=[predicted_color(int(y)) for y in pred["year"]],
                marker="s",
                s=30,
                edgecolors="black",
                lw=0.4,
                zorder=5,
                label="predicted",
            )

        # VVR threshold: distance from the centreline midpoint to the nearest
        # signaleringslijn near the region — approximate, but on the same axis
        # as everything else in this panel.
        cline = self.geometry.centrelines.get(loc_id)
        sgeom = self.geometry.polygons.get(loc_id)
        if cline is not None and sgeom is not None:
            local = self.signalering[self.signalering.intersects(sgeom.buffer(10))]
            if not local.empty:
                mid = cline.interpolate(0.5, normalized=True)
                ax.axhline(
                    local.distance(mid).min(),
                    color=VVR_PURPLE,
                    ls=":",
                    lw=1.5,
                    zorder=2,
                    label="signaleringslijn (≈ at region mid)",
                )

        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.legend(fontsize=7, loc="best", framealpha=0.8)


def _line_parts(geom):
    """Yield plottable LineStrings from any geometry (polygons → their rings)."""
    if geom is None or geom.is_empty:
        return
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from _line_parts(g)
    elif geom.geom_type == "Polygon":
        yield geom.exterior
    elif geom.geom_type in ("LineString", "LinearRing"):
        yield geom
