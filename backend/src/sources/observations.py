"""Bank-position observations and the adapters that produce them.

A bank observation is one number: how far the bank sits from the centreline of
its scope region, at one point in time. Every delivery format the project has
seen reduces to that, and the rest of the pipeline needs nothing else.

The selection rule is shared deliberately. Both sources take the **mean of the
N furthest candidates** from the centreline, which is the conservative
worst-case convention the project settled on for the point cloud (see
``scripts/viz_point_selection.py``). Keeping lines on the same rule means the
hybrid delivery produces velocities on the same footing as the height model.

TODO: revisit the centreline-distance convention itself. The hybrid delivery
ships p50/p90 distance statistics per region, implying a different summary than
"furthest N". Changing it shifts every velocity in the model, so it is a
deliberate modelling decision rather than an implementation detail.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src.sources.geometry import LOCATION_ID, ScopeGeometry, normalise_location_id

logger = logging.getLogger(__name__)

#: Canonical observation schema.
OBSERVATION_COLUMNS = [
    LOCATION_ID,
    "date",
    "dist_m",
    "n_candidates",
    "n_selected",
    "source",
]

#: Default number of furthest candidates averaged into one observation.
DEFAULT_N_POINTS = 3

#: Points sampled along each hybrid bank line before selecting the furthest N.
DEFAULT_N_SAMPLES = 20


@dataclass(frozen=True)
class BankObservations:
    """Bank distances from the centreline, one row per (location_id, date).

    Attributes:
        frame: DataFrame with :data:`OBSERVATION_COLUMNS`, sorted by
            (location_id, date).
    """

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = set(OBSERVATION_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"BankObservations is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def location_ids(self) -> set[str]:
        return set(self.frame[LOCATION_ID].unique())

    def to_dist_per_year(self, within_year: str = "median") -> pd.DataFrame:
        """Collapse to the legacy per-year shape.

        Returns the exact columns, ordering and dtypes that
        ``src.pipeline.region_split.build_region_split`` expects:
        ``location_id, year, dist_m, n_ok_pts, n_selected``.

        Args:
            within_year: How to combine several observations falling in the same
                calendar year — ``"median"`` (default), ``"mean"``, ``"max"`` or
                ``"last"``.

        The point cloud has one observation per year, so this argument has no
        effect on it and the output is identical whatever is passed. The hybrid
        delivery has up to five surveys in a year for a region, so the choice
        matters there.

        The default is deliberately **not** ``"max"``. Averaging the furthest
        candidates *within one survey* is a conservative reading of where the
        bank is at that moment; taking the furthest *across separate surveys* is
        a different thing — it selects the noisiest survey of the year and
        propagates that noise into the year-on-year velocity. Measured on the
        20260710 delivery, ``max`` inflates the spread of the target ``v_test``
        from 18.4 to 26.2 m/yr.

        TODO: collapsing to a calendar year discards most of the hybrid's
        temporal resolution (5,785 regions have four or more surveys). Carrying
        dates through and fitting on the full series is the change most likely
        to improve accuracy, but it means generalising the t1/t2/t3 triple in
        region_split, so it is deliberately out of scope here.
        """
        allowed = {"median", "mean", "max", "last"}
        if within_year not in allowed:
            raise ValueError(
                f"within_year must be one of {sorted(allowed)}, got {within_year!r}"
            )

        df = self.frame.copy()
        df["year"] = pd.to_datetime(df["date"]).dt.year.astype("int64")
        grouped = df.groupby([LOCATION_ID, "year"], sort=False)

        if within_year == "last":
            dist = grouped.apply(
                lambda g: g.sort_values("date")["dist_m"].iloc[-1], include_groups=False
            ).rename("dist_m")
        else:
            dist = grouped["dist_m"].agg(within_year)

        out = dist.reset_index().merge(
            grouped.agg(
                n_ok_pts=("n_candidates", "sum"), n_selected=("n_selected", "sum")
            ).reset_index(),
            on=[LOCATION_ID, "year"],
        )

        return (
            out[[LOCATION_ID, "year", "dist_m", "n_ok_pts", "n_selected"]]
            .sort_values([LOCATION_ID, "year"])
            .reset_index(drop=True)
            .astype(
                {
                    "year": "int64",
                    "dist_m": "float64",
                    "n_ok_pts": "float64",
                    "n_selected": "float64",
                }
            )
        )


class BankObservationSource(ABC):
    """A delivery format that can yield :class:`BankObservations`."""

    @abstractmethod
    def load(self) -> BankObservations:
        """Read the delivery and reduce it to bank observations."""


class HeightModelPointSource(BankObservationSource):
    """The ``punten_oever`` point cloud shipped with the WOCU deliveries.

    Each point already carries ``dist``, its distance from the centreline, so no
    geometry work is needed — only filtering and aggregation.

    Args:
        gpkg: GeoPackage containing the point layer.
        layer: Layer name.
        n_points: Number of furthest OK points averaged per (region, year).
        status_ok: Value of ``status`` marking a usable point.

    Note:
        ``dtm_date`` holds an integer **year**, not a date, so observations are
        dated to 1 January of that year — matching how the hybrid delivery
        represents the same height-model surveys.
    """

    SOURCE = "hoogtemodel"

    def __init__(
        self,
        gpkg: Path,
        layer: str = "punten_oever",
        n_points: int = DEFAULT_N_POINTS,
        status_ok: str = "OK",
    ) -> None:
        self.gpkg = Path(gpkg)
        self.layer = layer
        self.n_points = n_points
        self.status_ok = status_ok

    def load(self) -> BankObservations:
        pts = normalise_location_id(gpd.read_file(self.gpkg, layer=self.layer))
        ok = pts[pts["status"] == self.status_ok].copy()
        ok["year"] = ok["dtm_date"].astype(int)
        logger.info(
            "%s: %d points, %d with status %r",
            self.gpkg.name,
            len(pts),
            len(ok),
            self.status_ok,
        )

        agg = (
            ok.groupby([LOCATION_ID, "year"], group_keys=False)
            .apply(self._top_n_mean, n=self.n_points, include_groups=False)
            .reset_index()
        )

        agg["date"] = pd.to_datetime(agg["year"].astype(str) + "-01-01")
        agg["source"] = self.SOURCE
        frame = (
            agg[OBSERVATION_COLUMNS]
            .sort_values([LOCATION_ID, "date"])
            .reset_index(drop=True)
        )
        return BankObservations(frame)

    @staticmethod
    def _top_n_mean(group: pd.DataFrame, n: int) -> pd.Series:
        """Aggregate one (location_id, year) group into a single distance."""
        chosen = group.nlargest(n, "dist")
        return pd.Series(
            {
                "dist_m": chosen["dist"].mean(),
                "n_candidates": len(group),
                "n_selected": len(chosen),
            }
        )


class HybridLineSource(BankObservationSource):
    """The ``lines`` layer of the phase-2 hybrid delivery.

    Bank position arrives as a LineString with no distance attribute, so the
    distance is measured here: each line is sampled at evenly spaced points, the
    distance from each sample to the region's centreline is computed, and the
    furthest ``n_points`` are averaged. Several lines may describe one region on
    one date; their samples are pooled before selection.

    ``model_preference`` is **not** applied — the delivery has already resolved
    it, with every region's lines coming from a single model. The ``model``
    column is carried through as provenance in ``source``.

    Args:
        gpkg: The hybrid results GeoPackage.
        geometry: Centreline lookup. Regions absent from it cannot be measured.
        layer: Layer name holding the bank lines.
        n_points: Number of furthest samples averaged per (region, date).
        n_samples: Points sampled along each line.
    """

    def __init__(
        self,
        gpkg: Path,
        geometry: ScopeGeometry,
        layer: str = "lines",
        n_points: int = DEFAULT_N_POINTS,
        n_samples: int = DEFAULT_N_SAMPLES,
    ) -> None:
        self.gpkg = Path(gpkg)
        self.geometry = geometry
        self.layer = layer
        self.n_points = n_points
        self.n_samples = n_samples
        self.skipped_no_centreline: set[str] = set()

    def load(self) -> BankObservations:
        lines = normalise_location_id(gpd.read_file(self.gpkg, layer=self.layer))
        lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]

        centrelines = self.geometry.centrelines
        self.skipped_no_centreline = self.geometry.missing(lines[LOCATION_ID])
        if self.skipped_no_centreline:
            logger.warning(
                "%d of %d regions have no centreline and are skipped "
                "(mostly side channels absent from the phase-2 scope file)",
                len(self.skipped_no_centreline),
                lines[LOCATION_ID].nunique(),
            )
        usable = lines[~lines[LOCATION_ID].isin(self.skipped_no_centreline)].copy()
        if usable.empty:
            return BankObservations(pd.DataFrame(columns=OBSERVATION_COLUMNS))

        samples = self._sample_distances(usable, centrelines)
        return BankObservations(self._aggregate(samples))

    def _sample_distances(
        self, lines: gpd.GeoDataFrame, centrelines: gpd.GeoSeries
    ) -> pd.DataFrame:
        """Distance from each sampled point on each line to its centreline."""
        fractions = np.linspace(0.0, 1.0, self.n_samples)

        line_geoms = np.repeat(lines.geometry.values, self.n_samples)
        cline_geoms = np.repeat(
            centrelines.reindex(lines[LOCATION_ID]).values, self.n_samples
        )
        fracs = np.tile(fractions, len(lines))

        pts = shapely.line_interpolate_point(line_geoms, fracs, normalized=True)
        dist = shapely.distance(pts, cline_geoms)

        return pd.DataFrame(
            {
                LOCATION_ID: np.repeat(lines[LOCATION_ID].values, self.n_samples),
                "date": np.repeat(pd.to_datetime(lines["date"]).values, self.n_samples),
                "source": np.repeat(
                    lines.get("model", pd.Series("hybrid", index=lines.index)).values,
                    self.n_samples,
                ),
                "dist": dist,
            }
        ).dropna(subset=["dist"])

    def _aggregate(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Mean of the furthest n_points samples per (location_id, date)."""
        grouped = samples.groupby([LOCATION_ID, "date"], sort=False)

        top = (
            grouped["dist"]
            .nlargest(self.n_points)
            .groupby(level=[0, 1], sort=False)
            .mean()
            .rename("dist_m")
            .reset_index()
        )
        counts = grouped.agg(
            n_candidates=("dist", "size"), source=("source", "first")
        ).reset_index()

        out = top.merge(counts, on=[LOCATION_ID, "date"], how="left")
        out["n_selected"] = np.minimum(out["n_candidates"], self.n_points).astype(
            "float64"
        )
        out["n_candidates"] = out["n_candidates"].astype("float64")

        return (
            out[OBSERVATION_COLUMNS]
            .sort_values([LOCATION_ID, "date"])
            .reset_index(drop=True)
        )


def combine(*observations: BankObservations) -> BankObservations:
    """Concatenate observations from several sources.

    Later sources win where a (location_id, date) appears more than once, which
    lets a newer delivery override an older one for the regions it covers.
    """
    frames = [o.frame for o in observations if len(o)]
    if not frames:
        return BankObservations(pd.DataFrame(columns=OBSERVATION_COLUMNS))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=[LOCATION_ID, "date"], keep="last")
    return BankObservations(
        merged.sort_values([LOCATION_ID, "date"]).reset_index(drop=True)
    )
