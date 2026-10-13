"""Bank-position observations.

A bank observation is one number: how far the bank sits from the centreline of
its scope region, at one point in time. ``src.pipeline.hybrid_prep`` reduces
the hybrid lines to these; the rest of the pipeline needs nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.sources.geometry import LOCATION_ID

#: Canonical observation schema.
OBSERVATION_COLUMNS = [
    LOCATION_ID,
    "date",
    "dist_m",
    "n_candidates",
    "n_selected",
    "source",
]


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
