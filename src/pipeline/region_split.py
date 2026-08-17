"""Step 02 — Region split.

Turns the per-year distance table into per-region split records and separates
locations into three groups:

  train_test      (3 timestamps) → region_split output
  inference_only  (2 timestamps) → region_inference_only output
  missing_data    (1 timestamp)  → dropped

Also:
  - applies quality filter from summary_scope
  - joins is_nvo flag via spatial intersection with vvr_rates_of_change
  - computes erosion volume rates from erosion_vlakken_filtered
  - performs stratified 80/20 train/test split
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split

from src.sources.geometry import normalise_location_id

CLUSTERS = ["ijssel1", "ijssel2", "maas1", "maas2", "maas3", "rijn", "nederrijn"]

QUALITY_COL = "estimate_reliability_height_model"

SPLIT_COLS = [
    "cluster",
    "n_timestamps",
    "t1",
    "t2",
    "t3",
    "train_span_yr",
    "test_span_yr",
    "dist_t1",
    "dist_t2",
    "dist_t3",
    "v_train",
    "v_test",
    "is_nvo",
    "quality",
    "split",
    "erosion_vol_train_rate",
    "erosion_vol_test_rate",
    "erosion_vol_rate_t1",
]

INFERENCE_COLS = [
    "cluster",
    "n_timestamps",
    "t1",
    "t2",
    "train_span_yr",
    "dist_t1",
    "dist_t2",
    "v_train",
    "is_nvo",
    "quality",
]


def build_region_split(
    dist_per_year: pd.DataFrame,
    proc_gpkg: Path,
    test_size: float = 0.20,
    random_seed: int = 42,
    quality_col: str = QUALITY_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build region_split and region_inference_only from dist_per_year.

    Args:
        dist_per_year: Output of ``compute_bank_distances`` — (location_id, year, dist_m, ...).
        proc_gpkg: Processed GeoPackage with layers: summary_scope, vvr_rates_of_change,
                   erosion_vlakken_filtered.
        test_size: Fraction for the held-out test set (stratified by cluster).
        random_seed: Random seed for reproducibility.
        quality_col: Column in summary_scope used as quality gate (keep == 'OK').

    Returns:
        (region_split, region_inference_only) — both indexed by location_id.
    """
    ts_counts = dist_per_year.groupby("location_id")["year"].count()

    locs_2ts = ts_counts[ts_counts == 2].index
    eligible = ts_counts[ts_counts >= 3].index

    # ── Build pivot records ───────────────────────────────────────────────────
    split_records = (
        dist_per_year[dist_per_year["location_id"].isin(eligible)]
        .groupby("location_id")
        .apply(_build_split_row, include_groups=False)
    )
    inference_records = (
        dist_per_year[dist_per_year["location_id"].isin(locs_2ts)]
        .groupby("location_id")
        .apply(_build_split_row, include_groups=False)
    )

    # ── Quality filter ────────────────────────────────────────────────────────
    scope_quality = gpd.read_file(proc_gpkg, layer="summary_scope")
    scope_quality = normalise_location_id(scope_quality)
    # 20260330+ files have duplicate rows per location_id (one per type_oever segment);
    # deduplicate before building the quality map — quality value is the same across dupes.
    scope_quality = scope_quality.drop_duplicates(subset=["location_id"])
    quality_map = scope_quality.set_index("location_id")[quality_col]

    split_records["quality"] = split_records.index.map(quality_map)
    features_ok = split_records[split_records["quality"] == "OK"].copy()

    inference_records["quality"] = inference_records.index.map(quality_map)
    inference_ok = inference_records[inference_records["quality"] == "OK"].copy()

    # ── is_nvo via spatial join ───────────────────────────────────────────────
    vvr_polys = gpd.read_file(proc_gpkg, layer="vvr_rates_of_change")
    scope_geom = normalise_location_id(scope_quality)
    scope_ok = scope_geom[scope_geom["location_id"].isin(features_ok.index)][
        ["location_id", "geometry"]
    ].to_crs(vvr_polys.crs)
    nvo_ids = set(
        gpd.sjoin(
            scope_ok, vvr_polys[["geometry"]], how="inner", predicate="intersects"
        )["location_id"]
    )
    features_ok["is_nvo"] = features_ok.index.isin(nvo_ids)
    inference_ok["is_nvo"] = inference_ok.index.isin(nvo_ids)

    # ── Erosion volume rates ──────────────────────────────────────────────────
    features_ok = _join_erosion_volume(features_ok, proc_gpkg)

    # ── Stratified train/test split ───────────────────────────────────────────
    train_idx, test_idx = train_test_split(
        features_ok.index,
        test_size=test_size,
        random_state=random_seed,
        stratify=features_ok["cluster"],
    )
    features_ok["split"] = "train"
    features_ok.loc[test_idx, "split"] = "test"

    # ── Select output columns ─────────────────────────────────────────────────
    region_split = features_ok[[c for c in SPLIT_COLS if c in features_ok.columns]]
    region_inference = inference_ok[
        [c for c in INFERENCE_COLS if c in inference_ok.columns]
    ]

    return region_split, region_inference


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_cluster(loc_id: str) -> str:
    """Map a location_id to its cluster name based on prefix."""
    for c in CLUSTERS:
        if loc_id.startswith(c):
            return c
    return "other"


def _build_split_row(group: pd.DataFrame, n_use: int = 3) -> pd.Series:
    """Build one split record from the last n_use observations of a sorted group."""
    g = group.sort_values("year").tail(n_use).reset_index(drop=True)
    n = len(g)
    row: dict = {"cluster": get_cluster(group.name), "n_timestamps": n}

    for i, label in enumerate(["t1", "t2", "t3"][:n]):
        row[label] = int(g.loc[i, "year"])
        row[f"dist_{label}"] = g.loc[i, "dist_m"]

    if n >= 2:
        row["train_span_yr"] = row["t2"] - row["t1"]
        row["v_train"] = (row["dist_t2"] - row["dist_t1"]) / row["train_span_yr"]
    if n == 3:
        row["test_span_yr"] = row["t3"] - row["t2"]
        row["v_test"] = (row["dist_t3"] - row["dist_t2"]) / row["test_span_yr"]

    return pd.Series(row)


def _join_erosion_volume(features_ok: pd.DataFrame, proc_gpkg: Path) -> pd.DataFrame:
    """Join erosion volume rates from erosion_vlakken_filtered layer."""
    ev_raw = gpd.read_file(proc_gpkg, layer="erosion_vlakken_filtered")

    # Normalise year columns: older files use integer year_before/year_after;
    # newer files (20260330+) use date strings ('YYYY-01-01') in date_before/date_after.
    if "year_before" in ev_raw.columns and "year_after" in ev_raw.columns:
        ev_raw = ev_raw.rename(columns={"year_before": "_yb", "year_after": "_ya"})
    else:
        ev_raw["_yb"] = ev_raw["date_before"].astype(str).str[:4].astype(int)
        ev_raw["_ya"] = ev_raw["date_after"].astype(str).str[:4].astype(int)

    ev = (
        ev_raw[["location_id", "_yb", "_ya", "area", "erosion_volume"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    t_lookup = features_ok[["t1", "t2", "t3"]].copy()
    ev_joined = ev.merge(t_lookup, left_on="location_id", right_index=True, how="inner")

    train_vol = (
        ev_joined[
            (ev_joined["_yb"] == ev_joined["t1"])
            & (ev_joined["_ya"] == ev_joined["t2"])
        ]
        .groupby("location_id")["erosion_volume"]
        .sum()
        .rename("erosion_vol_train_m3")
    )
    test_vol = (
        ev_joined[
            (ev_joined["_yb"] == ev_joined["t2"])
            & (ev_joined["_ya"] == ev_joined["t3"])
        ]
        .groupby("location_id")["erosion_volume"]
        .sum()
        .rename("erosion_vol_test_m3")
    )

    features_ok = features_ok.join(train_vol).join(test_vol)
    features_ok["erosion_vol_train_m3"] = features_ok["erosion_vol_train_m3"].fillna(0)
    features_ok["erosion_vol_test_m3"] = features_ok["erosion_vol_test_m3"].fillna(0)
    features_ok["erosion_vol_train_rate"] = (
        features_ok["erosion_vol_train_m3"] / features_ok["train_span_yr"]
    )
    features_ok["erosion_vol_test_rate"] = (
        features_ok["erosion_vol_test_m3"] / features_ok["test_span_yr"]
    )
    features_ok["erosion_vol_rate_t1"] = features_ok["erosion_vol_train_rate"]

    return features_ok
