"""Step 03 — Feature engineering.

Enriches region_split and region_inference_only tables with all static and
dynamic features required by the LGB model.

Feature groups:
  1. Vegetation class  — dominant polygon area (WFS vegetatielegger)
  2. Land use          — dominant polygon area (BRP gewas)
  3. Soil group        — spatial overlay with BRO Bodemkaart (~30s)
  4. Nearest discharge station
  5. High-water window metrics per t1↔t2 and t2↔t3 windows
  6. Bend exposure     — loaded from pre-computed reference parquet
  7. Erosion volume rate
  8. Ordinal encoding  — all categorical columns

Output schema (29 columns, same as 20260314 reference):
  v_train, v_test, dist_t1/t2/t3, train_span_yr, test_span_yr,
  is_nvo, river(_enc), vegetation_class(_enc), land_use(_enc),
  erosion_vol_rate_t1, soil_group(_enc),
  n_events/max_rise_rate/drawdown_index/flood_days × t1/t2,
  bend_exposure_n5/n8, split, cluster
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

import src.constants as CONST

# ── Hydrology constants ───────────────────────────────────────────────────────

# High-water threshold (m³/s) per discharge station — used for event detection.
HIGH_WATER_THRESHOLD: dict[str, float] = {
    "lobith.bovenrijn.tolkamer": 6000,
    "millingenaanderijn": 4000,
    "pannerden.pannerdenschkanaal": 2000,
    "driel.boven": 1000,
    "hagestein.boven": 1000,
    "maastricht.borgharen.maas.beneden": 1000,
    "maaseik": 1000,
    "venlo": 1000,
    "megen.maas": 1000,
    "hank.bergschemaas": 1100,
    "westervoort.ijsselkop": 800,
    "westervoort": 800,
    "olst": 800,
    "genemuiden": 300,
}

DT_HOURS = 1 / 6   # 10-minute sampling interval expressed in hours
GAP_HOURS = 72     # gap below which two HW events are merged into one

HW_METRIC_COLS = ["n_events", "max_rise_rate", "drawdown_index"]
HW_WINDOW_COLS = (
    [f"{c}_t{t}" for t in [1, 2] for c in HW_METRIC_COLS]
    + ["flood_days_t1", "flood_days_t2"]
)

# Output column order — must match 20260314 reference schema
COLS_OUT = [
    "v_train", "v_test",
    "dist_t1", "dist_t2", "dist_t3",
    "train_span_yr", "test_span_yr",
    "is_nvo",
    "river", "river_enc",
    "vegetation_class", "vegetation_class_enc",
    "land_use", "land_use_enc",
    "erosion_vol_rate_t1",
    "soil_group", "soil_group_enc",
    "n_events_t1", "max_rise_rate_t1", "drawdown_index_t1", "flood_days_t1",
    "n_events_t2", "max_rise_rate_t2", "drawdown_index_t2", "flood_days_t2",
    "bend_exposure_n5", "bend_exposure_n8",
    "split", "cluster",
]

# Vegetation mix classes consolidated to 'Other'
RARE_VEG_CLASSES = {"90/10", "70/30", "50/50"}


def build_features(
    split: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    scope_gpkg: Path,
    veg_gpkg: Path,
    lu_gpkg: Path,
    soil_gpkg: Path,
    stations_gpkg: Path,
    discharge_dir: Path,
    reference_features_v2: Path,
    high_water_threshold: Optional[dict[str, float]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build region_features and region_inference_features from the split tables.

    Args:
        split:      region_split DataFrame (3-timestamp, indexed by location_id).
        inference:  region_inference_only DataFrame (2-timestamp, indexed by location_id).
        scope_gpkg: GeoPackage with vlakken_scope layer.
        veg_gpkg:   GeoPackage with rws_vegetatielegger layer.
        lu_gpkg:    GeoPackage with BrpGewas layer.
        soil_gpkg:  BRO Bodemkaart GeoPackage.
        stations_gpkg: GeoPackage with discharge_stations layer.
        discharge_dir:  Directory of cleaned discharge Parquet files (10-min resolution).
        reference_features_v2: Parquet with pre-computed bend_exposure columns.
        high_water_threshold:  Override for HIGH_WATER_THRESHOLD dict (optional).

    Returns:
        (region_features, region_inference_features) — both indexed by location_id.
    """
    hw_thresh = high_water_threshold or HIGH_WATER_THRESHOLD
    split = split.copy()
    inference = inference.copy()

    # River column derived from location_id prefix
    split["river"] = split.index.to_series().str.extract(r"^([a-z]+\d*)_")[0]
    inference["river"] = inference.index.to_series().str.extract(r"^([a-z]+\d*)_")[0]

    all_ids = split.index.union(inference.index)

    # ── Scope geometries ──────────────────────────────────────────────────────
    print("Loading scope geometries ...")
    scope = _load_scope(scope_gpkg, all_ids)

    # ── Feature group 1: vegetation class ────────────────────────────────────
    print("Assigning vegetation class ...")
    veg = gpd.read_file(veg_gpkg, layer="rws_vegetatielegger:vegetatieklassen")
    veg["area"] = veg.geometry.area
    dom_veg = (
        veg.sort_values("area", ascending=False)
        .groupby("scope_region_id")["vlklasse"]
        .first()
        .rename("vegetation_class")
    )
    dom_veg = dom_veg.apply(lambda x: "Other" if x in RARE_VEG_CLASSES else x)
    split["vegetation_class"] = split.index.map(dom_veg)
    inference["vegetation_class"] = inference.index.map(dom_veg)

    # ── Feature group 2: land use ─────────────────────────────────────────────
    print("Assigning land use ...")
    lu = gpd.read_file(lu_gpkg, layer="BrpGewas")
    lu["area"] = lu.geometry.area
    dom_lu = (
        lu.sort_values("area", ascending=False)
        .groupby("scope_region_id")["category"]
        .first()
        .rename("land_use")
    )
    split["land_use"] = split.index.map(dom_lu)
    inference["land_use"] = inference.index.map(dom_lu)

    # ── Feature group 3: soil group (~30s) ────────────────────────────────────
    print("Assigning soil group (spatial overlay, ~30s) ...")
    soil_poly = gpd.read_file(soil_gpkg, layer="soilarea")[["maparea_id", "geometry"]]
    soil_codes = gpd.read_file(soil_gpkg, layer="soilarea_soilunit")[
        ["maparea_id", "soilunit_code"]
    ]
    soil_poly = soil_poly.merge(soil_codes, on="maparea_id", how="left")
    joined_soil = gpd.overlay(
        scope.reset_index(),
        soil_poly[["soilunit_code", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    joined_soil["area"] = joined_soil.geometry.area
    dom_soil = (
        joined_soil.sort_values("area", ascending=False)
        .groupby("location_id")["soilunit_code"]
        .first()
        .map(soil_group_fn)
    )
    split["soil_group"] = split.index.map(dom_soil)
    inference["soil_group"] = inference.index.map(dom_soil)

    # ── Feature group 4: nearest discharge station ────────────────────────────
    print("Assigning nearest discharge station ...")
    stations = gpd.read_file(stations_gpkg, layer="discharge_stations")
    stations = stations.rename(columns={"CODE": "station_code"}).to_crs(scope.crs)
    scope_centroid = scope.copy()
    scope_centroid["geometry"] = scope_centroid.geometry.centroid
    nearest = gpd.sjoin_nearest(
        scope_centroid.reset_index(),
        stations[["station_code", "geometry"]],
        how="left",
        distance_col="station_dist_m",
    )
    station_map = nearest.set_index("location_id")["station_code"]
    split["nearest_station"] = split.index.map(station_map)
    inference["nearest_station"] = inference.index.map(station_map)

    # ── Feature group 5: high-water metrics ───────────────────────────────────
    print("Computing high-water window metrics ...")
    disc_raw, station_annual = _load_discharge(discharge_dir, hw_thresh)
    hw_metrics = _compute_all_hw_metrics(disc_raw, hw_thresh)

    hw_feat_split = split.apply(
        _hw_window_stats, axis=1,
        hw_metrics=hw_metrics, station_annual=station_annual, has_t3=True,
    )
    no_ev = hw_feat_split[HW_WINDOW_COLS].isna().all(axis=1)
    hw_feat_split.loc[no_ev, HW_WINDOW_COLS] = 0
    split[HW_WINDOW_COLS] = hw_feat_split[HW_WINDOW_COLS]

    hw_feat_inf = inference.apply(
        _hw_window_stats, axis=1,
        hw_metrics=hw_metrics, station_annual=station_annual, has_t3=False,
    )
    no_ev_inf = hw_feat_inf[HW_WINDOW_COLS].isna().all(axis=1)
    hw_feat_inf.loc[no_ev_inf, HW_WINDOW_COLS] = 0
    inference[HW_WINDOW_COLS] = hw_feat_inf[HW_WINDOW_COLS]

    # ── Feature group 6: bend exposure ───────────────────────────────────────
    print("Loading bend exposure from reference ...")
    curv_cols = ["bend_exposure_n5", "bend_exposure_n8"]
    feat_v2 = pd.read_parquet(reference_features_v2, columns=curv_cols)
    split[curv_cols] = feat_v2.reindex(split.index)[curv_cols]
    inference[curv_cols] = feat_v2.reindex(inference.index)[curv_cols]

    # ── Feature group 7: erosion volume rate ─────────────────────────────────
    if "erosion_vol_train_rate" in split.columns:
        split["erosion_vol_rate_t1"] = split["erosion_vol_train_rate"]
    elif "erosion_vol_rate_t1" not in split.columns:
        split["erosion_vol_rate_t1"] = np.nan
    inference["erosion_vol_rate_t1"] = 0.0

    # ── Feature group 8: ordinal encoding ────────────────────────────────────
    cat_cols = {
        "river": "river",
        "vegetation_class": "rws_vegetatielegger:vegetatieklassen_majority_class_vlklasse",
        "land_use": "BrpGewas_majority_class_category",
        "soil_group": "soil_group",
    }
    for df in [split, inference]:
        for col, key in cat_cols.items():
            df[f"{col}_enc"] = encode_col(df[col], key)

    # ── Assemble output ───────────────────────────────────────────────────────
    features_out = split[[c for c in COLS_OUT if c in split.columns]]
    inf_cols_out = [c for c in COLS_OUT if c not in ("v_test", "split") and c in inference.columns]
    inference_out = inference[inf_cols_out]

    return features_out, inference_out


# ── Public helpers ─────────────────────────────────────────────────────────────

def encode_col(series: pd.Series, category_key: str) -> pd.Series:
    """Ordinal-encode a categorical series using KNOWN_CATEGORIES from constants.

    Unknown values map to DEFAULT_UNKNOWN_CATEGORY_LABEL.
    """
    mapping = {v: i for i, v in enumerate(CONST.KNOWN_CATEGORIES[category_key])}
    return series.map(lambda x: mapping.get(x, CONST.DEFAULT_UNKNOWN_CATEGORY_LABEL))


def soil_group_fn(code: object) -> object:
    """Map a BRO soil unit code to a simplified soil group label."""
    if pd.isna(code):
        return np.nan
    c = str(code)
    if c[:2] in ("Rd", "Rn", "Ro"):
        return "River clay"
    elif c[0] == "Z":
        return "Sandy"
    elif c[:2] in ("Mv", "Mo", "pM") or c[0] == "b":
        return "Soft (peat/podzol)"
    elif c[:2] == "Mn":
        return "River sand-clay"
    else:
        return "Other"


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_scope(scope_gpkg: Path, all_ids) -> gpd.GeoDataFrame:
    scope_raw = gpd.read_file(scope_gpkg, layer="vlakken_scope")
    if "position_id" in scope_raw.columns and "location_id" not in scope_raw.columns:
        scope_raw = scope_raw.rename(columns={"position_id": "location_id"})
    scope_raw = scope_raw.set_index("location_id")[["geometry"]]
    return scope_raw.loc[scope_raw.index.intersection(all_ids)]


def _load_discharge(
    discharge_dir: Path, hw_thresh: dict
) -> tuple[pd.DataFrame, dict]:
    """Load cleaned discharge timeseries and pre-compute P90 annual flood days."""
    disc_files = sorted(discharge_dir.glob("*.parquet"))
    frames = []
    for f in disc_files:
        df_ = pd.read_parquet(f)
        df_["timestamp"] = pd.to_datetime(df_["timestamp"], utc=True)
        frames.append(df_)
    disc_raw = pd.concat(frames, ignore_index=True)

    disc_date = disc_raw.copy()
    disc_date["date"] = disc_date["timestamp"].dt.date
    disc_daily = (
        disc_date.groupby(["station_code", "date"])["discharge_m3s"].max().reset_index()
    )
    disc_daily["year"] = pd.to_datetime(disc_daily["date"]).dt.year

    station_annual = {}
    for code, grp in disc_daily.groupby("station_code"):
        p90 = grp["discharge_m3s"].quantile(0.90)
        station_annual[code] = grp.groupby("year").agg(
            days_above_p90=("discharge_m3s", lambda x: (x > p90).sum())
        )

    return disc_raw, station_annual


def _compute_all_hw_metrics(disc_raw: pd.DataFrame, hw_thresh: dict) -> pd.DataFrame:
    """Compute n_events, max_rise_rate, drawdown_index per (station_id, year)."""
    results = []
    for code, thresh in hw_thresh.items():
        sub = disc_raw[disc_raw["station_code"] == code]
        if sub.empty:
            continue
        results.append(_compute_hw_metrics_for_station(code, sub, thresh))
    return pd.concat(results, ignore_index=True).set_index(["station_id", "year"])


def _compute_hw_metrics_for_station(
    station_code: str, df: pd.DataFrame, q_thresh: float
) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    df["year"] = df["timestamp"].dt.year
    df["exceed"] = df["discharge_m3s"] > q_thresh
    df["dQ"] = df["discharge_m3s"].diff()
    df["dt_h"] = df["timestamp"].diff().dt.total_seconds() / 3600
    df["dQ_dt"] = np.where(df["dt_h"] > 0, df["dQ"] / df["dt_h"], np.nan)

    rows = []
    for year, grp in df.groupby("year"):
        g = grp.dropna(subset=["discharge_m3s"]).sort_values("timestamp")
        if g.empty:
            rows.append(
                {"station_id": station_code, "year": year,
                 "n_events": np.nan, "max_rise_rate": np.nan, "drawdown_index": np.nan}
            )
            continue

        exceed = g["exceed"].values
        ts = pd.to_datetime(g["timestamp"])
        # Detect and merge events separated by less than GAP_HOURS
        blocks: list[tuple] = []
        i = 0
        while i < len(exceed):
            if exceed[i]:
                start = i
                while i < len(exceed) and exceed[i]:
                    i += 1
                blocks.append((ts.iloc[start], ts.iloc[i - 1]))
            else:
                i += 1
        merged: list[tuple] = []
        for t0, t1_ in blocks:
            if merged and (t0 - merged[-1][1]).total_seconds() / 3600 < GAP_HOURS:
                merged[-1] = (merged[-1][0], t1_)
            else:
                merged.append((t0, t1_))

        rise_mask = g["exceed"] & (g["dQ_dt"] > 0)
        rec_mask = g["exceed"] & (g["dQ_dt"] < 0)
        max_rise = g.loc[rise_mask, "dQ_dt"].max() if rise_mask.any() else np.nan
        drawdown = (
            np.abs(g.loc[rec_mask, "dQ_dt"]).max() if rec_mask.any() else np.nan
        )
        rows.append(
            {"station_id": station_code, "year": year, "n_events": len(merged),
             "max_rise_rate": max_rise, "drawdown_index": drawdown}
        )
    return pd.DataFrame(rows)


def _hw_window_stats(
    row: pd.Series,
    hw_metrics: pd.DataFrame,
    station_annual: dict,
    has_t3: bool,
) -> pd.Series:
    """Compute mean HW metrics over the t1↔t2 and (optionally) t2↔t3 windows."""
    code = row.get("nearest_station")
    t1, t2 = row.get("t1"), row.get("t2")
    t3 = row.get("t3") if has_t3 else None

    nan_out = {f"{c}_t{t}": np.nan for t in [1, 2] for c in HW_METRIC_COLS}
    nan_out["flood_days_t1"] = np.nan
    nan_out["flood_days_t2"] = np.nan

    if pd.isna(code):
        return pd.Series(nan_out)

    years_t1 = (
        list(range(int(t1) + 1, int(t2)))
        if not (pd.isna(t1) or pd.isna(t2)) and int(t2) - int(t1) > 1
        else []
    )
    years_t2 = (
        list(range(int(t2) + 1, int(t3)))
        if t3 and not pd.isna(t3) and int(t3) - int(t2) > 1
        else []
    )

    out: dict = {}

    if code in hw_metrics.index.get_level_values(0):
        try:
            sub = hw_metrics.loc[code]
            sub = sub.to_frame().T if isinstance(sub, pd.Series) else sub
            for c in HW_METRIC_COLS:
                sub_t1 = sub.loc[sub.index.intersection(years_t1)] if years_t1 else pd.DataFrame()
                sub_t2 = sub.loc[sub.index.intersection(years_t2)] if years_t2 else pd.DataFrame()
                out[f"{c}_t1"] = sub_t1[c].mean() if len(sub_t1) > 0 else np.nan
                out[f"{c}_t2"] = sub_t2[c].mean() if len(sub_t2) > 0 else np.nan
        except KeyError:
            for c in HW_METRIC_COLS:
                out[f"{c}_t1"] = out[f"{c}_t2"] = np.nan
    else:
        for c in HW_METRIC_COLS:
            out[f"{c}_t1"] = out[f"{c}_t2"] = np.nan

    if code in station_annual:
        ann = station_annual[code]
        sub_t1 = ann.loc[ann.index.intersection(years_t1)] if years_t1 else pd.DataFrame()
        sub_t2 = ann.loc[ann.index.intersection(years_t2)] if years_t2 else pd.DataFrame()
        out["flood_days_t1"] = (
            sub_t1["days_above_p90"].mean() if len(sub_t1) > 0 else np.nan
        )
        out["flood_days_t2"] = (
            sub_t2["days_above_p90"].mean() if len(sub_t2) > 0 else np.nan
        )
    else:
        out["flood_days_t1"] = out["flood_days_t2"] = np.nan

    return pd.Series(out)
