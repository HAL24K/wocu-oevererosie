"""One-time caches for the loop-engineering harness (experiment branch).

Everything a loop iteration needs that does not depend on cleaning rules is
computed once and parked under ``data/03_features/loop/``:

  samples.parquet         every sampled point on every measurable line
  line_metrics.parquet    per-line geometry: length, chord, tortuosity
  static_features.parquet per-region features that never change with cleaning
                          (veg, land use, soil, station, bend, is_nvo, quality,
                          centreline length)
  hw_metrics.parquet      per (station, year) high-water metrics
  station_annual.parquet  per (station, year) days above P90
  ev_table.parquet        erosion-volume rows keyed (location_id, _yb, _ya)

The frozen holdout — the test region ids of the 20260820-hybrid-masked run —
goes to ``experiments/loop/frozen_test_regions.csv`` and is committed: every
variant in the experiment scores against these same regions.
"""

import logging
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src.erosion.region_inspector import RegionInspector
from src.pipeline.config import ExperimentConfig
from src.pipeline.feature_engineering import (
    HIGH_WATER_THRESHOLD,
    RARE_VEG_CLASSES,
    _compute_all_hw_metrics,
    _load_discharge,
    _load_scope,
    soil_group_fn,
)
from src.pipeline.region_split import QUALITY_COL
from src.sources.geometry import LOCATION_ID, normalise_location_id

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("caches")

cfg = ExperimentConfig(experiment="loop")
CACHE = cfg.data_dir / "03_features/loop"
CACHE.mkdir(parents=True, exist_ok=True)
EXP_DIR = cfg.data_dir.parent / "experiments/loop"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ── samples + line metrics ────────────────────────────────────────────────────
log.info("1 · samples (20 per line, distance to centreline)")
ri = RegionInspector()
samples = ri.samples.reset_index(names="line_idx")
samples.to_parquet(CACHE / "samples.parquet", index=False)
log.info(
    "    %s samples, %s lines, %s regions",
    f"{len(samples):,}",
    f"{samples.line_idx.nunique():,}",
    f"{samples[LOCATION_ID].nunique():,}",
)

log.info("2 · line metrics (length, chord, tortuosity)")
usable = ri.lines.loc[samples["line_idx"].unique()]
length = usable.geometry.length
# chord from the raw endpoints of each (possibly multi-part) line
first = usable.geometry.apply(lambda g: shapely.get_coordinates(g)[0])
last = usable.geometry.apply(lambda g: shapely.get_coordinates(g)[-1])
chord = np.hypot(
    first.str[0].astype(float) - last.str[0].astype(float),
    first.str[1].astype(float) - last.str[1].astype(float),
)
lm = pd.DataFrame(
    {
        LOCATION_ID: usable[LOCATION_ID],
        "date": usable["date"],
        "year": usable["year"],
        "model": usable["model"],
        "length": length,
        "chord": chord,
        "tortuosity": length / np.maximum(chord, 1.0),
    }
)
lm.index.name = "line_idx"
lm.reset_index().to_parquet(CACHE / "line_metrics.parquet", index=False)
log.info(
    "    tortuosity p50=%.2f p95=%.2f p99=%.2f",
    *lm["tortuosity"].quantile([0.5, 0.95, 0.99]),
)

# ── static per-region features ────────────────────────────────────────────────
log.info("3 · static per-region features")
all_ids = pd.Index(sorted(samples[LOCATION_ID].unique()), name=LOCATION_ID)
static = pd.DataFrame(index=all_ids)
static["river"] = static.index.to_series().str.extract(r"^([a-z]+\d*)_")[0]

scope = _load_scope(cfg.scope_gpkg, all_ids)

veg = gpd.read_file(cfg.veg_gpkg, layer="rws_vegetatielegger:vegetatieklassen")
veg["area"] = veg.geometry.area
dom_veg = (
    veg.sort_values("area", ascending=False)
    .groupby("scope_region_id")["vlklasse"]
    .first()
)
static["vegetation_class"] = static.index.map(dom_veg).map(
    lambda x: "Other" if x in RARE_VEG_CLASSES else x
)

lu = gpd.read_file(cfg.lu_gpkg, layer="BrpGewas")
lu["area"] = lu.geometry.area
static["land_use"] = static.index.map(
    lu.sort_values("area", ascending=False)
    .groupby("scope_region_id")["category"]
    .first()
)

log.info("    soil overlay (~1 min)")
soil_poly = gpd.read_file(cfg.soil_gpkg, layer="soilarea")[["maparea_id", "geometry"]]
soil_codes = gpd.read_file(cfg.soil_gpkg, layer="soilarea_soilunit")[
    ["maparea_id", "soilunit_code"]
]
soil_poly = soil_poly.merge(soil_codes, on="maparea_id", how="left")
joined = gpd.overlay(
    scope.reset_index(),
    soil_poly[["soilunit_code", "geometry"]],
    how="intersection",
    keep_geom_type=False,
)
joined["area"] = joined.geometry.area
static["soil_group"] = static.index.map(
    joined.sort_values("area", ascending=False)
    .groupby("location_id")["soilunit_code"]
    .first()
    .map(soil_group_fn)
)

stations = gpd.read_file(cfg.stations_gpkg, layer="discharge_stations")
stations = stations.rename(columns={"CODE": "station_code"}).to_crs(scope.crs)
cent = scope.copy()
cent["geometry"] = cent.geometry.centroid
nearest = gpd.sjoin_nearest(
    cent.reset_index(),
    stations[["station_code", "geometry"]],
    how="left",
    distance_col="station_dist_m",
)
static["nearest_station"] = static.index.map(
    nearest.drop_duplicates("location_id").set_index("location_id")["station_code"]
)

curv = pd.read_parquet(
    cfg.reference_features_v2, columns=["bend_exposure_n5", "bend_exposure_n8"]
)
static[["bend_exposure_n5", "bend_exposure_n8"]] = curv.reindex(static.index)

summary = normalise_location_id(gpd.read_file(cfg.proc_gpkg, layer="summary_scope"))
summary = summary.drop_duplicates(subset=["location_id"])
static["quality"] = static.index.map(summary.set_index("location_id")[QUALITY_COL])

vvr = gpd.read_file(cfg.proc_gpkg, layer="vvr_rates_of_change")
nvo_ids = set(
    gpd.sjoin(
        scope.reset_index().to_crs(vvr.crs),
        vvr[["geometry"]],
        how="inner",
        predicate="intersects",
    )["location_id"]
)
static["is_nvo"] = static.index.isin(nvo_ids)

static["cl_len"] = ri.geometry.centrelines.reindex(static.index).length

static.reset_index().to_parquet(CACHE / "static_features.parquet", index=False)
log.info(
    "    static features %s (quality OK: %d)",
    static.shape,
    (static["quality"] == "OK").sum(),
)

# ── high-water lookups ────────────────────────────────────────────────────────
log.info("4 · high-water metrics per (station, year)")
disc_raw, station_annual = _load_discharge(cfg.discharge_dir, HIGH_WATER_THRESHOLD)
hw = _compute_all_hw_metrics(disc_raw, HIGH_WATER_THRESHOLD)
hw.reset_index().to_parquet(CACHE / "hw_metrics.parquet", index=False)
ann = pd.concat(
    [df.assign(station_code=code).reset_index() for code, df in station_annual.items()],
    ignore_index=True,
)
ann.to_parquet(CACHE / "station_annual.parquet", index=False)
log.info("    hw_metrics %s · station_annual %s", hw.shape, ann.shape)

# ── erosion volume table ──────────────────────────────────────────────────────
log.info("5 · erosion volume table")
ev_raw = gpd.read_file(cfg.proc_gpkg, layer="erosion_vlakken_filtered")
if "year_before" in ev_raw.columns:
    ev_raw = ev_raw.rename(columns={"year_before": "_yb", "year_after": "_ya"})
else:
    ev_raw["_yb"] = ev_raw["date_before"].astype(str).str[:4].astype(int)
    ev_raw["_ya"] = ev_raw["date_after"].astype(str).str[:4].astype(int)
ev = (
    ev_raw[["location_id", "_yb", "_ya", "erosion_volume"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
ev.to_parquet(CACHE / "ev_table.parquet", index=False)
log.info("    ev rows %d", len(ev))

# ── frozen holdout ────────────────────────────────────────────────────────────
log.info("6 · frozen holdout from 20260820-hybrid-masked")
ref_split = pd.read_parquet(
    cfg.data_dir / "03_features/20260820-hybrid-masked/region_split.parquet"
)
frozen = ref_split[ref_split["split"] == "test"].index.to_series()
frozen.to_csv(EXP_DIR / "frozen_test_regions.csv", index=False, header=["location_id"])
log.info(
    "    %d frozen test regions → %s", len(frozen), EXP_DIR / "frozen_test_regions.csv"
)
log.info("done")
