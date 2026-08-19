"""Prepare + run the 20260820-hybrid-masked experiment.

Same as the 20260819-hybrid prep, plus:
  - kribben sample mask (Kribben_BKN, 10 m padding) in HybridLineSource
  - survival rule: a (region, date) observation needs >= 12 surviving samples
  - unchanged: within_year median, far-bank |v| > 50 m/yr region filter,
    bend_exposure fillna — so the comparison isolates the mask's effect.
Ends by running the pipeline (no export) and printing the model comparison
against 20260819-hybrid.
"""

import logging
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("prep")

import geopandas as gpd
import joblib
import pandas as pd

from src.pipeline.config import ExperimentConfig
from src.pipeline.feature_engineering import build_features
from src.pipeline.region_split import build_region_split
from src.sources import HybridLineSource, ScopeGeometry

cfg = ExperimentConfig(experiment="20260820-hybrid-masked")
cfg.features_dir.mkdir(parents=True, exist_ok=True)

HYBRID = cfg.data_dir / "02_processed/hybrid/hybrid_model_results_20260710.gpkg"
KRIBS = gpd.read_file(
    cfg.data_dir / "01_raw/scope/Levering_erosie_data.gpkg", layer="Kribben_BKN"
).to_crs(28992)
V_LIMIT = 50.0
MIN_SAMPLES = 12

log.info("1 · hybrid lines → observations (kribben mask, 10 m)")
src = HybridLineSource(
    HYBRID,
    geometry=ScopeGeometry(centreline_gpkg=cfg.raw_gpkg),
    mask=KRIBS,
    mask_buffer_m=10.0,
)
obs = src.load()
n_before = len(obs.frame)
frame = obs.frame[obs.frame["n_candidates"] >= MIN_SAMPLES]
log.info(
    "    survival rule (>=%d samples): %d of %d observations kept",
    MIN_SAMPLES,
    len(frame),
    n_before,
)
from src.sources.observations import BankObservations

dpy = BankObservations(frame.reset_index(drop=True)).to_dist_per_year(
    within_year="median"
)
log.info(
    "    %s region-years, %s regions", f"{len(dpy):,}", f"{dpy.location_id.nunique():,}"
)

d = dpy.sort_values(["location_id", "year"]).copy()
d["v"] = (
    d.groupby("location_id")["dist_m"].diff() / d.groupby("location_id")["year"].diff()
)
vmax = d.groupby("location_id")["v"].apply(lambda s: s.abs().max())
bad = set(vmax[vmax > V_LIMIT].index)
dpy = dpy[~dpy.location_id.isin(bad)].reset_index(drop=True)
log.info(
    "    far-bank filter (|v| > %.0f m/yr): %d regions excluded, %s remain",
    V_LIMIT,
    len(bad),
    f"{dpy.location_id.nunique():,}",
)
dpy.to_parquet(cfg.features_dir / "dist_per_year.parquet", index=False)

log.info("2 · region split")
split, inference = build_region_split(
    dpy, proc_gpkg=cfg.proc_gpkg, test_size=cfg.test_size, random_seed=cfg.seed
)
split.to_parquet(cfg.features_dir / "region_split.parquet")
inference.to_parquet(cfg.features_dir / "region_inference_only.parquet")
log.info(
    "    split %s · inference-only %s · v_test std %.2f",
    split.shape,
    inference.shape,
    split.v_test.std(),
)

log.info("3 · features")
feats, inf_feats = build_features(
    split,
    inference,
    scope_gpkg=cfg.scope_gpkg,
    veg_gpkg=cfg.veg_gpkg,
    lu_gpkg=cfg.lu_gpkg,
    soil_gpkg=cfg.soil_gpkg,
    stations_gpkg=cfg.stations_gpkg,
    discharge_dir=cfg.discharge_dir,
    reference_features_v2=cfg.reference_features_v2,
)
for df, name in [(feats, "region_features"), (inf_feats, "region_inference_features")]:
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(0.0)
    df.to_parquet(cfg.features_dir / f"{name}.parquet")
    log.info("    %s %s", name, df.shape)

log.info("4 · pipeline (train → predict, no export)")
rc = subprocess.call(
    [
        sys.executable,
        "-m",
        "src.pipeline",
        "--experiment",
        "20260820-hybrid-masked",
        "--resume",
        "--no-export",
    ]
)
log.info("pipeline exit code %d", rc)

log.info("5 · comparison vs 20260819-hybrid")
for exp in ("20260819-hybrid", "20260820-hybrid-masked"):
    b = joblib.load(cfg.data_dir / f"04_model_outputs/{exp}/bundle.joblib")
    res = (
        pd.DataFrame(b["results"]).T
        if not isinstance(b["results"], pd.DataFrame)
        else b["results"]
    )
    print(f"\n=== {exp} ===")
    print(res.round(3).to_string())
