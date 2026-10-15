"""Session GeoPackage for the asset-manager sessions, rebuilt from one model run.

Takes Etienne's ``wocu_sessies_14_15okt.gpkg`` as the template: the same regions, the same
layer names and fields (so his QGIS project and ``session_project_v2.py`` keep working),
with the run's own data:

- ``locaties``, ``vvr_vlak``, ``signaleringslijn``: copied unchanged;
- ``scoperegios_kruisingsjaar``: the template's regions with the run's crossing year and
  class (``exports/vvr_kruising_<run>.gpkg``); class "VVR, geen voorspelling" for regions
  that overlap a VVR but have no prediction, and no class for regions without a VVR (worked
  out here once, so QGIS does not have to test the overlap on every redraw);
- ``voorspelde_oeverlijnen``: the run's predicted lines per year, broken where a strip has
  no data (``exports/voorspelde_oeverlijnen_2026_2031_<run>.gpkg``), one row per piece,
  with ``eerste`` = 1 on the region's first year after its last measurement;
- ``gemeten_oeverlijnen``: every line of the run's hybrid in those regions, with
  ``gebruikt`` = 1 when the model kept it after cleaning and 0 when cleaning removed it,
  and ``laatste`` = 1 on the used lines of the region's latest survey date;
- ``andere_bron_oeverlijnen``: the lines of the source the hybrid did *not* choose for the
  scope area (from the hybrid's ``lines_all``), to compare with (asked for on 15 Oct).

Scope areas also get ``gekozen_bron`` (hoogtemodel / satelliet (SAM)).

Needs both exports of the run first (``export_vvr_crossing.py``, ``export_predicted_banklines.py``).

Run: ``PYTHONPATH=. uv run python scripts/export/session_geopackage.py --run 20261013-ref
        --template data/05_deliverables/wocu_sessies_14_15okt.gpkg --out <new.gpkg>``
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.pipeline.config import ExperimentConfig
from src.pipeline.hybrid_prep import load_lines
from src.sources.geometry import LOCATION_ID, ScopeGeometry

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--template", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
cfg = ExperimentConfig.from_run(args.run)
T = lambda layer: gpd.read_file(args.template, layer=layer)  # noqa: E731
args.out.unlink(missing_ok=True)


def write(gdf: gpd.GeoDataFrame, layer: str) -> None:
    gdf.to_file(args.out, layer=layer, driver="GPKG")
    print(f"{layer}: {len(gdf):,}")


for layer in ("locaties", "vvr_vlak", "signaleringslijn"):
    write(T(layer), layer)

# Regions: the template's set, the run's crossing.
regions = T("scoperegios_kruisingsjaar")[[LOCATION_ID, "waterlichaam", "geometry"]]
cross = gpd.read_file(cfg.exports_dir / f"vvr_kruising_{args.run}.gpkg")
cross = pd.DataFrame(cross.drop(columns="geometry")).drop_duplicates(LOCATION_ID)
regions = regions.merge(cross, on=LOCATION_ID, how="left")
predicted = regions["n_seg_vvr"].notna()
no_vvr = regions["n_seg_vvr"].fillna(0) == 0
regions.loc[no_vvr, ["kruisingsjaar", "kruising_klasse"]] = None
vvr_union = T("vvr_vlak").to_crs(regions.crs).union_all()
regions.loc[~predicted & regions.intersects(vvr_union), "kruising_klasse"] = (
    "VVR, geen voorspelling"
)
regions["gekozen_bron"] = regions["model_preference"].map(
    {"height": "hoogtemodel", "segmentation": "satelliet (SAM)"}
)
print(regions["kruising_klasse"].fillna("(geen VVR)").value_counts().to_string())
write(regions, "scoperegios_kruisingsjaar")
ids = set(regions[LOCATION_ID])

# Predicted lines: one row per piece, as LineStrings like the template.
pred = gpd.read_file(
    cfg.exports_dir / f"voorspelde_oeverlijnen_2026_2031_{args.run}.gpkg"
)
pred = pred[pred[LOCATION_ID].isin(ids)].explode(index_parts=False)
pred = pred[~pred.geometry.is_empty]
last_year = (
    pd.read_parquet(cfg.model_out_dir / "segment_predictions.parquet")
    .groupby(LOCATION_ID)["last_year"]
    .max()
)
first = (
    pred.assign(after=pred["year"] > pred[LOCATION_ID].map(last_year))
    .query("after")
    .groupby(LOCATION_ID)["year"]
    .min()
)
pred["eerste"] = (pred["year"] == pred[LOCATION_ID].map(first)).astype("int32")
write(
    pred[
        [
            LOCATION_ID,
            "year",
            "n_seg",
            "v_mean_m_per_yr",
            "dist_mean_m",
            "verplaatsing_tov_2026_m",
            "eerste",
            "geometry",
        ]
    ].reset_index(drop=True),
    "voorspelde_oeverlijnen",
)

# Measured lines: all of the run's hybrid in the template's regions, flagged by use.
measured_ids = set(T("gemeten_oeverlijnen")[LOCATION_ID]) | ids
lines = load_lines(cfg, ScopeGeometry(centreline_gpkg=cfg.raw_gpkg))
kept = set(
    pd.read_parquet(cfg.features_dir / "samples.parquet", columns=["line_idx"])[
        "line_idx"
    ].unique()
)
lines = lines[lines[LOCATION_ID].isin(measured_ids)]
measured = gpd.GeoDataFrame(
    {
        "date": pd.to_datetime(lines["date"]).dt.normalize(),
        "model": lines["model"],
        LOCATION_ID: lines[LOCATION_ID],
        "jaar": lines["date"].dt.year.astype("int32"),
        "gebruikt": lines.index.isin(kept).astype("int32"),
    },
    geometry=lines.geometry.values,
    crs=lines.crs,
).explode(index_parts=False)
latest = measured[measured["gebruikt"] == 1].groupby(LOCATION_ID)["date"].max()
measured["laatste"] = (
    (measured["gebruikt"] == 1)
    & (measured["date"] == measured[LOCATION_ID].map(latest))
).astype("int32")
write(measured.reset_index(drop=True), "gemeten_oeverlijnen")

# The other source: the hybrid's lines of the model it did not choose for the area.
chosen = gpd.read_file(cfg.hybrid_gpkg, layer="model_preference", ignore_geometry=True)
chosen = chosen.set_index(LOCATION_ID)["model_preference"]
chosen = chosen.map({"height": "hoogtemodel", "segmentation": "segmentation"})
other = gpd.read_file(cfg.hybrid_gpkg, layer="lines_all")
other = other[other[LOCATION_ID].isin(measured_ids)]
other = other[other["model"] != other[LOCATION_ID].map(chosen)]
other["date"] = pd.to_datetime(other["date"]).dt.normalize()
other["jaar"] = other["date"].dt.year.astype("int32")
write(
    other[["date", "model", LOCATION_ID, "jaar", "geometry"]].reset_index(drop=True),
    "andere_bron_oeverlijnen",
)
print(
    f"  used by the model: {measured['gebruikt'].sum():,}, "
    f"removed by cleaning: {(measured['gebruikt'] == 0).sum():,}"
)
print(f"→ {args.out}")
