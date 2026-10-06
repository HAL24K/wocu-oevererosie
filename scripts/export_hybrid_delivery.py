"""Write a reproduced hybrid run in the exact format of Luke's delivery, plus a list of differences.

Output 1: ``<stem>.gpkg`` with layers ``lines`` and ``model_preference``: same columns,
column order, geometry type and row order as the reference delivery.
Output 2: ``<stem>_differences.csv``: every region whose preference or winning lines differ
from the reference, with both sides' points and per-year line pieces and length.

Run: ``uv run python scripts/export_hybrid_delivery.py --repro …/hybrid_repro_nearest.gpkg
        --reference …/hybrid_model_results_nearest_20260710.gpkg --out-stem …/hybrid_model_results_nearest_repro``
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

ap = argparse.ArgumentParser()
ap.add_argument("--repro", required=True)
ap.add_argument("--reference", required=True)
ap.add_argument("--out-stem", required=True)
args = ap.parse_args()

ref_pref = pyogrio.read_dataframe(args.reference, layer="model_preference")
ref_lines = pyogrio.read_dataframe(args.reference, layer="lines")
pref = pyogrio.read_dataframe(args.repro, layer="model_preference")
lines = pyogrio.read_dataframe(args.repro, layer="lines")

order = {loc: i for i, loc in enumerate(ref_pref["location_id"])}
pref = pref.assign(_o=pref["location_id"].map(order)).sort_values("_o").drop(columns="_o")
pref = gpd.GeoDataFrame(pref[list(ref_pref.columns)], geometry=ref_pref.geometry.name, crs=ref_pref.crs)
lines = lines.assign(_o=lines["location_id"].map(order)).sort_values(["_o", "date", "model"]).drop(columns="_o")
lines = gpd.GeoDataFrame(lines[list(ref_lines.columns)], geometry=ref_lines.geometry.name, crs=ref_lines.crs)

gpkg = f"{args.out_stem}.gpkg"
pyogrio.write_dataframe(lines, gpkg, layer="lines", driver="GPKG")
pyogrio.write_dataframe(pref, gpkg, layer="model_preference", driver="GPKG", geometry_type="Unknown", append=True)
print(f"→ {gpkg}: lines {len(lines)}, model_preference {len(pref)}")


def per_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(length=df.geometry.length, year=pd.to_datetime(df["date"]).dt.year)
    return df.groupby(["location_id", "model", "year"]).agg(pieces=("length", "size"), length_m=("length", "sum"))


j = per_year(ref_lines).join(per_year(lines), lsuffix="_luke", rsuffix="_repro", how="outer")
line_diff = j[
    j["pieces_luke"].ne(j["pieces_repro"])
    | ~np.isclose(j["length_m_luke"].fillna(-1), j["length_m_repro"].fillna(-1), atol=0.01)
].reset_index()

p = ref_pref.set_index("location_id")[["model_preference", "points_height [0-10]", "points_segmentation [0-10]"]]
q = pref.set_index("location_id")[["model_preference", "points_height [0-10]", "points_segmentation [0-10]"]]
pc = p.join(q, lsuffix="_luke", rsuffix="_repro")
pc = pc[pc["model_preference_luke"].ne(pc["model_preference_repro"])].reset_index()
pc["what"] = "preference"
line_diff["what"] = "winner lines"
out = pd.concat([pc, line_diff], ignore_index=True, sort=False)
out = out[["what", "location_id", "model_preference_luke", "model_preference_repro",
           "points_height [0-10]_luke", "points_segmentation [0-10]_luke",
           "points_height [0-10]_repro", "points_segmentation [0-10]_repro",
           "model", "year", "pieces_luke", "pieces_repro", "length_m_luke", "length_m_repro"]]
csv = f"{args.out_stem}_differences.csv"
out.round(2).to_csv(csv, index=False)
print(f"→ {csv}: {len(pc)} preference differences, {len(line_diff)} region-years with different winner lines "
      f"in {line_diff['location_id'].nunique()} regions")
