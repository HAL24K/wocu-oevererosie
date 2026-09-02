"""Export every survey touched by the cleaning stack, one flag column per rule.

Extends export_geometry_flags.py (kronkelend/doolhof only) to the full e8
recipe. Each rule is applied *individually* to the base data, so a flag means
"this rule, on its own, removes samples/lines/surveys here" — matching the
2026-08-26 deck counts (rule_touched.png).

Rules → columns (True = touched):
  kribbenmasker      structure_mask, nationwide kribben + kunstwerken, 10 m
  kronkelend         max_tortuosity_line  (length/chord > 3, length > 30 m)
  verkeerde_oever    near_bank_line       (median dist < 0.25 x region ref)
  doolhof            maze_survey          (line length > 1.8 x centreline AND IQR > 20 m)
  fragment           fragment_survey      (< 25 % of the region covered)
  te_weinig_punten   min_samples_survey   (< 8 samples)
  tijd_uitschieter   temporal_outlier_survey (> 15 m off the detrended trend)

Output ``data/02_processed/triage/sam_filtered_<date>.gpkg``:
  - ``surveys``  every line of a touched survey; per-rule booleans, ``flags``
                 (comma-joined), ``n_rules``, ``volledig_weg`` (survey fully
                 removed by >= 1 rule alone)
  - ``vlakken``  scope polygons with per-rule survey counts
plus a CSV with one row per touched survey.

Run: ``uv run python scripts/export_filtered_sam.py``
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from experiments.loop.harness.harness import load_caches
from src.cleaning.rules import RuleContext, apply_rules, make_structure_geom
from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID, normalise_location_id

SURVEY = [LOCATION_ID, "date"]
RULES = {
    "kronkelend": ("max_tortuosity_line", {"max_tort": 3.0}),
    "verkeerde_oever": ("near_bank_line", {"frac": 0.25, "min_ref": 30.0}),
    "doolhof": ("maze_survey", {"max_ratio": 1.8, "min_iqr": 20.0}),
    "fragment": ("fragment_survey", {"min_cov": 0.25}),
    "te_weinig_punten": ("min_samples_survey", {"min_n": 8}),
    "tijd_uitschieter": (
        "temporal_outlier_survey",
        {"max_dev": 15.0, "min_surveys": 3, "detrend": True, "protect_min_years": 3},
    ),
}

caches = load_caches()
cfg = ExperimentConfig(experiment="loop")
samples = caches.samples
stamp = pd.Timestamp.today().strftime("%Y%m%d")
out_dir = cfg.data_dir / "02_processed/triage"

sg = cfg.data_dir / "02_processed/structures/structures.gpkg"
kribs = gpd.read_file(sg, layer="kribben").to_crs(28992)
kw = gpd.read_file(sg, layer="kunstwerken").to_crs(28992)
kw = kw[kw["categorie"].isin(["brug", "kade_damwand", "steiger_afmeer", "sluis_stuw"])]
structures = make_structure_geom(pd.concat([kribs[["geometry"]], kw[["geometry"]]]), 10.0)

base = samples.groupby(SURVEY).size().rename("n_before")
flags = base.to_frame()

for col, (rule, params) in [("kribbenmasker", ("structure_mask", {}))] + list(
    RULES.items()
):
    ctx = RuleContext(
        line_metrics=caches.line_metrics,
        cl_len=caches.static["cl_len"],
        structures=structures if rule == "structure_mask" else None,
    )
    after = apply_rules(samples, ctx, [(rule, params)])
    n_after = after.groupby(SURVEY).size().reindex(base.index, fill_value=0)
    flags[col] = n_after < base
    flags[f"_gone_{col}"] = n_after == 0
    print(f"{col:18s} surveys touched {int(flags[col].sum()):6d} · fully removed {int(flags[f'_gone_{col}'].sum()):5d}")

rule_cols = ["kribbenmasker", *RULES.keys()]
flags["n_rules"] = flags[rule_cols].sum(axis=1)
flags["volledig_weg"] = flags[[f"_gone_{c}" for c in rule_cols]].any(axis=1)
touched = flags[flags["n_rules"] > 0].drop(columns=[f"_gone_{c}" for c in rule_cols])
touched["flags"] = touched[rule_cols].apply(
    lambda r: ",".join(c for c in rule_cols if r[c]), axis=1
)
touched = touched.reset_index()

# ── line geometries of touched surveys ───────────────────────────────────────
lines = normalise_location_id(gpd.read_file(cfg.hybrid_gpkg, layer="lines"))
lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
lines["date"] = pd.to_datetime(lines["date"])
key = pd.MultiIndex.from_frame(touched[SURVEY])
sel = lines[pd.MultiIndex.from_frame(lines[SURVEY]).isin(key)].copy()
sel = sel.join(caches.line_metrics[["tortuosity", "length"]], how="left")
sel = sel.merge(touched.drop(columns="n_before"), on=SURVEY, how="left")
sel["date"] = sel["date"].dt.strftime("%Y-%m-%d")
sel[rule_cols + ["volledig_weg"]] = sel[rule_cols + ["volledig_weg"]].astype(int)

# ── region polygons with per-rule counts ─────────────────────────────────────
scope = normalise_location_id(gpd.read_file(cfg.scope_gpkg)).set_index(LOCATION_ID)
agg = {c: (c, "sum") for c in rule_cols}
per_region = (
    touched.groupby(LOCATION_ID)
    .agg(n_surveys=("flags", "size"), n_volledig_weg=("volledig_weg", "sum"), **agg)
    .join(scope.geometry, how="left")
)
vlakken = gpd.GeoDataFrame(per_region.reset_index(), geometry="geometry", crs=scope.crs)

gpkg = out_dir / f"sam_filtered_{stamp}.gpkg"
if gpkg.exists():
    gpkg.unlink()
sel.to_file(gpkg, layer="surveys", driver="GPKG")
vlakken.to_file(gpkg, layer="vlakken", driver="GPKG")
csv = out_dir / f"sam_filtered_{stamp}.csv"
touched.sort_values(["n_rules", LOCATION_ID], ascending=[False, True]).to_csv(csv, index=False)

print()
print("touched surveys:", len(touched), "· volledig weg:", int(touched.volledig_weg.sum()))
print("vlakken        :", len(vlakken))
print("model split    :", sel["model"].value_counts(dropna=False).to_dict())
print("→", gpkg)
print("→", csv)
