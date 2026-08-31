"""Export SAM surveys with suspect geometry for satellite-image review (Etienne).

Reproduces the two cleaning rules on the loop base data (same data the
2026-08-26 counts came from) and labels every flagged survey:

  - ``kronkelend``  max_tortuosity_line: >= 1 line with length/chord > 3
                    (and length > 30 m)
  - ``doolhof``     maze_survey: survey line length > 1.8 x centreline length
                    AND sample-distance IQR > 20 m
  - ``beide``       both

Output ``data/02_processed/triage/geometry_flags_<date>.gpkg``:
  - ``lijnen``    every line of a flagged survey; flag + per-survey metrics,
                  ``line_kronkelend`` marks the individual offending lines
  - ``vlakken``   scope polygons of flagged regions with survey counts
plus ``geometry_flags_<date>.csv`` (one row per flagged survey) to send along.

Run: ``uv run python scripts/export_geometry_flags.py``
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from experiments.loop.harness.harness import load_caches
from src.pipeline.config import ExperimentConfig
from src.sources.geometry import LOCATION_ID, normalise_location_id

TORT_MAX, TORT_MIN_LEN = 3.0, 30.0
MAZE_RATIO, MAZE_IQR = 1.8, 20.0
SURVEY = [LOCATION_ID, "date"]

caches = load_caches()
cfg = ExperimentConfig(experiment="loop")
stamp = pd.Timestamp.today().strftime("%Y%m%d")
out_dir = cfg.data_dir / "02_processed/triage"

lm = caches.line_metrics
s = caches.samples
cl_len = caches.static["cl_len"]

# ── kronkelend: per line, then per survey ─────────────────────────────────────
lm = lm.assign(line_kronkelend=(lm["tortuosity"] > TORT_MAX) & (lm["length"] > TORT_MIN_LEN))
kronkelend = set(
    map(tuple, lm[lm["line_kronkelend"]][SURVEY].drop_duplicates().itertuples(index=False))
)

# ── doolhof: per survey ──────────────────────────────────────────────────────
survey_lines = lm.groupby(SURVEY)["length"].sum().rename("total_line_len").reset_index()
survey_lines["length_ratio"] = survey_lines["total_line_len"] / survey_lines[
    LOCATION_ID
].map(cl_len)
q = s.groupby(SURVEY)["dist"].quantile([0.25, 0.75]).unstack()
survey_lines = survey_lines.merge(
    (q[0.75] - q[0.25]).rename("dist_iqr").reset_index(), on=SURVEY, how="left"
)
survey_lines["doolhof"] = (survey_lines["length_ratio"] > MAZE_RATIO) & (
    survey_lines["dist_iqr"] > MAZE_IQR
)
survey_lines["kronkelend"] = [
    t in kronkelend for t in map(tuple, survey_lines[SURVEY].itertuples(index=False))
]
flagged = survey_lines[survey_lines["doolhof"] | survey_lines["kronkelend"]].copy()
flagged["flag"] = "kronkelend"
flagged.loc[flagged["doolhof"] & ~flagged["kronkelend"], "flag"] = "doolhof"
flagged.loc[flagged["doolhof"] & flagged["kronkelend"], "flag"] = "beide"

# ── line geometries of flagged surveys ───────────────────────────────────────
lines = normalise_location_id(gpd.read_file(cfg.hybrid_gpkg, layer="lines"))
lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
lines["date"] = pd.to_datetime(lines["date"])
key = pd.MultiIndex.from_frame(flagged[SURVEY])
sel = lines[pd.MultiIndex.from_frame(lines[SURVEY]).isin(key)].copy()
# line-level metrics first: the metrics cache shares the lines index (line_idx)
sel = sel.join(lm[["tortuosity", "length", "line_kronkelend"]], how="left")
sel = sel.merge(
    flagged[SURVEY + ["flag", "length_ratio", "dist_iqr"]], on=SURVEY, how="left"
)
sel["model"] = sel.get("model", pd.NA)
sel["date"] = sel["date"].dt.strftime("%Y-%m-%d")

# ── region polygons with counts ──────────────────────────────────────────────
scope = normalise_location_id(gpd.read_file(cfg.scope_gpkg)).set_index(LOCATION_ID)
per_region = (
    flagged.groupby(LOCATION_ID)
    .agg(
        n_surveys=("flag", "size"),
        n_kronkelend=("kronkelend", "sum"),
        n_doolhof=("doolhof", "sum"),
        max_ratio=("length_ratio", "max"),
        max_iqr=("dist_iqr", "max"),
    )
    .join(scope.geometry, how="left")
)
per_region["flag"] = "kronkelend"
per_region.loc[(per_region.n_doolhof > 0) & (per_region.n_kronkelend == 0), "flag"] = "doolhof"
per_region.loc[(per_region.n_doolhof > 0) & (per_region.n_kronkelend > 0), "flag"] = "beide"
vlakken = gpd.GeoDataFrame(per_region.reset_index(), geometry="geometry", crs=scope.crs)

gpkg = out_dir / f"geometry_flags_{stamp}.gpkg"
if gpkg.exists():
    gpkg.unlink()
sel.to_file(gpkg, layer="lijnen", driver="GPKG")
vlakken.to_file(gpkg, layer="vlakken", driver="GPKG")
csv = out_dir / f"geometry_flags_{stamp}.csv"
flagged.sort_values(["flag", LOCATION_ID, "date"]).to_csv(csv, index=False)

print("surveys  :", flagged["flag"].value_counts().to_dict(), f"(totaal {len(flagged)})")
print("vlakken  :", vlakken["flag"].value_counts().to_dict(), f"(totaal {len(vlakken)})")
print(
    "regions kronkelend:", flagged[flagged.kronkelend][LOCATION_ID].nunique(),
    "· doolhof:", flagged[flagged.doolhof][LOCATION_ID].nunique(),
)
print("lijnen   :", len(sel), "→", gpkg)
print("csv      :", csv)
