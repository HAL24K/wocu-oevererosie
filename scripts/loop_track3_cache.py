"""Track-3 cache: dense samples (60/line) on the e8-kept lines.

Resolution needs sampling density that scales with R (20/line leaves ~2
samples per segment at R=10). Cleaning must not silently change though: the
kept-line set is exactly the e8 survivors from the canonical 20-sample table,
and the kribben mask is re-applied at sample level on the dense points.
Output: data/03_features/loop/samples_dense_e8.parquet
"""

import logging
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src.erosion.region_inspector import RegionInspector
from src.loop.harness import load_caches
from src.loop.multi_t import E8_RULES
from src.loop.rules import RuleContext, apply_rules, make_structure_geom
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("t3cache")

N_DENSE = 60

caches = load_caches()
kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
geom = make_structure_geom(kribs, 10.0)

log.info("1 · e8 survivors on the canonical 20-sample table")
ctx = RuleContext(
    line_metrics=caches.line_metrics, cl_len=caches.static["cl_len"], structures=geom
)
s20 = apply_rules(caches.samples, ctx, E8_RULES)
kept = np.sort(s20["line_idx"].unique())
log.info("    %d kept lines (of %d)", len(kept), caches.samples["line_idx"].nunique())

log.info("2 · dense resampling (%d per line)", N_DENSE)
ri = RegionInspector()
lines = ri.lines.loc[kept]
fractions = np.linspace(0.0, 1.0, N_DENSE)
line_geoms = np.repeat(lines.geometry.values, N_DENSE)
cline_geoms = np.repeat(
    ri.geometry.centrelines.reindex(lines[LOCATION_ID]).values, N_DENSE
)
pts = shapely.line_interpolate_point(
    line_geoms, np.tile(fractions, len(lines)), normalized=True
)


def rep(col):
    return np.repeat(lines[col].values, N_DENSE)


dense = pd.DataFrame(
    {
        "line_idx": np.repeat(lines.index.values, N_DENSE),
        LOCATION_ID: rep(LOCATION_ID),
        "date": rep("date"),
        "year": rep("year"),
        "model": rep("model"),
        "station": shapely.line_locate_point(cline_geoms, pts, normalized=True),
        "dist": shapely.distance(pts, cline_geoms),
    }
).dropna(subset=["dist"])

log.info("3 · kribben mask on dense points")
inside = shapely.contains(geom, pts[dense.index.values])
dense = dense[~inside]

out = caches.cache_dir / "samples_dense_e8.parquet"
dense.reset_index(drop=True).to_parquet(out, index=False)
log.info(
    "    %s samples, %s lines, %s regions → %s",
    f"{len(dense):,}",
    f"{dense.line_idx.nunique():,}",
    f"{dense[LOCATION_ID].nunique():,}",
    out,
)
