"""Padding sweep for the kribben sample mask.

Per-sample distance to the nearest krib is computed once; each padding P is
then a threshold. For every P the artefact metrics are recomputed from the
surviving samples:
  - multiline-gap regions (two banks drawn in one survey)
  - temporal-jump regions (year-collapsed series moves > 50 m/yr)
  - robust spread (IQR) of the yearly velocity across all regions
  - dead surveys (< 30% of station bins survive) and masked-sample fraction
The right P is where artefact curves flatten while coverage cost still rises.
"""

import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from src.erosion.region_inspector import RegionInspector
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEAL, GREY, ORANGE, BLUE = "#2ab3a3", "#444444", "#eb6834", "#2a78d6"
plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]

PADDINGS = [None, 0, 5, 10, 15, 20, 25, 30]
N_BINS = 20

ri = RegionInspector()
s = ri.samples.reset_index(names="line_idx")
s["bin"] = np.minimum((s["station"] * N_BINS).astype(int), N_BINS - 1)

kribs = gpd.read_file(
    ri.cfg.data_dir / "01_raw/scope/Levering_erosie_data.gpkg", layer="Kribben_BKN"
).to_crs(28992)

pts = gpd.GeoDataFrame(
    s[["line_idx"]], geometry=gpd.points_from_xy(s["x"], s["y"]), crs=28992
)
near = gpd.sjoin_nearest(
    pts, kribs[["geometry"]], how="left", max_distance=35.0, distance_col="d_krib"
)
s["d_krib"] = near.groupby(level=0)["d_krib"].min().values
print(f"samples within 35 m of a krib: {(s['d_krib'].notna()).mean():.1%}")


def metrics(surv: pd.DataFrame) -> dict:
    # per-line p50 from surviving samples (>=5 samples to count as a line)
    g = surv.groupby("line_idx")
    lp = pd.DataFrame(
        {
            LOCATION_ID: g[LOCATION_ID].first(),
            "date": g["date"].first(),
            "p50": g["dist"].median(),
            "n": g.size(),
        }
    )
    lp = lp[lp["n"] >= 5]

    per_date = lp.groupby([LOCATION_ID, "date"])["p50"].agg(["min", "max", "count"])
    per_date["gap"] = per_date["max"] - per_date["min"]
    per_date["ratio"] = per_date["max"] / per_date["min"].clip(lower=0.1)
    multi = (
        ((per_date["gap"] > 25) & (per_date["ratio"] > 1.5)).groupby(LOCATION_ID).any()
    )

    series = per_date["min"].reset_index()
    series["year"] = series["date"].dt.year
    yearly = (
        series.groupby([LOCATION_ID, "year"])["min"]
        .median()
        .reset_index()
        .sort_values([LOCATION_ID, "year"])
    )
    gr = yearly.groupby(LOCATION_ID)
    v = (gr["min"].diff() / gr["year"].diff()).dropna()
    jump = v.abs().groupby(yearly.loc[v.index, LOCATION_ID]).max() > 50

    surv_bins = surv.groupby([LOCATION_ID, "date"])["bin"].nunique()
    all_surveys = s.groupby([LOCATION_ID, "date"]).size()
    dead = len(all_surveys) - (surv_bins >= 0.3 * N_BINS).sum()

    return {
        "multiline_regions": int(multi.sum()),
        "jump_regions": int(jump.sum()),
        "v_iqr": float(v.quantile(0.75) - v.quantile(0.25)),
        "dead_surveys": int(dead),
        "masked_frac": 1.0 - len(surv) / len(s),
    }


rows = {}
for p in PADDINGS:
    surv = s if p is None else s[~(s["d_krib"] <= p)]
    rows["geen" if p is None else f"{p} m"] = metrics(surv)
    print("geen" if p is None else f"{p:>2} m", rows["geen" if p is None else f"{p} m"])

df = pd.DataFrame(rows).T
df.to_csv(ri.cfg.data_dir / "02_processed/triage/kribben_padding_sweep.csv")

fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), dpi=150)
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.16, wspace=0.30)
x = np.arange(len(df))
panels = [
    ("multiline_regions", "regio's met 2 oevers in één meetmoment", ORANGE),
    ("jump_regions", "regio's met |v| > 50 m/jr", BLUE),
    ("v_iqr", "IQR jaarsnelheid (m/jr)", GREY),
]
for ax, (col, title, c) in zip(axes, panels, strict=False):
    ax.plot(x, df[col], "-o", color=c, ms=5)
    ax.set_xticks(x, df.index, fontsize=8)
    ax.set_title(title, fontsize=10, color=TEAL, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("krib-padding", fontsize=9)
ax2 = axes[2].twinx()
ax2.plot(x, 100 * df["masked_frac"], "--s", color=ORANGE, ms=4, alpha=0.7)
ax2.set_ylabel("% samples gemaskeerd", fontsize=8, color=ORANGE)
ax2.tick_params(labelsize=8, colors=ORANGE)
ax2.spines[["top"]].set_visible(False)
fig.suptitle(
    "Kribben-masker: artefactmetrieken vs. padding",
    fontsize=12,
    color=GREY,
    fontweight="bold",
)
out = ri.cfg.data_dir / "02_processed/triage/kribben_padding_sweep.png"
fig.savefig(out, facecolor="white")
print(f"\nplot → {out}")
print(df.round(3).to_string())
