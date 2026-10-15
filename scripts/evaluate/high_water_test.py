"""Do SAM bank lines lie further inland at high water? (hypothesis Alexander, 13 Oct)

SAM segments water from land on a satellite photo, so its bank line is the waterline of
the day. Per strip the reference position is the median over all its SAM surveys; per
survey date and river the median offset of the strips from their reference is compared
with the discharge on that day (daily mean, ``data/02_context/discharge_daily``, as a
percentile per station so the rivers compare). Also: do the rivers have high water at
the same time?

Writes ``compare/high_water_test.png`` and ``compare/high_water_by_date.csv``.

Run: ``PYTHONPATH=. uv run python scripts/evaluate/high_water_test.py --run 20261013-preclean``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.config import ExperimentConfig  # noqa: E402

DAILY = Path("data/02_context/discharge_daily")
# river part of the scope id → gauging station
STATION = {
    "rijn": "lobith.bovenrijn.tolkamer",
    "nederrijn": "pannerden.pannerdenschkanaal",
    "ijssel1": "olst",
    "ijssel2": "olst",
    "maas2": "venlo",
    "maas3": "megen.maas",
}
RIVER = {
    "rijn": "Waal/Rijn",
    "nederrijn": "Neder-Rijn",
    "ijssel1": "IJssel",
    "ijssel2": "IJssel",
    "maas2": "Maas",
    "maas3": "Maas",
}

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--min-strips", type=int, default=200)
args = ap.parse_args()
cfg = ExperimentConfig.from_run(args.run)

obs = pd.read_parquet(cfg.features_dir / "segment_obs.parquet")
obs = obs[obs["model"] == "segmentation"].copy()
obs["part"] = obs["location_id"].str.split("_").str[0]
obs = obs[obs["part"].isin(STATION)]
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
ref = obs.groupby(["location_id", "seg"])["dist_m"].transform("median")
obs["offset"] = obs["dist_m"] - ref  # + = further inland than usual

# discharge percentile per station and day
q = {}
for st in set(STATION.values()):
    f = DAILY / f"{st}.parquet"
    if f.exists():
        d = pd.read_parquet(f).set_index("date")["discharge_m3s"]
        q[st] = pd.DataFrame({"q": d, "q_pct": d.rank(pct=True) * 100})
by = (
    obs.groupby(["part", "date"])
    .agg(offset=("offset", "median"), strips=("offset", "size"))
    .reset_index()
)
by = by[by["strips"] >= args.min_strips]
by["station"] = by["part"].map(STATION)
by["river"] = by["part"].map(RIVER)
by["q"] = [
    q[s]["q"].get(d, np.nan) if s in q else np.nan
    for s, d in zip(by["station"], by["date"], strict=True)
]
by["q_pct"] = [
    q[s]["q_pct"].get(d, np.nan) if s in q else np.nan
    for s, d in zip(by["station"], by["date"], strict=True)
]
by["month"] = by["date"].dt.month
by.to_csv(cfg.compare_dir / "high_water_by_date.csv", index=False)

print(
    "survey dates per river with discharge:",
    by.dropna(subset=["q"]).groupby("river").size().to_dict(),
)
for river, g in by.dropna(subset=["q_pct"]).groupby("river"):
    rho = g[["offset", "q_pct"]].corr(method="spearman").iloc[0, 1]
    hi, lo = (
        g[g["q_pct"] >= 75]["offset"].median(),
        g[g["q_pct"] <= 50]["offset"].median(),
    )
    print(
        f"{river:11s} n={len(g):3d}  Spearman offset~discharge {rho:+.2f} | median offset at "
        f"discharge >= p75: {hi:+.1f} m, <= p50: {lo:+.1f} m"
    )

# Are the rivers high at the same time? Correlation of daily discharge percentiles.
pct = pd.DataFrame({s: v["q_pct"] for s, v in q.items()}).dropna()
print("\ndaily discharge percentile correlation between stations (Spearman):")
print(pct.corr(method="spearman").round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
colors = {
    "Waal/Rijn": "#1f6fe0",
    "Neder-Rijn": "#2ab4a8",
    "IJssel": "#d97a2b",
    "Maas": "#8c0fd5",
}
for river, g in by.dropna(subset=["q_pct"]).groupby("river"):
    axes[0].scatter(
        g["q_pct"],
        g["offset"],
        s=np.clip(g["strips"] / 40, 8, 80),
        alpha=0.7,
        color=colors[river],
        label=river,
    )
axes[0].axhline(0, color="grey", lw=1)
axes[0].set_xlabel("afvoer op de opnamedag (percentiel per station)")
axes[0].set_ylabel("mediane verschuiving SAM-lijn (m, + = landinwaarts)")
axes[0].set_title("Verschuiving tegen afvoer")
axes[0].legend()
m = by.groupby(["month", "river"])["offset"].median().unstack()
m.plot.bar(ax=axes[1], color=[colors[c] for c in m.columns], width=0.8)
axes[1].axhline(0, color="grey", lw=1)
axes[1].set_xlabel("maand van opname")
axes[1].set_ylabel("mediane verschuiving (m)")
axes[1].set_title("Verschuiving per maand")
for st, v in q.items():
    axes[2].plot(v.index, v["q_pct"].rolling(15).mean(), lw=1, label=st.split(".")[0])
for d in sorted(by["date"].unique()):
    axes[2].axvline(d, color="grey", lw=0.3, alpha=0.4)
axes[2].set_title(
    "Afvoerpercentiel per station (15 dagen gemiddeld); grijs = opnamedagen"
)
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(cfg.compare_dir / "high_water_test.png", dpi=130)
print(f"→ {cfg.compare_dir / 'high_water_test.png'}")
