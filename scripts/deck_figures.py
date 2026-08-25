"""Figures for the 2026-08-26 update deck (docs/presentations).

Writes PNGs to docs/presentations/fig/. Sources: experiments/loop/ledger.csv,
run bundles, structures.gpkg, existing showcase PNGs.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import src.paths as PATHS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/presentations/fig"
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#2BB5A6"
DARK = "#1F3A3D"
GREY = "#9AA5A6"
ORANGE = "#E07A3F"
plt.rcParams.update(
    {
        "font.family": ["Verdana", "DejaVu Sans"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.labelcolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
    }
)


def _label(ax, bars, fmt="{:.2f}"):
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.05,
            fmt.format(b.get_height()),
            ha="center",
            va="bottom",
            fontsize=13,
            color=DARK,
            fontweight="bold",
        )


def fig_three_axes():
    """Where the gain came from — same frozen holdout for all three bars."""
    led = pd.read_csv(ROOT / "experiments/loop/ledger.csv")
    rows = {
        "v0-baseline": "19 aug\n(start)",
        "e8-final-protected": "as 1\nopschonen",
        "i1-traj2": "as 2\nhistorie-\nfeatures",
    }
    d = led.drop_duplicates("variant", keep="last").set_index("variant").loc[list(rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, title in (
        (axes[0], "lgb_mae", "gemiddelde fout, alle vlakken (m/jr)"),
        (axes[1], "lgb_tail_mae", "fout risicogevallen > 2 m/jr (m/jr)"),
    ):
        vals = d[col].values
        bars = ax.bar(list(rows.values()), vals, color=[GREY, TEAL, TEAL], width=0.6)
        _label(ax, bars)
        ax.set_title(title, color=DARK, fontsize=13, pad=12)
        ax.set_ylim(0, vals.max() * 1.25)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        # arrows with deltas
        for i in (0, 1):
            dv = vals[i + 1] - vals[i]
            ax.annotate(
                f"{dv:+.2f}",
                xy=(i + 0.5, max(vals[i], vals[i + 1]) * 1.08),
                ha="center",
                color=ORANGE,
                fontsize=12,
                fontweight="bold",
            )
    fig.suptitle(
        "Zelfde vaste testset (1.174 vlakken) · opschonen leverde 3× zoveel als modelleren",
        color=DARK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "three_axes.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_structures_effect():
    """Kribben + kunstwerken mask: honest runs before/after."""
    runs = {
        "20260822-grad": "alleen Waal/Nederrijn-\nkribben (oud)",
        "20260825-structures": "kribben landelijk\n+ bruggen/kades/steigers",
    }
    vals = {}
    for r in runs:
        b = joblib.load(PATHS.DATA_DIR / f"04_model_outputs/{r}/bundle.joblib")[
            "results"
        ]
        vals[r] = b["5 – LightGBM"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in (
        (axes[0], "test_mae", "gemiddelde fout, alle vlakken (m/jr)"),
        (axes[1], "test_tail_mae", "fout risicogevallen > 2 m/jr (m/jr)"),
    ):
        v = [vals[r][key] for r in runs]
        bars = ax.bar(list(runs.values()), v, color=[GREY, TEAL], width=0.55)
        _label(ax, bars)
        ax.set_title(title, color=DARK, fontsize=13, pad=12)
        ax.set_ylim(0, max(v) * 1.25)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.annotate(
            f"{(v[1] / v[0] - 1) * 100:+.0f} %",
            xy=(0.5, max(v) * 1.1),
            ha="center",
            color=ORANGE,
            fontsize=14,
            fontweight="bold",
        )
    fig.suptitle(
        "Eerlijke validatie · alleen de structurenlaag verschilt tussen de twee runs",
        color=DARK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "structures_effect.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_structures_map():
    scope = gpd.read_file(PATHS.DATA_DIR / "01_raw/scope/scope_fase2.gpkg").to_crs(
        28992
    )
    old = gpd.read_file(
        PATHS.DATA_DIR / "01_raw/scope/Levering_erosie_data.gpkg", layer="Kribben_BKN"
    ).to_crs(28992)
    new = gpd.read_file(
        PATHS.DATA_DIR / "02_processed/structures/structures.gpkg", layer="kribben"
    )
    kw = gpd.read_file(
        PATHS.DATA_DIR / "02_processed/structures/structures.gpkg", layer="kunstwerken"
    )
    kw = kw[kw.categorie.isin(["brug", "kade_damwand", "steiger_afmeer", "sluis_stuw"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, title in zip(axes, ("maart – augustus", "nu")):
        scope.plot(ax=ax, color="#DDE7E8", linewidth=0)
        ax.set_title(title, color=DARK, fontsize=14)
        ax.set_axis_off()
        ax.set_aspect("equal")
    old.centroid.plot(ax=axes[0], color=ORANGE, markersize=2)
    new.centroid.plot(
        ax=axes[1], color=ORANGE, markersize=2, label=f"kribben ({len(new):,})"
    )
    kw.centroid.plot(
        ax=axes[1],
        color=DARK,
        markersize=2,
        label=f"bruggen/kades/steigers/sluizen ({len(kw):,})",
    )
    axes[0].text(
        0.02,
        0.02,
        f"kribben: {len(old):,}\nalleen Waal en Nederrijn-Lek",
        transform=axes[0].transAxes,
        color=DARK,
        fontsize=12,
        va="bottom",
    )
    axes[1].text(
        0.02,
        0.02,
        f"kribben: {len(new):,} (IJssel, Rijntakken, Maas)\nkunstwerken gemaskeerd: {len(kw):,}",
        transform=axes[1].transAxes,
        color=DARK,
        fontsize=12,
        va="bottom",
    )
    xmin, ymin, xmax, ymax = scope.total_bounds
    for ax in axes:
        ax.set_xlim(xmin - 5000, xmax + 5000)
        ax.set_ylim(ymin - 5000, ymax + 5000)
    fig.suptitle("Structurenlaag: van heuristiek naar feit", color=DARK, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "structures_map.png", dpi=200, facecolor="white")
    plt.close(fig)


def crops():
    """Crop existing showcase PNGs to presentation-sized panels."""
    show = Image.open(
        PATHS.DATA_DIR / "03_features/loop/variants/j5-R5-traj2/showcase_2027.png"
    )
    w, h = show.size
    # bottom row, panel 1 (ijssel1_l_0580_0590) and panel 4 (rijn_l_6420_6430)
    row_top, row_bot = int(h * 0.585), int(h * 0.985)
    show.crop((int(w * 0.02), row_top, int(w * 0.255), row_bot)).save(
        OUT / "r5_ijssel.png"
    )
    show.crop((int(w * 0.75), row_top, int(w * 0.995), row_bot)).save(
        OUT / "r5_rijn.png"
    )
    show.crop((int(w * 0.27), int(h * 0.04), int(w * 0.49), int(h * 0.55))).save(
        OUT / "r5_rijn_1220.png"
    )

    kb = Image.open(
        PATHS.DATA_DIR / "02_processed/triage/kribben_mask_before_after_osm.png"
    )
    w, h = kb.size
    # row 2, left pair (nederrijn_r_5210_5220): voor | na
    kb.crop((int(w * 0.03), int(h * 0.265), int(w * 0.505), int(h * 0.51))).save(
        OUT / "krib_before_after.png"
    )


if __name__ == "__main__":
    fig_three_axes()
    fig_structures_effect()
    fig_structures_map()
    crops()
    print("→", OUT, sorted(p.name for p in OUT.glob("*.png")))
