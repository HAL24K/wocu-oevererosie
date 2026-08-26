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
        "Zelfde vaste testset (1 174 vlakken) · opschonen leverde 3× zoveel als modelleren",
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
        ax=axes[1], color=ORANGE, markersize=2, label=f"kribben ({f"{len(new):,}".replace(",", chr(8239))})"
    )
    kw.centroid.plot(
        ax=axes[1],
        color=DARK,
        markersize=2,
        label=f"bruggen/kades/steigers/sluizen ({f"{len(kw):,}".replace(",", chr(8239))})",
    )
    axes[0].text(
        0.02,
        0.02,
        f"kribben: {f"{len(old):,}".replace(",", chr(8239))}\nalleen Waal en Nederrijn-Lek",
        transform=axes[0].transAxes,
        color=DARK,
        fontsize=12,
        va="bottom",
    )
    axes[1].text(
        0.02,
        0.02,
        f"kribben: {f"{len(new):,}".replace(",", chr(8239))} (IJssel, Rijntakken, Maas)\nkunstwerken gemaskeerd: {f"{len(kw):,}".replace(",", chr(8239))}",
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


def _ledger():
    led = pd.read_csv(ROOT / "experiments/loop/ledger.csv")
    return led.drop_duplicates("variant", keep="last").set_index("variant")


def _two_panel(labels, mae, tail, colors, title, out, deltas=True, unit="m/jr"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, vals, t in (
        (axes[0], mae, f"gemiddelde fout, alle vlakken ({unit})"),
        (axes[1], tail, f"fout risicogevallen > 2 m/jr ({unit})"),
    ):
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        _label(ax, bars)
        ax.set_title(t, color=DARK, fontsize=13, pad=12)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=11)
        if deltas:
            for i in range(len(vals) - 1):
                ax.annotate(
                    f"{vals[i + 1] - vals[i]:+.2f}",
                    xy=(i + 0.5, max(vals[i], vals[i + 1]) * 1.08),
                    ha="center",
                    color=ORANGE,
                    fontsize=12,
                    fontweight="bold",
                )
    fig.suptitle(title, color=DARK, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / out, dpi=200, facecolor="white")
    plt.close(fig)


def fig_cleaning_ladder():
    d = _ledger().loc[["v0-baseline", "e8-final-protected"]]
    _two_panel(
        [
            "19 aug-model op de\nvaste testset: 3.99\n(= 4.12 op eigen split)",
            "+ opschoonregels\n(7 regels, herstel i.p.v. weggooien)",
        ],
        list(d.lgb_mae),
        list(d.lgb_tail_mae),
        [GREY, TEAL],
        "As 1 · zelfde vaste testset (1 174 vlakken) · resolutie 1 · dekking 0.94",
        "cleaning_ladder.png",
    )


def fig_structures_ablation():
    d = _ledger().loc[["s1-e8-nomask", "s2-e8-oldkribben", "s3-e8-newstructures"]]
    _two_panel(
        [
            "geen structuren-\nmasker",
            "kribben Waal +\nNederrijn (1 922)",
            "kribben landelijk +\nbruggen/kades/steigers",
        ],
        list(d.lgb_mae),
        list(d.lgb_tail_mae),
        [GREY, TEAL, TEAL],
        "Opschoonregels aan · alleen het structurenmasker verschilt · vaste testset",
        "structures_ablation.png",
    )


def fig_history():
    d = _ledger().loc[["e8-final-protected", "i1-traj2", "k5-H2-multi-w"]]
    labels = [
        "na opschonen\n(1 174 vlakken)",
        "+ historie-features\n(1 174 vlakken)",
        "+ paren ≥ 2 jaar\n(715 vlakken · andere meetlat)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, col, ncol, t in (
        (axes[0], "lgb_mae", "naive_mae", "gemiddelde fout, alle vlakken (m/jr)"),
        (axes[1], "lgb_tail_mae", None, "fout risicogevallen > 2 m/jr (m/jr)"),
    ):
        vals = list(d[col])
        bars = ax.bar(labels, vals, color=[GREY, TEAL, ORANGE], width=0.6)
        _label(ax, bars)
        if ncol:
            nv = list(d[ncol])
            ax.plot(range(3), nv, "_", color=DARK, ms=40, mew=2.5)
            for k, (v, n) in enumerate(zip(vals, nv)):
                ax.text(k, n + 0.08, f"naïef {n:.2f}\nskill {100 * (1 - v / n):+.0f} %", ha="center", fontsize=9.5, color=DARK)
        ax.set_title(t, color=DARK, fontsize=13, pad=12)
        ax.set_ylim(0, max(vals + (list(d[ncol]) if ncol else [])) * 1.4)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9.5)
    fig.suptitle("As 2 · vaste testset · resolutie 1 · oranje = andere meetlat (de naïeve fout zakt mee)", color=DARK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "history.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_resolution_sweep():
    d = _ledger().loc[[f"deck-R{r}-traj2" for r in (1, 2, 5, 10, 20)]]
    R = [1, 2, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(
        R, d.lgb_mae, "-o", color=TEAL, lw=2.5, ms=8, label="fout per segment (m/jr)"
    )
    ax.plot(
        R,
        d.core_mae,
        "--s",
        color=DARK,
        lw=2,
        ms=7,
        label="fout per vlak na samenvoegen (max) (m/jr)",
    )
    ax.plot(
        R,
        d.lgb_tail_mae,
        "-o",
        color=ORANGE,
        lw=2,
        ms=7,
        label="fout risicogevallen per segment (m/jr)",
    )
    for x, y in zip(R, d.lgb_mae):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color=TEAL,
            fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xticks(R)
    ax.set_xticklabels([f"R = {r}\n≈ {round(100 / r)} m" for r in R])
    ax.set_ylim(0, 5)
    ax.set_xlabel("segmenten per vlak (vlak ≈ 100 m)")
    ax.legend(frameon=False, fontsize=11)
    ax.axvspan(15, 30, color=GREY, alpha=0.12)
    ax.text(
        20,
        4.6,
        "60 punten per lijn:\nsegmenten raken leeg",
        ha="center",
        fontsize=10,
        color=GREY,
    )
    fig.suptitle(
        "As 3 · resolutie: fijner = nauwkeuriger, tot de lijn-bemonstering op is",
        color=DARK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "resolution_sweep.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_horizon_sweep():
    d = _ledger().loc[[f"hz-R{r}" for r in (1, 2, 5, 10, 20)]]
    R = [1, 2, 5, 10, 20]
    starved = [0, 1.7, 5.9, 9.1, 30.9]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axvspan(14, 28, color=GREY, alpha=0.12)
    ax.text(20, 3.75, "bemonstering op:\n31 % segmenten zonder meting", ha="center", fontsize=10, color=GREY)
    ax.plot(R, d.lgb_tail_mae, "-o", color=ORANGE, lw=2, ms=7, label="fout risicogevallen > 2 m/jr")
    ax.plot(R, d.naive_mae, "--s", color=GREY, lw=2, ms=7, label="naïef (gemiddelde)")
    ax.plot(R, d.lgb_mae, "-o", color=TEAL, lw=2.5, ms=8, label="fout per segment")
    for x, y, n in zip(R, d.lgb_mae, d.naive_mae):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -18), ha="center", color=TEAL, fontweight="bold")
        ax.annotate(f"skill {100 * (1 - y / n):+.0f} %", (x, n), textcoords="offset points", xytext=(0, 9), ha="center", color=DARK, fontsize=9.5)
    ax.axvline(10, color=DARK, lw=1, ls=":")
    ax.text(10.4, 2.35, "keuze: R = 10", color=DARK, fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(R)
    ax.set_xticklabels([f"R = {r}\n≈ {round(100 / r)} m\n{s:.0f} % leeg" for r, s in zip(R, starved)])
    ax.set_ylim(0, 4)
    ax.set_xlabel("segmenten per vlak (vlak ≈ 100 m) · segment-datums zonder meting")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle("As 3 · segment × paren ≥ 2 jaar · vaste testset · m/jr", color=DARK, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "horizon_sweep.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_rule_ablation():
    led = _ledger()
    rules = {
        "r5-minsamples8": "+ te weinig punten",
        "r2-nearbank": "+ verkeerde oever",
        "r1-tortuosity": "+ kronkelende lijn",
        "r3-maze": "+ doolhof",
        "r4-fragment": "+ fragment",
        "r6-temporal": "+ uitschieter in de tijd",
        "r0-newstructures": "+ kribben & kunstwerken landelijk",
    }
    order = sorted(rules, key=lambda v: -led.loc[v].lgb_mae)
    keys = ["v0-baseline"] + order + ["e8-final-protected"]
    labels = ["19 aug (= 4.12 eigen split)"] + [rules[k] for k in order] + ["alle regels samen"]
    colors = [GREY] + [TEAL if k != "r0-newstructures" else DARK for k in order] + [ORANGE]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    for ax, col, t in ((axes[0], "lgb_mae", "gemiddelde fout, alle vlakken (m/jr)"), (axes[1], "lgb_tail_mae", "fout risicogevallen > 2 m/jr (m/jr)")):
        vals = [led.loc[k][col] for k in keys]
        bars = ax.bar(range(len(keys)), vals, color=colors, width=0.65)
        _label(ax, bars)
        ax.set_title(t, color=DARK, fontsize=12, pad=8, loc="left")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    cov = [led.loc[k].coverage_core for k in keys]
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels([f"{l}\ndekking {c:.2f}" for l, c in zip(labels, cov)], rotation=28, ha="right", fontsize=9.5)
    fig.suptitle("Elke regel apart op het 19 aug-model · vaste testset (1 174 vlakken) · resolutie 1 · donker = kost dekking, geen foutwinst", color=DARK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "rule_ablation.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_matrix():
    led = _ledger()
    steps = [
        ("19 aug (uitgangspunt)", led.loc["v0-baseline"], GREY),
        ("+ opschoonregels", led.loc["e8-final-protected"], TEAL),
        ("+ kribben & kunstwerken landelijk", led.loc["s3-e8-newstructures"], TEAL),
        ("+ historie-features", led.loc["i1-traj2"], TEAL),
        ("+ paren ≥ 2 jaar (andere meetlat)", led.loc["k5-H2-multi-w"], ORANGE),
        ("+ resolutie R = 10 (per segment)", led.loc["hz-R10"], ORANGE),
    ]
    labels = [s[0] for s in steps][::-1]
    mae = [s[1].lgb_mae for s in steps][::-1]
    tail = [s[1].lgb_tail_mae for s in steps][::-1]
    cols = [s[2] for s in steps][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, vals, t in (
        (axes[0], mae, "gemiddelde fout (m/jr)"),
        (axes[1], tail, "fout risicogevallen > 2 m/jr (m/jr)"),
    ):
        bars = ax.barh(labels, vals, color=cols, height=0.6)
        for b in bars:
            ax.text(
                b.get_width() + 0.05,
                b.get_y() + b.get_height() / 2,
                f"{b.get_width():.2f}",
                va="center",
                fontsize=12,
                color=DARK,
                fontweight="bold",
            )
        ax.set_title(t, color=DARK, fontsize=13)
        ax.set_xlim(0, max(vals) * 1.2)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", labelsize=12)
    fig.suptitle(
        "Alles op dezelfde vaste testset · oranje = andere meetlat (paren ≥ 2 jaar; de naïeve fout zakt mee)",
        color=DARK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "matrix.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_three_axes()
    fig_structures_effect()
    fig_structures_map()
    crops()
    fig_cleaning_ladder()
    fig_structures_ablation()
    fig_history()
    fig_resolution_sweep()
    fig_horizon_sweep()
    fig_rule_ablation()
    fig_matrix()
    print("→", OUT, sorted(p.name for p in OUT.glob("*.png")))
