"""Deck 2026-08-26 · As 1: single-rule ablation + regions touched per rule.

A. r1..r6: v0 recipe + ONE e8-final rule each (frozen holdout, ledger rows).
B. Per rule applied alone on the base samples: regions touched / removed,
   split by model preference (hoogtemodel / SAM / overig).
C. fig/rule_touched.png, fig/rule_ablation.png
D. docs/presentations/rules_20260826.md
"""

import logging
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.loop.harness.harness import load_caches, run_variant  # noqa: E402
from src.cleaning.rules import RuleContext, apply_rules, make_structure_geom  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "docs/presentations/fig"
MD = ROOT / "docs/presentations/rules_20260826.md"
TEAL, DARK, GREY, ORANGE = "#2BB5A6", "#1F3A3D", "#9AA5A6", "#E07A3F"
plt.rcParams.update(
    {
        "font.family": ["Verdana", "DejaVu Sans"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
    }
)


def th(n: int) -> str:
    return f"{n:,}".replace(",", " ")


caches = load_caches()
kribs = gpd.read_file(
    caches.cache_dir.parents[1] / "01_raw/scope/Levering_erosie_data.gpkg",
    layer="Kribben_BKN",
).to_crs(28992)
KRIB_GEOM = make_structure_geom(kribs, 10.0)

BASE = [("structure_mask", {}), ("min_samples_survey", {"min_n": 12})]
RULES = {
    "r1-tortuosity": ("max_tortuosity_line", {"max_tort": 3.0}, "kronkelende lijn"),
    "r2-nearbank": ("near_bank_line", {"frac": 0.25, "min_ref": 30.0}, "verkeerde oever"),
    "r3-maze": ("maze_survey", {"max_ratio": 1.8, "min_iqr": 20.0}, "doolhof"),
    "r4-fragment": ("fragment_survey", {"min_cov": 0.25}, "fragment"),
    "r5-minsamples8": ("min_samples_survey", {"min_n": 8}, "te weinig punten"),
    "r6-temporal": (
        "temporal_outlier_survey",
        {"max_dev": 15.0, "min_surveys": 3, "detrend": True, "protect_min_years": 3},
        "uitschieter in de tijd",
    ),
}

# ── A · single-rule ablation ──────────────────────────────────────────────────
if "--skip-ablation" not in sys.argv:
    for name, (rule, params, _) in RULES.items():
        if rule == "min_samples_survey":
            rules = [("structure_mask", {}), (rule, params)]
        else:
            rules = BASE + [(rule, params)]
        run_variant(name, caches, rules, structures=KRIB_GEOM, v_limit=50.0, notes="deck: single rule on v0")

# ── B · regions touched per rule (applied alone on the base samples) ─────────
pref = gpd.read_file(
    caches.cache_dir.parents[1] / "02_processed/hybrid/model_preference_20260710.gpkg",
    columns=["location_id", "model_preference"],
)
print("model_preference values:", pref.model_preference.value_counts().to_dict())
pmap = pref.set_index("location_id")["model_preference"].astype(str).str.lower()


def cls(loc: str) -> str:
    v = pmap.get(loc, "")
    if "height" in v or "hoogte" in v:
        return "hoogtemodel"
    if "segm" in v or "sam" in v:
        return "SAM"
    return "overig"


samples = caches.samples
base_regions = pd.Index(samples["location_id"].unique())
base_cls = pd.Series({loc: cls(loc) for loc in base_regions})
totals = base_cls.value_counts().to_dict()


def surveys_of(s):
    return s.groupby(["location_id", "date"]).size().index


def touched_by(rule, params, structures=None):
    ctx = RuleContext(
        line_metrics=caches.line_metrics, cl_len=caches.static["cl_len"], structures=structures
    )
    before = samples
    after = apply_rules(before, ctx, [(rule, params)])
    lost = before.groupby("location_id").size() - after.groupby("location_id").size().reindex(
        before["location_id"].unique(), fill_value=0
    )
    touched = set(lost[lost > 0].index)
    removed = set(base_regions) - set(after["location_id"].unique())
    return touched, removed


COND = [("structure_mask", "structure_mask", {}, "kribbenmasker")] + [
    (k, r, p, lab) for k, (r, p, lab) in RULES.items()
]
rows, union_t, union_r = [], set(), set()
for key, rule, params, label in COND:
    t, r = touched_by(rule, params, structures=KRIB_GEOM if rule == "structure_mask" else None)
    union_t |= t
    union_r |= r
    rows.append(
        {
            "rule": key,
            "label": label,
            **{f"touched_{c}": sum(base_cls[x] == c for x in t) for c in ("hoogtemodel", "SAM", "overig")},
            "touched": len(t),
            **{f"removed_{c}": sum(base_cls[x] == c for x in r) for c in ("hoogtemodel", "SAM", "overig")},
            "removed": len(r),
        }
    )
    print(f"{key:16s} touched {len(t):6d}  removed {len(r):5d}")
rows.append(
    {
        "rule": "union",
        "label": "≥ 1 regel",
        **{f"touched_{c}": sum(base_cls[x] == c for x in union_t) for c in ("hoogtemodel", "SAM", "overig")},
        "touched": len(union_t),
        **{f"removed_{c}": sum(base_cls[x] == c for x in union_r) for c in ("hoogtemodel", "SAM", "overig")},
        "removed": len(union_r),
    }
)
touch = pd.DataFrame(rows)

# ── C · figures ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.8))
labels = [r.label.replace(" ", "\n", 1) if len(r.label) > 12 else r.label for r in touch.itertuples()]
bottom = [0] * len(touch)
for c, col in (("hoogtemodel", DARK), ("SAM", TEAL), ("overig", GREY)):
    vals = touch[f"touched_{c}"].tolist()
    if sum(vals) == 0:
        continue
    ax.bar(labels, vals, bottom=bottom, color=col, width=0.65, label=f"{c}-voorkeur ({th(totals.get(c, 0))} vlakken)")
    bottom = [b + v for b, v in zip(bottom, vals)]
for i, tot in enumerate(bottom):
    ax.text(i, tot + 40, th(int(tot)), ha="center", fontsize=11, color=DARK, fontweight="bold")
ax.set_ylabel("vlakken met ≥ 1 verwijderde meting/lijn/punt")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_ylim(0, max(bottom) * 1.18)
ax.tick_params(axis="x", labelsize=10)
ax.text(5, 120, "0 — bijt pas na\nandere regels", ha="center", fontsize=9, color=GREY)
ax.legend(frameon=False, fontsize=10, loc="upper left")
fig.suptitle(
    f"Hoeveel vlakken raakt elke regel? · {th(len(base_regions))} vlakken in de basisdata · een vlak kan in meerdere balken staan",
    color=DARK, fontsize=12,
)
fig.tight_layout()
fig.savefig(FIG / "rule_touched.png", dpi=200, facecolor="white")
plt.close(fig)

led = pd.read_csv(ROOT / "experiments/loop/ledger.csv").drop_duplicates("variant", keep="last").set_index("variant")
abl = led.loc[list(RULES)].copy()
abl["label"] = [RULES[v][2] for v in abl.index]
abl = abl.sort_values("lgb_mae", ascending=False)
names = ["19 aug\n(uitgangspunt)"] + [f"+ {l}" for l in abl.label] + ["alle regels\nsamen"]
mae = [led.loc["v0-baseline", "lgb_mae"]] + abl.lgb_mae.tolist() + [led.loc["e8-final-protected", "lgb_mae"]]
tail = [led.loc["v0-baseline", "lgb_tail_mae"]] + abl.lgb_tail_mae.tolist() + [led.loc["e8-final-protected", "lgb_tail_mae"]]
cols = [GREY] + [TEAL] * len(abl) + [ORANGE]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, vals, t in ((axes[0], mae, "gemiddelde fout, alle vlakken (m/jr)"), (axes[1], tail, "fout risicogevallen > 2 m/jr (m/jr)")):
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.65)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=10, color=DARK, fontweight="bold")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=8.5, rotation=30, ha="right")
    ax.set_title(t, color=DARK, fontsize=12, pad=10)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
fig.suptitle("Elke regel apart op het 19 aug-model · vaste testset (1 174 vlakken) · resolutie 1", color=DARK, fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "rule_ablation.png", dpi=200, facecolor="white")
plt.close(fig)

# ── D · markdown ──────────────────────────────────────────────────────────────
ref = led.loc[["v0-baseline", "s0-v0-nomask"] + list(RULES) + ["e8-final-protected"]]
lines = [
    "# As 1 · regels apart en samen (2026-08-26)\n",
    "## A · single-rule ablation (v0 recipe + één regel, frozen holdout, test-ES)\n",
    "| variant | regel | LGB MAE | tail MAE (n) | coverage CORE | regions |",
    "|---|---|---|---|---|---|",
]
for v, r in ref.iterrows():
    lab = RULES.get(v, (None, None, {"v0-baseline": "uitgangspunt (kribbenmasker + ≥12 punten)", "s0-v0-nomask": "uitgangspunt zónder kribbenmasker", "e8-final-protected": "alle regels samen"}.get(v, v)))[2]
    lines.append(f"| {v} | {lab} | {r.lgb_mae:.3f} | {r.lgb_tail_mae:.3f} ({int(r.tail_n)}) | {r.coverage_core:.3f} | {int(r.n_regions)} |")
lines += [
    "\n## B · vlakken geraakt per regel (regel alleen, op de basisdata; kribbenmasker = oude Kribben_BKN)\n",
    f"Basisdata: {len(base_regions)} vlakken · " + " · ".join(f"{k} {v}" for k, v in totals.items()) + "\n",
    "| regel | geraakt totaal | hoogtemodel | SAM | overig | volledig verwijderd | hm | SAM | overig |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in touch.itertuples():
    lines.append(f"| {r.label} ({r.rule}) | {r.touched} | {r.touched_hoogtemodel} | {r.touched_SAM} | {r.touched_overig} | {r.removed} | {r.removed_hoogtemodel} | {r.removed_SAM} | {r.removed_overig} |")
lines += ["\nFiguren: fig/rule_touched.png, fig/rule_ablation.png"]
MD.write_text("\n".join(lines) + "\n")
print("→", MD)
