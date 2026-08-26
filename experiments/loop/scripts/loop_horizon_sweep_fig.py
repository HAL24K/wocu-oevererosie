"""Figure + markdown for the segment × horizon sweep (ledger rows hz-R*)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TEAL, DARK, GREY, ORANGE = "#2BB5A6", "#1F3A3D", "#9AA5A6", "#E07A3F"
plt.rcParams.update({"font.family": ["Verdana", "DejaVu Sans"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})

led = pd.read_csv(ROOT / "experiments/loop/ledger.csv").drop_duplicates("variant", keep="last")
hz = led[led.variant.str.match(r"^hz-R\d+$")].copy()
hz["R"] = hz.variant.str.extract(r"R(\d+)").astype(int)
hz = hz.sort_values("R")
hz["starved"] = 1 - hz["coverage_core"]
hz["skill"] = 1 - hz.lgb_mae / hz.naive_mae


def th(n):
    return f"{int(n):,}".replace(",", " ")


fig, ax = plt.subplots(figsize=(11, 4.8))
R = hz.R.values
ax.plot(R, hz.lgb_mae, "-o", color=TEAL, lw=2.5, ms=8, label="fout per segment (m/jr)")
ax.plot(R, hz.lgb_tail_mae, "-o", color=ORANGE, lw=2, ms=7, label="fout risicogevallen > 2 m/jr (m/jr)")
ax.plot(R, hz.naive_mae, "--s", color=GREY, lw=2, ms=6, label="naïef (gemiddelde) (m/jr)")
for x, y in zip(R, hz.lgb_mae):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -18), ha="center",
                color=TEAL, fontweight="bold")
starved = hz[hz.starved > 0.05]
if len(starved):
    ax.axvspan(starved.R.min() / 1.35, R.max() * 1.6, color=GREY, alpha=0.12)
    ax.text(starved.R.min(), ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 4.5,
            "> 5 % segmenten zonder metingen\n(60 punten per lijn)", ha="left", va="top",
            fontsize=10, color=GREY)
ax.set_xscale("log")
ax.set_xticks(R)
ax.set_xticklabels([f"R = {r}\n≈ {round(100 / r)} m" for r in R])
ax.set_xlim(R.min() / 1.4, R.max() * 1.6)
ax.set_ylim(0, max(hz.lgb_tail_mae.max(), 3.5) * 1.15)
ax.set_xlabel("segmenten per vlak (vlak ≈ 100 m)")
ax.legend(frameon=False, fontsize=11, loc="upper right")
fig.suptitle("Segment × horizon ≥ 2 jaar · vaste testset · fijner blijft beter tot de bemonstering op is",
             color=DARK, fontsize=13)
fig.tight_layout()
out = ROOT / "docs/presentations/fig/horizon_sweep.png"
fig.savefig(out, dpi=200, facecolor="white")

lines = ["# Segment × horizon (≥ 2 jaar) sweep — 2026-08-26", "",
         "Frozen holdout, e8 obs, traj2 segment features, all ≥2-yr origin/end pairs for training, "
         "span-weighted (clip 2–5), test = latest ≥2-yr pair per segment; test-set early stopping "
         "(same as every other ledger row). Ledger variants `hz-R<R>`. R=1 reproduces k5-H2-multi-w.", "",
         "| R | ≈ m/segment | n_train | n_test | starved | MAE | tail (n) | naïef | skill | R² | pos_err_med (m) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in hz.iterrows():
    lines.append(f"| {r.R} | {round(100 / r.R)} | {th(r.n_train)} | {th(r.n_test)} | {r.starved:.1%} | "
                 f"**{r.lgb_mae:.2f}** | {r.lgb_tail_mae:.2f} ({int(r.tail_n)}) | {r.naive_mae:.2f} | "
                 f"{r.skill:.2f} | {r.lgb_r2:.2f} | {r.pos_err_med:.2f} |")
(ROOT / "docs/presentations/horizon_sweep_20260826.md").write_text("\n".join(lines) + "\n")
print(out)
print("\n".join(lines[5:]))
