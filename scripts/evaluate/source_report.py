"""Error per source and per VVR, for one or more runs, as a PDF with tables and charts.

For each run, the segment model's test strips are split by
- source: Luke's original choice per region (height model or SAM), so the
  groups mean the same in every run, also when a run forces one model;
- VVR: the region touches a VVR polygon (is_nvo);
and scored for three predictors: LightGBM (the model), persistence (the strip's
own last speed) and the naive mean (the average speed of all training rows).

Each run is scored on its own observations: a run on "height everywhere" is
tested against height-model lines, one on "SAM everywhere" against SAM lines.
There is no independent ground truth, so a lower MAE can also mean noisier or
smoother observations; the last page shows how far the runs' observed speeds
differ on the same strips.

Run: ``PYTHONPATH=. uv run python scripts/evaluate/source_report.py
        --base 20261013-ref --runs 20261013-force-height 20261013-force-sam``
Writes ``<base>/compare/source_report.pdf`` and ``source_report.csv``.
"""

from __future__ import annotations

import argparse
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src.pipeline.config import ExperimentConfig  # noqa: E402

LUKE = "03_hybrid/luke_20260710/hybrid_model_results_nearest_20260710.gpkg"
COLORS = {"LightGBM": "#E07B39", "persistence": "#7A8FA6", "naive": "#C9CED6"}

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--runs", nargs="*", default=[])
args = ap.parse_args()
names = [args.base, *args.runs]
base = ExperimentConfig.from_run(args.base)
out_dir = base.compare_dir
out_dir.mkdir(parents=True, exist_ok=True)

luke_pref = (
    pyogrio.read_dataframe(
        base.data_dir / LUKE,
        layer="model_preference",
        columns=["location_id", "model_preference"],
        read_geometry=False,
    )
    .set_index("location_id")["model_preference"]
    .map({"height": "height-model regions", "segmentation": "SAM regions"})
)


def load(name: str) -> tuple[pd.DataFrame, dict]:
    cfg = ExperimentConfig.from_run(name)
    t = pd.read_parquet(cfg.model_out_dir / "segment_test.parquet")
    split = pd.read_parquet(cfg.features_dir / "region_split.parquet")
    t["vvr"] = t["location_id"].map(split["is_nvo"]).fillna(False).astype(bool)
    t["source"] = t["location_id"].map(luke_pref).fillna("not in Luke's hybrid")
    if "persist" not in t:
        t["persist"] = np.nan  # runs before 13 Oct did not keep it per strip
    record = json.loads((cfg.run_dir / "run.json").read_text())
    return t, record


def mae_table(t: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "all test strips": t,
        "VVR regions": t[t["vvr"]],
        "height-model regions": t[t["source"] == "height-model regions"],
        "SAM regions": t[t["source"] == "SAM regions"],
        "VVR, height-model regions": t[t["vvr"] & (t["source"] == "height-model regions")],
        "VVR, SAM regions": t[t["vvr"] & (t["source"] == "SAM regions")],
    }
    rows = []
    for name, d in groups.items():
        rows.append(
            {
                "group": name,
                "strips": len(d),
                "regions": d["location_id"].nunique(),
                "LightGBM": (d["pred"] - d["v_test"]).abs().mean(),
                "persistence": (d["persist"] - d["v_test"]).abs().mean(),
                "naive": (d["naive"] - d["v_test"]).abs().mean(),
            }
        )
    out = pd.DataFrame(rows).set_index("group")
    out["skill vs naive"] = 1 - out["LightGBM"] / out["naive"]
    return out


data = {n: load(n) for n in names}
tables = {n: mae_table(t) for n, (t, _) in data.items()}
pd.concat(tables, names=["run"]).round(4).to_csv(out_dir / "source_report.csv")


def text_page(pdf, title, lines, fontsize=9.5):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.94, title, fontsize=16, weight="bold")
    fig.text(0.05, 0.89, "\n".join(lines), fontsize=fontsize, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(out_dir / "source_report.pdf") as pdf:
    # ── page 1: what was run ────────────────────────────────────────────────
    lines = []
    for n, (t, rec) in data.items():
        c = rec.get("counts", {})
        reg = rec["metrics"]["region"]
        lines += [
            f"{n}",
            *textwrap.wrap(rec.get("notes", ""), 110, initial_indent="  ", subsequent_indent="  "),
            f"  regions modelled {c.get('regions_train', 0) + c.get('regions_test', 0):,} "
            f"(test {c.get('regions_test', 0):,}) · predicted only {c.get('regions_inference_only', 0):,} · "
            f"test strips {len(t):,}",
            "  region model, test MAE (m/yr): "
            + " · ".join(
                f"{k.split(' – ')[1]} {v['test_mae']:.2f}"
                for k, v in reg.items()
                if k.split(" – ")[0] in ("0", "1", "5")
            ),
            "",
        ]
    lines += textwrap.wrap(
        "How to read: each run is tested on its own observations (height-model lines, SAM "
        "lines, or Luke's mix). 'Height-model regions' and 'SAM regions' are Luke's original "
        "choice per region, the same in every run. Persistence = the strip's own last speed; "
        "naive = the mean speed of all training rows. Segment = 20 strips of ~5 m per region.",
        120,
    )
    text_page(pdf, "Error per source and per VVR — runs", lines)

    # ── page 2: the tables ──────────────────────────────────────────────────
    lines = []
    for n, tab in tables.items():
        lines.append(f"{n}   (segment model, test strips, MAE in m/yr)")
        show = tab.copy()
        for col in ("LightGBM", "persistence", "naive"):
            show[col] = show[col].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
        show["skill vs naive"] = tab["skill vs naive"].map(lambda v: "—" if pd.isna(v) else f"{v:+.0%}")
        show["strips"] = show["strips"].map("{:,}".format)
        lines += show.to_string().splitlines() + [""]
    text_page(pdf, "MAE per group", lines, fontsize=8.5)

    # ── page 3: bars ────────────────────────────────────────────────────────
    groups = ["all test strips", "VVR regions", "height-model regions", "SAM regions"]
    fig, axes = plt.subplots(1, len(groups), figsize=(11.69, 5.5), sharey=True)
    for ax, g in zip(axes, groups, strict=True):
        x = np.arange(len(names))
        for i, model in enumerate(COLORS):
            vals = [tables[n].loc[g, model] for n in names]
            ax.bar(x + (i - 1) * 0.27, vals, 0.27, color=COLORS[model], label=model)
            for xi, v in zip(x, vals, strict=True):
                if pd.notna(v) and model == "LightGBM":
                    ax.text(xi - 0.27, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, [n.replace("20261013-", "") for n in names], rotation=30, ha="right", fontsize=8)
        ax.set_title(g, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("MAE, m/yr (segment model, test strips)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Model vs baselines, per source and for VVR regions", fontsize=13, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    fig.savefig(out_dir / "source_report_bars.png", dpi=150)
    plt.close(fig)

    # ── page 4: error distributions, VVR regions ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.5))
    for ax, g in zip(axes, ["VVR regions", "all test strips"], strict=True):
        for n in names:
            t = data[n][0]
            d = t[t["vvr"]] if g == "VVR regions" else t
            e = np.sort((d["pred"] - d["v_test"]).abs().to_numpy())
            ax.plot(e, np.linspace(0, 1, len(e)), label=n.replace("20261013-", ""))
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("absolute error, m/yr")
        ax.set_ylabel("share of test strips")
        ax.set_title(f"{g}: error distribution (LightGBM)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── page 5: how different are the observations on shared strips ─────────
    if args.runs:
        lines = []
        t0 = data[args.base][0].set_index(["location_id", "seg"])
        for n in args.runs:
            t1 = data[n][0].set_index(["location_id", "seg"])
            j = t0.join(t1, lsuffix="_base", rsuffix="_run", how="inner")
            dv = (j["v_test_base"] - j["v_test_run"]).abs()
            lines += [
                f"{n} vs {args.base}: {len(j):,} strips tested in both",
                f"  observed speed differs: median {dv.median():.2f} · mean {dv.mean():.2f} · "
                f"p90 {dv.quantile(0.9):.2f} m/yr; identical on {(dv < 1e-9).mean():.0%} of strips",
                f"  MAE on these strips, each against its own observations: base "
                f"{(j['pred_base'] - j['v_test_base']).abs().mean():.3f} · run "
                f"{(j['pred_run'] - j['v_test_run']).abs().mean():.3f}",
                f"  base prediction against the run's observations: "
                f"{(j['pred_base'] - j['v_test_run']).abs().mean():.3f} · run prediction against "
                f"the base's observations: {(j['pred_run'] - j['v_test_base']).abs().mean():.3f}",
                "",
            ]
        text_page(pdf, "Same strips, different observations", lines)

print(pd.concat(tables, names=["run"]).round(3).to_string())
print("→", out_dir / "source_report.pdf")
