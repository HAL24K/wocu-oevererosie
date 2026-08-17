"""Self-contained HTML run report: the browsable output a notebook used to give.

One file per experiment, written next to the model outputs. Figures are
matplotlib PNGs embedded as base64, tables are the head of each dataframe the
run produced — so a run can be inspected in a browser without a kernel, and
two runs can be compared by opening two files.
"""

from __future__ import annotations

import base64
import datetime
import io
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#22303a"
MUTED = "#6b7e82"

_CSS = """
body{background:#fcfcfb;color:#22303a;font-family:-apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,sans-serif;margin:0;font-size:15px;line-height:1.55}
.wrap{max-width:1060px;margin:0 auto;padding:36px 28px 80px}
h1{font-size:1.7rem;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:1.15rem;margin:44px 0 10px;padding-top:18px;border-top:1px solid #e1e0d9}
.sub{color:#6b7e82;margin:0 0 8px;font-size:13px}
table{border-collapse:collapse;font-size:12.5px;margin:10px 0;
font-variant-numeric:tabular-nums}
th{background:#f0efec;text-align:left;padding:5px 10px;font-weight:600;
border-bottom:1px solid #d5d4cc;white-space:nowrap}
td{padding:4px 10px;border-bottom:1px solid #eceae4}
tr:hover td{background:#f6f5f1}
.tw{overflow-x:auto;border:1px solid #e1e0d9;border-radius:6px;background:#fff;
padding:0 4px;display:inline-block;max-width:100%;vertical-align:top;margin-right:14px}
img{max-width:100%;border:1px solid #e1e0d9;border-radius:6px;background:#fff;
margin:6px 14px 6px 0;vertical-align:top}
.ok{color:#2c6238;font-weight:600}.fail{color:#9e2f1c;font-weight:600}
.meta{display:flex;gap:26px;flex-wrap:wrap;color:#6b7e82;font-size:13px;margin:8px 0 0}
.meta b{color:#22303a}
code{background:#f0efec;padding:1px 5px;border-radius:3px;font-size:.9em}
"""


def _fig(width=7.2, height=3.4):
    fig, ax = plt.subplots(figsize=(width, height), dpi=115)
    ax.grid(axis="y", color="#e6e4dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    return fig, ax


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _table(df: pd.DataFrame, rows: int = 8, caption: str = "") -> str:
    shown = df.head(rows)
    cap = (
        f'<p class="sub">{caption} — showing {len(shown):,} of {len(df):,} rows</p>'
        if caption
        else ""
    )
    html = shown.to_html(border=0, float_format=lambda x: f"{x:,.3f}", na_rep="")
    return f'{cap}<div class="tw">{html}</div>'


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_run_report(result, bank_positions) -> None:
    """Render the run report for a completed :class:`RunResult`."""
    cfg = result.config
    parts: list[str] = []
    add = parts.append

    # ── header ────────────────────────────────────────────────────────────────
    total = sum(result.timings.values())
    add(f"<h1>Experiment {cfg.experiment}</h1>")
    add(
        f'<div class="meta"><span>generated <b>{datetime.datetime.now():%Y-%m-%d %H:%M}</b></span>'
        f"<span>commit <b>{_git_sha()}</b></span>"
        f"<span>total <b>{total / 60:.1f} min</b></span>"
        f"<span>horizon <b>{cfg.start_year}–{cfg.end_year}</b></span>"
        f"<span>model <b>{cfg.model_name}</b></span></div>"
    )

    # ── acceptance up front ──────────────────────────────────────────────────
    add("<h2>Acceptance checks</h2>")
    rows = "".join(
        f'<tr><td>{name}</td><td class="{"ok" if ok else "fail"}">'
        f"{'OK' if ok else 'FAIL'}</td><td>{detail}</td></tr>"
        for name, ok, detail in result.acceptance
    )
    add(
        f'<div class="tw"><table><tr><th>check</th><th></th><th>detail</th></tr>{rows}</table></div>'
    )

    # ── step 1 ────────────────────────────────────────────────────────────────
    dpy = result.dist_per_year
    add(
        f"<h2>1 · Bank distances <span class='sub'>({result.timings.get('1 · bank distances', 0):.0f}s)</span></h2>"
    )
    per_year = dpy.groupby("year")["location_id"].nunique()
    fig, ax = _fig()
    ax.bar(per_year.index.astype(str), per_year.values, color=BLUE, width=0.62)
    ax.set_title("regions observed per year", fontsize=10, color=INK)
    img1 = _png(fig)
    fig, ax = _fig()
    ax.hist(dpy["dist_m"], bins=60, color=BLUE)
    ax.set_title("dist_m distribution (m from centreline)", fontsize=10, color=INK)
    add(f'<img src="{img1}" width="440"><img src="{_png(fig)}" width="440">')
    add(_table(dpy, caption="dist_per_year"))

    # ── step 2 ────────────────────────────────────────────────────────────────
    spl = result.region_split
    add(
        f"<h2>2 · Region split <span class='sub'>({result.timings.get('2 · region split', 0):.0f}s)</span></h2>"
    )
    counts = spl.groupby(["cluster", "split"]).size().unstack(fill_value=0)
    fig, ax = _fig()
    ax.bar(counts.index, counts.get("train", 0), color=BLUE, width=0.6, label="train")
    ax.bar(
        counts.index,
        counts.get("test", 0),
        bottom=counts.get("train", 0),
        color=ORANGE,
        width=0.6,
        label="test",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("regions per cluster (train/test)", fontsize=10, color=INK)
    img1 = _png(fig)
    fig, ax = _fig()
    ax.scatter(
        spl["v_train"], spl["v_test"], s=4, alpha=0.25, color=BLUE, edgecolors="none"
    )
    lim = np.nanpercentile(np.abs(pd.concat([spl.v_train, spl.v_test])), 99.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axline((0, 0), slope=1, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("v_train (m/yr)", fontsize=9)
    ax.set_ylabel("v_test (m/yr)", fontsize=9)
    ax.set_title("historic vs target velocity per region", fontsize=10, color=INK)
    add(f'<img src="{img1}" width="440"><img src="{_png(fig)}" width="440">')
    add(
        f"<p class='sub'>{len(spl):,} train/test regions · "
        f"{len(result.region_inference):,} inference-only · "
        f"{int(spl['is_nvo'].sum()):,} NVO</p>"
    )
    add(_table(spl.reset_index(), caption="region_split"))

    # ── step 3 ────────────────────────────────────────────────────────────────
    feats = result.region_features
    add(
        f"<h2>3 · Features <span class='sub'>({result.timings.get('3 · feature engineering', 0):.0f}s)</span></h2>"
    )
    nulls = feats.isnull().sum()
    nulls = nulls[nulls > 0].sort_values()
    if len(nulls):
        fig, ax = _fig(height=0.5 + 0.35 * len(nulls))
        ax.barh(nulls.index, nulls.values, color=ORANGE)
        ax.set_title(f"null counts (of {len(feats):,} regions)", fontsize=10, color=INK)
        add(f'<img src="{_png(fig)}" width="440">')
    add(
        _table(
            feats.reset_index(), caption=f"region_features ({feats.shape[1]} columns)"
        )
    )

    # ── step 4 ────────────────────────────────────────────────────────────────
    add(
        f"<h2>4 · Models <span class='sub'>({result.timings.get('4 · train', 0):.0f}s)</span></h2>"
    )
    cmp = (
        pd.DataFrame(result.train_results)
        .T[["test_mae", "test_rmse", "test_r2", "test_tail_mae"]]
        .sort_values("test_mae")
    )
    fig, ax = _fig(height=0.5 + 0.4 * len(cmp))
    ax.barh(cmp.index[::-1], cmp["test_mae"][::-1], color=BLUE)
    ax.set_title("test MAE (m/yr) — lower is better", fontsize=10, color=INK)
    add(f'<img src="{_png(fig)}" width="470">')
    add(
        _table(
            cmp.reset_index(names="model"), rows=len(cmp), caption="model comparison"
        )
    )
    add(
        "<p class='sub'>Caveat carried from the reference run: LightGBM early-stops on the "
        "test set, so its test metrics are optimistic; see README → Known issues.</p>"
    )

    # ── step 5 ────────────────────────────────────────────────────────────────
    pred = result.predictions
    add(
        f"<h2>5 · Predictions <span class='sub'>({result.timings.get('5 · iterative prediction', 0):.0f}s"
        f" + {result.timings.get('6 · geometry', 0):.0f}s geometry)</span></h2>"
    )
    fig, ax = _fig()
    for year, color in [(cfg.start_year, BLUE), (cfg.end_year, ORANGE)]:
        v = pred.loc[pred.year == year, "velocity_m_per_yr"]
        ax.hist(v, bins=80, alpha=0.6, label=str(year), color=color)
    ax.set_xlim(*np.nanpercentile(pred["velocity_m_per_yr"], [0.2, 99.8]))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        "predicted velocity distribution, first vs last year", fontsize=10, color=INK
    )
    img1 = _png(fig)

    nvo_locs = bank_positions.loc[bank_positions.is_nvo == 1, "location_id"].unique()
    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(nvo_locs, size=min(8, len(nvo_locs)), replace=False)
    fig, ax = _fig()
    for loc in sample:
        traj = pred[pred.location_id == loc].sort_values("year")
        ax.plot(
            traj["year"], traj["predicted_dist_m"], linewidth=1.4, alpha=0.85, label=loc
        )
    ax.legend(frameon=False, fontsize=7, ncols=2)
    ax.set_title("sample NVO trajectories — predicted dist (m)", fontsize=10, color=INK)
    add(f'<img src="{img1}" width="440"><img src="{_png(fig)}" width="440">')

    crossed = result.vvr_crossing[result.vvr_crossing["crossing_year"].notna()]
    if len(crossed):
        fig, ax = _fig()
        years = crossed["crossing_year"].astype(int).value_counts().sort_index()
        ax.bar(years.index.astype(str), years.values, color=ORANGE, width=0.62)
        ax.set_title("signaleringslijn crossings per year", fontsize=10, color=INK)
        add(f'<img src="{_png(fig)}" width="470">')
    add(
        f"<p class='sub'>{len(pred):,} predictions · {pred['location_id'].nunique():,} regions · "
        f"{len(crossed):,} crossings within {cfg.start_year}–{cfg.end_year}</p>"
    )

    # ── config & timings ─────────────────────────────────────────────────────
    add("<h2>Configuration</h2>")
    cfg_df = pd.DataFrame(list(cfg.to_table().items()), columns=["setting", "value"])
    tim_df = pd.DataFrame(list(result.timings.items()), columns=["step", "seconds"])
    add(_table(cfg_df, rows=len(cfg_df)))
    add(_table(tim_df, rows=len(tim_df)))

    html = (
        f"<!-- generated by src.pipeline.report -->\n<title>Run {cfg.experiment}</title>"
        f"<style>{_CSS}</style><div class='wrap'>{''.join(parts)}</div>"
    )
    cfg.report_path.write_text(html)
