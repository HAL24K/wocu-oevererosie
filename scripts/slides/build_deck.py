"""Build the predictive-model slide deck (HTML) from a run's figures and numbers.

Fills ``scripts/slides/deck/deck_template.html`` with the run's images (as data
URIs), the feature descriptions (``src.analysis.feature_labels``), two charts
computed from the run's own numbers (``compare/source_report.csv``) and the
region data for the live explorer (``figures/methodology_data.json``). Writes
``<run>/figures/deck/wocu_voorspellend_model.html``, which is published as the
Claude artifact; ``--pdf`` also prints it to PDF with Chrome as an offline backup.
``--short`` leaves out the slides marked ``data-skip-short`` (the 6-slide session version,
``wocu_sessie_kort.html``). The map slide shows a stretch of the session it is for
(``--sessie rijntakken|maas``), cropped from the ``scripts/export/session_project_v2.py``
renders (``--renders``).

Run: ``PYTHONPATH=. uv run python scripts/slides/build_deck.py --run 20261013-ref [--short]
[--sessie rijntakken|maas] [--pdf <out.pdf>]``
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from src.analysis.feature_labels import FEATURES
from src.pipeline.config import ExperimentConfig

HERE = Path(__file__).parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TEAL, GREY, INK, MUTED = "#2ab4a8", "#c9cecd", "#333332", "#7d8688"

# Shown on the features slide, in this order (the model uses all of FEATURES).
SHOWN = {
    "geschiedenis": [
        "v_train",
        "theil_v",
        "dist_t2",
        "span_hist_yr",
        "n_seg_years",
        "last_gap_yr",
        "v_recent",
        "accel_v",
        "resid_last",
        "frac_sam",
    ],
    "positie": ["seg_center", "seg_edge"],
    "context": [
        "river_enc",
        "is_nvo",
        "soil_group_enc",
        "vegetation_class_enc",
        "land_use_enc",
        "bend_exposure_n8",
        "erosion_vol_rate_t1",
        "n_events_t2",
        "flood_days_t2",
    ],
}

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--pdf", type=Path, help="also print the deck to this PDF (Chrome)")
ap.add_argument("--short", action="store_true", help="session version without the detail slides")
ap.add_argument("--sessie", choices=["rijntakken", "maas"], default="rijntakken")
ap.add_argument(
    "--renders",
    type=Path,
    default=Path("data/05_deliverables/sessie_20261014/renders"),
    help="stretch renders of session_project_v2.py for the map slide",
)
args = ap.parse_args()
cfg = ExperimentConfig.from_run(args.run)
assets = cfg.figures_dir / "deck_assets"
RENDERS = args.renders
# Map slide per session: (render, crop box in pixels of the 1400×1000 render, caption)
KAART = {
    # Wageningen shows every class at once; drawn on the 2024 photo (the current one
    # was taken at high water) with qgis_render_regions.py --aerial
    "rijntakken": (
        "wageningen_beide.png",
        (50, 0, 1310, 1000),
        "Wageningen, Neder-Rijn (beide oevers) · luchtfoto PDOK 2024",
    ),
    "maas": (
        "Broekhuizerweerd.png",
        (200, 0, 1200, 900),
        "Broekhuizerweerd, Maas · luchtfoto PDOK actueel",
    ),
}


def uri(path: Path) -> str:
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[
        path.suffix[1:].lower()
    ]
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def features(group: str) -> str:
    out = []
    for f in SHOWN[group]:
        short, meaning, _, _ = FEATURES[f]
        if group == "positie":
            out.append(
                f"<b>{html.escape(short)}</b><span>{html.escape(meaning)}</span>"
            )
        else:
            out.append(
                f"<b>{html.escape(short)}</b><span>{html.escape(meaning)}</span>"
            )
    return "".join(out)


def nl(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


report = pd.read_csv(cfg.compare_dir / "source_report.csv")
report = report[report["run"] == cfg.experiment].set_index("group")
skill_all = report.loc["all test strips", "skill vs naive"]
counts = json.loads((cfg.run_dir / "run.json").read_text())["counts"]
seg_metrics = json.loads((cfg.model_out_dir / "segment_metrics.json").read_text())
seg_test = pd.read_parquet(cfg.model_out_dir / "segment_test.parquet")


def results_chart() -> str:
    """Grouped bars: model against the historical average, per group of regions."""
    t = pd.read_csv(cfg.compare_dir / "source_report.csv")
    t = t[t["run"] == cfg.experiment].set_index("group")
    rows = [
        ("all test strips", "Alle vlakken"),
        ("VVR regions", "VVR-vlakken"),
        ("height-model regions", "Hoogtemodel"),
        ("SAM regions", "SAM"),
    ]
    W, L, B, top = 900, 70, 320, 30
    ymax = 2.5
    y = lambda v: B - (B - top) * v / ymax  # noqa: E731
    s = [
        f'<line x1="{L}" y1="{B}" x2="{W - 10}" y2="{B}" stroke="{GREY}" stroke-width="2"/>'
    ]
    for tick in (0, 0.5, 1.0, 1.5, 2.0, 2.5):
        s.append(
            f'<line x1="{L}" y1="{y(tick)}" x2="{W - 10}" y2="{y(tick)}" stroke="#eef1f0" stroke-width="1.5"/>'
        )
        s.append(
            f'<text x="{L - 12}" y="{y(tick) + 7}" font-size="20" fill="{MUTED}" text-anchor="end">{nl(tick, 1)}</text>'
        )
    gw = (W - 10 - L) / len(rows)
    for i, (key, label) in enumerate(rows):
        m, n = t.loc[key, "LightGBM"], t.loc[key, "naive"]
        cx = L + gw * (i + 0.5)
        for dx, v, col in ((-44, m, TEAL), (6, n, GREY)):
            s.append(
                f'<rect x="{cx + dx}" y="{y(v)}" width="38" height="{B - y(v)}" fill="{col}" rx="3"/>'
            )
            s.append(
                f'<text x="{cx + dx + 19}" y="{y(v) - 8}" font-size="19" fill="{INK}" text-anchor="middle">{nl(v)}</text>'
            )
        s.append(
            f'<text x="{cx}" y="{B + 32}" font-size="22" font-weight="700" fill="{INK}" text-anchor="middle">{label}</text>'
        )
        s.append(
            f'<text x="{cx}" y="{B + 60}" font-size="20" fill="{TEAL}" text-anchor="middle">+{round((1 - m / n) * 100)} %</text>'
        )
    s.append(
        f'<rect x="{W - 420}" y="0" width="18" height="18" fill="{TEAL}"/><text x="{W - 394}" y="16" font-size="20" fill="{INK}">model</text>'
    )
    s.append(
        f'<rect x="{W - 300}" y="0" width="18" height="18" fill="{GREY}"/><text x="{W - 274}" y="16" font-size="20" fill="{INK}">historisch gemiddelde</text>'
    )
    s.append(f'<text x="{L}" y="16" font-size="20" fill="{MUTED}">fout in m/jr</text>')
    return "".join(s)


def progress_chart() -> str:
    """Skill against the historical average, March → October."""
    steps = [
        ("Maart", -20, "hoogtemodel, één snelheid per scopevlak"),
        ("September", 35, "SAM + hoogtemodel"),
        ("Oktober", round(skill_all * 100), "opschonen vóór hybride"),
    ]
    x0, scale = 330, 6.5
    s = [
        f'<line x1="{x0}" y1="0" x2="{x0}" y2="150" stroke="{MUTED}" stroke-width="2"/>'
    ]
    for i, (when, v, what) in enumerate(steps):
        yy = 12 + i * 46
        w = abs(v) * scale
        col = "#d0442a" if v < 0 else TEAL
        xs = x0 - w if v < 0 else x0
        s.append(
            f'<text x="0" y="{yy + 24}" font-size="22" font-weight="700" fill="{INK}">{when}</text>'
        )
        s.append(
            f'<rect x="{xs}" y="{yy}" width="{w}" height="32" fill="{col}" rx="3"/>'
        )
        lx = xs - 10 if v < 0 else xs + w + 10
        s.append(
            f'<text x="{lx}" y="{yy + 24}" font-size="22" font-weight="700" fill="{col}" text-anchor="{"end" if v < 0 else "start"}">{v:+d} %</text>'
        )
        if v > 0:
            s.append(
                f'<text x="{xs + w + 90}" y="{yy + 24}" font-size="19" fill="{MUTED}">{what}</text>'
            )
        else:
            s.append(
                f'<text x="{x0 + 12}" y="{yy + 24}" font-size="19" fill="{MUTED}">{what}</text>'
            )
    return "".join(s)


def kaart_uri() -> str:
    """The session's map picture, cropped and saved as JPEG in deck_assets."""
    from PIL import Image

    name, box, _ = KAART[args.sessie]
    img = Image.open(RENDERS / name).convert("RGB")
    if box:
        img = img.crop(box)
    out = assets / f"kaart_{args.sessie}.jpg"
    img.save(out, quality=85)
    return uri(out)


page = (HERE / "deck" / "deck_template.html").read_text()
if args.short:
    page = re.sub(
        r"<!-- [^>]*-->\s*<section [^>]*data-skip-short>.*?</section>\s*",
        "",
        page,
        flags=re.S,
    )
    page = page.replace(
        "<title>WOCU Voorspellend Model</title>",
        f"<title>WOCU Sessie {args.sessie.capitalize()}</title>",
    )
repl = {
    "{{DRONE}}": uri(assets / "header_block.png"),
    "{{HEADER}}": uri(assets / "header_block.png"),
    "{{REGION}}": uri(assets / "region_processing.jpg"),
    "{{FAILURES}}": uri(assets / "failure_modes.jpg"),
    "{{SHAP}}": uri(assets / "shap.png"),
    "{{KAART}}": kaart_uri(),
    "{{KAART_NAAM}}": KAART[args.sessie][2],
    "{{FEAT_geschiedenis}}": features("geschiedenis"),
    "{{FEAT_positie}}": features("positie"),
    "{{FEAT_context}}": features("context"),
    "{{RESULTS_CHART}}": results_chart(),
    "{{ALL_MAE}}": nl(report.loc["all test strips", "LightGBM"]),
    "{{ALL_NAIVE}}": nl(report.loc["all test strips", "naive"]),
    "{{ALL_SKILL}}": str(round(skill_all * 100)),
    "{{VVR_MAE}}": nl(report.loc["VVR regions", "LightGBM"]),
    "{{VVR_NAIVE}}": nl(report.loc["VVR regions", "naive"]),
    "{{VVR_SKILL}}": str(round(report.loc["VVR regions", "skill vs naive"] * 100)),
    "{{N_REGIONS}}": thousands(counts["regions_train"] + counts["regions_test"]),
    "{{N_TRAIN}}": thousands(counts["regions_train"]),
    "{{N_TEST}}": thousands(counts["regions_test"]),
    "{{N_EXAMPLES}}": thousands(seg_metrics["n_train"]),
    "{{N_TEST_STRIPS}}": thousands(len(seg_test)),
    "{{N_TEST_REGIONS}}": thousands(seg_test["location_id"].nunique()),
    "{{PROGRESS_CHART}}": progress_chart(),
    "/*REGIONS*/null": "null"
    if args.short
    else (cfg.figures_dir / "methodology_data.json").read_text(),
}
for k, v in repl.items():
    if k in page:
        page = page.replace(k, v)
    else:
        assert args.short, k
name = (
    f"wocu_sessie_kort_{args.sessie}.html"
    if args.short
    else "wocu_voorspellend_model.html"
)
out = cfg.figures_dir / "deck" / name
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(page)
print(f"→ {out} ({out.stat().st_size / 1e6:.1f} MB)")

if args.pdf:
    # The artifact host adds the document skeleton; a local print needs one.
    printable = out.with_name("print.html")
    printable.write_text(
        "<!doctype html><html><head><meta charset='utf-8'></head><body>"
        + page
        + "</body></html>"
    )
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=8000",
            f"--print-to-pdf={args.pdf}",
            printable.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    printable.unlink()
    print(f"→ {args.pdf}")
