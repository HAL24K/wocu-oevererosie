"""Session deck for asset managers (from 15 Oct 2026): less model, more warning light.

Fills ``scripts/slides/deck/sessie_template.html`` with: the header images of the run's
``deck_assets``, a scope-region picture (``scope_region_figure.py``), the five-step
pictures of three example regions (``warning_light_steps.py``: keys ``kruist``,
``binnen``, ``voorbij``) and a map picture of a session stretch
(``session_project_v2.py --render``). ``--appendix`` appends background slides after the
demo, taken from a built full deck (``build_deck.py``), behind an "Achtergrond" divider.
``--pdf`` also prints it with Chrome.

Run: ``PYTHONPATH=. uv run python scripts/slides/build_session_deck.py --run 20261014-stable
        --sessie Maas --datum "15 oktober 2026" --scope <jpg> --scope-naam Broekhuizerweerd
        --steps <dir> --jaar-kruist 2027 --kaart <png> --kaart-naam "…" --out <html>``
"""

from __future__ import annotations

import argparse
import base64
import io
import subprocess
from pathlib import Path

from PIL import Image

from src.pipeline.config import ExperimentConfig

HERE = Path(__file__).parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--sessie", required=True)
ap.add_argument("--datum", required=True)
ap.add_argument("--scope", type=Path, required=True)
ap.add_argument("--scope-naam", required=True)
ap.add_argument("--steps", type=Path, required=True)
ap.add_argument("--jaar-kruist", required=True)
ap.add_argument("--kaart", type=Path, required=True)
ap.add_argument("--kaart-naam", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--pdf", type=Path)
ap.add_argument(
    "--appendix", type=Path, help="built full deck to take background slides from"
)
ap.add_argument(
    "--appendix-slides",
    nargs="*",
    default=[
        "Een regio verwerken",
        "Opschonen",
        "Kenmerken",
        "Trainen en testen",
        "Resultaten",
        "Wat drijft de voorspelling",
    ],
)
args = ap.parse_args()
assets = ExperimentConfig.from_run(args.run).figures_dir / "deck_assets"


def uri(path: Path, max_px: int | None = None) -> str:
    img = Image.open(path).convert("RGB")
    if max_px:
        img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=84)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def steps(key: str) -> str:
    return "".join(
        f'<img src="{uri(args.steps / f"{key}_{i}.jpg", 700)}" alt="Stap {i}">'
        for i in range(1, 6)
    )


repl = {
    "{{SESSIE}}": args.sessie,
    "{{DATUM}}": args.datum,
    "{{DRONE}}": png_uri(assets / "header_block.png"),
    "{{HEADER}}": png_uri(assets / "header_block.png"),
    "{{SCOPE}}": uri(args.scope, 1400),
    "{{SCOPE_NAAM}}": args.scope_naam,
    "{{STEPS_KRUIST}}": steps("kruist"),
    "{{STEPS_BINNEN}}": steps("binnen"),
    "{{STEPS_VOORBIJ}}": steps("voorbij"),
    "{{JAAR_KRUIST}}": args.jaar_kruist,
    # the class colour of that year, as on the map
    "{{KLEUR_KRUIST}}": "#d7301f"
    if int(args.jaar_kruist) <= 2027
    else "#ef6548"
    if int(args.jaar_kruist) <= 2029
    else "#fdbb47",
    "{{KAART}}": uri(args.kaart, 1600),
    "{{KAART_NAAM}}": args.kaart_naam,
}
page = (HERE / "deck" / "sessie_template.html").read_text()
for k, v in repl.items():
    assert k in page, k
    page = page.replace(k, v)
if args.appendix:
    full = args.appendix.read_text()
    css = full[full.index("<style>") : full.index("</style>") + len("</style>")]
    sections = []
    for title in args.appendix_slides:
        a = full.index(f'<section class="slide" data-title="{title}"')
        sections.append(full[a : full.index("</section>", a) + len("</section>")])
    divider = (
        '<section class="slide" data-title="Achtergrond">'
        '<div class="hdr"><img src="' + repl["{{HEADER}}"] + '" alt=""></div>'
        '<div style="position:absolute;left:200px;top:400px;right:140px;display:grid;gap:24px">'
        '<div style="font-size:120px;font-weight:700;color:var(--teal)">Achtergrond</div>'
        '<div style="font-size:40px">Hoe het model werkt, voor wie meer wil weten</div>'
        "</div></section>"
    )
    # The full deck's styles first, so the session deck's own rules win.
    page = page.replace("<style>", css + "\n<style>", 1)
    end = page.index("</div></div>\n\n<nav")
    page = page[:end] + "\n".join([divider, *sections]) + "\n\n" + page[end:]
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(page)
print(f"→ {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

if args.pdf:
    printable = args.out.with_name("_print.html")
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
            printable.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    printable.unlink()
    print(f"→ {args.pdf}")
