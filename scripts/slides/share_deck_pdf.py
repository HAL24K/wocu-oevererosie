"""PDF twin of ``share_deck.js``: the same spec, laid out as 1920×1080 pages with Pillow,
for when PowerPoint cannot export (or is not at hand).

Run: ``uv run python scripts/slides/share_deck_pdf.py <spec.json> <out.pdf>``
"""

import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

spec = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
W, H = 1920, 1080
TEAL, INK, MUTED = (42, 180, 168), (51, 51, 50), (125, 134, 136)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
f = lambda path, size: ImageFont.truetype(path, size)  # noqa: E731
pad = lambda n: f"{n:02d}" if spec["n_slides"] >= 10 else str(n)  # noqa: E731


def text(draw, xy, s, font, fill, width_chars):
    x, y = xy
    for para in s.split("\n"):
        for line in textwrap.wrap(para, width_chars) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += int(font.size * 1.3)
        y += int(font.size * 0.4)
    return y


def deck(i):
    return (
        Image.open(Path(spec["slides_dir"]) / f"slide-{pad(i)}.png")
        .convert("RGB")
        .resize((W, H))
    )


pages = [deck(i) for i in range(1, spec["insert_after"] + 1)]
intro = Image.new("RGB", (W, H), TEAL)
d = ImageDraw.Draw(intro)
d.text((115, 290), spec["maps_title"], font=f(BOLD, 70), fill="white")
text(d, (115, 520), spec["maps_intro"], f(FONT, 32), "white", 95)
pages.append(intro)
for file, name, sub, note in spec["maps"]:
    page = Image.new("RGB", (W, H), "white")
    page.paste(Image.open(file).convert("RGB").resize((1512, H)), (0, 0))
    d = ImageDraw.Draw(page)
    y = text(d, (1555, 96), name, f(BOLD, 34), TEAL, 16)
    y = text(d, (1555, max(y + 20, 384)), sub, f(FONT, 23), MUTED, 24)
    if note:
        text(d, (1555, max(y + 20, 576)), note, f(FONT, 23), INK, 24)
    pages.append(page)
pages += [deck(i) for i in range(spec["insert_after"] + 1, spec["n_slides"] + 1)]
pages[0].save(out, save_all=True, append_images=pages[1:], resolution=96)
print(f"→ {out} ({len(pages)} pages)")
