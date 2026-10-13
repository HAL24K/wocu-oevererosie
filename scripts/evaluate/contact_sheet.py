"""Put the close-ups of one or more forecast renders side by side (one row per render).

Run: ``uv run python scripts/evaluate/contact_sheet.py <out.jpg> <render dir> [<render dir> ...]``
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

out, *dirs = sys.argv[1:]
names = ["zuilichem_oost", "zuilichem_west", "druten", "elst", "wageningen_zuid"]
w, h = 560, 400
sheet = Image.new("RGB", (w * len(names), (h + 24) * len(dirs)), "white")
draw = ImageDraw.Draw(sheet)
for r, d in enumerate(dirs):
    for c, n in enumerate(names):
        p = Path(d) / "closeup" / f"{n}.png"
        if p.exists():
            sheet.paste(
                Image.open(p).convert("RGB").resize((w, h)), (c * w, r * (h + 24) + 24)
            )
        draw.text((c * w + 6, r * (h + 24) + 6), f"{Path(d).name} · {n}", fill="black")
sheet.save(out, quality=82)
print(out)
