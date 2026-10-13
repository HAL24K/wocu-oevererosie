"""Key slide: the predictive model's workflow, in native PowerPoint shapes.

Built on the WOCU template (the March "Overzicht" slide, layout "Nieuw
hoofdstuk"): every element is an editable shape, nothing is a picture, so the
slide can be tweaked by hand afterwards. Each block is a named group. Left the
input (height model and SAM lines; the steps before our model in grey), in the
middle the loop (prepare, choose features, train and test, evaluate, back),
right the output for the user (a sketch until the viewer is decided). No
numbers on this slide.

Run: ``PYTHONPATH=. uv run python scripts/slides/workflow_slide.py [--out …pptx]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

DRIVE = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-alexander.bakker@morainewater.com/My Drive/WOCU Oevererosie/presentaties"
)
TEMPLATE = DRIVE / "_template/WOCU_template_assetmanagement.pptx"
OUT = DRIVE / "2026-10-14_sessies_assetmanagement/01_werkwijze.pptx"
BASE_SLIDE = 2  # 0-based: the March "Overzicht · Voorspellend model overzicht" slide

TEAL = RGBColor(0x2A, 0xB4, 0xA8)
INK = RGBColor(0x33, 0x33, 0x32)
MUTED = RGBColor(0x8A, 0x91, 0x93)
LIGHT = RGBColor(0xEE, 0xF1, 0xF0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREYED = RGBColor(0xC9, 0xCE, 0xCD)
WATER = RGBColor(0xCF, 0xE3, 0xEC)
SAM = RGBColor(0x2E, 0x7D, 0x5B)
HEIGHT = RGBColor(0x49, 0x6A, 0x8E)
PRED = RGBColor(0xD9, 0x7A, 0x2B)
RED, AMBER, GREEN = (
    RGBColor(0xD0, 0x44, 0x2A),
    RGBColor(0xE8, 0xB0, 0x3A),
    RGBColor(0x5B, 0xAE, 0x6A),
)
FONT = "Ebrima"

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=OUT)
args = ap.parse_args()


# ── deck from the template: keep only the base slide, clear its content ──────
prs = Presentation(TEMPLATE)
ids = prs.slides._sldIdLst
for i, sld in reversed(list(enumerate(list(ids)))):
    if i != BASE_SLIDE:
        prs.part.drop_rel(sld.get(qn("r:id")))
        ids.remove(sld)
slide = prs.slides[0]
for sh in list(slide.shapes):
    if not (sh.name in ("Graphic 15", "Titel 1", "Titel 5") or sh.is_placeholder):
        sh._element.getparent().remove(sh._element)
for sh in slide.shapes:
    if sh.name in ("Titel 1", "Titel 5"):
        runs = sh.text_frame.paragraphs[0].runs
        runs[0].text = (
            "Aanpak" if sh.name == "Titel 1" else "Van meting naar voorspelling"
        )
        for r in runs[1:]:
            r.text = ""


# ── helpers ───────────────────────────────────────────────────────────────────
# Shapes are drawn into CANVAS[-1]: the slide, or a group inside `with group(...)`.
CANVAS = [slide.shapes]


class group:
    """Draw the enclosed shapes into one named group, so a block moves as one."""

    def __init__(self, name):
        self.grp = slide.shapes.add_group_shape()
        self.grp.name = name

    def __enter__(self):
        CANVAS.append(self.grp.shapes)

    def __exit__(self, *exc):
        CANVAS.pop()


def box(
    x,
    y,
    w,
    h,
    fill=None,
    line=None,
    shape=MSO_SHAPE.ROUNDED_RECTANGLE,
    dash=False,
    lw=1.0,
    radius=0.12,
):
    s = CANVAS[-1].add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = 4  # dash
    s.shadow.inherit = False
    return s


def text(x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """Text box; ``lines`` is one (text, style) pair or a list of them."""
    tb = CANVAS[-1].add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0.02))
    for i, (t, kw) in enumerate([lines] if isinstance(lines, tuple) else lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = t
        r.font.name = FONT
        r.font.size = Pt(kw.get("size", 11))
        r.font.bold = kw.get("bold", False)
        r.font.color.rgb = kw.get("color", INK)
    return tb


def arrow(x1, y1, x2, y2, color=TEAL, lw=2.0, dash=False, name=None):
    c = CANVAS[-1].add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    c.line.color.rgb = color
    c.line.width = Pt(lw)
    if dash:
        c.line.dash_style = 4
    tail = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"))
    tail.set("type", "triangle")
    if name:
        c.name = name
    return c


def polyline(points, color, lw=1.6, x0=0.0, y0=0.0, dash=False):
    """Open freeform through (x, y) inches, offset by (x0, y0)."""
    fb = CANVAS[-1].build_freeform(
        Inches(x0 + points[0][0]), Inches(y0 + points[0][1]), scale=1.0
    )
    fb.add_line_segments(
        [(Inches(x0 + x), Inches(y0 + y)) for x, y in points[1:]], close=False
    )
    s = fb.convert_to_shape()
    s.fill.background()
    s.line.color.rgb = color
    s.line.width = Pt(lw)
    if dash:
        s.line.dash_style = 4
    s.shadow.inherit = False
    return s


def region_tile(x, y, w=1.15, h=0.78):
    """Pictogram of a scope region: water on the left, the region as a dashed box."""
    box(x, y, w, h, fill=LIGHT, radius=0.08)
    box(x, y, w * 0.28, h, fill=WATER, shape=MSO_SHAPE.RECTANGLE)
    box(
        x + w * 0.2,
        y + h * 0.12,
        w * 0.7,
        h * 0.76,
        line=MUTED,
        dash=True,
        lw=0.75,
        shape=MSO_SHAPE.RECTANGLE,
    )


def bank(x, y, w, h, color, offset, amp=0.05, phase=0.0, lw=1.6):
    """A gently meandering bank line, top to bottom of a tile."""
    import math

    n = 14
    pts = [
        (
            w * (0.36 + offset + amp * math.sin(phase + 2.4 * math.pi * i / (n - 1))),
            h * (0.12 + 0.76 * i / (n - 1)),
        )
        for i in range(n)
    ]
    return polyline(pts, color, lw, x0=x, y0=y)


TOP = 2.35  # content starts below the title

# ── input bookend ────────────────────────────────────────────────────────────
text(0.35, TOP, 2.3, 0.3, ("INVOER", {"bold": True, "color": MUTED, "size": 10}))
with group("Invoer: hoogtemodel en SAM"):
    region_tile(0.4, TOP + 0.4)
    bank(0.4, TOP + 0.4, 1.15, 0.78, HEIGHT, 0.04, amp=0.05, lw=2.2)
    text(
        1.62,
        TOP + 0.45,
        1.25,
        0.7,
        [
            ("Hoogtemodel", {"bold": True}),
            ("2015–2025 · Luke", {"size": 9, "color": MUTED}),
        ],
    )
    region_tile(0.4, TOP + 1.38)
    for off, ph in ((0.0, 0.0), (0.08, 0.9), (0.2, 2.1)):
        bank(0.4, TOP + 1.38, 1.15, 0.78, SAM, off, amp=0.04, phase=ph, lw=1.4)
    text(
        1.62,
        TOP + 1.43,
        1.25,
        0.7,
        [
            ("SAM-satellietlijnen", {"bold": True}),
            ("2023–2026 · Lars", {"size": 9, "color": MUTED}),
        ],
    )

with group("Vooraf (grijs): opschonen SAM en hybride model"):
    box(0.4, TOP + 2.45, 2.45, 1.05, line=GREYED, dash=True, lw=1.0)
    text(
        0.5,
        TOP + 2.5,
        2.3,
        0.95,
        [
            ("Vooraf, door partners", {"bold": True, "color": MUTED, "size": 10}),
            ("SAM-lijnen opschonen", {"color": MUTED, "size": 10}),
            ("Hybride model: per regio de beste bron", {"color": MUTED, "size": 10}),
        ],
    )
arrow(1.6, TOP + 2.2, 1.6, TOP + 2.42, color=GREYED, lw=1.5)
elbow = CANVAS[-1].add_connector(
    MSO_CONNECTOR.ELBOW,
    Inches(2.88),
    Inches(TOP + 2.97),
    Inches(3.36),
    Inches(TOP + 0.15 + 1.55 / 2),
)
elbow.line.color.rgb = GREYED
elbow.line.width = Pt(2)
etree.SubElement(elbow.line._get_or_add_ln(), qn("a:tailEnd")).set("type", "triangle")
elbow.name = "Invoer naar stap 1"

# ── the loop ──────────────────────────────────────────────────────────────────
LX, LY, BW, BH, GX, GY = 3.4, TOP + 0.15, 2.4, 1.75, 0.75, 0.8
steps = [
    (LX, LY, "1  Voorbereiden", "Opschonen en per regio 20 strookjes van ±5 m"),
    (
        LX + BW + GX,
        LY,
        "2  Kenmerken kiezen",
        "Geschiedenis, positie en context per strookje",
    ),
    (
        LX + BW + GX,
        LY + BH + GY,
        "3  Trainen en testen",
        "Leren op bekende regio's, toetsen op regio's die het model nooit zag",
    ),
    (
        LX,
        LY + BH + GY,
        "4  Evalueren",
        "Slechtst voorspelde regio's bekijken: wat ging er mis?",
    ),
]
for i, (x, y, title, sub) in enumerate(steps):
    with group(f"Stap {title}"):
        box(x, y, BW, BH, fill=WHITE, line=TEAL, lw=1.5)
        ix, iy = x + 0.15, y + 0.18
        if i == 0:  # region tile with strips
            region_tile(ix, iy, 0.75, 0.55)
            for k in range(5):
                box(
                    ix + 0.2 + k * 0.105,
                    iy + 0.07,
                    0.05,
                    0.41,
                    fill=TEAL,
                    shape=MSO_SHAPE.RECTANGLE,
                )
        elif i == 1:  # feature list
            for k, wdt in enumerate([0.6, 0.45, 0.3, 0.2]):
                box(
                    ix,
                    iy + 0.04 + k * 0.13,
                    wdt,
                    0.08,
                    fill=TEAL if k < 2 else GREYED,
                    shape=MSO_SHAPE.RECTANGLE,
                )
        elif i == 2:  # train / test tiles
            for dx, dy, col in [
                (0.0, 0.0, TEAL),
                (0.08, 0.08, TEAL),
                (0.45, 0.0, PRED),
            ]:
                box(
                    ix + dx,
                    iy + dy,
                    0.32,
                    0.42,
                    fill=LIGHT,
                    line=col,
                    lw=1.2,
                    radius=0.1,
                )
        else:  # magnifier over a region
            region_tile(ix, iy, 0.75, 0.55)
            box(ix + 0.38, iy + 0.12, 0.3, 0.3, line=INK, lw=1.8, shape=MSO_SHAPE.OVAL)
            polyline([(0.64, 0.38), (0.78, 0.52)], INK, lw=2.4, x0=ix, y0=iy)
        text(x + 0.98, y + 0.2, BW - 1.03, 0.6, (title, {"bold": True, "size": 12}))
        text(x + 0.15, y + 0.9, BW - 0.3, 0.75, (sub, {"size": 10.5}))

# cycle arrows 1→2→3→4→1
arrow(LX + BW + 0.05, LY + BH / 2, LX + BW + GX - 0.05, LY + BH / 2)
arrow(LX + BW + GX + BW / 2, LY + BH + 0.05, LX + BW + GX + BW / 2, LY + BH + GY - 0.05)
arrow(LX + BW + GX - 0.05, LY + BH + GY + BH / 2, LX + BW + 0.05, LY + BH + GY + BH / 2)
arrow(LX + BW / 2, LY + BH + GY - 0.05, LX + BW / 2, LY + BH + 0.05)
# feedback 4 → 2 (new rules, other features), dashed
arrow(
    LX + BW - 0.1,
    LY + BH + GY + 0.1,
    LX + BW + GX + 0.1,
    LY + BH - 0.1,
    color=PRED,
    lw=1.8,
    dash=True,
    name="Terugkoppeling naar kenmerken",
)
text(
    LX + BW + 0.32,
    LY + BH + 0.25,
    1.2,
    0.5,
    [
        ("nieuwe regels,", {"size": 9, "color": PRED}),
        ("andere kenmerken", {"size": 9, "color": PRED}),
    ],
)
text(
    LX,
    LY + 2 * BH + GY + 0.12,
    2 * BW + GX,
    0.3,
    (
        "Herhalen zolang de slechtste regio's iets leren over de data",
        {"size": 10, "color": MUTED, "bold": True},
    ),
    align=PP_ALIGN.CENTER,
)

# ── output bookend ───────────────────────────────────────────────────────────
OX = LX + 2 * BW + GX + 0.55
arrow(LX + 2 * BW + GX + 0.08, LY + BH + GY / 2, OX - 0.08, LY + BH + GY / 2, lw=2.5)
text(
    OX, TOP, 2.2, 0.3, ("UITVOER  (schets)", {"bold": True, "color": MUTED, "size": 10})
)
with group("Uitvoer: schets van de viewer"):
    box(OX, TOP + 0.4, 2.05, 2.6, fill=LIGHT, line=GREYED, lw=1.0, radius=0.06)
    polyline(
        [(0.25, 0.0), (0.45, 0.6), (0.4, 1.3), (0.6, 2.0), (0.55, 2.6)],
        WATER,
        lw=18,
        x0=OX,
        y0=TOP + 0.4,
    )
    for k, col in enumerate([GREEN, GREEN, AMBER, RED, AMBER, GREEN]):
        box(
            OX + 0.95,
            TOP + 0.55 + k * 0.4,
            0.75,
            0.32,
            fill=col,
            shape=MSO_SHAPE.RECTANGLE,
        )
    polyline(
        [(0.88, 0.1), (0.84, 0.8), (0.9, 1.6), (0.86, 2.45)],
        PRED,
        lw=2,
        x0=OX,
        y0=TOP + 0.4,
        dash=True,
    )
    slider = CANVAS[-1].add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(OX + 0.15),
        Inches(TOP + 3.25),
        Inches(OX + 1.9),
        Inches(TOP + 3.25),
    )
    slider.line.color.rgb = MUTED
    slider.line.width = Pt(1.5)
    box(OX + 1.05, TOP + 3.17, 0.16, 0.16, fill=TEAL, shape=MSO_SHAPE.OVAL)
    text(
        OX,
        TOP + 3.38,
        2.05,
        0.25,
        ("2026          2031          2040", {"size": 8, "color": MUTED}),
        align=PP_ALIGN.CENTER,
    )
    text(
        OX,
        TOP + 3.7,
        2.1,
        0.9,
        [
            ("Voor de gebruiker", {"bold": True}),
            (
                "Per regio het jaar waarin de oever de VVR bereikt, en de voorspelde oeverlijn per jaar",
                {"size": 9.5},
            ),
        ],
    )

args.out.parent.mkdir(parents=True, exist_ok=True)
prs.save(args.out)
print("→", args.out)
