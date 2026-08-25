"""Build the 2026-08-26 update deck (docs/presentations/20260826_update.pptx).

Storyline: docs/presentations/20260826_storyline.md. Figures from
scripts/deck_figures.py plus rasterised slides of the March and August decks
(scratchpad; passed via --renders).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs/presentations/fig"
OUT = ROOT / "docs/presentations/20260826_update.pptx"

TEAL = RGBColor(0x2B, 0xB5, 0xA6)
DARK = RGBColor(0x1F, 0x3A, 0x3D)
GREY = RGBColor(0x6B, 0x77, 0x78)
ORANGE = RGBColor(0xE0, 0x7A, 0x3F)
W, H = Inches(13.333), Inches(7.5)


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    # ── primitives ──────────────────────────────────────────────────────────
    def slide(self, title: str, subtitle: str | None = None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(0.8))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, TEAL
        if subtitle:
            sb = s.shapes.add_textbox(
                Inches(0.6), Inches(1.05), Inches(12), Inches(0.6)
            )
            q = sb.text_frame.paragraphs[0]
            q.text = subtitle
            q.font.size, q.font.color.rgb = Pt(17), DARK
        ft = s.shapes.add_textbox(Inches(0.6), Inches(7.0), Inches(12), Inches(0.3))
        f = ft.text_frame.paragraphs[0]
        f.text = f"Oevererosietool · update 26-08-2026 · {self.n}"
        f.font.size, f.font.color.rgb = Pt(10), GREY
        return s

    def bullets(self, s, items, left, top, width, height, size=16, gap=6):
        tb = s.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame
        tf.word_wrap = True
        first = True
        for it in items:
            level, text = it if isinstance(it, tuple) else (0, it)
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            bold = text.startswith("**")
            text = text.strip("*")
            p.text = ("• " if level == 0 else "– ") + text if not bold else text
            p.level = level
            p.font.size = Pt(size - 2 * level)
            p.font.bold = bold
            p.font.color.rgb = DARK
            p.space_after = Pt(gap)
        return tb

    def image(self, s, path, left, top, width=None, height=None):
        return s.shapes.add_picture(str(path), left, top, width=width, height=height)

    def note(self, s, text, left, top, width, size=12, color=GREY, italic=True):
        tb = s.shapes.add_textbox(left, top, width, Inches(0.6))
        tb.text_frame.word_wrap = True
        p = tb.text_frame.paragraphs[0]
        p.text = text
        p.font.size, p.font.color.rgb, p.font.italic = Pt(size), color, italic
        return tb

    def table(self, s, rows, left, top, width, col_widths=None, size=14):
        nr, nc = len(rows), len(rows[0])
        shp = s.shapes.add_table(nr, nc, left, top, width, Inches(0.4) * nr)
        t = shp.table
        if col_widths:
            for i, cw in enumerate(col_widths):
                t.columns[i].width = cw
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                cell = t.cell(r, c)
                cell.text = str(val)
                p = cell.text_frame.paragraphs[0]
                p.font.size = Pt(size)
                p.font.bold = r == 0 or (c == 0)
                p.font.color.rgb = DARK
                if c > 0:
                    p.alignment = PP_ALIGN.CENTER
                cell.fill.solid()
                cell.fill.fore_color.rgb = (
                    RGBColor(0xE6, 0xF5, 0xF3) if r == 0 else RGBColor(0xFF, 0xFF, 0xFF)
                )
        return shp

    def speaker(self, s, text):
        s.notes_slide.notes_text_frame.text = text

    def save(self):
        self.prs.save(OUT)
        return OUT


CAND = FIG / "candidates"
PICKS = {
    "single_line": "maas3_l_18580_18590",
    "resolution": "maas3_r_8930_8940",
    "r_sweep": "maas3_r_8930_8940",
    "cleanup": "nederrijn_r_4780_4790",
}


def _col_boxes(d, s, cols, top=Inches(1.8), accent=None):
    for i, (head, items) in enumerate(cols):
        left = Inches(0.6 + i * 4.2)
        box = s.shapes.add_shape(1, left, top, Inches(3.9), Inches(0.6))
        box.fill.solid()
        box.fill.fore_color.rgb = ORANGE if accent == i else TEAL
        box.line.fill.background()
        p = box.text_frame.paragraphs[0]
        p.text = head
        p.font.size, p.font.bold, p.font.color.rgb = (
            Pt(18),
            True,
            RGBColor(255, 255, 255),
        )
        d.bullets(s, items, left, top + Inches(0.8), Inches(3.9), Inches(3.8), size=15)


def build(renders: Path):
    d = Deck()

    # 1 ── title
    s = d.prs.slides.add_slide(d.blank)
    d.n += 1
    bg = s.shapes.add_shape(1, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = TEAL
    bg.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(3))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Oevererosietool — van lijn naar oever"
    p.font.size, p.font.bold, p.font.color.rgb = Pt(40), True, RGBColor(255, 255, 255)
    for line, sz in (
        ("Voortgang voorspellend model · wat we vonden · wat we nodig hebben", 20),
        ("Update 26 augustus 2026", 16),
    ):
        q = tf.add_paragraph()
        q.text = line
        q.font.size, q.font.color.rgb = Pt(sz), RGBColor(255, 255, 255)
        q.space_before = Pt(12)

    # 2 ── maart: één lijn is geen oever
    s = d.slide(
        "Waar we in maart stonden",
        "Eén afstand per scopevlak — en een oever is geen rechte lijn",
    )
    d.image(
        s,
        CAND / "single_line_wavy" / f"{PICKS['single_line']}.png",
        Inches(0.6),
        Inches(1.65),
        height=Inches(5.2),
    )
    d.bullets(
        s,
        [
            "Alleen hoogtemodel (AHN3 / AHN4 / AHN5), 3–4 jaar tussen opnames",
            "Per vlak: één afstand tot de hartlijn per opname (gestippeld)",
            "De gemeten oever (groen) buigt 90°; de 'oever' van het model ligt 56 m verderop",
            "Fout 0,94 m/jr — maar 'neem het gemiddelde' deed het beter (0,78)",
            "Risicogevallen (> 2 m/jr): 4,6 m/jr · 7.444 vlakken",
        ],
        Inches(6.4),
        Inches(1.9),
        Inches(6.6),
        Inches(4.5),
        size=16,
    )
    d.note(
        s,
        "paars = VVR / signaleringslijn · zwart = hartlijn · groen = gemeten oeverlijnen per jaar (donker = recenter)",
        Inches(6.4),
        Inches(6.2),
        Inches(6.6),
    )
    d.speaker(
        s,
        "De val: het maartcijfer zag er goed uit omdat de meetlat mild was (3–4 jaar) én omdat één getal per vlak veel verbergt.",
    )

    # 3 ── hybride levering + tabel van vorige week
    s = d.slide(
        "Wat er sindsdien binnenkwam",
        "Hybride levering: SAM + hoogtemodel · tot 11 meetmomenten · 1 jaar tussen t-punten",
    )
    d.image(s, renders / "aug-2.png", Inches(0.6), Inches(1.6), width=Inches(7.3))
    rows = [
        ["", "maart 2026\nalleen hoogtemodel", "augustus 2026\nhybride"],
        ["Baseline (naïef gemiddelde) — fout test", "0,78", "4,79"],
        ["LightGBM — fout test", "0,94", "4,12"],
        ["LightGBM t.o.v. baseline", "20 % slechter", "14 % beter"],
        ["Spreiding doelvariabele (m/jr)", "2,2", "11,4"],
        ["Tijd tussen t-punten", "3–4 jaar", "1 jaar"],
        ["Regio's met voorspellingen", "7.444", "8.006"],
    ]
    d.table(
        s,
        rows,
        Inches(8.1),
        Inches(1.7),
        Inches(4.9),
        [Inches(2.5), Inches(1.2), Inches(1.2)],
        size=11,
    )
    d.note(
        s,
        "Dezelfde meetruis, gedeeld door een 4× kleiner tijdsverschil: het doel wordt 4× ruiziger en élke fout groeit mee — ook die van de baseline. Relatief verslaat het model nu voor het eerst de baseline.",
        Inches(8.1),
        Inches(5.0),
        Inches(4.9),
        size=12,
        color=DARK,
        italic=False,
    )

    # 4 ── drie assen + uitgangspunt
    s = d.slide(
        "Drie assen om te verbeteren",
        "Uitgangspunt: het model van 19 augustus, resolutie 1, één vaste testset van 1.174 vlakken",
    )
    _col_boxes(
        d,
        s,
        [
            (
                "1 · Opschonen",
                [
                    "Verkeerde-oever-detecties weghalen zónder de meting weg te gooien",
                    "Kribben, bruggen, nevengeulen, doolhoven, fragmenten, uitschieters in de tijd",
                ],
            ),
            (
                "2 · Meerdere t-punten",
                [
                    "Snelheid als trend over álle metingen",
                    "Versnelling, terugkeer naar gemiddelde, recente helling als feature",
                ],
            ),
            (
                "3 · Resolutie",
                [
                    "Meerdere segmenten per vlak in plaats van één afstand",
                    "Gemeten én voorspelde oever als lijn",
                ],
            ),
        ],
    )
    d.note(
        s,
        "Leeswijzer: op deze vaste testset scoort het model van 19 augustus 3,99 m/jr (4,12 op de eigen split van toen). Alle cijfers hierna zijn op deze set, zodat stappen optelbaar zijn.",
        Inches(0.6),
        Inches(6.1),
        Inches(12),
        size=13,
        color=DARK,
        italic=False,
    )

    # 5 ── as 1: opschoonregels
    s = d.slide(
        "As 1 · Opschonen: van 3,99 naar 2,27",
        "Zeven regels — herstellen in plaats van weggooien",
    )
    d.image(s, FIG / "cleaning_ladder.png", Inches(0.6), Inches(1.6), width=Inches(7.6))
    d.image(s, renders / "aug-3.png", Inches(8.3), Inches(1.6), width=Inches(4.7))
    d.bullets(
        s,
        [
            "Kronkelende lijnen, verkeerde oever, doolhoven, fragmenten, te weinig punten, uitschieters t.o.v. de trend",
            "Een uitschieter kost één meting, niet het hele vlak: +511 vlakken t.o.v. de oude |v| > 50-filter",
            "Elke variant visueel gecontroleerd (50+ varianten, dekking blijft 0,94)",
        ],
        Inches(0.6),
        Inches(5.0),
        Inches(7.6),
        Inches(1.9),
        size=13,
        gap=4,
    )
    d.note(
        s,
        "Kribben · meeroeverig · bruggen · doolhoven · randen",
        Inches(8.3),
        Inches(4.4),
        Inches(4.7),
        size=11,
    )

    # 6 ── as 1: kribben & kunstwerken
    s = d.slide(
        "As 1 · Kribben en kunstwerken: nu een laag, geen gok meer",
        "Deze week geleverd: kribben landelijk (1.922 → 4.698) + 6.386 kunstwerken",
    )
    cleanup = CAND / "structures_cleanup" / f"{PICKS['cleanup']}.png"
    if not cleanup.exists():
        cleanup = FIG / "krib_before_after.png"
    d.image(s, cleanup, Inches(0.6), Inches(1.6), width=Inches(7.4))
    d.image(
        s, FIG / "structures_ablation.png", Inches(8.1), Inches(1.6), width=Inches(5.0)
    )
    d.bullets(
        s,
        [
            "Op de vaste testset is de laag metrisch neutraal: de opschoonregels vangen deze artefacten al",
            "Wél: 218 vlakken vallen af waarvan de 'oever' een constructie was — juistheid en dekking, geen foutwinst",
            "Les: wat een regel kan afleiden, hoeft niet gelabeld · wat een regel níet kan zien, wel",
        ],
        Inches(8.1),
        Inches(4.4),
        Inches(5.0),
        Inches(2.5),
        size=12,
        gap=4,
    )
    d.note(
        s,
        "links: zonder masker · rechts: met kribben + bruggen/kades/steigers (10 m) · grijs = constructie · paars = VVR",
        Inches(0.6),
        Inches(6.3),
        Inches(7.4),
        size=11,
    )

    # 7 ── as 2
    s = d.slide(
        "As 2 · Meerdere t-punten: van 2,27 naar 2,13",
        "Historie-features op resolutie 1 — en de horizon als meetlat voor het alarm",
    )
    d.image(s, FIG / "history.png", Inches(0.6), Inches(1.6), width=Inches(7.4))
    d.bullets(
        s,
        [
            "**Features uit de hele meetreeks",
            "Trend (Theil–Sen), spreiding om de trend, versnelling, terugkeer naar gemiddelde, recente helling",
            "**Wat níet werkte",
            "Trainen op losse jaar-op-jaar stappen (sub-jaar ruis), huber-loss (beter gemiddeld, slechter op de staart), debiet",
            "**Horizon ≥ 2 jaar — voor het alarm",
            "Hoogtemodel-paren zijn altijd 3–5 jaar uit elkaar; SAM-paren 1 jaar → ruis",
            "Zelfde doel (m/jr), zelfde fout, alleen paren ≥ 2 jaar: 1 vlak met 6 metingen = 10 leervoorbeelden",
        ],
        Inches(8.2),
        Inches(1.7),
        Inches(4.9),
        Inches(5.2),
        size=13,
        gap=4,
    )

    # 8 ── as 3 (1): what it looks like
    s = d.slide(
        "As 3 · Resolutie: van scalar naar oeverlijn",
        "Eén afstand per vlak verbergt dat 20 m hard erodeert en 80 m stil ligt",
    )
    d.image(
        s,
        CAND / "resolution" / f"{PICKS['resolution']}.png",
        Inches(0.6),
        Inches(1.6),
        height=Inches(5.3),
    )
    d.bullets(
        s,
        [
            "**Wat je ziet",
            "groen = gemeten oever per segment (2026) · bruin = voorspeld 2027 per segment",
            "gestippeld = de oude voorspelling: één getal voor het hele vlak",
            "paars = VVR / signaleringslijn",
            "**Dit vlak",
            "Eén getal zegt −9,7 m/jr; de vijf segmenten lopen van −17,1 tot +1,4",
            "Het bovenste segment nadert de VVR; de rest niet — het alarm hoort per segment te kijken",
            "**Vandaag",
            "VVR-jaar = één drempel per vlak (het verst gelegen punt van de signaleringslijn) tegen één scalar → per definitie laat",
        ],
        Inches(6.6),
        Inches(1.7),
        Inches(6.5),
        Inches(5.2),
        size=13,
        gap=4,
    )

    # 9 ── as 3 (2): R sweep
    s = d.slide(
        "As 3 · Hoe fijn kan het?",
        "Zelfde vlak, R = 1 … 100 · en de fout op de vaste testset per R",
    )
    d.image(
        s,
        CAND / "r_sweep" / f"{PICKS['r_sweep']}.png",
        Inches(0.6),
        Inches(1.5),
        width=Inches(6.3),
    )
    d.image(
        s, FIG / "resolution_sweep.png", Inches(7.0), Inches(1.5), width=Inches(6.1)
    )
    d.bullets(
        s,
        [
            "De segment-representatie (bruin) volgt de gemeten oever vanaf R ≈ 5–10; bij R = 50–100 raken segmenten leeg (rood)",
            "Fout per segment daalt van 2,25 (R = 1) naar 1,88 (R = 5) en 1,61 (R = 20) — kleiner wordt nauwkeuriger, niet ruiziger",
            "Terug-samengevoegd per vlak blijft de fout ≈ 2,1–2,2: het vlak-getal zelf heeft een vloer (0,5–1,1 m/jr representatiefout)",
            "Grens ligt bij de lijnbemonstering (60 punten per lijn), niet bij het model · praktisch: R = 5–10 (≈ 10–20 m)",
        ],
        Inches(0.6),
        Inches(5.2),
        Inches(12.5),
        Inches(1.8),
        size=12,
        gap=3,
    )

    # 10 ── matrix
    s = d.slide(
        "Alles bij elkaar", "Stap voor stap, op één vaste testset — waar zit de winst?"
    )
    d.image(s, FIG / "matrix.png", Inches(0.6), Inches(1.5), width=Inches(12.1))
    d.bullets(
        s,
        [
            "Opschonen: −1,72 m/jr (−43 %) · historie-features: −0,14 · structurenlaag: ±0 (dekking/juistheid) · resolutie: andere eenheid, −0,25 per segment",
            "**Het model was nooit de bottleneck — de labels waren het. En de regels raken op: wat overblijft, kan alleen een mens zien.",
        ],
        Inches(0.6),
        Inches(5.8),
        Inches(12.1),
        Inches(1.2),
        size=14,
        gap=4,
    )
    d.speaker(s, "Dit is de hele presentatie in één plaatje.")

    # 11 ── scorebord eerlijk
    s = d.slide(
        "Scorebord — eerlijk gemeten",
        "Alleen binnen een rij vergelijken: elke rij is een andere meetlat",
    )
    rows = [
        ["meetlat", "naïef", "LightGBM", "risicogevallen > 2 m/jr\nLGB / naïef"],
        [
            "maart · hoogtemodel · 3–4 jr · per vlak",
            "0,78",
            "0,94  (−20 %)",
            "4,6 / 5,5",
        ],
        ["19 aug · hybride · 1 jr · per vlak", "4,79", "4,12  (+14 %)", "≈ 8,3"],
        [
            "nu · zelfde 1-jr meetlat · eerlijke validatie",
            "2,86",
            "2,47  (+13 %)",
            "5,0 / 6,3",
        ],
        ["nu · segment · ≥ 2 jr · eerlijke validatie", "1,79", "1,23  (+31 %)", "3,5"],
    ]
    d.table(
        s,
        rows,
        Inches(0.6),
        Inches(1.8),
        Inches(12.1),
        [Inches(5.3), Inches(1.6), Inches(2.4), Inches(2.8)],
        size=14,
    )
    d.bullets(
        s,
        [
            "Eerlijk: de testset speelt geen rol meer bij het trainen (in maart en augustus wél) — strenger, en tóch lager",
            "Vlakken met voorspelling 8.006 → 10.721",
            "1,23 is m/jr op een gladdere meetlat — lees het als '31 % beter dan niets doen', niet als '2× beter dan 2,47'",
            "Toezegging 10× (op 4,12): nu 1,7× op die exacte meetlat, met strengere validatie",
        ],
        Inches(0.6),
        Inches(4.4),
        Inches(12),
        Inches(2.5),
        size=14,
    )

    # 12 ── wat we vragen
    s = d.slide("Wat we vragen", "Labelen is trial-and-error — geen eenmalige levering")
    _col_boxes(
        d,
        s,
        [
            (
                "Hoogtemodel-engineers",
                [
                    "De vijfdeling (scope_coverage) meeleveren",
                    "Per lijn een vlag: 'dit is een constructie / verkeerde oever'",
                    "Bekende randgevallen: meeroeverig, hoogwater",
                ],
            ),
            (
                "SAM-engineers",
                [
                    "Kwaliteitsvlag per meting",
                    "Vooral: méér jaren — elk jaar maakt het model vanzelf beter",
                    "Waar mogelijk: één lijn per meetmoment",
                ],
            ),
            (
                "Samen",
                [
                    "Labelronde in QGIS: wij leveren de gesorteerde lijst twijfelgevallen, jullie het oordeel",
                    "Eerste ronde: ~200 vlakken, twee uur",
                    "Daarna meten we opnieuw",
                ],
            ),
        ],
        accent=2,
    )

    # 13 ── asset management
    s = d.slide(
        "Wat dit oplevert voor asset management",
        "Van 'dit vlak gemiddeld' naar 'deze 20 meter'",
    )
    d.image(s, FIG / "r5_rijn_1220.png", Inches(0.6), Inches(1.7), height=Inches(4.9))
    d.bullets(
        s,
        [
            "Per segment een verwachte positie ≥ 2 jaar vooruit (48.000 segmenten, 10.000 vlakken)",
            "Een vlak kleurt rood zodra énig segment de signaleringslijn nadert — met het segment erbij",
            "Kribvakken en meerdelige vlakken worden daarmee vanzelf goed behandeld",
            "Bestaande stoplicht-kaart en tijdslider blijven; dit komt er als laag bij",
            "**Status: segmentvoorspellingen staan klaar; koppeling aan het VVR-alarm is de volgende stap",
        ],
        Inches(5.6),
        Inches(1.9),
        Inches(7.5),
        Inches(4.5),
        size=15,
    )

    # 14 ── samenvattend
    s = d.slide("Samenvattend")
    d.bullets(
        s,
        [
            "**Model: klaar en eerlijk gevalideerd",
            "Hybride bron, opschoonregels, historie-features, testset buiten de training",
            "**Van lijn naar oever",
            "Vijf segmenten per vlak; gemeten én voorspelde oever als lijn; positiefout ≈ 1 m",
            "**Bottleneck: gelabelde input, niet het algoritme",
            "Opschonen leverde 12× zoveel als modelleren; de regels raken op — wat overblijft vraagt een oordeel",
            "**Volgende stap",
            "Labelronde met SAM- en hoogtemodel-engineers · VVR-alarm op segmentniveau · daarna opnieuw meten",
        ],
        Inches(0.6),
        Inches(1.6),
        Inches(12),
        Inches(5.2),
        size=18,
        gap=8,
    )

    return d.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--renders", required=True, help="dir with march-NN.png / aug-N.png"
    )
    a = ap.parse_args()
    print("→", build(Path(a.renders)))
