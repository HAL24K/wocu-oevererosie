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

    # 2 ── maart
    s = d.slide(
        "Waar we in maart stonden",
        "Eén rechte lijn per scopevlak, 3–4 jaar tussen metingen",
    )
    d.image(s, renders / "march-03.png", Inches(0.6), Inches(1.7), width=Inches(7.6))
    d.bullets(
        s,
        [
            "Alleen hoogtemodel (AHN3 / AHN4 / AHN5)",
            "Oever = één afstand tot de hartlijn per vlak",
            "Fout 0,94 m/jr — maar het model verloor van 'neem het gemiddelde' (0,78)",
            "Risicogevallen (> 2 m/jr): 4,6 m/jr",
            "7.444 vlakken met voorspelling",
        ],
        Inches(8.5),
        Inches(1.9),
        Inches(4.4),
        Inches(4.5),
        size=16,
    )
    d.note(
        s,
        "De fout leek laag omdat de input simpel was: rechte lijnen, lange tijdstappen.",
        Inches(8.5),
        Inches(5.6),
        Inches(4.4),
    )
    d.speaker(
        s,
        "Zet de val: het maartcijfer zag er goed uit omdat de meetlat mild was — 3 à 4 jaar tussen metingen dempt de meetruis.",
    )

    # 3 ── hybride levering
    s = d.slide(
        "Wat er sindsdien binnenkwam",
        "Hybride levering: SAM + hoogtemodel, tot 11 meetmomenten, 1 jaar tussen t-punten",
    )
    d.image(s, renders / "aug-2.png", Inches(0.6), Inches(1.7), width=Inches(7.9))
    d.bullets(
        s,
        [
            "16.377 scopevlakken, 91.531 oeverlijnen",
            "Fout sprong naar 4,1 m/jr — niet omdat het model slechter werd:",
            (1, "dezelfde meetruis ÷ 4× kleiner tijdsverschil = 4× ruiziger doel"),
            (1, "ook de baseline ging van 0,78 naar 4,8"),
            "Voor het eerst verslaat het model de baseline (14 %)",
            "Nieuw: 12 % van de meetmomenten heeft méér dan één lijn; fragmenten en 'doolhoven'",
        ],
        Inches(8.7),
        Inches(1.9),
        Inches(4.3),
        Inches(4.8),
        size=15,
    )

    # 4 ── drie assen
    s = d.slide(
        "Drie assen om te verbeteren",
        "Alle drie doorlopen, elk gemeten tegen dezelfde vaste testset van 1.174 vlakken",
    )
    cols = [
        (
            "1 · Opschonen",
            [
                "Verkeerde-oever-detecties verwijderen zónder de hele meting weg te gooien",
                "Kribben, bruggen, nevengeulen, doolhoven, fragmenten, uitschieters in de tijd",
                "Ruim 50 varianten, elk visueel gecontroleerd",
            ],
        ),
        (
            "2 · Meerdere t-punten",
            [
                "Snelheid als trend over álle metingen",
                "Versnelling, terugkeer naar gemiddelde, recente helling als feature",
                "Voorspellen op ≥ 2 jaar in plaats van jaar-op-jaar",
            ],
        ),
        (
            "3 · Resolutie",
            [
                "Vijf segmenten per vlak in plaats van één afstand",
                "Gemeten én voorspelde oever als lijn",
                "Alarm zodra énig deel de signaleringslijn nadert",
            ],
        ),
    ]
    for i, (head, items) in enumerate(cols):
        left = Inches(0.6 + i * 4.2)
        box = s.shapes.add_shape(1, left, Inches(1.8), Inches(3.9), Inches(0.6))
        box.fill.solid()
        box.fill.fore_color.rgb = TEAL
        box.line.fill.background()
        p = box.text_frame.paragraphs[0]
        p.text = head
        p.font.size, p.font.bold, p.font.color.rgb = (
            Pt(18),
            True,
            RGBColor(255, 255, 255),
        )
        d.bullets(s, items, left, Inches(2.6), Inches(3.9), Inches(3.8), size=15)

    # 5 ── resolutie
    s = d.slide(
        "As 3 · Resolutie: van scalar naar oeverlijn",
        "Eén afstand per vlak verbergt dat 20 m hard erodeert en 80 m stil ligt",
    )
    d.image(s, FIG / "r5_ijssel.png", Inches(0.6), Inches(1.75), height=Inches(4.6))
    d.image(s, FIG / "r5_rijn.png", Inches(5.4), Inches(1.75), height=Inches(4.6))
    d.bullets(
        s,
        [
            "**Wat je ziet",
            "groen = gemeten oever per segment (2026)",
            "bruin = voorspelde oever 2027 per segment",
            "gestippeld = de oude voorspelling: één getal",
            "**Wat het oplevert",
            "IJssel-vlak: één getal zegt −6,7 m/jr; de segmenten lopen van −10,8 tot +7,0",
            "De fout wordt níet groter als segmenten kleiner worden",
            "Mediane positiefout op de horizon: 1,1 m",
        ],
        Inches(9.9),
        Inches(1.8),
        Inches(3.3),
        Inches(5),
        size=13,
        gap=4,
    )
    d.note(
        s,
        "R = 5 segmenten (≈ 20 m) · dichte bemonstering 60 punten per lijn · elk segment eigen meetreeks",
        Inches(0.6),
        Inches(6.45),
        Inches(9),
    )

    # 6 ── horizon
    s = d.slide(
        "As 2 · Voorspellen op twee jaar of langer",
        "Zelfde doel (m/jr), zelfde fout, alleen andere paren van metingen",
    )
    d.bullets(
        s,
        [
            "**Het probleem met één jaar",
            "Hoogtemodel: opnames 3–5 jaar uit elkaar → snelheid is stabiel",
            "SAM: jaarlijks → dezelfde meetruis gedeeld door 1 jaar → snelheid is grotendeels ruis",
            "De huidige testset mengt beide; het 1-jaars-cijfer is daardoor 'ruis-gedomineerd'",
            "**De regel",
            "Train en test alleen op paren van metingen ≥ 2 jaar uit elkaar — en gebruik álle zulke paren",
            "Eén vlak met 6 metingen levert 10 leervoorbeelden in plaats van 1",
            "Wordt vanzelf beter met elk SAM-jaar (3-jaars-paren zijn er nu nog te weinig)",
            "**Wat dit is en niet is",
            "Dit is hoe het alarm gescoord wordt, niet een ander model: de 1-jaars-cijfers blijven bestaan",
        ],
        Inches(0.6),
        Inches(1.8),
        Inches(12),
        Inches(5),
        size=16,
    )

    # 7 ── scorebord
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
        Inches(1.9),
        Inches(12.1),
        [Inches(5.3), Inches(1.6), Inches(2.4), Inches(2.8)],
        size=14,
    )
    d.bullets(
        s,
        [
            "Eerlijk: de testset speelt geen rol meer bij het trainen (in maart en augustus wél) — strenger, en tóch lager",
            "Binnen de hybride 1-jaars-meetlat: 4,12 → 2,47 en ≈ 8,3 → 5,0; vlakken met voorspelling 8.006 → 10.721",
            "Het 1,23 is m/jr op een gladdere meetlat — lees het als '31 % beter dan niets doen', niet als '2× beter dan 2,47'",
            "Toezegging 10× (op 4,12): nu 1,7× op die exacte meetlat, met strengere validatie",
        ],
        Inches(0.6),
        Inches(4.4),
        Inches(12),
        Inches(2.5),
        size=14,
    )

    # 8 ── pivot
    s = d.slide(
        "Waar kwam de winst vandaan?",
        "Het model was nooit de bottleneck — de labels waren het",
    )
    d.image(s, FIG / "three_axes.png", Inches(0.8), Inches(1.7), width=Inches(11.7))
    d.note(
        s,
        "As 1 (opschonen): −1,72 m/jr · As 2 (modelleren): −0,14 m/jr · As 3 (resolutie) verandert de meetlat en staat hier niet in",
        Inches(0.8),
        Inches(6.5),
        Inches(11.7),
        size=13,
    )
    d.speaker(
        s,
        "Dit is de hele presentatie in één plaatje. Alles ervoor is bewijs, alles erna is gevolg.",
    )

    # 9 ── wat opschonen betekende
    s = d.slide(
        "Wat 'opschonen' betekende",
        "Zeven regels — elk een gok naar iets dat iemand in deze zaal zéker weet",
    )
    d.image(s, renders / "aug-3.png", Inches(0.6), Inches(1.7), width=Inches(8.2))
    d.bullets(
        s,
        [
            "Is dit een krib, brug, steiger of kade?",
            "Is dit de rivieroever of een nevengeul, plas of haven?",
            "Is dit één oever of de overkant?",
            "Is deze meting 15 m van de trend een fout of een echte gebeurtenis?",
            "**Elke gok kost dekking of maakt fouten",
            "5,9 % van de metingen in 'normale' vlakken kwam uit een nevengeul",
            "Kribben-masker alleen: 60 % van de lijnpunten bij een krib was géén oever",
        ],
        Inches(9.0),
        Inches(1.8),
        Inches(4.2),
        Inches(5),
        size=14,
        gap=5,
    )

    # 10 ── structuren: kaart
    s = d.slide(
        "De eerste gok is al ingewisseld",
        "Deze week geleverd: kribben landelijk + kunstwerken (bruggen, kades, steigers, sluizen)",
    )
    d.image(s, FIG / "structures_map.png", Inches(0.6), Inches(1.6), width=Inches(8.3))
    d.bullets(
        s,
        [
            "Kribben: 1.922 → 4.698 bij de scope (IJssel en Maas waren er niet)",
            "Kunstwerken: 6.386 bij de scope, 3.884 gemaskeerd",
            "Geen heuristiek meer, maar een laag",
            "**Dit is het patroon: elke aanname vervangen door een feit",
        ],
        Inches(9.1),
        Inches(1.9),
        Inches(4.1),
        Inches(4),
        size=15,
    )

    # 11 ── structuren: effect
    s = d.slide(
        "…en het is meteen meetbaar",
        "Twee identieke runs, alleen de structurenlaag verschilt",
    )
    d.image(
        s, FIG / "structures_effect.png", Inches(0.6), Inches(1.6), width=Inches(7.4)
    )
    d.image(
        s, FIG / "krib_before_after.png", Inches(8.2), Inches(1.7), width=Inches(4.9)
    )
    d.bullets(
        s,
        [
            "Risicogevallen: 5,9 → 5,0 m/jr (−15 %)",
            "224 vlakken vallen af: hun 'oever' wás een constructie",
            "Segmentmodel bewoog niet (1,15 → 1,24): binnen de spreiding tussen splits",
        ],
        Inches(0.6),
        Inches(5.0),
        Inches(7.4),
        Inches(1.8),
        size=14,
    )
    d.note(
        s,
        "Nederrijn: kribvak vóór (links) en ná (rechts) het 10 m-masker — groen = gemeten lijnen, paars = signaleringslijn",
        Inches(8.2),
        Inches(5.0),
        Inches(4.9),
        size=11,
    )

    # 12 ── wat we vragen
    s = d.slide("Wat we vragen", "Labelen is trial-and-error — geen eenmalige levering")
    cols = [
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
                "Daarna meten we opnieuw — en zien we hoe ver opschonen alleen komt",
            ],
        ),
    ]
    for i, (head, items) in enumerate(cols):
        left = Inches(0.6 + i * 4.2)
        box = s.shapes.add_shape(1, left, Inches(1.8), Inches(3.9), Inches(0.6))
        box.fill.solid()
        box.fill.fore_color.rgb = ORANGE if i == 2 else TEAL
        box.line.fill.background()
        p = box.text_frame.paragraphs[0]
        p.text = head
        p.font.size, p.font.bold, p.font.color.rgb = (
            Pt(18),
            True,
            RGBColor(255, 255, 255),
        )
        d.bullets(s, items, left, Inches(2.6), Inches(3.9), Inches(3.8), size=15)

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
            "Opschonen leverde 3× zoveel als modelleren; elke regel is een gok die jullie zeker weten",
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
