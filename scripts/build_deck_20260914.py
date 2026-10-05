"""Build the 14 Sep 2026 internal-session deck (docs/presentations/20260914_interne_sessie.pptx).

Two halves: (1) modelverbetering, (2) eindproduct. Prep and numbers on the
Notion page "2026-09-14 Interne sessie — modelverbetering & eindproduct".
Figures: docs/presentations/fig/*_20260914.png (+ vvr_kruising_map_20260913.png).
Segment-model VVR skill per variant is read from
data/04_model_outputs/segment_vvr_skill_20260914.csv when present.

Run: ``uv run python scripts/build_deck_20260914.py``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs/presentations/fig"
OUT = ROOT / "docs/presentations/20260914_interne_sessie.pptx"
FOOT = "Oevererosietool · interne sessie 14-09-2026"

TEAL = RGBColor(0x2B, 0xB5, 0xA6)
DARK = RGBColor(0x1F, 0x3A, 0x3D)
GREY = RGBColor(0x6B, 0x77, 0x78)
ORANGE = RGBColor(0xE0, 0x7A, 0x3F)
WHITE = RGBColor(255, 255, 255)
W, H = Inches(13.333), Inches(7.5)
NB = chr(8239)

SKILL = ROOT / "data/04_model_outputs/segment_vvr_skill_20260914.csv"
sk = pd.read_csv(SKILL, index_col=0) if SKILL.exists() else None


def pct(run, col):
    if sk is None or run not in sk.index:
        return "…"
    return f"{sk.loc[run, col]*100:+.0f} %"


def num(run, col, fmt="{:.2f}"):
    if sk is None or run not in sk.index:
        return "…"
    return fmt.format(sk.loc[run, col])


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, title, subtitle=None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(0.8))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, TEAL
        if subtitle:
            sb = s.shapes.add_textbox(Inches(0.6), Inches(1.05), Inches(12), Inches(0.6))
            q = sb.text_frame.paragraphs[0]
            q.text = subtitle
            q.font.size, q.font.color.rgb = Pt(17), DARK
        ft = s.shapes.add_textbox(Inches(0.6), Inches(7.0), Inches(12), Inches(0.3))
        f = ft.text_frame.paragraphs[0]
        f.text = f"{FOOT} · {self.n}"
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
            p.text = text if bold else ("• " if level == 0 else "– ") + text
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

    def table(self, s, rows, left, top, width, col_widths=None, size=14, first_bold=True):
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
                p.font.bold = r == 0 or (c == 0 and first_bold)
                p.font.color.rgb = DARK
                if c > 0:
                    p.alignment = PP_ALIGN.CENTER
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xE6, 0xF5, 0xF3) if r == 0 else WHITE
        return shp

    def boxes(self, s, cols, top=Inches(1.8), accent=None, size=15, height=Inches(4.2)):
        for i, (head, items) in enumerate(cols):
            left = Inches(0.6 + i * 4.2)
            box = s.shapes.add_shape(1, left, top, Inches(3.9), Inches(0.6))
            box.fill.solid()
            box.fill.fore_color.rgb = ORANGE if accent == i else TEAL
            box.line.fill.background()
            p = box.text_frame.paragraphs[0]
            p.text = head
            p.font.size, p.font.bold, p.font.color.rgb = Pt(18), True, WHITE
            self.bullets(s, items, left, top + Inches(0.8), Inches(3.9), height, size=size)

    def placeholder(self, s, text, left, top, width, height):
        box = s.shapes.add_shape(1, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xF1, 0xF4, 0xF4)
        box.line.color.rgb = GREY
        p = box.text_frame.paragraphs[0]
        p.text = text
        p.font.size, p.font.color.rgb, p.font.italic = Pt(13), GREY, True
        p.alignment = PP_ALIGN.CENTER
        return box

    def divider(self, title, sub):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        bg = s.shapes.add_shape(1, 0, 0, W, H)
        bg.fill.solid()
        bg.fill.fore_color.rgb = TEAL
        bg.line.fill.background()
        tb = s.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(3))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.size, p.font.bold, p.font.color.rgb = Pt(40), True, WHITE
        q = tf.add_paragraph()
        q.text = sub
        q.font.size, q.font.color.rgb = Pt(20), WHITE
        q.space_before = Pt(12)
        return s

    def speaker(self, s, text):
        s.notes_slide.notes_text_frame.text = text

    def save(self):
        self.prs.save(OUT)
        return OUT


def build():
    d = Deck()
    B = "20260907-near-fz"

    # 1 title
    d.divider(
        "Oevererosietool — waar we staan, en hoe het product eruit gaat zien",
        "Interne sessie 14 september 2026 · deel 1: modelverbetering · deel 2: eindproduct",
    )

    # 2 waar we staan
    s = d.slide("Waar we staan", "Strookjesmodel (R = 20, ≈ 5 m): één voorspelde oeverlijn per regio per jaar")
    rows = [
        ["", "waarde", "toelichting"],
        ["Bevroren testset (aug)", "3.99 → 2.27 → 2.13 → 1.37 → 0.85 m/jr", "opschonen → historie → paren ≥ 2 jr → R = 20 (verschillende meetlatten, zie 26 aug)"],
        ["Eerlijke run 7 sep", f"{num(B,'mae')} m/jr · skill {pct(B,'skill')}", "bevroren split · opgeschoonde levering (`_nearest`) · lengte-bemonstering"],
        ["Alleen strookjes in regio's mét VVR", f"{num(B,'vvr_mae')} m/jr · skill {pct(B,'vvr_skill')}", f"{num(B,'vvr_rows','{:.0f}')} testrijen in {num(B,'vvr_regions','{:.0f}')} regio's · naïef {num(B,'vvr_naive')}"],
        ["Dekking", f"≥ 1 strookje met ≥ 3 meetjaren: 7{NB}112 regio's (59 %) · met VVR 1{NB}187 (82 %)", f"met ≥ 2 jaren: 9{NB}784 (81 %) · eerlijk te scoren 3{NB}738 (31 %)"],
    ]
    d.table(s, rows, Inches(0.6), Inches(1.8), Inches(12.1), [Inches(3.3), Inches(3.9), Inches(4.9)], size=13)
    d.bullets(
        s,
        [
            "Skill = hoeveel beter dan 'neem het historisch gemiddelde als snelheid'; m/jr zijn tussen meetlatten niet vergelijkbaar",
            "Te bespreken: is skill op VVR-regio's de maat waar we op sturen?",
        ],
        Inches(0.6), Inches(4.6), Inches(12), Inches(2), size=15, gap=6,
    )

    # 3 funnel
    s = d.slide("Van scope naar voorspelling", "Alle regio's naast de regio's met een VVR")
    d.image(s, FIG / "funnel_segment_vvr_20260914.png", Inches(0.6), Inches(1.5), width=Inches(8.6))
    d.bullets(
        s,
        [
            "Eis besproken 13 sep: minstens één strookje met drie meetjaren (de oorspronkelijke drie-tijdpunten-eis, nu per 5 m)",
            "VVR-regio's zijn beter gedekt dan gemiddeld: 82 % haalt de eis, 43 % is eerlijk te scoren",
            "Onze cleaning kost 577 regio's, bijna alles door twee regels uit het scalaire tijdperk: 'fragment' (< 25 % van de regio gedekt) en 'te weinig punten' (< 8 samples)",
            "Te bespreken: die twee regels versoepelen voor strookjes, of vóór het hybride model zetten? En ≥ 2 of ≥ 3 jaren als eis?",
        ],
        Inches(9.4), Inches(1.5), Inches(3.7), Inches(5.4), size=11, gap=4,
    )

    # 4 hybride voorkeur
    s = d.slide("Welke bron kiest het hybride model?", "100 slechtst voorspelde regio's, gekleurd op de gekozen bron")
    d.image(s, FIG / "hybrid_pref_worst100_seg_20260914.png", Inches(0.6), Inches(1.5), width=Inches(8.6))
    d.bullets(
        s,
        [
            "85 van de 100 slechtste hebben SAM-voorkeur, tegen 61 % van alle gescoorde regio's",
            "Alle 85 via de puntentelling (0 via de datagate); 44 met 1–2 punten verschil",
            "Gemiddelde fout 3.0× hoger bij SAM-voorkeur (1.05 vs 0.35 m/jr) — deels terrein",
            "De telling weegt geometrie: afstand, gladheid, steilheid, aantal jaren. Niet tijd, niet voorspelbaarheid.",
            "Te bespreken: both-lines export (beide bronnen per regio) — validatie én circa 2 200 extra regio's",
        ],
        Inches(9.4), Inches(1.7), Inches(3.7), Inches(5), size=12, gap=6,
    )
    d.speaker(s, "Luke's model: gates (< 2 jaar = automatisch verlies), dan 7 criteria met punten, winnaar krijgt alles, export bevat alleen de winnaar.")

    # 5 Etienne
    s = d.slide("Wat Etienne zag op de satellietbeelden", "100 slechtst voorspelde SAM-regio's — ná onze filtering; hij beoordeelde 56, de 44 andere zijn niet bekeken")
    d.image(s, FIG / "etienne_causes_20260914.png", Inches(0.6), Inches(1.5), width=Inches(8.6))
    d.bullets(
        s,
        [
            "De regio's zijn de slechtste ná cleaning: de meting die de fout veroorzaakt kwam overal door onze regels heen",
            "Teal = een regel verwijderde wél andere metingen in die regio",
            "Bewolking vooral IJssel, schaduw vooral Maas; SAM accepteerde beelden tot 10 % bewolking",
            "Schaduw stond telkens op één beeld; schepen wachten bij sluizen; zand = werkzaamheden",
            "Twee regio's 'Erosie': echte, grote erosie die het model miste",
        ],
        Inches(9.4), Inches(1.7), Inches(3.7), Inches(5), size=12, gap=6,
    )

    # 6 examples
    s = d.slide("Zeven oorzaken, één voorbeeld elk")
    d.image(s, FIG / "cause_examples_20260914.png", Inches(0.6), Inches(1.2), width=Inches(12.1))

    # 7 oorzaak → fix → waar in de keten
    s = d.slide("Per oorzaak: mogelijke fix, en waar in de keten", "Wat vóór het hybride model kan, hoeft het hybride model niet te leren")
    rows = [
        ["oorzaak", "mogelijke fix", "waar", "wie"],
        ["Bewolking (≈ 23 van 56)", "bewolkingsdrempel 10 % → 0–5 % bij de SAM-verwerking; herverwerking IJssel", "vóór hybride", "Lars / Sytze"],
        ["Nevengeul (12)", "masker: officiële laag + vegetatielegger 'Water'", "vóór hybride", "wij"],
        ["Schepen bij sluizen (5)", "ruimer masker rond sluis_stuw", "vóór hybride", "wij"],
        ["Schaduw (8)", "per beeld; geen geometrische regel — model-toets op de nieuwste meting?", "ná hybride", "wij"],
        ["Werkzaamheden (2) / ingrepen", "zand op het beeld; Satellietdataportaal als check bij hoogtemodel-uitschieters", "ná hybride", "wij + van Oord"],
        ["Verkeerde bron gekozen (85 van 100)", "both-lines export; onze maskers vóór de puntentelling", "hybride", "Luke + wij"],
        ["Onze eigen dekkingsregels (577 regio's)", "'fragment' en 'te weinig punten' versoepelen voor strookjes", "ná hybride", "wij"],
        ["Te weinig meetjaren", "meer DTM-jaren? SAM-jaren via both-lines", "bron", "Luke / Joost"],
    ]
    d.table(s, rows, Inches(0.6), Inches(1.7), Inches(12.1), [Inches(3.3), Inches(5.0), Inches(1.7), Inches(2.1)], size=12)
    d.bullets(s, ["De temporele regel (uitschieters t.o.v. de trend per regio) blijft zoals hij is; aanscherpen op 13 sep leverde niets meetbaars op", "Te bespreken: volgorde, en wat wij zelf oppakken tegenover wat we vragen"], Inches(0.6), Inches(5.7), Inches(12), Inches(1.2), size=13, gap=4)

    # 9 nevengeul + sluizen
    s = d.slide("Nevengeul en sluizen: vóór het hybride model", "Beide zijn maskers op de lijnen, geen modelregels")
    d.boxes(
        s,
        [
            ("Nevengeul", [
                "Officiële laag zit al in onze scope-data: 105 vlakken, 49.7 km²",
                "8 van Etienne's 12 nevengeul-regio's liggen erbinnen; in 6 raakt een masker ≥ 10 % van de metingen",
                "De andere 4: strangen en plassen — vegetatielegger 'Water' erbij",
                f"1{NB}765 regio's raken een nevengeul: metingen maskeren, geen regio's",
            ]),
            ("Sluizen", [
                "Schepen wachten bij sluizen; 4 van 5 gevallen raakten we al deels",
                "Masker rond sluis_stuw nu 10 m; kunstwerkenlaag hebben we",
            ]),
            ("Te bespreken", [
                "Wie past het toe: wij op de levering, of Lars/Luke vóór de puntentelling?",
                "Zelfde masker voor het hoogtemodel?",
                "Toets: skill op VVR-strookjes voor/na, plus de echte-erosie-regio's",
            ]),
        ],
        accent=2, size=13,
    )

    # 10 vragen aan partners
    s = d.slide("Te bespreken met partners")
    d.boxes(
        s,
        [
            ("Lars / Sytze (SAM)", [
                "Bewolkingsdrempel 10 % → 0–5 %: haalbaar, en voor welke jaren?",
                "Herverwerking IJssel met wolkmasker of infrarood?",
                "Kwaliteitsvlag per meting meeleveren?",
            ]),
            ("Luke (hybride)", [
                "Both-lines export: beide bronnen per regio?",
                "Onze maskers vóór de puntentelling toepassen?",
                "Test-fout als criterium naast geometrie?",
            ]),
            ("Rik / Joost / Pam", [
                "Komen er meer DTM-jaren?",
                "Bij welke snelheid en VVR-afstand grijpt RWS in?",
                "10–20 bekende probleemlocaties voor kalibratie?",
                "Vervanging Etienne tot ≈ 30 sep?",
            ]),
        ],
        size=14,
    )

    # 11 hoe meten we vooruitgang
    s = d.slide("Hoe meten we vooruitgang — en wat zeggen we in oktober?")
    rows = [
        ["", "was", "nu"],
        ["Bevroren testset, strookjes ≥ 2 jr", "3.99 (19 aug)", "0.85"],
        ["Skill t.o.v. naïef (eerlijke run)", "−20 % (maart)", pct(B, "skill")],
        ["Skill op VVR-strookjes", "—", pct(B, "vvr_skill")],
        ["Regio's met voorspelling (≥ 3 meetjaren in ≥ 1 strookje)", f"7{NB}444 (maart)", f"7{NB}112 (bij ≥ 2 jaren: 9{NB}784)"],
    ]
    d.table(s, rows, Inches(0.6), Inches(1.6), Inches(6.6), [Inches(3.4), Inches(1.4), Inches(1.8)], size=12)
    d.bullets(
        s,
        [
            "Te bespreken",
            ("Eén toets voor elke opschoonstap: skill op VVR-strookjes op de bevroren split, plus de echte-erosie-regio's?"),
            ("Standaard omzetten naar de opgeschoonde levering en lengte-bemonstering; develop pushen; PR naar main"),
            ("Oktober: een factor, of skill en dekking?"),
            ("Idee: alle oude tijdpunten gebruiken om de níeuwste meting te voorspellen — als toets op die meting (uitschieter of echte verschuiving?)"),
            ("Bijwerken met nieuwe leveringen: één pipeline-run per levering (bestaat); wat ontbreekt: versiebeheer en een verschil-overzicht tussen runs"),
        ],
        Inches(7.3), Inches(1.7), Inches(5.8), Inches(5), size=14, gap=7,
    )

    # 12 divider
    d.divider("Deel 2 — het eindproduct", "Stoplicht 1 bestaat als GIS-lagen. Hoe komt het bij de gebruiker?")

    # 13 stoplicht 1
    s = d.slide("Stoplicht 1", "Kruisingsjaar per regio: de voorspelde oever bereikt de landzijde van de VVR in minstens één strookje")
    d.image(s, FIG / "vvr_kruising_map_20260913.png", Inches(0.6), Inches(1.5), height=Inches(5.4))
    d.bullets(
        s,
        [
            f"1{NB}325 regio's met een VVR: 132 al gekruist · 12 in 2027 · 47 vóór 2031 · 642 niet vóór 2050",
            f"Voorspelde oeverlijn per jaar 2026–2031: 9{NB}759 regio's, 58{NB}554 lijnen",
            "Kaart: rood = snel; de reds liggen bovenop, dus dichte trajecten ogen roder dan de telling",
            "Te bespreken: dit strookjes-jaar of het regio-jaar uit de pipeline (144 vs 91 vóór 2027)?",
        ],
        Inches(7.4), Inches(1.7), Inches(5.7), Inches(5), size=13, gap=6,
    )

    # 14 wat de gebruiker ziet
    s = d.slide("Wat de gebruiker ziet (QGIS-prototype)", "Per regio: gemeten geschiedenis, voorspelde oever per jaar, kruisingsjaar, bron — en waarom het mis kan zijn")
    d.placeholder(s, "screenshot QGIS: kruisingskaart, rand = bron", Inches(0.6), Inches(1.6), Inches(6.0), Inches(4.2))
    d.placeholder(s, "screenshot QGIS: één regio, oever loopt jaar voor jaar landinwaarts", Inches(6.9), Inches(1.6), Inches(6.0), Inches(4.2))
    d.note(s, "vulling = kruisingsjaar rood → groen · rand zwart = SAM, blauw = hoogtemodel · Etienne's oorzaken als ring · gestreept = door ons weggefilterd", Inches(0.6), Inches(5.9), Inches(12), size=11)
    d.bullets(s, ["Te bespreken: is het stoplicht dé weergave, of ook de bewegende oever (voorspeld per jaar én de gemeten jaren terug)? Altijd per jaar, of ook kleinere stappen? Bekende erosielocaties als voorbeeld"], Inches(0.6), Inches(6.25), Inches(12.1), Inches(0.6), size=12, gap=2)

    # 15 urgentie × vertrouwen
    s = d.slide("Urgentie én vertrouwen?", "Van de 91 vroege alarmen staan er 19 op regio's met een fout boven 5 m/jr; 6 000 voorspellingen zijn niet te scoren")
    rows = [
        ["", "vertrouwen hoog", "vertrouwen midden", "vertrouwen laag / niet te scoren"],
        ["kruising ≤ 2027", "rood", "rood met vraagteken", "eerst de data bekijken"],
        ["kruising 2028–2035", "oranje / geel", "geel met vraagteken", "eerst de data bekijken"],
        ["kruising later / niet", "groen", "groen, licht", "grijs: onvoldoende data"],
    ]
    d.table(s, rows, Inches(0.6), Inches(1.8), Inches(12.1), [Inches(3.0), Inches(2.8), Inches(2.9), Inches(3.4)], size=13)
    d.bullets(
        s,
        [
            "Vertrouwen zou kunnen komen uit: fout van de regio, dekking van strookjes, datakwaliteitsvlaggen",
            "Regio's zonder voorspelling (19 %): hoe tonen we die?",
            "Te bespreken: één stoplicht, of stoplicht plus vertrouwen? En welke klassen?",
        ],
        Inches(0.6), Inches(4.3), Inches(12), Inches(2.5), size=14, gap=6,
    )

    # 16 vorm van oplevering
    s = d.slide("Vorm van oplevering", "Juni-planning: testoplevering half–eind september, launchparty half oktober")
    rows = [
        ["optie", "klaar", "eigenaar", "opmerking"],
        ["GeoPackage + QGIS-stijlen + PDF-galerij", "nu", "wij", "de stijlen leggen lagen, attributen en kleuren vast"],
        ["Concept viewer", "fase 2-deliverable", "Etienne (verlof tot ≈ 30 sep)", "wie pakt het op, of schuift het?"],
        ["Webtool", "fase 3", "consortium", "buiten scope voor oktober"],
    ]
    d.table(s, rows, Inches(0.6), Inches(1.7), Inches(12.1), [Inches(3.6), Inches(1.8), Inches(2.9), Inches(3.8)], size=13)
    d.bullets(
        s,
        ["Te bespreken: wat is de testoplevering van deze maand, en aan wie?"],
        Inches(0.6), Inches(4.0), Inches(12), Inches(1), size=15,
    )

    # 17 open punten
    s = d.slide("Open punten voor vandaag")
    d.boxes(
        s,
        [
            ("Deel 1 · model", [
                "Maat: skill op VVR-strookjes?",
                "Toets per opschoonstap",
                "Volgorde: nevengeul, sluis, temporele regel",
                "Wie vraagt wat aan Lars, Luke, Rik",
                "Oktober: wat zeggen we",
            ]),
            ("Deel 2 · product", [
                "Vorm van de testoplevering",
                "Stoplicht met of zonder vertrouwen",
                "Welk kruisingsjaar",
                "RWS-vragen: wie en wanneer",
                "Viewer: wachten op Etienne of overnemen",
            ]),
            ("Parkeerplaats / logistiek", [
                "Hoe leggen we deze sessie vast?",
                "Soft launch: wanneer, en is er een overleg vóór die tijd?",
                "18/21 sep-sessie: datum of schrappen",
                "Verslag weekly 9 sep zodra opname er is",
                "Volledige segment-CV",
            ]),
        ],
        accent=2, size=13,
    )

    return d.save()


if __name__ == "__main__":
    print("→", build())
