"""Build the 30 Sep 2026 WOCU weekly deck in Dutch (docs/presentations/20260930_wocu_weekly_nl.pptx).

Six slides: situation, root causes, the odd-one-out survey, first mask results,
asks to Luke / Etienne / Pam, and the road to the 14 Oct pre-meeting. Reuses the
14 Sep deck's ``Deck`` helpers. Figures: docs/presentations/fig/*_20260930.png;
mask numbers from the mask experiment's CSVs.

Run: ``PYTHONPATH=. uv run python scripts/build_deck_20260930.py --mask-results … --jetty-results …``
"""

from __future__ import annotations

import argparse

import pandas as pd
from pptx.util import Inches

import scripts.build_deck_20260914 as base

base.FOOT = "Oevererosietool · WOCU weekly 30-09-2026"
FIG = base.FIG
OUT = base.ROOT / "docs/presentations/20260930_wocu_weekly_nl.pptx"

ap = argparse.ArgumentParser()
ap.add_argument("--mask-results", required=True)
ap.add_argument("--jetty-results", required=True)
args = ap.parse_args()
mr = pd.read_csv(args.mask_results, index_col=0)
mj = pd.read_csv(args.jetty_results, index_col=0)


d = base.Deck()

s = d.slide(
    "Waar we staan, en wat we vandaag nodig hebben",
    "De slechtste voorspellingen komen door een handvol slechte invoerlijnen — de meeste vóór het hybride model op te lossen",
)
d.bullets(
    s,
    [
        "**Sinds 14 september**",
        "Etienne's oorzaken gekoppeld aan de 100 slechtste SAM-regio's, in één galerij gesorteerd op oorzaak",
        "Een controle die per regio de meting vindt die afwijkt van de rest, langs de hele oever",
        "Eerste test van twee maskers: de RWS-legger nevengeulen, en ruimere buffers rond sluizen en aanlegplaatsen",
        "**Vandaag nodig**",
        "Luke: de lijnen die het hybride model ingaan, beide modellen per regio — en de nieuwe hoogtedata Rijntakken",
        "Etienne: het bewolkingspercentage op elke SAM-lijn (of exports bij 0 % en 5 %), en een blik op de maskers",
        "**Volgende mijlpaal**",
        "Voorbespreking 14 oktober met collega's van RWS en Van Oord van buiten de kerngroep",
    ],
    Inches(0.6), Inches(1.8), Inches(12), Inches(5), size=18, gap=8,
)

s = d.slide(
    "Waarom de slechtste regio's fout gaan",
    "100 slechtst voorspelde SAM-regio's ná onze opschoning — Etienne beoordeelde er 56 op de satellietbeelden",
)
d.image(s, FIG / "causes_chart_nl_20260930.png", Inches(0.6), Inches(1.6), height=Inches(5.3))
d.bullets(
    s,
    [
        "Al deze regio's kwamen door onze huidige regels: de slechte lijn lijkt een gewone oeverlijn",
        "Bewolking (20) en schaduw (8) zijn beeldkwaliteit — op te lossen bij de bron",
        "Nevengeulen (12) en afgemeerde schepen (5) zijn geometrisch — op te lossen met een masker",
        "85 van de 100 slechtste hebben SAM-voorkeur; allemaal gewonnen op de puntentelling, 44 met maar 1–2 punten verschil",
        "Twee zijn echte erosie die het model miste",
    ],
    Inches(7.0), Inches(1.8), Inches(5.8), Inches(5), size=15, gap=8,
)

s = d.slide(
    "De slechte lijn is meestal het nieuwste beeld",
    "Rood = het stuk van een meting dat 8 m of meer afwijkt van de meerderheid van de metingen in die regio",
)
d.image(s, FIG / "cloud_panels_20260930.png", Inches(0.4), Inches(1.55), width=Inches(8.6))
d.bullets(
    s,
    [
        "Bewolking: in 14 van de 20 regio's is de uitschieter de laatste meting (maart–april 2026), 12–38 m ernaast",
        "Schaduw: steeds naar de rivier toe — een lijn in het water",
        "De bestaande trendregel mist ze: ze blijven onder de drempel van 15 m",
        "Grens: echte erosie op het laatste beeld ziet er hetzelfde uit, tot het volgende beeld het bevestigt",
        "Vandaar de vraag: met het bewolkingspercentage per lijn filteren we ze bij de bron",
    ],
    Inches(9.2), Inches(1.7), Inches(3.9), Inches(5.2), size=13, gap=7,
)

s = d.slide(
    "Eerste resultaten maskers: nevengeulen ja, schepen nog niet",
    "Maskers toegepast op de opgeschoonde metingen van de huidige run · nog niet door het model gehaald",
)
d.image(s, FIG / "side_channel_mask_check_nl_20260930.png", Inches(0.4), Inches(1.55), height=Inches(5.35))
side = mr.loc["side: legger +10 m"]
jet = mj.loc["jetty: 100 m"]
d.bullets(
    s,
    [
        "**Nevengeulen — RWS-legger nevengeulen, +10 m**",
        f"Raakt {int(side['target hit >=10%'])} van Etienne's {int(side['target regions'])} nevengeulregio's en verwijdert daar 25–83 % van de punten",
        f"Raakt ook {int(side['all other regions touched'])} andere regio's; bij een steekproef lijken de meeste dezelfde fout "
        "(de lijn ligt langs de nevengeul, ver van de rivieroever) — Etienne een steekproef laten bekijken",
        "De andere 7: strangen en plassen die niet in de legger staan",
        "**Schepen — niet bij sluizen, maar afgemeerd**",
        "4 van de 5 scheepsregio's liggen 22–125 m van een aanlegsteiger; sluizen liggen 225 m tot 4 km weg",
        f"Een buffer van 100 m rond steigers raakt er {int(jet['target hit >=10%'])} van de 5, maar ook {int(jet['all other regions touched'])} andere regio's — te grof; vraagt een slimmere regel",
    ],
    Inches(7.4), Inches(1.6), Inches(5.6), Inches(5.3), size=13, gap=6,
)
d.note(s, "Links: 3 van Etienne's nevengeulregio's (DOEL) en 9 willekeurige andere regio's waar het masker ≥ 50 % raakt; rood = verwijderd.",
       Inches(7.4), Inches(6.3), Inches(5.6), size=10)

s = d.slide("Wat we van jullie nodig hebben", "Lars, Sytze en Joost zijn er niet — Etienne en Luke, kunnen jullie helpen?")
d.table(
    s,
    [
        ["wie", "wat", "waarom"],
        ["Luke", "De lijnen die het hybride model ingaan ná jouw opschoning — beide modellen per regio, vóór de puntentelling",
         "De voorkeur valideren; ~2.200 extra modelleerbare regio's. De ruwe invoer hebben we, deze set niet"],
        ["Luke", "Nieuwe hoogtedata Rijntakken: voor ons beschikbaar? tijd om te verwerken? onderdeel van de demo over 5 weken?",
         "Een recente meting voor elke regio in de Rijntakken, welk model ook wint"],
        ["Etienne", "Bewolkingspercentage als attribuut op elke SAM-lijn (anders: exports bij 0 % en 5 %) — kun jij Lars' pipeline draaien?",
         "De grootste enkele verbetering: 20 van de 56 beoordeelde fouten zijn bewolking"],
        ["Etienne", "Blik op de maskers: juiste laag voor nevengeulen, juiste breedte rond steigers?", "Voordat ze de cijfers veranderen"],
        ["Pam", "Toegang tot de Claap-opnames na 2 september", "Notities om mee verder te werken"],
    ],
    Inches(0.6), Inches(1.7), Inches(12.1), [Inches(1.3), Inches(6.2), Inches(4.6)], size=13,
)

s = d.slide("Op weg naar 14 oktober", "Voorbespreking met collega's van RWS en Van Oord van buiten de kerngroep — een eerste blik van buiten")
d.boxes(
    s,
    [
        ("Laten zien", [
            "Skill in VVR-regio's — waar de tool ertoe doet",
            "De 50 slechtste VVR-regio's, met oorzaken",
            "Fout op hoogtemodel- tegenover SAM-lijnen",
        ]),
        ("Als het lukt", [
            "Effect van de maskers op de skill",
            "Bewolkingsfilter, als het attribuut er is",
            "Rijntakken met de nieuwe hoogtedata",
        ]),
        ("Vandaag afspreken", [
            "Wie komt er op 14 oktober, en wat verwachten ze",
            "Datum soft launch en de sessie met de assetmanagers",
            "Wanneer ons deel van het rapport Fase 2 af moet",
        ]),
    ],
    top=Inches(1.9), accent=0, size=15, height=Inches(4.2),
)

d.prs.save(OUT)
print("→", OUT)
