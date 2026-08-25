# Storyline update 2026-08-26 13:00 — "Van lijn naar oever"

Doel: 1) resultaten sinds maart, 2) waarom gelabelde input nu de bottleneck is.
Volgorde: eerst resultaten (4–6), dán de pivot (7) — anders klinkt "de data" als excuus.

1. **Waar we in maart stonden** — rechte lijn per vlak, 3–4 jr tussen metingen, fout 0.78 (risico 4.6), model verloor van gemiddelde. *(maart slide 3 + 12)*
2. **Wat er binnenkwam** — hybride SAM+hoogtemodel, 16.377 vlakken, tot 11 meetmomenten, 1 jr tussen t-punten; fout 4.1 door 4× ruiziger doel; model verslaat baseline. *(huidige slide 1–2)*
3. **Drie assen** — opschonen · t-punten · resolutie; alle drie tegen dezelfde vaste testset.
4. **Resolutie: van scalar naar oeverlijn** — R=1 vs R=5, gemeten én voorspeld; fout wordt niet groter als segmenten kleiner worden (mediane positiefout 1.1 m). *(FIGUUR: zelfde regio maart-lijn vs R=5)*
5. **Horizon ≥2 jr** — per segment positie ≥2 jr vooruit: R² 0.49, mediane positiefout 1.1 m, 72 % beter dan niets; wordt beter met elk SAM-jaar.
6. **Scorebord (eerlijk) — alleen binnen een rij vergelijken**
   | meetlat | naïef | LGB | staart LGB / naïef |
   |---|---|---|---|
   | maart: hoogtemodel, 3–4 jr, per vlak, test-ES | 0.78 | 0.94 (−20 %) | 4.61 / 5.52 |
   | 19 aug: hybride, 1 jr, per vlak, test-ES | 4.79 | 4.12 | ~8.3–9.1 |
   | nu, zelfde 1-jr meetlat, eerlijk | 2.86 | **2.47** | **5.00** / 6.29 |
   | nu, segment ≥2 jr (R=5), eerlijk | 1.79 | **1.23** | **3.51** |
   Zeg: 1.2 is m/jr op een gladdere meetlat — claim "31 % beter dan naïef, mediane positiefout 1.1 m (p90 6.5)", niet "2× beter dan 2.47". 10×: was op 4.12; nu 1.7× op die meetlat mét strengere validatie; benoem dat zelf.
7. **PIVOT — waar kwam de winst vandaan?** — 3.99 → 2.27 (opschonen) → 2.13 (modelleren). Opschonen = 3× modelleren. Labels waren de bottleneck. *(FIGUUR: 3 balken)*
8. **Wat opschonen betekende** — 7 regels = 7 gokken naar wat de zaal zeker weet; 5.9 % nevengeul-contaminatie. *(huidige slide 3 + nevengeulen)*
9. **Eerste gok ingewisseld** — kribben 1.922 → 4.698, 6.386 kunstwerken; aanname → feit. *(FIGUUR: IJssel-krib voor/na; structures.gpkg in QGIS)*
10. **Wat we vragen** — hoogtemodel: scope_coverage + constructie/verkeerde-oever-vlag per lijn; SAM: kwaliteitsvlag + meer jaren; samen: labelronde in QGIS (wij lijst, zij oordeel; trial-and-error).
11. **Voor asset management** — per segment positie → vlak rood als *enig* deel de signaleringslijn nadert, met segment erbij.
12. **Samenvattend** — model klaar en eerlijk; van lijn naar oever; bottleneck = gelabelde input; volgende stap labelen, dan opnieuw meten.

Te maken figuren: slide 4, 7, 9. Bronnen: notebooks/inspector, ledger.csv, structures.gpkg.
