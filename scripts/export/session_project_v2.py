"""Restyle the asset-manager session QGIS project and draw each stretch with QGIS itself.

Reads Etienne's ``Presentatie_Oktober2026.qgz`` (which opens ``wocu_sessies_14_15okt.gpkg``
next to it) and writes ``Presentatie_Oktober2026_v2.qgz`` beside it, leaving the original
and the GeoPackage untouched. Changes:

- regions coloured on ``kruising_klasse`` (8 classes) at 60 % opacity; regions without a
  crossing year split into "VVR, no prediction" (solid grey) and "no VVR" (no fill), instead of the
  same green as "no crossing before 2050";
- measured lines thin and solid on a warm scale (2015 dark orange → 2026 bright yellow),
  predicted lines dashed from light to dark blue, signal line purple dash-dot on top;
- VVR area as a light purple fill without outline (its outline is the signal line);
- background: PDOK aerial "Actueel" with OSM on top at 50 % (Etienne's tip), and the
  OSM → RD New transformation stored so QGIS does not ask;
- one spatial bookmark per session stretch;
- with ``--data``: the layers read that GeoPackage (built by ``session_geopackage.py`` from one
  model run) instead of the template's, and the measured lines are split into the lines
  the model used (on) and the lines cleaning removed (grey, off).

Runs with the Python of QGIS 4.2, the version Etienne saves with (3.44 cannot read the
4.2 project CRS and writes a project without one, which puts the layers off Africa):
``PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/qgis-pycache QT_QPA_PLATFORM=offscreen
/Applications/QGIS-final-4_2_3.app/Contents/MacOS/python
scripts/export/session_project_v2.py <folder with the .qgz> [--render <png dir>]``
(Both variables, and ``sys.dont_write_bytecode`` below, matter: cache files written inside
the app break its code signature, and macOS then refuses to open QGIS as "damaged".)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write cache files into the signed QGIS app

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from qgis.core import (  # noqa: E402
    QgsApplication,
    QgsBookmark,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFillSymbol,
    QgsLineSymbol,
    QgsMapRendererSequentialJob,
    QgsMapSettings,
    QgsProject,
    QgsRasterLayer,
    QgsReferencedRectangle,
    QgsRendererCategory,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QSize  # noqa: E402
from qgis.PyQt.QtGui import QColor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument(
    "--render",
    type=Path,
    help="write one PNG per session stretch here, as the project shows it",
)
ap.add_argument(
    "--name", default="Presentatie_Oktober2026_v2.qgz", help="project file to write"
)
ap.add_argument(
    "--data",
    help="GeoPackage in the same folder to read instead of wocu_sessies_14_15okt.gpkg",
)
ap.add_argument(
    "--order",
    nargs="*",
    default=[],
    help="stretch names in presenting order; their bookmarks get a number",
)
ap.add_argument(
    "--aerial",
    choices=["actueel", "2024"],
    default="actueel",
    help="aerial photo on at start (the current one shows high water on the Maas)",
)
args = ap.parse_args()

app = QgsApplication([], False)
app.initQgis()
project = QgsProject.instance()
assert project.read(str(args.folder / "Presentatie_Oktober2026.qgz"))
assert project.crs().authid() == "EPSG:28992", (
    f"project CRS not read: {project.crs().authid()!r}"
)
layer = {lyr.name(): lyr for lyr in project.mapLayers().values()}
TEMPLATE_GPKG = "wocu_sessies_14_15okt.gpkg"
if args.data:
    assert (args.folder / args.data).exists(), args.data
    for lyr in layer.values():
        src = lyr.source()
        if TEMPLATE_GPKG in src:
            name = src.split("layername=")[1].split("|")[0]
            lyr.setDataSource(
                f"{args.folder / args.data}|layername={name}", lyr.name(), "ogr"
            )
            assert lyr.isValid(), f"{name} not found in {args.data}"
root = project.layerTreeRoot()

# Regions: crossing class, as in the September style; "VVR, geen voorspelling" comes from
# session_geopackage.py, an empty class means no VVR. (A category expression testing the
# overlap with the VVR layer took minutes per redraw, so the GeoPackage carries it.)
CLASSES = [
    ("reeds gekruist (2026)", "139,0,0"),
    ("2027", "215,48,31"),
    ("2028-2029", "239,101,72"),
    ("2030-2031", "253,187,71"),
    ("2032-2035", "254,224,139"),
    ("2036-2042", "166,217,106"),
    ("2043-2050", "77,175,74"),
    ("geen kruising voor 2050", "26,120,55"),
]


def fill(rgb: str) -> QgsFillSymbol:
    return QgsFillSymbol.createSimple(
        {
            "color": f"{rgb},255",
            "outline_color": "35,35,35,255",
            "outline_width": "0.3",
        }
    )


cats = [QgsRendererCategory(v, fill(c), v) for v, c in CLASSES]
# No fill for "VVR, no prediction" either: grey regions drew too much attention in
# the first session (14 Oct); they stay recognisable by their outline.
cats.append(
    QgsRendererCategory(
        "VVR, geen voorspelling",
        QgsFillSymbol.createSimple(
            {
                "color": "0,0,0,0",
                "outline_color": "170,170,170,255",
                "outline_width": "0.3",
                "outline_style": "dash",
            }
        ),
        "VVR, geen voorspelling",
    )
)
cats.append(
    QgsRendererCategory(
        None,
        QgsFillSymbol.createSimple(
            {
                "color": "0,0,0,0",
                "outline_color": "220,220,220,255",
                "outline_width": "0.2",
            }
        ),
        "geen VVR",
    )
)
regions = layer["scoperegios_kruisingsjaar"]
regions.setRenderer(QgsCategorizedSymbolRenderer("kruising_klasse", cats))
regions.setOpacity(0.6)

# Readable field names when a scope area is clicked (Identify).
ALIASES = {
    "kruising_klasse": "kruisingsklasse",
    "kruisingsjaar": "jaar oever bij signaleringslijn",
    "gekozen_bron": "gekozen bron",
    "voorbij_lengte_m": "lengte voorbij de lijn (m)",
    "voorbij_aandeel_pct": "deel van het scopevlak voorbij de lijn (%)",
    "voorbij_afstand_med_m": "afstand voorbij de lijn, mediaan (m)",
    "voorbij_afstand_max_m": "afstand voorbij de lijn, grootste (m)",
    "ruimte_tot_lijn_min_m": "ruimte tot de lijn, kleinste (m)",
    "v_mean_m_per_yr": "voorspelde verplaatsing (m/jaar)",
    "vvr_breedte_m": "breedte VVR (m)",
    "pct_vvr_op_2026": "deel van de VVR al ingenomen (%)",
}


def alias(lyr):
    for field, label in ALIASES.items():
        i = lyr.fields().indexOf(field)
        if i >= 0:
            lyr.setFieldAlias(i, label)


alias(regions)

# Second view (15 Oct): only the coloured outline, so the base map and lines stay
# visible when inspecting a red scope area.
OUTLINE = "scopevlakken, alleen rand"
outline = regions.clone()
outline.setName(OUTLINE)
outline.setOpacity(1.0)
outline.setRenderer(
    QgsCategorizedSymbolRenderer(
        "kruising_klasse",
        [
            QgsRendererCategory(
                v,
                QgsFillSymbol.createSimple(
                    {
                        "color": "0,0,0,0",
                        "outline_color": f"{c},255",
                        "outline_width": "1.2",
                    }
                ),
                v,
            )
            for v, c in CLASSES
        ],
    )
)
alias(outline)
project.addMapLayer(outline, False)
root.addLayer(outline)
layer[OUTLINE] = outline

# Measured lines: warm colours only (purple is the signal line, blue the predictions),
# height-model years darker, newest SAM year brightest.
MEASURED = {
    2015: "153,52,4",
    2016: "178,70,6",
    2020: "204,95,14",
    2021: "224,122,24",
    2023: "240,152,36",
    2024: "250,184,52",
    2025: "254,216,74",
    2026: "255,255,60",
}
measured = layer["gemeten_oeverlijnen"]
measured.setRenderer(
    QgsCategorizedSymbolRenderer(
        "jaar",
        [
            QgsRendererCategory(
                y,
                QgsLineSymbol.createSimple(
                    {"line_color": f"{c},255", "line_width": "0.4"}
                ),
                f"gemeten {y}",
            )
            for y, c in MEASURED.items()
        ],
    )
)
REMOVED = "gemeten, verwijderd bij opschonen"
if args.data:
    removed = measured.clone()
    removed.setName(REMOVED)
    removed.setSubsetString('"gebruikt" = 0')
    removed.setRenderer(
        QgsSingleSymbolRenderer(
            QgsLineSymbol.createSimple(
                {"line_color": "200,200,200,255", "line_width": "0.3"}
            )
        )
    )
    project.addMapLayer(removed, False)
    root.addLayer(removed)
    layer[REMOVED] = removed
    measured.setSubsetString('"gebruikt" = 1')
    measured.setName("gemeten oeverlijnen (gebruikt)")
    layer[measured.name()] = measured
EARLIER = "eerdere gemeten oeverlijnen"
if args.data and measured.fields().indexOf("laatste") >= 0:
    # Default view since 15 Oct: only the latest survey, thick and yellow; the
    # earlier ones (by year) one click away.
    earlier = measured.clone()
    earlier.setName(EARLIER)
    earlier.setSubsetString('"gebruikt" = 1 AND "laatste" = 0')
    project.addMapLayer(earlier, False)
    root.addLayer(earlier)
    layer[EARLIER] = earlier
    measured.setSubsetString('"gebruikt" = 1 AND "laatste" = 1')
    measured.setName("laatst gemeten oeverlijn")
    measured.setRenderer(
        QgsSingleSymbolRenderer(
            QgsLineSymbol.createSimple(
                {"line_color": "255,255,60,255", "line_width": "1.0"}
            )
        )
    )
    layer[measured.name()] = measured

# Predicted lines: dashed, light cyan (2026) to dark blue (2031).
PREDICTED = {
    2026: "160,240,255",
    2027: "110,210,245",
    2028: "70,175,235",
    2029: "40,135,220",
    2030: "25,95,200",
    2031: "15,55,170",
}
predicted = layer["voorspelde_oeverlijnen"]
predicted.setRenderer(
    QgsCategorizedSymbolRenderer(
        "year",
        [
            QgsRendererCategory(
                y,
                QgsLineSymbol.createSimple(
                    {
                        "line_color": f"{c},255",
                        "line_width": "0.6",
                        "line_style": "dash",
                    }
                ),
                f"voorspeld {y}",
            )
            for y, c in PREDICTED.items()
        ],
    )
)

OTHER = "andere bron (niet gekozen)"
if args.data:
    other = QgsVectorLayer(
        f"{args.folder / args.data}|layername=andere_bron_oeverlijnen", OTHER, "ogr"
    )
    if other.isValid():
        other.setRenderer(
            QgsSingleSymbolRenderer(
                QgsLineSymbol.createSimple(
                    {
                        "line_color": "120,215,255,255",
                        "line_width": "0.5",
                        "line_style": "dot",
                    }
                )
            )
        )
        project.addMapLayer(other, False)
        root.addLayer(other)
        layer[OTHER] = other

ALL_YEARS = "voorspelde oeverlijnen per jaar"
if predicted.fields().indexOf("eerste") >= 0:
    # Default view since 15 Oct: only the first predicted year after the last survey.
    all_years = predicted.clone()
    all_years.setName(ALL_YEARS)
    project.addMapLayer(all_years, False)
    root.addLayer(all_years)
    layer[ALL_YEARS] = all_years
    predicted.setSubsetString('"eerste" = 1')
    predicted.setName("eerste voorspelde oeverlijn")
    predicted.setRenderer(
        QgsSingleSymbolRenderer(
            QgsLineSymbol.createSimple(
                {
                    "line_color": "31,111,224,255",
                    "line_width": "0.9",
                    "line_style": "dash",
                }
            )
        )
    )
    layer[predicted.name()] = predicted

signal = layer["signaleringslijn"]
signal.setRenderer(
    QgsSingleSymbolRenderer(
        QgsLineSymbol.createSimple(
            {
                "line_color": "140,15,213,255",
                "line_width": "0.8",
                "line_style": "dash dot",
            }
        )
    )
)
vvr = layer["vvr_vlak"]
vvr.setRenderer(
    QgsSingleSymbolRenderer(
        QgsFillSymbol.createSimple({"color": "140,15,213,60", "outline_style": "no"})
    )
)

locs = layer["locaties"]
locs.setRenderer(
    QgsSingleSymbolRenderer(
        QgsFillSymbol.createSimple(
            {
                "color": "0,0,0,0",
                "outline_color": "255,255,255,255",
                "outline_width": "0.8",
            }
        )
    )
)

# OSM (EPSG:3857) to RD New: store the current Amersfoort-to-WGS 84 (4) operation in the
# project, so QGIS stops asking which transformation to use.
src, dst = (QgsCoordinateReferenceSystem(c) for c in ("EPSG:3857", "EPSG:28992"))
context = project.transformContext()
op = QgsCoordinateTransform(src, dst, project).coordinateOperation()
if not op:
    from qgis.core import QgsDatumTransform

    ops = [
        o
        for o in QgsDatumTransform.operations(src, dst)
        if "Amersfoort to WGS 84 (4)" in o.name
    ]
    op = ops[0].proj
context.addCoordinateOperation(src, dst, op, True)
project.setTransformContext(context)
print("3857 → 28992:", op[:60], "…")

# Background: the latest PDOK aerial ("Actueel") with OSM on top at 50 %.
ACTUEEL = "Luchtfoto Actueel Ortho 8cm RGB"
if ACTUEEL not in layer:
    actueel = QgsRasterLayer(
        "crs=EPSG%3A28992&format=image%2Fjpeg&layers=Actueel_orthoHR&styles=default"
        "&tileMatrixSet=EPSG%3A28992&url=https%3A%2F%2Fservice.pdok.nl%2Fhwh%2Fluchtfotorgb"
        "%2Fwmts%2Fv1_0%3Frequest%3DGetCapabilities%26service%3DWMTS",
        ACTUEEL,
        "wms",
    )
    assert actueel.isValid(), "PDOK Actueel aerial did not load"
    project.addMapLayer(actueel, False)
    root.addLayer(actueel)
    layer[ACTUEEL] = actueel
layer["OSM Standard"].renderer().setOpacity(0.5)

# Layer order (top first) and what is on at start.
ORDER = {
    "signaleringslijn": True,
    predicted.name(): True,
    **({ALL_YEARS: False} if ALL_YEARS in layer else {}),
    measured.name(): True,
    **({EARLIER: False} if EARLIER in layer else {}),
    **({REMOVED: False} if args.data else {}),
    **({OTHER: False} if OTHER in layer else {}),
    "vvr_vlak": False,
    "locaties": True,
    OUTLINE: False,
    "scoperegios_kruisingsjaar": True,
    "OSM Standard": True,
    ACTUEEL: args.aerial == "actueel",
    "Luchtfoto 2024 Ortho 8cm RGB": args.aerial == "2024",
    "Luchtfoto 2023 Ortho 8cm RGB": False,
    "Luchtfoto 2022 Ortho 8cm RGB": False,
    "Luchtfoto 2021 Ortho 8cm RGB": False,
}
for i, (name, on) in enumerate(ORDER.items()):
    node = root.findLayer(layer[name].id())
    clone = node.clone()
    clone.setItemVisibilityChecked(on)
    root.insertChildNode(i, clone)
    node.parent().removeChildNode(node)

regions.setName("scopevlakken")  # "regio" clashes with RWS's own regions (15 Oct)

# One bookmark per stretch.
manager = project.bookmarkManager()
for b in list(manager.bookmarks()):
    manager.removeBookmark(b.id())
stretches: dict[str, list] = {}
for f in locs.getFeatures():
    stretches.setdefault((f["sessie"], f["locatie"]), []).append(f.geometry())
extents = {}
ORDER_OF = {n: i for i, n in enumerate(args.order)}
for (sessie, name), geoms in sorted(
    stretches.items(), key=lambda kv: (kv[0][0], ORDER_OF.get(kv[0][1], 99), kv[0][1])
):
    ext = geoms[0].boundingBox()
    for g in geoms[1:]:
        ext.combineExtentWith(g.boundingBox())
    ext = ext.buffered(max(ext.width(), ext.height()) * 0.08)
    extents[(sessie, name)] = ext
    bm = QgsBookmark()
    nr = f"{ORDER_OF[name] + 1}. " if name in ORDER_OF else ""
    bm.setName(f"{sessie[:6]} · {nr}{name}")
    bm.setGroup(sessie)
    bm.setExtent(QgsReferencedRectangle(ext, locs.crs()))
    manager.addBookmark(bm)

out = args.folder / args.name
assert project.write(str(out))
print(f"→ {out}")

if args.render:
    args.render.mkdir(parents=True, exist_ok=True)
    # OSM's tile server blocks scripted downloads, so the pictures leave the overlay out.
    order = [
        n.layer()
        for n in root.findLayers()
        if n.isVisible() and n.layer().name() != "OSM Standard"
    ]
    for (_sessie, name), ext in extents.items():
        ms = QgsMapSettings()
        ms.setLayers(order)
        ms.setDestinationCrs(locs.crs())
        ms.setOutputSize(QSize(1400, 1000))
        ms.setExtent(ext)
        ms.setBackgroundColor(QColor("white"))
        job = QgsMapRendererSequentialJob(ms)
        job.start()
        job.waitForFinished()
        png = args.render / (
            name.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        )
        job.renderedImage().save(str(png))
        print(png)

app.exitQgis()
