"""Close-up pictures of named scope regions from a session QGIS project (QGIS 4.2 Python).

Each argument ``name=id1,id2,...`` draws the combined extent of those regions (plus a
margin) with the project's visible layers, OSM left out (its tile server blocks scripted
downloads). Used by ``forecast_render.py`` to check forecasts by eye.

Run: ``PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen
/Applications/QGIS-final-4_2_3.app/Contents/MacOS/python scripts/evaluate/qgis_render_regions.py
<project.qgz> <out dir> [--aerial="Luchtfoto 2024 Ortho 8cm RGB"]
zuilichem_oost=rijn_l_8360_8370,rijn_l_8370_8380``
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write cache files into the signed QGIS app

from pathlib import Path  # noqa: E402

from qgis.core import (  # noqa: E402
    QgsApplication,
    QgsFeatureRequest,
    QgsMapRendererSequentialJob,
    QgsMapSettings,
    QgsProject,
)
from qgis.PyQt.QtCore import QSize  # noqa: E402
from qgis.PyQt.QtGui import QColor  # noqa: E402

args = sys.argv[1:]
aerial = next((a.split("=", 1)[1] for a in args if a.startswith("--aerial=")), None)
project_path, out_dir, *groups = [a for a in args if not a.startswith("--aerial=")]
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
app = QgsApplication([], False)
app.initQgis()
project = QgsProject.instance()
assert project.read(project_path)
regions = project.mapLayersByName("scoperegios_kruisingsjaar")[0]
layers = [
    n.layer()
    for n in project.layerTreeRoot().findLayers()
    if n.isVisible() and n.layer().name() != "OSM Standard"
]
if aerial:  # swap the visible aerial photo for another year
    layers = [lyr for lyr in layers if not lyr.name().startswith("Luchtfoto")]
    layers.append(project.mapLayersByName(aerial)[0])
for group in groups:
    name, ids = group.split("=")
    ids = ids.split(",")
    expr = '"location_id" IN (' + ",".join(f"'{i}'" for i in ids) + ")"
    ext = None
    for f in regions.getFeatures(QgsFeatureRequest().setFilterExpression(expr)):
        b = f.geometry().boundingBox()
        ext = b if ext is None else (ext.combineExtentWith(b) or ext)
    if ext is None:
        print(f"{name}: no regions found")
        continue
    ext = ext.buffered(max(ext.width(), ext.height()) * 0.1)
    ms = QgsMapSettings()
    ms.setLayers(layers)
    ms.setDestinationCrs(project.crs())
    ms.setOutputSize(QSize(1400, 1000))
    ms.setExtent(ext)
    ms.setBackgroundColor(QColor("white"))
    job = QgsMapRendererSequentialJob(ms)
    job.start()
    job.waitForFinished()
    job.renderedImage().save(str(out / f"{name}.png"))
    print(out / f"{name}.png")
app.exitQgis()
