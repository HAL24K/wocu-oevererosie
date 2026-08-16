"""Phase 1 code. Not used by the current pipeline — kept for reference only.

This is the ``DataCollector → WFSDataBundler → DataHandler → BaselineErosionModel``
stack built in late 2025. It fetches WFS layers per prediction region at run time
and generates features inside ``DataHandler``. It is what ``README.md`` and
``ARCHITECTURE.md`` described for a long time, and what the 13 open items in the
old root ``TODO.md`` were about.

**Nothing here is on the path that produces predictions.** The live pipeline is:

    src.sources     read a delivery into bank observations
    src.pipeline    bank_distances → region_split → feature_engineering → train
    src.model       export_utils, feature_shifter, predictor
    src.erosion     centerline_utils, export, plot_utils

The two do not share code beyond ``src.constants`` and ``src.paths``; no module
outside this package imports anything inside it.

Why it is kept rather than deleted:

* ``DataCollector`` / ``WFSDataBundler`` are the only working WFS clients in the
  repository. If contextual features ever need refreshing from PDOK/RWS, this is
  where that logic lives.
* Its tests (``tests/legacy/``) still pass, apart from three that depend on live
  WFS responses which have since drifted.

Delete it once the WFS layers are confirmed frozen, or once the feature set is
rebuilt on pre-bundled GeoPackages for good.
"""
