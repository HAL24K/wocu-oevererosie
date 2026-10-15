"""Dutch labels for the strip model's features: short name, meaning, source, group.

One place for the SHAP figure and the feature-introduction slides, so they use
the same words. Groups follow how the model sees a strip: its own history, its
place in the region, and context of the region.
"""

from __future__ import annotations

GROUPS = {
    "geschiedenis": "Geschiedenis van het strookje",
    "positie": "Positie in het scopevlak",
    "context": "Context van het scopevlak",
}

# feature: (short label, meaning, data source, group)
FEATURES: dict[str, tuple[str, str, str, str]] = {
    # the strip's own measurements
    "dist_t2": (
        "Huidige afstand",
        "Afstand van de oever tot de middenlijn bij de laatste meting",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "v_train": (
        "Laatste snelheid",
        "Verplaatsing tussen de twee laatste meetjaren (m/jr)",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "train_span_yr": (
        "Tijd tussen metingen",
        "Jaren tussen de twee laatste meetjaren",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "test_span_yr": (
        "Voorspelhorizon",
        "Hoeveel jaar vooruit voorspeld wordt",
        "Afgeleid",
        "geschiedenis",
    ),
    "n_seg_years": (
        "Aantal meetjaren",
        "Meetjaren van dit strookje",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "n_hist": (
        "Aantal metingen",
        "Alle metingen van het strookje tot nu toe",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "nyears_hist": (
        "Aantal jaren",
        "Jaren met minstens één meting van het strookje",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "span_hist_yr": (
        "Lengte meetreeks",
        "Jaren tussen eerste en laatste meting",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "theil_v": (
        "Trend",
        "Robuuste trend over de hele meetreeks (Theil–Sen, m/jr)",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "theil_resid_std": (
        "Spreiding rond trend",
        "Hoe grillig de metingen rond de trend liggen",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "accel_v": (
        "Versnelling",
        "Trend tweede helft min trend eerste helft",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "last_gap_yr": (
        "Tijd sinds vorige meting",
        "Jaren tussen de laatste twee metingen",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "frac_sam": (
        "Aandeel SAM",
        "Deel van de metingen uit satellietbeelden (SAM)",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "resid_last": (
        "Afwijking laatste meting",
        "Hoe ver de laatste meting van de trend ligt",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "v_recent": (
        "Recente snelheid",
        "Snelheid tussen de laatste twee metingen",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    "slope_recent": (
        "Recente trend",
        "Trend over de laatste drie metingen",
        "Oeverlijnen (hybride)",
        "geschiedenis",
    ),
    # where the strip sits
    "seg_center": (
        "Positie langs de oever",
        "Plaats van het strookje in het scopevlak (0–1)",
        "Afgeleid",
        "positie",
    ),
    "seg_edge": (
        "Afstand tot rand scopevlak",
        "Hoe dicht het strookje bij de rand van het scopevlak ligt",
        "Afgeleid",
        "positie",
    ),
    # context of the region
    "is_nvo": (
        "NVO",
        "Natuurvriendelijke oever met vrije ruimte (VVR)",
        "VVR-vlakken (RWS)",
        "context",
    ),
    "river_enc": (
        "Rivier",
        "Riviertak (Maas, Rijn, Nederrijn, IJssel; nevengeul)",
        "Scope (WOCU)",
        "context",
    ),
    "vegetation_class_enc": (
        "Vegetatie",
        "Dominante vegetatieklasse",
        "Vegetatielegger (RWS)",
        "context",
    ),
    "land_use_enc": (
        "Landgebruik",
        "Dominant landgebruik achter de oever",
        "BRP",
        "context",
    ),
    "soil_group_enc": (
        "Bodem",
        "Bodemgroep (klei, zand-klei, zand, veen)",
        "BRO bodemkaart",
        "context",
    ),
    "bend_exposure_n5": (
        "Bocht (kort)",
        "Binnen- of buitenbocht over ~500 m",
        "Middenlijn rivier",
        "context",
    ),
    "bend_exposure_n8": (
        "Bocht (lang)",
        "Binnen- of buitenbocht over ~800 m",
        "Middenlijn rivier",
        "context",
    ),
    "erosion_vol_rate_t1": (
        "Erosievolume",
        "Volumetrische erosie in de vorige periode",
        "Hoogtemodel (Luke)",
        "context",
    ),
    "n_events_t2": (
        "Hoogwaters",
        "Aantal hoogwatergebeurtenissen in de recente periode",
        "Afvoer RWS",
        "context",
    ),
    "max_rise_rate_t2": (
        "Stijgsnelheid",
        "Snelste stijging van de afvoer, recente periode",
        "Afvoer RWS",
        "context",
    ),
    "drawdown_index_t2": (
        "Terugval na piek",
        "Hoe snel het water na een piek zakt",
        "Afvoer RWS",
        "context",
    ),
    "flood_days_t2": (
        "Hoogwaterdagen",
        "Dagen boven de P90-afvoer, recente periode",
        "Afvoer RWS",
        "context",
    ),
}


def label(feature: str) -> str:
    return FEATURES.get(feature, (feature,))[0]
