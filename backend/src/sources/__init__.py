"""Input adapters for bank-position observations.

The pipeline downstream of step 01 only ever needs one thing: a distance from
the centreline per scope region per point in time. Everything else about the
delivery format — points vs lines, integer years vs real dates, which detection
model produced it — is an input concern and is resolved here.

    HeightModelPointSource   punten_oever point cloud   (phase 1 / phase 2 raw)
    HybridLineSource         hybrid `lines` layer        (phase 2 hybrid)
            │
            ▼
    BankObservations         location_id, date, dist_m, n_candidates,
                             n_selected, source
            │
            ▼
    .to_dist_per_year()      the legacy shape consumed unchanged by
                             src.pipeline.region_split.build_region_split

Named `sources` rather than `io` so it cannot shadow the standard library.
"""

from src.sources.geometry import ScopeGeometry
from src.sources.observations import (
    OBSERVATION_COLUMNS,
    BankObservations,
    BankObservationSource,
    HeightModelPointSource,
    HybridLineSource,
)

__all__ = [
    "OBSERVATION_COLUMNS",
    "BankObservations",
    "BankObservationSource",
    "HeightModelPointSource",
    "HybridLineSource",
    "ScopeGeometry",
]
