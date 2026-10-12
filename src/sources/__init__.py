"""Scope geometry and bank-position observations.

    ScopeGeometry      scope regions and their centrelines
    BankObservations   location_id, date, dist_m, n_candidates, n_selected, source
                       → .to_dist_per_year(), the shape build_region_split reads

Named `sources` rather than `io` so it cannot shadow the standard library.
"""

from src.sources.geometry import ScopeGeometry
from src.sources.observations import OBSERVATION_COLUMNS, BankObservations

__all__ = ["OBSERVATION_COLUMNS", "BankObservations", "ScopeGeometry"]
