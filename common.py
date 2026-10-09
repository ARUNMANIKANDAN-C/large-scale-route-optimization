"""Shared definitions: truck types, parameters, small helpers.

All times are in HOURS measured from t0 (the earliest Available_Time in the data).
All distances are in km, weights in kg, areas in m^2.
"""
from dataclasses import dataclass, field
from typing import Optional, Set, FrozenSet

HAZARD_CLASSES = ["type_1", "type_2", "non_danger"]


@dataclass
class TruckType:
    name: str
    area: float            # inner loading area (m^2)         -> A_t
    weight_cap: float      # weight capacity (kg)              -> W_t
    cost_km: float         # cost per km                       -> c_t
    speed: float = 40.0    # km/h                              -> s
    count: Optional[int] = None   # K_t : number of trucks available (None = unlimited)
    fixed_cost: float = 0.0       # optional cost per truck used


# Returns the default list of truck types and their specifications based on the Kaggle dataset.
def default_fleet():
    # Values from the Kaggle dataset description (inner size = length x width)
    return [
        TruckType("16.5 m", 16.1 * 2.5, 10000, 3, 40.0, None),
        TruckType("12.5 m", 12.1 * 2.5, 5000, 2, 40.0, None),
        TruckType("9.6 m", 9.1 * 2.3, 2000, 1, 40.0, None),
    ]


# Returns default hazardous materials incompatibility pairs where non-danger cannot mix with hazardous.
def default_incompatible():
    """Hazard rule agreed in the project: type_1 and type_2 may share a truck,
    non_danger may never share with type_1 / type_2."""
    return {frozenset(("non_danger", "type_1")), frozenset(("non_danger", "type_2"))}


@dataclass
class Params:
    max_stops: int = 3             # N
    unload_h: float = 1.0          # M  (hours spent unloading at each stop)
    stop_cost: float = 500.0       # f  (fixed cost per stop)
    delta_h: float = 4.0           # Delta (max availability gap inside one truck, hours)
    return_to_depot: bool = False  # one-way routes by default (as in the Kaggle sample)
    allow_multi_stop: bool = True  # heuristic: try to merge trucks of nearby cities
    incompatible: Set[FrozenSet[str]] = field(default_factory=default_incompatible)

    # Checks whether two hazard classes are compatible to share the same truck.
    def compatible(self, a: str, b: str) -> bool:
        return a == b or frozenset((a, b)) not in self.incompatible


# Filters and returns truck types that currently have available inventory.
def available_types(fleet):
    """Truck types that may be used (count is None = unlimited, or count > 0)."""
    return [t for t in fleet if t.count is None or t.count > 0]

