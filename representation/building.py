"""Building and Storey: lightweight containers organising floor plans into a building.

The geometric content lives in `FloorPlanGraph`; these are ordering/bookkeeping wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from matplotlib.figure import Figure

from representation.floor_plan import FloorPlanGraph
from representation.visualisation import plot_building


@dataclass
class Storey:
    """A single building level: a floor plan and its height range, in metres."""

    z_min: float
    z_max: float
    floor_plan: FloorPlanGraph = field(default_factory=FloorPlanGraph)

    def __post_init__(self) -> None:
        if self.z_max <= self.z_min:
            raise ValueError(f"z_max ({self.z_max}) must exceed z_min ({self.z_min})")


class Building:
    """An ordered stack of storeys."""

    def __init__(self) -> None:
        self.storeys: list[Storey] = []

    def add_storey(self, z_min: float, z_max: float) -> Storey:
        storey = Storey(z_min=z_min, z_max=z_max)
        self.storeys.append(storey)
        return storey

    def plot(self, **kwargs) -> Figure:
        return plot_building(self.storeys, **kwargs)
