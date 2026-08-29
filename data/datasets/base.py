"""BenchmarkDataset: the shared interface every benchmark dataset wrapper implements.

Per docs/data.md section 5 -- ground truth floor plans are available for some datasets
(or some items within a dataset, e.g. 3DSES's unlabeled test-area scans) but not others,
so `__getitem__` returns `FloorPlanGraph | None` rather than requiring ground truth.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from representation.floor_plan import FloorPlanGraph
from simulator.engine import FloorPointCloud


@runtime_checkable
class BenchmarkDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[FloorPointCloud, FloorPlanGraph | None]: ...

    def split(self, ratio: tuple[float, ...]) -> tuple[Self, ...]: ...
