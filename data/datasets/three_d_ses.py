"""3DSES benchmark dataset wrapper, per docs/data.md section 5.

3DSES ships colorized TLS point clouds of a real building as `.npy` arrays (columns
`X, Y, Z, R, G, B, intensity[, semantic_label]`) inside two zip archives:
`3DSES.zip`, containing nested `Bronze`/`Silver`/`Gold` training-difficulty tiers
(Bronze superset-of-Silver superset-of-Gold, with progressively coarser/noisier
labels), and `3DSES_test_area.zip`, containing 3 unlabeled scans held out for
evaluation. Both are large (7.3 GB / 220 MB) and are never fully extracted here --
individual `.npy` members are streamed out of the zip on demand.

Ground truth here comes from RANSAC line-fitting on wall-labelled points
(`data.datasets.wall_fitting`), not from the dataset's shipped CAD/BIM model
(`3DSES_cad_model.obj`). That model uses a different, undocumented coordinate frame
from the released point clouds (confirmed empirically -- no registration transform is
published anywhere, and the dataset's own reference dataloader does not apply one
either), so using it as ground truth would require solving an unverified registration
problem first. See the project plan for this module for the full investigation --
CAD-based ground truth is a deliberately deferred follow-up, not an oversight.

A `.npy` "scan" is one scanner acquisition, not necessarily one storey -- some scans
(observed directly on the locally-downloaded test-area files) span multiple floors,
most likely where a stairwell or atrium is visible from the scan position. Each scan
is therefore run through the existing `segment_storeys` before use, and each detected
storey becomes its own dataset item.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.datasets.wall_fitting import build_floor_plan_graph, fit_wall_segments, infer_wall_label
from data.normalisation import normalise
from data.preprocessing import extract_height_slice, preprocess, segment_storeys
from representation.floor_plan import FloorPlanGraph
from simulator.engine import DEFAULT_HEIGHTS, FloorPointCloud

_VARIANT_ARCHIVES: dict[str, tuple[str, str | None]] = {
    "gold": ("3DSES.zip", "Gold"),
    "silver": ("3DSES.zip", "Silver"),
    "bronze": ("3DSES.zip", "Bronze"),
    "test": ("3DSES_test_area.zip", None),
}

_LABELED_VARIANTS = frozenset({"gold", "silver", "bronze"})

DEFAULT_WALL_SLICE_HEIGHT = 1.0  # metres above the storey floor
DEFAULT_WALL_SLICE_TOLERANCE = 0.1  # metres


@dataclass(frozen=True)
class _ScanRef:
    """One storey-segment within one `.npy` archive member."""

    member_name: str
    storey_index: int


def _read_member(zip_path: Path, member_name: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive, archive.open(member_name) as fh:
        return np.load(io.BytesIO(fh.read()))


class ThreeDSESDataset:
    """`BenchmarkDataset` wrapper for the 3DSES indoor LiDAR dataset.

    `variant` selects which archive/tier to draw scans from: `"gold"`, `"silver"`,
    `"bronze"` (all inside `3DSES.zip`, semantic-labelled), or `"test"` (inside
    `3DSES_test_area.zip`, unlabelled -- `__getitem__` returns `None` ground truth
    for these). `root` is the local directory containing both zip archives
    (`data/3DSES/` in this repo).

    Building the index (`_build_index`) reads every member's Z column once to detect
    storey boundaries -- for `bronze` (42 large scans) this can take a while, since
    zip members are compressed and can't be memory-mapped; there is no eager point
    loading beyond that.
    """

    def __init__(
        self,
        variant: str,
        root: Path,
        *,
        heights: Sequence[float] = DEFAULT_HEIGHTS,
        wall_slice_height: float = DEFAULT_WALL_SLICE_HEIGHT,
        wall_slice_tolerance: float = DEFAULT_WALL_SLICE_TOLERANCE,
        _items: Sequence[_ScanRef] | None = None,
    ) -> None:
        if variant not in _VARIANT_ARCHIVES:
            raise ValueError(
                f"Unknown 3DSES variant {variant!r}; expected one of {sorted(_VARIANT_ARCHIVES)}"
            )
        self._variant = variant
        self._root = Path(root)
        archive_name, folder = _VARIANT_ARCHIVES[variant]
        self._zip_path = self._root / archive_name
        self._folder = folder
        self._heights = heights
        self._wall_slice_height = wall_slice_height
        self._wall_slice_tolerance = wall_slice_tolerance

        self._items = list(_items) if _items is not None else self._build_index()

    def _member_names(self) -> list[str]:
        with zipfile.ZipFile(self._zip_path) as archive:
            names = archive.namelist()
        prefix = f"{self._folder}/" if self._folder else ""
        return sorted(
            name
            for name in names
            if name.startswith(prefix) and name.endswith(".npy") and name != prefix
        )

    def _build_index(self) -> list[_ScanRef]:
        items: list[_ScanRef] = []
        for member_name in self._member_names():
            raw = _read_member(self._zip_path, member_name)
            storeys = segment_storeys(np.asarray(raw, dtype=np.float64))
            items.extend(
                _ScanRef(member_name=member_name, storey_index=i) for i in range(len(storeys))
            )
        return items

    def split(self, ratio: Sequence[float]) -> tuple[ThreeDSESDataset, ...]:
        """Split into `len(ratio)` datasets over disjoint index ranges, e.g. train/val/test."""
        if not np.isclose(sum(ratio), 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got {ratio}")

        n = len(self._items)
        boundaries = np.round(np.cumsum(ratio) * n).astype(int)
        splits = []
        start = 0
        for boundary in boundaries:
            splits.append(
                ThreeDSESDataset(
                    self._variant,
                    self._root,
                    heights=self._heights,
                    wall_slice_height=self._wall_slice_height,
                    wall_slice_tolerance=self._wall_slice_tolerance,
                    _items=self._items[start:boundary],
                )
            )
            start = boundary
        return tuple(splits)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[FloorPointCloud, FloorPlanGraph | None]:
        item = self._items[idx]
        raw = _read_member(self._zip_path, item.member_name)
        storeys = segment_storeys(np.asarray(raw, dtype=np.float64))
        storey_points = storeys[item.storey_index]

        xyz = storey_points[:, :3]
        normalised_xyz, _transform = normalise(xyz)
        point_cloud = preprocess(normalised_xyz, heights=self._heights)

        floor_plan: FloorPlanGraph | None = None
        if self._variant in _LABELED_VARIANTS:
            labels = storey_points[:, -1]
            wall_label = infer_wall_label(normalised_xyz, labels)
            wall_points_3d = normalised_xyz[labels == wall_label]
            wall_points_2d = extract_height_slice(
                wall_points_3d, self._wall_slice_height, self._wall_slice_tolerance
            )
            raw_segments = fit_wall_segments(wall_points_2d)
            floor_plan = build_floor_plan_graph(raw_segments)

        return point_cloud, floor_plan

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
