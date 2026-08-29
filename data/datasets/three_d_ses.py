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

The floor plan is a property of the room, not of any one scan height, so wall points
are line-fitted after projecting the storey's *entire* Z range onto the XY plane --
not from a single height slice. A single slice would make the fitted ground truth
depend on an arbitrary height choice, and would be vulnerable to whatever happens to
occlude that particular band (a window, a door, furniture) even though the wall is
solid at other heights. `self._heights` (default `simulator.engine.DEFAULT_HEIGHTS`)
is a separate concept -- it controls what heights `FloorPointCloud`'s own slices are
taken at, mirroring the Simulator's multi-height LiDAR sweep, and has no bearing on
ground truth construction.

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
from data.preprocessing import preprocess, segment_storeys
from representation.floor_plan import FloorPlanGraph
from simulator.engine import DEFAULT_HEIGHTS, FloorPointCloud

_VARIANT_ARCHIVES: dict[str, tuple[str, str | None]] = {
    "gold": ("3DSES.zip", "Gold"),
    "silver": ("3DSES.zip", "Silver"),
    "bronze": ("3DSES.zip", "Bronze"),
    "test": ("3DSES_test_area.zip", None),
}

_LABELED_VARIANTS = frozenset({"gold", "silver", "bronze"})


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
            # Project every wall-labelled point across the storey's full Z range onto
            # XY, rather than slicing at one height -- the floor plan is a property of
            # the room, not of an arbitrarily chosen scan height (see module docstring).
            wall_points_2d = np.asarray(normalised_xyz[labels == wall_label][:, :2])
            raw_segments = fit_wall_segments(wall_points_2d)
            floor_plan = build_floor_plan_graph(raw_segments)

        return point_cloud, floor_plan

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
