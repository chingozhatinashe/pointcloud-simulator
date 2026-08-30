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
solid at other heights. `self._height_fractions` (default `DEFAULT_HEIGHT_FRACTIONS`)
is a separate concept -- it controls what heights `FloorPointCloud`'s own slices are
taken at, mirroring the Simulator's multi-height LiDAR sweep, and has no bearing on
ground truth construction. Each fraction is scaled by the storey's own floor-to-ceiling
extent rather than being an absolute height, and each `(storey, fraction)` pair is its
own dataset item -- `len(self) == len(self._items) * len(height_fractions)` -- so every
height-fraction item of a storey must share the *same* `FloorPlanGraph` object rather
than a freshly rebuilt (RANSAC is not deterministic-by-construction) equal one; see
`_load_storey`.

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
from simulator.engine import FloorPointCloud

_VARIANT_ARCHIVES: dict[str, tuple[str, str | None]] = {
    "gold": ("3DSES.zip", "Gold"),
    "silver": ("3DSES.zip", "Silver"),
    "bronze": ("3DSES.zip", "Bronze"),
    "test": ("3DSES_test_area.zip", None),
}

_LABELED_VARIANTS = frozenset({"gold", "silver", "bronze"})

# Fractions of each storey's own floor-to-ceiling extent (post-normalisation Z range,
# floor at 0) rather than absolute metres -- see the module docstring. Roughly mirrors
# simulator.engine.DEFAULT_HEIGHTS for a typical ~2.5m storey (0.3m/1.0m/1.8m ->
# ~0.12/0.4/0.72).
DEFAULT_HEIGHT_FRACTIONS: tuple[float, ...] = (0.1, 0.4, 0.7)


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

    `height_fractions` -- see the module docstring -- makes each `(storey, fraction)`
    pair its own dataset item. `_load_storey` caches the most recently loaded storey's
    normalised points, height, and ground-truth `FloorPlanGraph` (a size-1 cache, not a
    full dataset cache, to keep memory bounded to one storey's raw points at a time,
    matching this loader's streamed on-demand design above) so that the
    `len(height_fractions)` consecutive `__getitem__` calls for the same storey --
    indices are storey-major, so this is the normal access pattern -- share the same
    `FloorPlanGraph` object and skip re-reading/re-segmenting/re-fitting.
    """

    def __init__(
        self,
        variant: str,
        root: Path,
        *,
        height_fractions: Sequence[float] = DEFAULT_HEIGHT_FRACTIONS,
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
        self._height_fractions = height_fractions

        self._items = list(_items) if _items is not None else self._build_index()

        self._cache_item_index: int | None = None
        self._cache_normalised_xyz: np.ndarray | None = None
        self._cache_storey_height: float | None = None
        self._cache_floor_plan: FloorPlanGraph | None = None

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
                    height_fractions=self._height_fractions,
                    _items=self._items[start:boundary],
                )
            )
            start = boundary
        return tuple(splits)

    def __len__(self) -> int:
        return len(self._items) * len(self._height_fractions)

    def _load_storey(self, item_index: int) -> tuple[np.ndarray, float, FloorPlanGraph | None]:
        """Normalised points, floor-to-ceiling height, and ground truth for one storey.

        See the class docstring -- this is a size-1 cache keyed by `item_index`, not a
        full dataset cache.
        """
        if self._cache_item_index == item_index:
            return self._cache_normalised_xyz, self._cache_storey_height, self._cache_floor_plan

        item = self._items[item_index]
        raw = _read_member(self._zip_path, item.member_name)
        storeys = segment_storeys(np.asarray(raw, dtype=np.float64))
        storey_points = storeys[item.storey_index]

        xyz = storey_points[:, :3]
        normalised_xyz, _transform = normalise(xyz)
        # z_min of xyz maps to 0 under normalise() (translate_to_origin is per-axis),
        # so the storey's own floor-to-ceiling extent is just z_max.
        storey_height = float(normalised_xyz[:, 2].max())

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

        self._cache_item_index = item_index
        self._cache_normalised_xyz = normalised_xyz
        self._cache_storey_height = storey_height
        self._cache_floor_plan = floor_plan
        return normalised_xyz, storey_height, floor_plan

    def __getitem__(self, idx: int) -> tuple[FloorPointCloud, FloorPlanGraph | None]:
        n_fractions = len(self._height_fractions)
        item_index, fraction_index = divmod(idx, n_fractions)

        normalised_xyz, storey_height, floor_plan = self._load_storey(item_index)
        height = self._height_fractions[fraction_index] * storey_height
        point_cloud = preprocess(normalised_xyz, heights=[height])

        return point_cloud, floor_plan

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
