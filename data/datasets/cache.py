"""HDF5 caching for `BenchmarkDataset` items.

`ThreeDSESDataset.__getitem__` (and any future benchmark wrapper that derives ground
truth the same way, e.g. S3DIS via `data.datasets.wall_fitting`) rebuilds each
`(FloorPointCloud, FloorPlanGraph | None)` pair from scratch on every call -- re-reading
raw archive bytes, re-running storey segmentation, re-running RANSAC wall fitting. This
module lets that work be paid once and reloaded lazily afterwards.

Same HDF5-backed incremental-write/lazy-read pattern as `data.synthetic`'s
`DatasetGenerator`/`SyntheticDataset`, generalised for the `BenchmarkDataset` protocol
(`data.datasets.base`) rather than tied to synthetic `(G, D)` generation: tuple order is
`(FloorPointCloud, FloorPlanGraph | None)` rather than `(FloorPlanGraph, FloorPointCloud)`,
and ground truth is optional -- some items (e.g. 3DSES's unlabeled `"test"` variant) have
none, tracked via a per-item `floor_plans/has_ground_truth` flag rather than assuming
every item has a graph. `point_clouds/floor_plan_bounds` is stored directly (rather than
recomputed from the floor plan on load, as `SyntheticDataset` does) since it must survive
the round trip even when there is no floor plan to recompute it from.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from data.datasets.base import BenchmarkDataset
from representation.floor_plan import J_MAX, K_MAX, FloorPlanGraph
from simulator.engine import FloorPointCloud, PointCloudSlice

_EMPTY_SEGMENTS = np.zeros((K_MAX, 4), dtype=np.float32)
_EMPTY_SEGMENT_MASK = np.zeros((K_MAX,), dtype=bool)
_EMPTY_JUNCTIONS = np.zeros((J_MAX, 2), dtype=np.float32)
_EMPTY_JUNCTION_MASK = np.zeros((J_MAX,), dtype=bool)


def cache_benchmark_dataset(dataset: BenchmarkDataset, output_path: Path) -> None:
    """Stream every `(FloorPointCloud, FloorPlanGraph | None)` item of `dataset` to HDF5.

    If `output_path` already exists (e.g. from an interrupted previous run), resumes
    from its current sample count rather than re-deriving items already cached -- useful
    for large, slow-to-derive variants (e.g. 3DSES's `bronze` tier).
    """
    output_path = Path(output_path)
    n_samples = len(dataset)

    if output_path.exists():
        with h5py.File(output_path, "r") as f:
            start_index = int(f["metadata/n_samples"][()])
    else:
        start_index = 0

    if start_index >= n_samples:
        return

    if not output_path.exists():
        point_cloud, floor_plan = dataset[0]
        with h5py.File(output_path, "w") as f:
            _create_layout(f, point_cloud)
            _append_item(f, 0, point_cloud, floor_plan)
            f["metadata/n_samples"][()] = 1
        start_index = max(start_index, 1)

    with h5py.File(output_path, "r+") as f:
        for index in range(start_index, n_samples):
            point_cloud, floor_plan = dataset[index]
            _append_item(f, index, point_cloud, floor_plan)
            f["metadata/n_samples"][()] = index + 1


def _create_layout(f: h5py.File, point_cloud: FloorPointCloud) -> None:
    num_heights = len(point_cloud.slices)
    num_scanners = point_cloud.scanner_positions.shape[0]

    metadata = f.create_group("metadata")
    metadata.create_dataset("n_samples", data=0, dtype="i8")
    metadata.create_dataset("num_heights", data=num_heights, dtype="i8")
    metadata.create_dataset("num_scanners", data=num_scanners, dtype="i8")

    floor_plans = f.create_group("floor_plans")
    floor_plans.create_dataset(
        "segments",
        shape=(0, K_MAX, 4),
        maxshape=(None, K_MAX, 4),
        dtype="f4",
        chunks=(1, K_MAX, 4),
    )
    floor_plans.create_dataset(
        "junctions",
        shape=(0, J_MAX, 2),
        maxshape=(None, J_MAX, 2),
        dtype="f4",
        chunks=(1, J_MAX, 2),
    )
    floor_plans.create_dataset(
        "segment_masks",
        shape=(0, K_MAX),
        maxshape=(None, K_MAX),
        dtype="bool",
        chunks=(1, K_MAX),
    )
    floor_plans.create_dataset(
        "junction_masks",
        shape=(0, J_MAX),
        maxshape=(None, J_MAX),
        dtype="bool",
        chunks=(1, J_MAX),
    )
    floor_plans.create_dataset(
        "has_ground_truth", shape=(0,), maxshape=(None,), dtype="bool", chunks=(1,)
    )

    point_clouds = f.create_group("point_clouds")
    vlen_f32 = h5py.vlen_dtype(np.dtype("float32"))
    point_clouds.create_dataset("points_3d", shape=(0,), maxshape=(None,), dtype=vlen_f32)
    point_clouds.create_dataset(
        "slice_heights", shape=(0, num_heights), maxshape=(None, num_heights), dtype="f4"
    )
    point_clouds.create_dataset(
        "scanner_positions",
        shape=(0, num_scanners, 2),
        maxshape=(None, num_scanners, 2),
        dtype="f4",
        # num_scanners is legitimately 0 for datasets that don't carry scanner
        # positions (e.g. ThreeDSESDataset) -- an explicit chunks=(1, 0, 2) is
        # rejected by h5py ("chunk dimensions must be positive"), so let it
        # auto-pick a valid chunk shape instead of specifying one manually.
        chunks=True,
    )
    point_clouds.create_dataset(
        "floor_plan_bounds", shape=(0, 4), maxshape=(None, 4), dtype="f4", chunks=(1, 4)
    )


def _append_item(
    f: h5py.File, index: int, point_cloud: FloorPointCloud, floor_plan: FloorPlanGraph | None
) -> None:
    datasets = [
        f["floor_plans/segments"],
        f["floor_plans/junctions"],
        f["floor_plans/segment_masks"],
        f["floor_plans/junction_masks"],
        f["floor_plans/has_ground_truth"],
        f["point_clouds/points_3d"],
        f["point_clouds/slice_heights"],
        f["point_clouds/scanner_positions"],
        f["point_clouds/floor_plan_bounds"],
    ]
    if datasets[0].shape[0] <= index:
        for dataset in datasets:
            dataset.resize(index + 1, axis=0)

    if floor_plan is not None:
        f["floor_plans/segments"][index] = np.asarray(floor_plan.to_segment_array())
        f["floor_plans/junctions"][index] = np.asarray(floor_plan.to_junction_array())
        f["floor_plans/segment_masks"][index] = np.asarray(floor_plan.segment_mask())
        f["floor_plans/junction_masks"][index] = np.asarray(floor_plan.junction_mask())
    else:
        f["floor_plans/segments"][index] = _EMPTY_SEGMENTS
        f["floor_plans/junctions"][index] = _EMPTY_JUNCTIONS
        f["floor_plans/segment_masks"][index] = _EMPTY_SEGMENT_MASK
        f["floor_plans/junction_masks"][index] = _EMPTY_JUNCTION_MASK
    f["floor_plans/has_ground_truth"][index] = floor_plan is not None

    f["point_clouds/points_3d"][index] = np.asarray(
        point_cloud.points_3d, dtype=np.float32
    ).reshape(-1)
    f["point_clouds/slice_heights"][index] = np.asarray(
        [sl.height for sl in point_cloud.slices], dtype=np.float32
    )
    f["point_clouds/scanner_positions"][index] = np.asarray(
        point_cloud.scanner_positions, dtype=np.float32
    )
    f["point_clouds/floor_plan_bounds"][index] = np.asarray(
        point_cloud.floor_plan_bounds, dtype=np.float32
    )


class CachedBenchmarkDataset:
    """Lazily-loading, index-splittable `BenchmarkDataset` backed by a file written by
    `cache_benchmark_dataset`.
    """

    def __init__(self, path: Path, indices: Sequence[int] | None = None) -> None:
        self._path = Path(path)
        if indices is None:
            with h5py.File(self._path, "r") as f:
                indices = range(int(f["metadata/n_samples"][()]))
        self._indices = list(indices)

    def split(self, ratio: Sequence[float]) -> tuple[CachedBenchmarkDataset, ...]:
        """Split into `len(ratio)` datasets over disjoint index ranges, e.g. train/val/test."""
        if not np.isclose(sum(ratio), 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got {ratio}")

        n = len(self._indices)
        boundaries = np.round(np.cumsum(ratio) * n).astype(int)
        splits = []
        start = 0
        for boundary in boundaries:
            splits.append(CachedBenchmarkDataset(self._path, self._indices[start:boundary]))
            start = boundary
        return tuple(splits)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[FloorPointCloud, FloorPlanGraph | None]:
        real_index = self._indices[idx]
        with h5py.File(self._path, "r") as f:
            segments = np.asarray(f["floor_plans/segments"][real_index])
            segment_mask = np.asarray(f["floor_plans/segment_masks"][real_index])
            junctions = np.asarray(f["floor_plans/junctions"][real_index])
            junction_mask = np.asarray(f["floor_plans/junction_masks"][real_index])
            has_ground_truth = bool(f["floor_plans/has_ground_truth"][real_index])
            points_3d_flat = np.asarray(f["point_clouds/points_3d"][real_index])
            slice_heights = np.asarray(f["point_clouds/slice_heights"][real_index])
            scanner_positions = np.asarray(f["point_clouds/scanner_positions"][real_index])
            floor_plan_bounds = np.asarray(f["point_clouds/floor_plan_bounds"][real_index])

        floor_plan = (
            FloorPlanGraph.from_arrays(segments, segment_mask, junctions, junction_mask)
            if has_ground_truth
            else None
        )

        points_3d = points_3d_flat.reshape(-1, 3)
        slices = []
        for height in slice_heights:
            mask = np.isclose(points_3d[:, 2], height)
            slices.append(
                PointCloudSlice(height=float(height), points=jnp.asarray(points_3d[mask, :2]))
            )

        point_cloud = FloorPointCloud(
            slices=slices,
            points_3d=jnp.asarray(points_3d),
            scanner_positions=jnp.asarray(scanner_positions),
            floor_plan_bounds=jnp.asarray(floor_plan_bounds),
        )
        return point_cloud, floor_plan

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
