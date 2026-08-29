"""DatasetGenerator and SyntheticDataset: HDF5-backed synthetic (G, D) pair management.

Per docs/data.md section 2, but with generation logic kept out of this module: the
Data component has no dependency on the Simulator or Inference components (per
docs/data.md's own Overview), so `DatasetGenerator` does not sample floor-plan/noise
priors or call `simulator.engine.simulate()` itself. It assumes a `FloorPlanGraph` and
`FloorPointCloud` have already been produced -- by a caller that combines
`inference.prior` and `simulator.engine` (e.g. an experiments/ script) -- and takes an
iterable of `(FloorPlanGraph, FloorPointCloud)` pairs, streaming each one to HDF5 as
it arrives. This keeps the module's only job as storage/retrieval machinery.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from representation.floor_plan import J_MAX, K_MAX, FloorPlanGraph
from simulator.engine import FloorPointCloud, PointCloudSlice


class DatasetGenerator:
    """Streams `(FloorPlanGraph, FloorPointCloud)` pairs to an HDF5 file.

    Assumes each pair has already been produced elsewhere (e.g. by sampling a
    floor-plan prior and calling the Simulator) -- this class's only job is
    incremental storage, never accumulating the full dataset in memory, and
    resuming an interrupted run from wherever it left off.
    """

    def generate(
        self,
        samples: Iterable[tuple[FloorPlanGraph, FloorPointCloud]],
        n_samples: int,
        output_path: Path,
    ) -> None:
        """Consume up to `n_samples` pairs from `samples`, streaming each to disk.

        If `output_path` already exists (e.g. from an interrupted previous run),
        resumes from its current sample count: the first
        `existing_count` items yielded by `samples` are skipped, and generation
        continues until `n_samples` total are stored.
        """
        output_path = Path(output_path)

        if output_path.exists():
            with h5py.File(output_path, "r") as f:
                start_index = int(f["metadata/n_samples"][()])
        else:
            start_index = 0

        if start_index >= n_samples:
            return

        samples_iter = iter(samples)
        for _ in range(start_index):
            next(samples_iter, None)

        if not output_path.exists():
            floor_plan, point_cloud = next(samples_iter)
            with h5py.File(output_path, "w") as f:
                self._create_layout(f, floor_plan, point_cloud)
                self._append_sample(f, 0, floor_plan, point_cloud)
                f["metadata/n_samples"][()] = 1
            start_index = 1

        with h5py.File(output_path, "r+") as f:
            for index in range(start_index, n_samples):
                floor_plan, point_cloud = next(samples_iter)
                self._append_sample(f, index, floor_plan, point_cloud)
                f["metadata/n_samples"][()] = index + 1

    @staticmethod
    def _create_layout(
        f: h5py.File, floor_plan: FloorPlanGraph, point_cloud: FloorPointCloud
    ) -> None:
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
            chunks=(1, num_scanners, 2),
        )

    @staticmethod
    def _append_sample(
        f: h5py.File, index: int, floor_plan: FloorPlanGraph, point_cloud: FloorPointCloud
    ) -> None:
        datasets = [
            f["floor_plans/segments"],
            f["floor_plans/junctions"],
            f["floor_plans/segment_masks"],
            f["floor_plans/junction_masks"],
            f["point_clouds/points_3d"],
            f["point_clouds/slice_heights"],
            f["point_clouds/scanner_positions"],
        ]
        if datasets[0].shape[0] <= index:
            for dataset in datasets:
                dataset.resize(index + 1, axis=0)

        f["floor_plans/segments"][index] = np.asarray(floor_plan.to_segment_array())
        f["floor_plans/junctions"][index] = np.asarray(floor_plan.to_junction_array())
        f["floor_plans/segment_masks"][index] = np.asarray(floor_plan.segment_mask())
        f["floor_plans/junction_masks"][index] = np.asarray(floor_plan.junction_mask())
        f["point_clouds/points_3d"][index] = np.asarray(
            point_cloud.points_3d, dtype=np.float32
        ).reshape(-1)
        f["point_clouds/slice_heights"][index] = np.asarray(
            [sl.height for sl in point_cloud.slices], dtype=np.float32
        )
        f["point_clouds/scanner_positions"][index] = np.asarray(
            point_cloud.scanner_positions, dtype=np.float32
        )


class SyntheticDataset:
    """Lazily-loading, index-splittable wrapper around a generated HDF5 dataset."""

    def __init__(self, path: Path, indices: Sequence[int] | None = None) -> None:
        self._path = Path(path)
        if indices is None:
            with h5py.File(self._path, "r") as f:
                indices = range(int(f["metadata/n_samples"][()]))
        self._indices = list(indices)

    @classmethod
    def from_file(cls, path: Path) -> SyntheticDataset:
        return cls(path)

    def split(self, ratio: Sequence[float]) -> tuple[SyntheticDataset, ...]:
        """Split into `len(ratio)` datasets over disjoint index ranges, e.g. train/val/test."""
        if not np.isclose(sum(ratio), 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got {ratio}")

        n = len(self._indices)
        boundaries = np.round(np.cumsum(ratio) * n).astype(int)
        splits = []
        start = 0
        for boundary in boundaries:
            splits.append(SyntheticDataset(self._path, self._indices[start:boundary]))
            start = boundary
        return tuple(splits)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[FloorPlanGraph, FloorPointCloud]:
        real_index = self._indices[idx]
        with h5py.File(self._path, "r") as f:
            segments = np.asarray(f["floor_plans/segments"][real_index])
            segment_mask = np.asarray(f["floor_plans/segment_masks"][real_index])
            junctions = np.asarray(f["floor_plans/junctions"][real_index])
            junction_mask = np.asarray(f["floor_plans/junction_masks"][real_index])
            points_3d_flat = np.asarray(f["point_clouds/points_3d"][real_index])
            slice_heights = np.asarray(f["point_clouds/slice_heights"][real_index])
            scanner_positions = np.asarray(f["point_clouds/scanner_positions"][real_index])

        floor_plan = FloorPlanGraph.from_arrays(segments, segment_mask, junctions, junction_mask)
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
            floor_plan_bounds=floor_plan.bounds(),
        )
        return floor_plan, point_cloud

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
