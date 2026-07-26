"""Tests for data.synthetic: HDF5-backed DatasetGenerator and SyntheticDataset."""

from __future__ import annotations

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from data.synthetic import DatasetGenerator, SyntheticDataset
from representation.floor_plan import FloorPlanGraph
from simulator.engine import FloorPointCloud, PointCloudSlice


def _square_room(size: float) -> FloorPlanGraph:
    graph = FloorPlanGraph()
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    junctions = [graph.add_junction(corner) for corner in corners]
    for start, end in zip(junctions, junctions[1:] + junctions[:1]):
        graph.add_segment(start.id, end.id)
    return graph


def _make_pair(seed: int, num_scanners: int = 1) -> tuple[FloorPlanGraph, FloorPointCloud]:
    """A small synthetic (FloorPlanGraph, FloorPointCloud) pair, varying by `seed`."""
    size = 2.0 + seed * 0.5
    floor_plan = _square_room(size)

    rng = np.random.default_rng(seed)
    heights = [0.3, 1.0]
    slices = []
    for height in heights:
        n = 5 + seed
        points = rng.uniform(0.0, size, size=(n, 2))
        slices.append(PointCloudSlice(height=height, points=jnp.asarray(points, dtype=jnp.float32)))

    points_3d = jnp.concatenate(
        [
            jnp.column_stack(
                [sl.points, jnp.full((sl.points.shape[0],), sl.height, dtype=jnp.float32)]
            )
            for sl in slices
        ]
    )
    scanner_positions = jnp.asarray(
        rng.uniform(0.0, size, size=(num_scanners, 2)), dtype=jnp.float32
    )
    point_cloud = FloorPointCloud(
        slices=slices,
        points_3d=points_3d,
        scanner_positions=scanner_positions,
        floor_plan_bounds=jnp.array([0.0, 0.0, size, size]),
    )
    return floor_plan, point_cloud


class TestDatasetGenerator:
    def test_generates_file_with_correct_metadata_and_shapes(self, tmp_path):
        output_path = tmp_path / "dataset.h5"
        samples = (_make_pair(i) for i in range(3))

        DatasetGenerator().generate(samples, n_samples=3, output_path=output_path)

        with h5py.File(output_path, "r") as f:
            assert int(f["metadata/n_samples"][()]) == 3
            assert f["floor_plans/segments"].shape[0] == 3
            assert f["floor_plans/junctions"].shape[0] == 3
            assert f["point_clouds/points_3d"].shape[0] == 3
            assert f["point_clouds/slice_heights"].shape == (3, 2)
            assert f["point_clouds/scanner_positions"].shape == (3, 1, 2)

    def test_no_op_when_n_samples_already_reached(self, tmp_path):
        output_path = tmp_path / "dataset.h5"
        DatasetGenerator().generate((_make_pair(i) for i in range(2)), 2, output_path)

        # a generator that would raise if actually consumed
        def _poison():
            raise AssertionError("should not be consumed")
            yield  # pragma: no cover

        DatasetGenerator().generate(_poison(), n_samples=2, output_path=output_path)

        with h5py.File(output_path, "r") as f:
            assert int(f["metadata/n_samples"][()]) == 2

    def test_resumes_from_existing_sample_count(self, tmp_path):
        output_path = tmp_path / "dataset.h5"

        DatasetGenerator().generate(
            (_make_pair(i) for i in range(2)), n_samples=2, output_path=output_path
        )

        # a fresh, deterministic generator reproducing the same sequence by index;
        # the first 2 items are skipped since they're already stored
        DatasetGenerator().generate(
            (_make_pair(i) for i in range(5)), n_samples=5, output_path=output_path
        )

        dataset = SyntheticDataset.from_file(output_path)
        assert len(dataset) == 5
        for i in range(5):
            floor_plan, _ = dataset[i]
            expected_size = 2.0 + i * 0.5
            positions = np.array([j.position for j in floor_plan.junctions.values()])
            np.testing.assert_allclose(positions.max(), expected_size, atol=1e-4)


class TestSyntheticDataset:
    def test_round_trips_floor_plan_topology_and_point_cloud(self, tmp_path):
        output_path = tmp_path / "dataset.h5"
        original_floor_plan, original_point_cloud = _make_pair(0)
        DatasetGenerator().generate(
            iter([(original_floor_plan, original_point_cloud)]), 1, output_path
        )

        dataset = SyntheticDataset.from_file(output_path)
        assert len(dataset) == 1
        floor_plan, point_cloud = dataset[0]

        assert len(floor_plan.junctions) == len(original_floor_plan.junctions)
        assert len(floor_plan.segments) == len(original_floor_plan.segments)

        # every original segment's endpoint pair should reappear as a segment in the
        # reconstructed graph (junction ids may have been renumbered)
        def _edge_positions(graph: FloorPlanGraph) -> set:
            return {
                frozenset(
                    (
                        tuple(
                            np.round(np.asarray(graph.junctions[seg.junction_start].position), 4)
                        ),
                        tuple(np.round(np.asarray(graph.junctions[seg.junction_end].position), 4)),
                    )
                )
                for seg in graph.segments.values()
            }

        original_edge_positions = _edge_positions(original_floor_plan)
        reconstructed_edge_positions = _edge_positions(floor_plan)
        assert original_edge_positions == reconstructed_edge_positions

        assert len(point_cloud.slices) == len(original_point_cloud.slices)
        for original_slice, recovered_slice in zip(original_point_cloud.slices, point_cloud.slices):
            assert recovered_slice.height == pytest.approx(original_slice.height, abs=1e-6)
            assert recovered_slice.points.shape == original_slice.points.shape
        np.testing.assert_allclose(
            np.asarray(point_cloud.scanner_positions),
            np.asarray(original_point_cloud.scanner_positions),
            atol=1e-5,
        )

    def test_len_and_iteration(self, tmp_path):
        output_path = tmp_path / "dataset.h5"
        DatasetGenerator().generate((_make_pair(i) for i in range(4)), 4, output_path)

        dataset = SyntheticDataset.from_file(output_path)
        assert len(dataset) == 4
        assert len(list(dataset)) == 4

    def test_split_produces_disjoint_ranges_matching_ratios(self, tmp_path):
        output_path = tmp_path / "dataset.h5"
        DatasetGenerator().generate((_make_pair(i) for i in range(10)), 10, output_path)

        dataset = SyntheticDataset.from_file(output_path)
        train, val, test = dataset.split((0.6, 0.2, 0.2))

        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2
        assert len(train) + len(val) + len(test) == len(dataset)

    def test_split_rejects_ratios_not_summing_to_one(self, tmp_path):
        output_path = tmp_path / "dataset.h5"
        DatasetGenerator().generate((_make_pair(i) for i in range(3)), 3, output_path)

        dataset = SyntheticDataset.from_file(output_path)
        with pytest.raises(ValueError):
            dataset.split((0.5, 0.6))
