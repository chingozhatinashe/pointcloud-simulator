"""Tests for data.preprocessing: storey segmentation, slicing, outlier removal,
downsampling, and the top-level preprocess() orchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.preprocessing import (
    compute_bounds,
    downsample_voxel,
    extract_height_slice,
    preprocess,
    remove_outliers,
    segment_storeys,
)
from simulator.engine import FloorPointCloud, PointCloudSlice


def _dense_band(z_start: float, z_end: float, n: int = 3000, x: float = 0.0, y: float = 0.0):
    z = np.linspace(z_start, z_end, n)
    return np.column_stack([np.full(n, x), np.full(n, y), z])


class TestSegmentStoreys:
    def test_single_storey_returns_all_points(self):
        points = _dense_band(0.0, 3.0)
        result = segment_storeys(points)
        assert len(result) == 1
        assert result[0].shape[0] == points.shape[0]

    def test_two_storeys_with_gap_are_split(self):
        storey_a = _dense_band(0.0, 3.0)
        storey_b = _dense_band(3.5, 6.5)
        points = np.concatenate([storey_a, storey_b])

        result = segment_storeys(points)

        assert len(result) == 2
        assert result[0].shape[0] == storey_a.shape[0]
        assert result[1].shape[0] == storey_b.shape[0]
        assert result[0][:, 2].max() <= 3.25
        assert result[1][:, 2].min() >= 3.25

    def test_three_storeys_are_split(self):
        storeys = [_dense_band(0.0, 3.0), _dense_band(3.5, 6.5), _dense_band(7.0, 10.0)]
        points = np.concatenate(storeys)

        result = segment_storeys(points)

        assert len(result) == 3
        for original, recovered in zip(storeys, result):
            assert recovered.shape[0] == original.shape[0]

    def test_sparse_edge_of_cloud_does_not_create_a_spurious_boundary(self):
        # a handful of stray points at the very top edge shouldn't be split off,
        # since a valley needs occupied bins on *both* sides
        dense = _dense_band(0.0, 3.0)
        stray = np.array([[0.0, 0.0, 3.05], [0.0, 0.0, 3.06]])
        points = np.concatenate([dense, stray])

        result = segment_storeys(points)
        assert len(result) == 1


class TestExtractHeightSlice:
    def test_keeps_points_within_tolerance_and_projects_to_2d(self):
        points = np.array(
            [
                [1.0, 2.0, 1.0],  # within tolerance (exactly at height)
                [3.0, 4.0, 1.04],  # within tolerance
                [5.0, 6.0, 1.05],  # exactly at the tolerance boundary -- excluded
                [7.0, 8.0, 1.5],  # far outside tolerance
            ]
        )
        result = extract_height_slice(points, height=1.0, slice_tolerance=0.05)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_no_points_in_range_returns_empty(self):
        points = np.array([[0.0, 0.0, 5.0]])
        result = extract_height_slice(points, height=1.0, slice_tolerance=0.05)
        assert result.shape == (0, 2)


class TestRemoveOutliers:
    def test_removes_a_distant_outlier(self):
        rng = np.random.default_rng(0)
        cluster = rng.normal(loc=[0.0, 0.0], scale=0.01, size=(100, 2))
        outlier = np.array([[100.0, 100.0]])
        points = np.concatenate([cluster, outlier])

        cleaned = remove_outliers(points, k=10, std_ratio=2.0)

        assert cleaned.shape[0] < points.shape[0]
        assert not np.any(np.all(np.isclose(cleaned, [100.0, 100.0]), axis=1))

    def test_returns_unchanged_when_fewer_points_than_k(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        cleaned = remove_outliers(points, k=10)
        np.testing.assert_allclose(cleaned, points)


class TestDownsampleVoxel:
    def test_collapses_dense_cluster_to_few_points(self):
        rng = np.random.default_rng(0)
        points = rng.uniform(-0.005, 0.005, size=(200, 2))
        downsampled = downsample_voxel(points, voxel_size=0.02)
        assert downsampled.shape[0] < points.shape[0]

    def test_empty_input_returns_empty(self):
        points = np.zeros((0, 2))
        result = downsample_voxel(points, voxel_size=0.02)
        assert result.shape == (0, 2)


class TestComputeBounds:
    def test_matches_known_extent(self):
        points = np.array([[0.0, 1.0, 9.0], [4.0, -2.0, 9.0], [2.0, 3.0, 9.0]])
        bounds = compute_bounds(points)
        np.testing.assert_allclose(np.asarray(bounds), [0.0, -2.0, 4.0, 3.0])


class TestPreprocess:
    def _wall_point_cloud(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        low = np.column_stack(
            [
                rng.uniform(0.0, 4.0, 500),
                np.zeros(500),
                np.full(500, 0.3) + rng.normal(0, 0.001, 500),
            ]
        )
        high = np.column_stack(
            [
                rng.uniform(0.0, 4.0, 500),
                np.zeros(500),
                np.full(500, 1.0) + rng.normal(0, 0.001, 500),
            ]
        )
        return np.concatenate([low, high])

    def test_returns_floor_point_cloud_with_matching_slices(self):
        points = self._wall_point_cloud()
        result = preprocess(points, heights=[0.3, 1.0])

        assert isinstance(result, FloorPointCloud)
        assert len(result.slices) == 2
        assert [sl.height for sl in result.slices] == [0.3, 1.0]
        for sl in result.slices:
            assert isinstance(sl, PointCloudSlice)
            assert sl.points.shape[1] == 2
            assert sl.points.shape[0] > 0

    def test_points_3d_and_bounds_are_consistent(self):
        points = self._wall_point_cloud()
        result = preprocess(points, heights=[0.3, 1.0])

        total_points = sum(sl.points.shape[0] for sl in result.slices)
        assert result.points_3d.shape == (total_points, 3)
        assert result.scanner_positions.shape == (0, 2)
        np.testing.assert_allclose(np.asarray(result.floor_plan_bounds)[:2], [0.0, 0.0], atol=0.05)

    def test_rejects_empty_heights(self):
        with pytest.raises(ValueError):
            preprocess(self._wall_point_cloud(), heights=[])

    def test_rejects_empty_points(self):
        with pytest.raises(ValueError):
            preprocess(np.zeros((0, 3)), heights=[1.0])
