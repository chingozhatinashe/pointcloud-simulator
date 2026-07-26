"""Tests for simulator.engine: the top-level simulate() forward model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from representation.floor_plan import FloorPlanGraph
from simulator.engine import (
    DEFAULT_HEIGHTS,
    DEFAULT_SLICE_TOL,
    DEFAULT_VOXEL_SIZE,
    FloorPointCloud,
    PointCloudSlice,
    simulate,
    simulate_segments,
)
from simulator.noise_model import HeightDependentNoiseModel, NoiseModel
from simulator.scanner import ScannerConfig, generate_rays_multi

ZERO_NOISE = NoiseModel(
    range_sigma=0.0, angular_sigma=0.0, outlier_rate=0.0, missing_rate=0.0, height_sigma=0.0
)


def _square_room(size: float = 2.0) -> FloorPlanGraph:
    graph = FloorPlanGraph()
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    junctions = [graph.add_junction(corner) for corner in corners]
    for start, end in zip(junctions, junctions[1:] + junctions[:1]):
        graph.add_segment(start.id, end.id)
    return graph


class TestConstants:
    def test_default_heights(self):
        assert tuple(DEFAULT_HEIGHTS) == (0.3, 1.0, 1.8)

    def test_default_voxel_size_and_slice_tol(self):
        assert DEFAULT_VOXEL_SIZE == 0.02
        assert DEFAULT_SLICE_TOL == 0.05


class TestSimulateValidation:
    def test_rejects_empty_floor_plan(self):
        with pytest.raises(ValueError):
            simulate(
                FloorPlanGraph(),
                jnp.array([[1.0, 1.0]]),
                [1.0],
                ZERO_NOISE,
                jax.random.PRNGKey(0),
            )

    def test_rejects_empty_heights(self):
        with pytest.raises(ValueError):
            simulate(_square_room(), jnp.array([[1.0, 1.0]]), [], ZERO_NOISE, jax.random.PRNGKey(0))


class TestSimulateHardOcclusion:
    def test_single_height_single_scanner_hits_room_boundary(self):
        room = _square_room(size=2.0)
        result = simulate(
            room,
            jnp.array([[1.0, 1.0]]),
            [1.0],
            ZERO_NOISE,
            jax.random.PRNGKey(0),
            num_rays=64,
        )

        assert isinstance(result, FloorPointCloud)
        assert len(result.slices) == 1
        sl = result.slices[0]
        assert isinstance(sl, PointCloudSlice)
        assert sl.height == 1.0
        assert sl.points.shape == (64, 2)

        points = np.asarray(sl.points)
        on_boundary = (
            np.isclose(points[:, 0], 0.0, atol=1e-3)
            | np.isclose(points[:, 0], 2.0, atol=1e-3)
            | np.isclose(points[:, 1], 0.0, atol=1e-3)
            | np.isclose(points[:, 1], 2.0, atol=1e-3)
        )
        assert np.all(on_boundary)

    def test_multiple_heights_produce_matching_slices(self):
        room = _square_room(size=2.0)
        heights = [0.3, 1.0, 1.8]
        result = simulate(
            room,
            jnp.array([[1.0, 1.0]]),
            heights,
            ZERO_NOISE,
            jax.random.PRNGKey(0),
            num_rays=32,
        )

        assert len(result.slices) == 3
        assert [sl.height for sl in result.slices] == heights
        for sl in result.slices:
            assert sl.points.shape == (32, 2)

    def test_points_3d_matches_slice_concatenation_with_correct_heights(self):
        room = _square_room(size=2.0)
        heights = [0.3, 1.8]
        result = simulate(
            room,
            jnp.array([[1.0, 1.0]]),
            heights,
            ZERO_NOISE,
            jax.random.PRNGKey(1),
            num_rays=16,
        )

        total_points = sum(sl.points.shape[0] for sl in result.slices)
        assert result.points_3d.shape == (total_points, 3)

        points_3d = np.asarray(result.points_3d)
        first_slice_count = result.slices[0].points.shape[0]
        np.testing.assert_allclose(points_3d[:first_slice_count, 2], 0.3, atol=1e-6)
        np.testing.assert_allclose(points_3d[first_slice_count:, 2], 1.8, atol=1e-6)

    def test_multiple_scanners_aggregate_rays(self):
        room = _square_room(size=2.0)
        positions = jnp.array([[0.5, 0.5], [1.5, 1.5]])
        result = simulate(room, positions, [1.0], ZERO_NOISE, jax.random.PRNGKey(2), num_rays=20)
        assert result.slices[0].points.shape == (40, 2)  # 2 scanners * 20 rays, no noise loss
        np.testing.assert_allclose(np.asarray(result.scanner_positions), np.asarray(positions))

    def test_floor_plan_bounds_matches_room_corners(self):
        room = _square_room(size=2.0)
        result = simulate(
            room, jnp.array([[1.0, 1.0]]), [1.0], ZERO_NOISE, jax.random.PRNGKey(0), num_rays=8
        )
        np.testing.assert_allclose(np.asarray(result.floor_plan_bounds), [0.0, 0.0, 2.0, 2.0])

    def test_missing_rate_reduces_point_count(self):
        room = _square_room(size=2.0)
        noisy = NoiseModel(
            range_sigma=0.0, angular_sigma=0.0, outlier_rate=0.0, missing_rate=0.5, height_sigma=0.0
        )
        result = simulate(
            room, jnp.array([[1.0, 1.0]]), [1.0], noisy, jax.random.PRNGKey(3), num_rays=200
        )
        assert result.slices[0].points.shape[0] < 200

    def test_full_outlier_rate_keeps_every_ray(self):
        room = _square_room(size=2.0)
        all_clutter = NoiseModel(
            range_sigma=0.0, angular_sigma=0.0, outlier_rate=1.0, missing_rate=0.0, height_sigma=0.0
        )
        result = simulate(
            room, jnp.array([[1.0, 1.0]]), [1.0], all_clutter, jax.random.PRNGKey(4), num_rays=50
        )
        assert result.slices[0].points.shape == (50, 2)

    def test_height_dependent_noise_model_varies_missing_rate_per_slice(self):
        room = _square_room(size=2.0)
        model = HeightDependentNoiseModel(
            range_sigma=lambda h: 0.0,
            angular_sigma=lambda h: 0.0,
            outlier_rate=lambda h: 0.0,
            missing_rate=lambda h: 0.0 if h < 1.0 else 1.0,
            height_sigma=lambda h: 0.0,
        )
        result = simulate(
            room,
            jnp.array([[1.0, 1.0]]),
            [0.3, 1.8],
            model,
            jax.random.PRNGKey(5),
            num_rays=40,
        )
        assert result.slices[0].points.shape[0] == 40  # low slice: no missing returns
        assert result.slices[1].points.shape[0] == 0  # high slice: all returns missing


class TestSimulateDifferentiableMode:
    def test_runs_and_returns_full_fixed_size_points(self):
        room = _square_room(size=2.0)
        result = simulate(
            room,
            jnp.array([[1.0, 1.0]]),
            [1.0],
            ZERO_NOISE,
            jax.random.PRNGKey(0),
            differentiable=True,
            num_rays=32,
        )
        assert result.slices[0].points.shape == (32, 2)
        assert bool(jnp.all(jnp.isfinite(result.slices[0].points)))


class TestFloorPointCloudPlot:
    @pytest.fixture(autouse=True)
    def _close_figures(self):
        yield
        plt.close("all")

    def test_mode_3d_returns_axes(self):
        room = _square_room(size=2.0)
        result = simulate(
            room, jnp.array([[1.0, 1.0]]), [1.0], ZERO_NOISE, jax.random.PRNGKey(0), num_rays=16
        )
        ax = result.plot(room, mode="3d")
        assert ax.name == "3d"

    def test_mode_slices_returns_figure_with_one_axes_per_slice(self):
        room = _square_room(size=2.0)
        result = simulate(
            room,
            jnp.array([[1.0, 1.0]]),
            [0.3, 1.0, 1.8],
            ZERO_NOISE,
            jax.random.PRNGKey(0),
            num_rays=16,
        )
        fig = result.plot(room, mode="slices")
        assert len(fig.axes) == 3

    def test_mode_coverage_returns_axes(self):
        room = _square_room(size=2.0)
        result = simulate(
            room, jnp.array([[1.0, 1.0]]), [1.0], ZERO_NOISE, jax.random.PRNGKey(0), num_rays=16
        )
        ax = result.plot(room, mode="coverage")
        assert ax.name == "rectilinear"

    def test_unknown_mode_raises(self):
        room = _square_room(size=2.0)
        result = simulate(
            room, jnp.array([[1.0, 1.0]]), [1.0], ZERO_NOISE, jax.random.PRNGKey(0), num_rays=16
        )
        with pytest.raises(ValueError):
            result.plot(room, mode="bogus")


class TestSimulateSegments:
    def test_jit_compilable_and_matches_manual_shapes(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        config = ScannerConfig(height=1.0, num_rays=16)
        positions = jnp.array([[1.0, 1.0], [0.5, 0.5]])
        origins, directions = generate_rays_multi(positions, config)
        bounds = jnp.array([0.0, 0.0, 2.0, 2.0])

        points, valid = simulate_segments(
            origins, directions, segments, mask, ZERO_NOISE, bounds, jax.random.PRNGKey(0)
        )
        assert points.shape == (32, 2)
        assert valid.shape == (32,)
        assert bool(jnp.all(valid))

    def test_differentiable_wrt_segment_geometry(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        config = ScannerConfig(height=1.0, num_rays=8)
        positions = jnp.array([[1.0, 1.0]])
        origins, directions = generate_rays_multi(positions, config)
        bounds = jnp.array([0.0, 0.0, 2.0, 2.0])

        def summed_distance_to_walls(moved_segments):
            points, valid = simulate_segments(
                origins,
                directions,
                moved_segments,
                mask,
                ZERO_NOISE,
                bounds,
                jax.random.PRNGKey(0),
                differentiable=True,
                beta=50.0,
            )
            return jnp.sum(points)

        grad = jax.grad(summed_distance_to_walls)(segments)
        assert grad.shape == segments.shape
        assert bool(jnp.any(grad != 0.0))
        assert bool(jnp.all(jnp.isfinite(grad)))
