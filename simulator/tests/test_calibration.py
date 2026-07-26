"""Tests for simulator.calibration: Chamfer distance and noise-model calibration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from representation.floor_plan import FloorPlanGraph
from simulator.calibration import (
    calibrate_noise_model,
    chamfer_distance,
    height_slice_consistency,
    point_density_by_wall,
)
from simulator.engine import FloorPointCloud, PointCloudSlice, simulate
from simulator.noise_model import NoiseModel, NoiseModelPrior

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


def _cloud_from_points(points: jnp.ndarray, height: float = 1.0) -> FloorPointCloud:
    z = jnp.full((points.shape[0],), height)
    points_3d = jnp.column_stack([points, z])
    return FloorPointCloud(
        slices=[PointCloudSlice(height=height, points=points)],
        points_3d=points_3d,
        scanner_positions=jnp.zeros((1, 2)),
        floor_plan_bounds=jnp.array([0.0, 0.0, 2.0, 2.0]),
    )


class TestChamferDistance:
    def test_identical_clouds_have_zero_distance(self):
        points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
        cloud = _cloud_from_points(points)
        assert chamfer_distance(cloud, cloud) == pytest.approx(0.0, abs=1e-6)

    def test_shifted_cloud_has_positive_distance_matching_shift(self):
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        shifted = points + jnp.array([0.0, 1.0])
        cloud_a = _cloud_from_points(points)
        cloud_b = _cloud_from_points(shifted)
        # every point's nearest neighbour in the other cloud is its shifted counterpart
        assert chamfer_distance(cloud_a, cloud_b) == pytest.approx(1.0, abs=1e-5)

    def test_is_symmetric(self):
        a = _cloud_from_points(jnp.array([[0.0, 0.0], [3.0, 3.0]]))
        b = _cloud_from_points(jnp.array([[0.1, 0.0], [1.0, 1.0], [3.1, 3.0]]))
        assert chamfer_distance(a, b) == pytest.approx(chamfer_distance(b, a), abs=1e-6)

    def test_rejects_empty_point_cloud(self):
        empty = _cloud_from_points(jnp.zeros((0, 2)))
        nonempty = _cloud_from_points(jnp.array([[0.0, 0.0]]))
        with pytest.raises(ValueError):
            chamfer_distance(empty, nonempty)
        with pytest.raises(ValueError):
            chamfer_distance(nonempty, empty)


class TestPointDensityByWall:
    def test_empty_floor_plan_returns_empty_dict(self):
        assert point_density_by_wall(jnp.array([[0.0, 0.0]]), FloorPlanGraph()) == {}

    def test_empty_points_returns_zero_density_for_every_wall(self):
        room = _square_room(2.0)
        density = point_density_by_wall(jnp.zeros((0, 2)), room)
        assert set(density) == set(room.segments)
        assert all(value == 0.0 for value in density.values())

    def test_points_on_one_wall_only_are_assigned_there(self):
        room = _square_room(2.0)
        # junctions are added in order (0,0),(2,0),(2,2),(0,2) -> ids 0,1,2,3, and
        # segments are added as consecutive pairs -> segment 0 is the bottom wall
        # (0,0)-(2,0).
        bottom_wall_id = 0
        points = jnp.array([[0.5, 0.0], [1.0, 0.0], [1.5, 0.0]])
        density = point_density_by_wall(points, room)

        assert density[bottom_wall_id] == pytest.approx(3 / 2.0, abs=1e-5)  # 3 points / 2m wall
        for sid, value in density.items():
            if sid != bottom_wall_id:
                assert value == 0.0


class TestHeightSliceConsistency:
    def test_raises_on_no_slices(self):
        room = _square_room(2.0)
        empty_cloud = FloorPointCloud(
            slices=[],
            points_3d=jnp.zeros((0, 3)),
            scanner_positions=jnp.zeros((1, 2)),
            floor_plan_bounds=jnp.array([0.0, 0.0, 2.0, 2.0]),
        )
        with pytest.raises(ValueError):
            height_slice_consistency(empty_cloud, room)

    def test_zero_for_floor_plan_with_no_segments(self):
        cloud = _cloud_from_points(jnp.array([[0.0, 0.0]]))
        assert height_slice_consistency(cloud, FloorPlanGraph()) == 0.0

    def test_identical_density_across_slices_is_zero(self):
        room = _square_room(2.0)
        result = simulate(
            room,
            jnp.array([[1.0, 1.0]]),
            [0.3, 1.0, 1.8],
            ZERO_NOISE,
            jax.random.PRNGKey(0),
            num_rays=64,
        )
        assert height_slice_consistency(result, room) == pytest.approx(0.0, abs=1e-4)

    def test_varying_density_across_slices_is_positive(self):
        room = _square_room(2.0)
        low_noise = NoiseModel(
            range_sigma=0.0, angular_sigma=0.0, outlier_rate=0.0, missing_rate=0.0, height_sigma=0.0
        )
        high_missing = NoiseModel(
            range_sigma=0.0, angular_sigma=0.0, outlier_rate=0.0, missing_rate=0.8, height_sigma=0.0
        )
        slice_low = simulate(
            room, jnp.array([[1.0, 1.0]]), [0.3], low_noise, jax.random.PRNGKey(0), num_rays=64
        ).slices[0]
        slice_high = simulate(
            room, jnp.array([[1.0, 1.0]]), [1.8], high_missing, jax.random.PRNGKey(1), num_rays=64
        ).slices[0]

        combined = FloorPointCloud(
            slices=[slice_low, slice_high],
            points_3d=jnp.zeros((0, 3)),
            scanner_positions=jnp.array([[1.0, 1.0]]),
            floor_plan_bounds=jnp.array([0.0, 0.0, 2.0, 2.0]),
        )
        assert height_slice_consistency(combined, room) > 0.0


class TestCalibrateNoiseModel:
    def test_rejects_real_point_cloud_with_no_slices(self):
        room = _square_room(2.0)
        empty_cloud = FloorPointCloud(
            slices=[],
            points_3d=jnp.zeros((0, 3)),
            scanner_positions=jnp.zeros((1, 2)),
            floor_plan_bounds=jnp.array([0.0, 0.0, 2.0, 2.0]),
        )
        prior = NoiseModelPrior(
            range_sigma=(0.0, 0.01),
            angular_sigma=(0.0, 0.01),
            outlier_rate=(0.0, 0.1),
            missing_rate=(0.0, 0.1),
            height_sigma=(0.0, 0.01),
        )
        with pytest.raises(ValueError):
            calibrate_noise_model(
                room, empty_cloud, jnp.array([[1.0, 1.0]]), prior, jax.random.PRNGKey(0)
            )

    def test_rejects_non_positive_num_candidates(self):
        room = _square_room(2.0)
        real = simulate(
            room, jnp.array([[1.0, 1.0]]), [1.0], ZERO_NOISE, jax.random.PRNGKey(0), num_rays=16
        )
        prior = NoiseModelPrior(
            range_sigma=(0.0, 0.01),
            angular_sigma=(0.0, 0.01),
            outlier_rate=(0.0, 0.1),
            missing_rate=(0.0, 0.1),
            height_sigma=(0.0, 0.01),
        )
        with pytest.raises(ValueError):
            calibrate_noise_model(
                room,
                real,
                jnp.array([[1.0, 1.0]]),
                prior,
                jax.random.PRNGKey(0),
                num_candidates=0,
            )

    def test_returns_noise_model_achieving_low_chamfer_distance(self):
        room = _square_room(2.0)
        positions = jnp.array([[1.0, 1.0]])
        heights = [1.0]
        real = simulate(room, positions, heights, ZERO_NOISE, jax.random.PRNGKey(0), num_rays=64)

        prior = NoiseModelPrior(
            range_sigma=(0.0, 0.05),
            angular_sigma=(0.0, 0.01),
            outlier_rate=(0.0, 0.3),
            missing_rate=(0.0, 0.3),
            height_sigma=(0.0, 0.02),
        )
        calibrated = calibrate_noise_model(
            room, real, positions, prior, jax.random.PRNGKey(1), num_candidates=30
        )
        assert isinstance(calibrated, NoiseModel)

        simulated = simulate(
            room, positions, heights, calibrated, jax.random.PRNGKey(2), num_rays=64
        )
        distance = chamfer_distance(simulated, real)
        assert distance < 0.05
