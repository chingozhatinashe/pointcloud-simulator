"""Tests for simulator.ray_caster: hard and soft occlusion ray-segment intersection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representation.floor_plan import FloorPlanGraph
from simulator.ray_caster import cast_rays
from simulator.scanner import ScannerConfig, generate_rays


def _square_room(size: float = 2.0) -> FloorPlanGraph:
    """An axis-aligned square room from (0, 0) to (size, size)."""
    graph = FloorPlanGraph()
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    junctions = [graph.add_junction(corner) for corner in corners]
    for start, end in zip(junctions, junctions[1:] + junctions[:1]):
        graph.add_segment(start.id, end.id)
    return graph


class TestHardOcclusion:
    def test_cardinal_rays_hit_expected_walls(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        origin = jnp.array([[1.0, 1.0]] * 4)
        directions = jnp.array(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )  # east, west, north, south from room centre

        distances, hits, points = cast_rays(origin, directions, segments, mask)

        assert bool(jnp.all(hits))
        np.testing.assert_allclose(np.asarray(distances), [1.0, 1.0, 1.0, 1.0], atol=1e-4)
        expected_points = np.array([[2.0, 1.0], [0.0, 1.0], [1.0, 2.0], [1.0, 0.0]])
        np.testing.assert_allclose(np.asarray(points), expected_points, atol=1e-4)

    def test_full_sweep_hits_only_room_boundary(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        config = ScannerConfig(height=1.0, num_rays=360)
        origins, directions = generate_rays(jnp.array([1.0, 1.0]), config)

        distances, hits, points = cast_rays(origins, directions, segments, mask)

        assert bool(jnp.all(hits))
        assert bool(jnp.all(jnp.isfinite(distances)))
        points_np = np.asarray(points)
        # every hit point must lie on one of the four walls of the [0, 2] x [0, 2] room
        on_boundary = (
            np.isclose(points_np[:, 0], 0.0, atol=1e-3)
            | np.isclose(points_np[:, 0], 2.0, atol=1e-3)
            | np.isclose(points_np[:, 1], 0.0, atol=1e-3)
            | np.isclose(points_np[:, 1], 2.0, atol=1e-3)
        )
        assert np.all(on_boundary)
        assert np.all(points_np >= -1e-3)
        assert np.all(points_np <= 2.0 + 1e-3)

    def test_ray_with_no_segments_in_range_misses(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        # ray from outside the room, pointing further away
        origin = jnp.array([[10.0, 10.0]])
        direction = jnp.array([[1.0, 0.0]])

        distances, hits, points = cast_rays(origin, direction, segments, mask)

        assert not bool(hits[0])
        assert bool(jnp.isinf(distances[0]))
        assert bool(jnp.all(jnp.isnan(points[0])))

    def test_segment_mask_excludes_padded_rows(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()  # padded to K_MAX with zero rows
        mask = room.segment_mask()
        origin = jnp.array([[1.0, 1.0]])
        direction = jnp.array([[1.0, 0.0]])

        distances, hits, _ = cast_rays(origin, direction, segments, mask)
        assert bool(hits[0])
        np.testing.assert_allclose(float(distances[0]), 1.0, atol=1e-4)

    def test_defaults_to_all_valid_when_mask_omitted(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        origin = jnp.array([[1.0, 1.0]])
        direction = jnp.array([[1.0, 0.0]])

        distances, hits, _ = cast_rays(origin, direction, segments)
        assert bool(hits[0])
        np.testing.assert_allclose(float(distances[0]), 1.0, atol=1e-4)


class TestSoftOcclusion:
    def test_agrees_with_hard_occlusion_for_large_beta(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        config = ScannerConfig(height=1.0, num_rays=64)
        origins, directions = generate_rays(jnp.array([1.0, 1.0]), config)

        hard_distances, hard_hits, _ = cast_rays(origins, directions, segments, mask)
        soft_distances, soft_hits, _ = cast_rays(
            origins, directions, segments, mask, differentiable=True, beta=1e4
        )

        assert bool(jnp.all(hard_hits == soft_hits))
        np.testing.assert_allclose(
            np.asarray(soft_distances), np.asarray(hard_distances), atol=1e-2
        )

    def test_no_hit_case_returns_inf_and_nan_point(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        origin = jnp.array([[10.0, 10.0]])
        direction = jnp.array([[1.0, 0.0]])

        distances, hits, points = cast_rays(origin, direction, segments, mask, differentiable=True)
        assert not bool(hits[0])
        assert bool(jnp.isinf(distances[0]))
        assert bool(jnp.all(jnp.isnan(points[0])))

    def test_differentiable_wrt_segment_geometry(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        origin = jnp.array([[1.0, 1.0]])
        direction = jnp.array([[1.0, 0.0]])  # heads straight for the east wall

        def soft_distance_to_east_wall(moved_segments):
            distances, _, _ = cast_rays(
                origin, direction, moved_segments, mask, differentiable=True, beta=50.0
            )
            return distances[0]

        grad = jax.grad(soft_distance_to_east_wall)(segments)
        assert bool(jnp.any(grad != 0.0))
        assert jnp.all(jnp.isfinite(grad))


class TestJaxCompatibility:
    def test_cast_rays_is_jit_compilable(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        config = ScannerConfig(height=1.0, num_rays=32)
        origins, directions = generate_rays(jnp.array([1.0, 1.0]), config)

        jitted = jax.jit(lambda o, d: cast_rays(o, d, segments, mask))
        distances, hits, points = jitted(origins, directions)
        assert distances.shape == (32,)
        assert hits.shape == (32,)
        assert points.shape == (32, 2)

    def test_cast_rays_is_vmappable_over_scanner_positions(self):
        room = _square_room(size=2.0)
        segments = room.to_segment_array()
        mask = room.segment_mask()
        config = ScannerConfig(height=1.0, num_rays=16)
        positions = jnp.array([[1.0, 1.0], [0.5, 0.5], [1.5, 1.5]])
        origins, directions = jax.vmap(generate_rays, in_axes=(0, None))(positions, config)

        batched_cast = jax.vmap(cast_rays, in_axes=(0, 0, None, None))
        distances, hits, points = batched_cast(origins, directions, segments, mask)

        assert distances.shape == (3, 16)
        assert hits.shape == (3, 16)
        assert points.shape == (3, 16, 2)
        assert bool(jnp.all(hits))


def _single_wall(start=(0.0, 0.0), end=(2.0, 0.0)) -> FloorPlanGraph:
    graph = FloorPlanGraph()
    a = graph.add_junction(start)
    b = graph.add_junction(end)
    graph.add_segment(a.id, b.id)
    return graph


@pytest.mark.parametrize("differentiable", [False, True])
def test_ray_collinear_with_wall_does_not_report_a_hit(differentiable):
    wall = _single_wall()
    segments = wall.to_segment_array()
    mask = wall.segment_mask()
    origin = jnp.array([[-1.0, 0.0]])
    direction = jnp.array([[1.0, 0.0]])  # collinear with the wall

    distances, hits, _ = cast_rays(origin, direction, segments, mask, differentiable=differentiable)
    # parallel-to-segment rays are treated as non-intersecting (denom ~ 0)
    assert not bool(hits[0])
    assert bool(jnp.isinf(distances[0]))
