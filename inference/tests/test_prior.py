"""Tests for inference.prior: the five log P(G) terms and the procedural sampler."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from inference.config import PriorConfig
from inference.prior import (
    Prior,
    _bsp_subdivide,
    _gaussian_log_pdf,
    _junction_deviation,
    _polygon_area,
    _Rect,
    _rects_to_floor_plan,
    complexity_log_prob,
    junction_regularity_log_prob,
    log_prob,
    min_length_log_prob,
    orthogonality_log_prob,
    room_size_log_prob,
    sample,
)
from representation.floor_plan import FloorPlanGraph


def _square_room(size: float = 2.0) -> FloorPlanGraph:
    graph = FloorPlanGraph()
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    junctions = [graph.add_junction(corner) for corner in corners]
    for start, end in zip(junctions, junctions[1:] + junctions[:1]):
        graph.add_segment(start.id, end.id)
    return graph


def _rotated_wall(angle_degrees: float, length: float = 1.0) -> FloorPlanGraph:
    graph = FloorPlanGraph()
    angle = np.radians(angle_degrees)
    a = graph.add_junction((0.0, 0.0))
    b = graph.add_junction((length * np.cos(angle), length * np.sin(angle)))
    graph.add_segment(a.id, b.id)
    return graph


class TestComplexityLogProb:
    def test_matches_formula(self):
        room = _square_room()
        assert complexity_log_prob(room, lambda_complexity=0.5) == pytest.approx(-0.5 * 4)

    def test_zero_lambda_gives_zero(self):
        room = _square_room()
        assert complexity_log_prob(room, lambda_complexity=0.0) == 0.0


class TestOrthogonalityLogProb:
    def test_axis_aligned_room_has_zero_penalty(self):
        room = _square_room()
        assert orthogonality_log_prob(room, lambda_orthogonality=1.0) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_45_degree_wall_has_maximum_penalty(self):
        wall = _rotated_wall(45.0)
        penalty = orthogonality_log_prob(wall, lambda_orthogonality=1.0)
        assert penalty == pytest.approx(-np.pi / 4, abs=1e-6)

    def test_penalty_scales_with_lambda(self):
        wall = _rotated_wall(45.0)
        assert orthogonality_log_prob(wall, lambda_orthogonality=2.0) == pytest.approx(
            2 * orthogonality_log_prob(wall, lambda_orthogonality=1.0)
        )


class TestMinLengthLogProb:
    def test_all_segments_long_enough_is_zero(self):
        room = _square_room(size=2.0)
        assert min_length_log_prob(room, min_segment_length=0.1) == 0.0

    def test_short_segment_is_negative_infinity(self):
        wall = _rotated_wall(0.0, length=0.05)
        assert min_length_log_prob(wall, min_segment_length=0.1) == float("-inf")


class TestJunctionDeviation:
    def test_no_incident_segments_or_one_is_zero(self):
        assert _junction_deviation([]) == 0.0
        assert _junction_deviation([0.0]) == 0.0

    def test_perfect_l_junction_is_zero(self):
        assert _junction_deviation([0.0, np.pi / 2]) == pytest.approx(0.0, abs=1e-9)

    def test_perfect_t_junction_is_zero(self):
        assert _junction_deviation([0.0, np.pi / 2, np.pi]) == pytest.approx(0.0, abs=1e-9)

    def test_irregular_junction_is_positive(self):
        assert _junction_deviation([0.0, np.radians(50), np.radians(200)]) > 0.0


class TestJunctionRegularityLogProb:
    def test_square_room_has_zero_penalty(self):
        room = _square_room()
        assert junction_regularity_log_prob(room, lambda_junction=1.0) == pytest.approx(
            0.0, abs=1e-6
        )


class TestPolygonAreaAndGaussianLogPdf:
    def test_polygon_area_of_unit_square(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert _polygon_area(positions) == pytest.approx(1.0)

    def test_gaussian_log_pdf_at_mean(self):
        value = _gaussian_log_pdf(5.0, mean=5.0, std=2.0)
        expected = -0.5 * np.log(2 * np.pi * 4.0)
        assert value == pytest.approx(expected)


class TestRoomSizeLogProb:
    def test_matches_gaussian_log_pdf_for_single_room(self):
        room = _square_room(size=3.0)  # area = 9
        value = room_size_log_prob(room, room_area_mean=9.0, room_area_std=2.0)
        assert value == pytest.approx(_gaussian_log_pdf(9.0, 9.0, 2.0))

    def test_no_rooms_is_zero(self):
        wall = _rotated_wall(0.0)
        assert room_size_log_prob(wall, room_area_mean=10.0, room_area_std=5.0) == 0.0


class TestLogProb:
    def test_all_terms_disabled_is_zero(self):
        room = _square_room()
        config = PriorConfig(
            use_complexity=False,
            use_orthogonality=False,
            use_min_length=False,
            use_junction_regularity=False,
            use_room_size=False,
        )
        assert log_prob(room, config) == 0.0

    def test_short_segment_is_negative_infinity_regardless_of_other_terms(self):
        wall = _rotated_wall(0.0, length=0.01)
        config = PriorConfig(min_segment_length=0.1)
        assert log_prob(wall, config) == float("-inf")

    def test_disabling_min_length_ignores_short_segments(self):
        wall = _rotated_wall(0.0, length=0.01)
        config = PriorConfig(
            use_min_length=False,
            min_segment_length=0.1,
            use_complexity=True,
            use_orthogonality=False,
            use_junction_regularity=False,
            use_room_size=False,
        )
        assert log_prob(wall, config) == pytest.approx(
            complexity_log_prob(wall, config.lambda_complexity)
        )


class TestBspSubdivide:
    def test_single_target_returns_original_rect(self):
        rect = _Rect(0.0, 0.0, 4.0, 3.0)
        leaves = _bsp_subdivide(rect, target_count=1, max_splits=4, rng_key=jax.random.PRNGKey(0))
        assert leaves == [rect]

    def test_splits_reach_target_count(self):
        rect = _Rect(0.0, 0.0, 10.0, 10.0)
        leaves = _bsp_subdivide(rect, target_count=4, max_splits=10, rng_key=jax.random.PRNGKey(0))
        assert len(leaves) == 4

    def test_leaves_partition_the_total_area(self):
        rect = _Rect(0.0, 0.0, 10.0, 6.0)
        leaves = _bsp_subdivide(rect, target_count=5, max_splits=10, rng_key=jax.random.PRNGKey(1))
        total_area = sum((r.x_max - r.x_min) * (r.y_max - r.y_min) for r in leaves)
        assert total_area == pytest.approx(60.0)

    def test_max_splits_caps_leaf_count(self):
        rect = _Rect(0.0, 0.0, 10.0, 10.0)
        leaves = _bsp_subdivide(rect, target_count=100, max_splits=2, rng_key=jax.random.PRNGKey(0))
        assert len(leaves) == 3  # 2 splits -> 3 leaves


class TestRectsToFloorPlan:
    def test_single_rect_is_a_valid_room(self):
        floor_plan = _rects_to_floor_plan([_Rect(0.0, 0.0, 2.0, 2.0)])
        assert len(floor_plan.junctions) == 4
        assert len(floor_plan.segments) == 4
        assert len(floor_plan.cycles()) == 1

    def test_adjacent_rects_share_one_wall_not_two(self):
        rects = [_Rect(0.0, 0.0, 2.0, 2.0), _Rect(2.0, 0.0, 4.0, 2.0)]
        floor_plan = _rects_to_floor_plan(rects)
        assert len(floor_plan.junctions) == 6  # 4 + 4 - 2 shared corners
        assert len(floor_plan.segments) == 7  # 4 + 4 - 1 shared wall
        assert len(floor_plan.cycles()) == 2

    def test_t_junction_from_one_edge_spanning_two_neighbours_is_planar(self):
        # a wide rect below two stacked rects of half its width: the wide rect's
        # top edge spans both narrower rects above it, landing a corner on its
        # interior rather than at an endpoint (the T-junction case that motivated
        # _split_segments_at_interior_junctions)
        rects = [
            _Rect(0.0, 0.0, 4.0, 2.0),
            _Rect(0.0, 2.0, 2.0, 4.0),
            _Rect(2.0, 2.0, 4.0, 4.0),
        ]
        floor_plan = _rects_to_floor_plan(rects)

        assert floor_plan.is_planar()
        assert len(floor_plan.cycles()) == 3
        # no duplicate segments between any pair of junctions
        pairs = [
            frozenset((s.junction_start, s.junction_end)) for s in floor_plan.segments.values()
        ]
        assert len(pairs) == len(set(pairs))


class TestSample:
    def test_is_planar_across_many_seeds(self):
        # regression test for two bugs found during development: BSP subdivision
        # can produce a T-junction (one rectangle's edge spans two smaller
        # neighbours), and independent per-junction perturbation can occasionally
        # push a junction across a nearby wall for small rooms
        config = PriorConfig(
            bounding_box_min_size=3.0, bounding_box_max_size=8.0, room_count_rate=5.0
        )
        for seed in range(200):
            floor_plan = sample(jax.random.PRNGKey(seed), config)
            assert floor_plan.is_planar(), f"non-planar sample at seed {seed}"

    def test_produces_a_nonempty_floor_plan(self):
        config = PriorConfig(bounding_box_min_size=3.0, bounding_box_max_size=6.0)
        floor_plan = sample(jax.random.PRNGKey(0), config)
        assert len(floor_plan.junctions) > 0
        assert len(floor_plan.segments) > 0
        assert len(floor_plan.cycles()) >= 1

    def test_different_keys_give_different_layouts(self):
        config = PriorConfig(
            bounding_box_min_size=3.0, bounding_box_max_size=8.0, room_count_rate=5.0
        )
        a = sample(jax.random.PRNGKey(0), config)
        b = sample(jax.random.PRNGKey(1), config)
        assert len(a.junctions) != len(b.junctions) or len(a.segments) != len(b.segments)

    def test_zero_perturbation_sigma_keeps_axis_aligned_walls(self):
        config = PriorConfig(
            bounding_box_min_size=3.0,
            bounding_box_max_size=8.0,
            room_count_rate=5.0,
            orthogonality_perturbation_sigma=0.0,
        )
        floor_plan = sample(jax.random.PRNGKey(2), config)
        assert orthogonality_log_prob(floor_plan, lambda_orthogonality=1.0) == pytest.approx(
            0.0, abs=1e-6
        )


class TestPriorClass:
    def test_log_prob_matches_free_function(self):
        config = PriorConfig()
        room = _square_room()
        prior = Prior(config)
        assert prior.log_prob(room) == log_prob(room, config)

    def test_sample_matches_free_function_given_same_key(self):
        config = PriorConfig(bounding_box_min_size=3.0, bounding_box_max_size=6.0)
        prior = Prior(config)
        key = jax.random.PRNGKey(0)
        a = prior.sample(key)
        b = sample(key, config)
        assert len(a.junctions) == len(b.junctions)
        assert len(a.segments) == len(b.segments)
