"""Tests for inference.evaluate: metrics against ground truth."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from inference.evaluate import (
    evaluate,
    junction_localisation_error,
    room_count_accuracy,
    segment_chamfer_distance,
    topology_accuracy,
)
from inference.result import InferenceResult
from representation.floor_plan import FloorPlanGraph


def _square_room(size: float = 2.0, offset: tuple[float, float] = (0.0, 0.0)) -> FloorPlanGraph:
    graph = FloorPlanGraph()
    ox, oy = offset
    corners = [(ox, oy), (ox + size, oy), (ox + size, oy + size), (ox, oy + size)]
    junctions = [graph.add_junction(corner) for corner in corners]
    for start, end in zip(junctions, junctions[1:] + junctions[:1]):
        graph.add_segment(start.id, end.id)
    return graph


def _dangling_wall() -> FloorPlanGraph:
    graph = FloorPlanGraph()
    a = graph.add_junction((0.0, 0.0))
    b = graph.add_junction((1.0, 0.0))
    graph.add_segment(a.id, b.id)
    return graph


def _make_result(map_estimate: FloorPlanGraph, samples: list[FloorPlanGraph] | None = None):
    samples = samples if samples is not None else []
    return InferenceResult(
        samples=samples,
        log_weights=jnp.zeros((len(samples),)),
        map_estimate=map_estimate,
        junction_position_std=jnp.zeros((4, 2)),
        n_segments_distribution=jnp.array([0, 0, 0, 0, 1]),
        method="rjmcmc",
        n_likelihood_evaluations=10,
        runtime_seconds=1.0,
        acceptance_rates={},
        ess=1.0,
        log_posterior_trace=jnp.array([-1.0]),
    )


class TestSegmentChamferDistance:
    def test_identical_graphs_have_zero_distance(self):
        room = _square_room()
        assert segment_chamfer_distance(room, room) == pytest.approx(0.0, abs=1e-6)

    def test_shifted_graph_has_positive_distance(self):
        a = _square_room(offset=(0.0, 0.0))
        b = _square_room(offset=(1.0, 0.0))
        distance = segment_chamfer_distance(a, b)
        assert distance == pytest.approx(1.0, abs=1e-6)

    def test_raises_when_either_graph_has_no_segments(self):
        room = _square_room()
        empty = FloorPlanGraph()
        with pytest.raises(ValueError):
            segment_chamfer_distance(empty, room)
        with pytest.raises(ValueError):
            segment_chamfer_distance(room, empty)


class TestJunctionLocalisationError:
    def test_identical_graphs_have_zero_error(self):
        room = _square_room()
        assert junction_localisation_error(room, room) == pytest.approx(0.0, abs=1e-6)

    def test_shifted_graph_has_expected_error(self):
        a = _square_room(offset=(0.0, 0.0))
        b = _square_room(offset=(0.5, 0.0))
        error = junction_localisation_error(a, b)
        assert error == pytest.approx(0.5, abs=1e-6)

    def test_raises_when_either_graph_has_no_junctions(self):
        room = _square_room()
        empty = FloorPlanGraph()
        with pytest.raises(ValueError):
            junction_localisation_error(empty, room)
        with pytest.raises(ValueError):
            junction_localisation_error(room, empty)


class TestRoomCountAccuracy:
    def test_map_only_matching_room_count_is_perfect(self):
        room = _square_room()
        result = _make_result(map_estimate=room)
        assert room_count_accuracy(result, room) == 1.0

    def test_map_only_mismatched_room_count_is_zero(self):
        result = _make_result(map_estimate=_dangling_wall())
        assert room_count_accuracy(result, _square_room()) == 0.0

    def test_fraction_of_samples_with_correct_room_count(self):
        true_room = _square_room()
        samples = [_square_room(), _square_room(offset=(5.0, 5.0)), _dangling_wall()]
        result = _make_result(map_estimate=samples[0], samples=samples)
        assert room_count_accuracy(result, true_room) == pytest.approx(2 / 3)


class TestTopologyAccuracy:
    def test_identical_topology_is_perfect(self):
        room = _square_room()
        result = _make_result(map_estimate=room)
        assert topology_accuracy(result, room) == 1.0

    def test_different_topology_is_zero(self):
        result = _make_result(map_estimate=_dangling_wall())
        assert topology_accuracy(result, _square_room()) == 0.0

    def test_fraction_of_samples_with_correct_topology(self):
        true_room = _square_room()
        samples = [_square_room(offset=(5.0, 5.0)), _dangling_wall()]
        result = _make_result(map_estimate=samples[0], samples=samples)
        assert topology_accuracy(result, true_room) == pytest.approx(0.5)


class TestEvaluate:
    def test_perfect_recovery_report(self):
        room = _square_room()
        result = _make_result(map_estimate=room, samples=[room])
        report = evaluate(result, room)
        assert report.chamfer_distance_segments == pytest.approx(0.0, abs=1e-6)
        assert report.junction_localisation_error == pytest.approx(0.0, abs=1e-6)
        assert report.room_count_accuracy == 1.0
        assert report.topology_accuracy == 1.0

    def test_rejects_ground_truth_with_no_geometry(self):
        result = _make_result(map_estimate=_square_room())
        with pytest.raises(ValueError):
            evaluate(result, FloorPlanGraph())

    def test_rejects_map_estimate_with_no_geometry(self):
        result = _make_result(map_estimate=FloorPlanGraph())
        with pytest.raises(ValueError):
            evaluate(result, _square_room())
