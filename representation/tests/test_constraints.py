"""Tests for representation.constraints: hard validators and soft penalty functions."""

from __future__ import annotations

import pytest

from representation.constraints import (
    ConstraintViolation,
    bounds_penalty,
    length_penalty,
    planarity_penalty,
    valence_penalty,
    validate_all,
    validate_bounds,
    validate_junction_valence,
    validate_min_segment_length,
    validate_planarity,
)
from representation.floor_plan import FloorPlanGraph


def _square_room() -> FloorPlanGraph:
    """A single 1m x 1m room: four junctions, four segments, all valence 2, planar."""
    graph = FloorPlanGraph()
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    junctions = [graph.add_junction(corner) for corner in corners]
    for start, end in zip(junctions, junctions[1:] + junctions[:1]):
        graph.add_segment(start.id, end.id)
    return graph


def _crossing_x() -> FloorPlanGraph:
    """Two segments crossing at a point that is not a shared junction."""
    graph = FloorPlanGraph()
    a = graph.add_junction((0.0, 0.0))
    b = graph.add_junction((1.0, 1.0))
    c = graph.add_junction((0.0, 1.0))
    d = graph.add_junction((1.0, 0.0))
    graph.add_segment(a.id, b.id)
    graph.add_segment(c.id, d.id)
    return graph


def _dangling_wall() -> FloorPlanGraph:
    """Two junctions joined by one segment -- both endpoints have valence 1."""
    graph = FloorPlanGraph()
    a = graph.add_junction((0.0, 0.0))
    b = graph.add_junction((1.0, 0.0))
    graph.add_segment(a.id, b.id)
    return graph


def _short_wall() -> FloorPlanGraph:
    """A single segment shorter than the default minimum length."""
    graph = FloorPlanGraph()
    a = graph.add_junction((0.0, 0.0))
    b = graph.add_junction((0.05, 0.0))
    graph.add_segment(a.id, b.id)
    return graph


class TestPlanarity:
    def test_valid_graph_has_zero_penalty(self):
        assert planarity_penalty(_square_room()) == 0.0

    def test_valid_graph_passes_validation(self):
        validate_planarity(_square_room())  # should not raise

    def test_crossing_segments_have_positive_penalty(self):
        assert planarity_penalty(_crossing_x()) > 0.0

    def test_crossing_segments_raise(self):
        with pytest.raises(ConstraintViolation):
            validate_planarity(_crossing_x())


class TestMinSegmentLength:
    def test_valid_graph_has_zero_penalty(self):
        assert length_penalty(_square_room()) == 0.0

    def test_valid_graph_passes_validation(self):
        validate_min_segment_length(_square_room())  # should not raise

    def test_short_segment_has_positive_penalty(self):
        assert length_penalty(_short_wall()) > 0.0

    def test_short_segment_raises(self):
        with pytest.raises(ConstraintViolation):
            validate_min_segment_length(_short_wall())

    def test_custom_threshold_accepted(self):
        graph = _short_wall()
        assert length_penalty(graph, min_length=0.01) == 0.0
        validate_min_segment_length(graph, min_length=0.01)  # should not raise


class TestJunctionValence:
    def test_valid_graph_has_zero_penalty(self):
        assert valence_penalty(_square_room()) == 0.0

    def test_valid_graph_passes_validation(self):
        validate_junction_valence(_square_room())  # should not raise

    def test_dangling_junctions_have_positive_penalty(self):
        assert valence_penalty(_dangling_wall()) > 0.0

    def test_dangling_junctions_raise(self):
        with pytest.raises(ConstraintViolation):
            validate_junction_valence(_dangling_wall())

    def test_custom_min_valence_accepted(self):
        assert valence_penalty(_dangling_wall(), min_valence=1) == 0.0
        validate_junction_valence(_dangling_wall(), min_valence=1)  # should not raise


class TestBounds:
    def test_none_bounds_is_a_no_op(self):
        graph = _square_room()
        assert bounds_penalty(graph, None) == 0.0
        validate_bounds(graph, None)  # should not raise

    def test_graph_within_bounds_has_zero_penalty(self):
        graph = _square_room()
        assert bounds_penalty(graph, (0.0, 0.0, 1.0, 1.0)) == 0.0

    def test_graph_within_bounds_passes_validation(self):
        validate_bounds(_square_room(), (0.0, 0.0, 1.0, 1.0))  # should not raise

    def test_graph_outside_bounds_has_positive_penalty(self):
        graph = _square_room()
        assert bounds_penalty(graph, (0.0, 0.0, 0.5, 0.5)) > 0.0

    def test_graph_outside_bounds_raises(self):
        with pytest.raises(ConstraintViolation):
            validate_bounds(_square_room(), (0.0, 0.0, 0.5, 0.5))


class TestValidateAll:
    def test_valid_graph_passes(self):
        validate_all(_square_room())  # should not raise

    def test_valid_graph_passes_with_bounds(self):
        validate_all(_square_room(), bounds=(0.0, 0.0, 1.0, 1.0))  # should not raise

    def test_planarity_violation_raises_first(self):
        with pytest.raises(ConstraintViolation):
            validate_all(_crossing_x())

    def test_length_violation_raises(self):
        with pytest.raises(ConstraintViolation):
            validate_all(_short_wall())

    def test_valence_violation_raises(self):
        with pytest.raises(ConstraintViolation):
            validate_all(_dangling_wall())

    def test_bounds_violation_raises(self):
        with pytest.raises(ConstraintViolation):
            validate_all(_square_room(), bounds=(0.0, 0.0, 0.5, 0.5))
