import jax.numpy as jnp
import pytest

from representation.floor_plan import J_MAX, K_MAX, FloorPlanGraph


def square_room() -> FloorPlanGraph:
    """A unit square: junctions at the corners, walls along the edges."""
    graph = FloorPlanGraph()
    a = graph.add_junction((0.0, 0.0))
    b = graph.add_junction((1.0, 0.0))
    c = graph.add_junction((1.0, 1.0))
    d = graph.add_junction((0.0, 1.0))
    graph.add_segment(a.id, b.id)
    graph.add_segment(b.id, c.id)
    graph.add_segment(c.id, d.id)
    graph.add_segment(d.id, a.id)
    return graph


class TestConstruction:
    def test_add_junction_assigns_incrementing_ids(self):
        graph = FloorPlanGraph()
        j0 = graph.add_junction((0.0, 0.0))
        j1 = graph.add_junction((1.0, 1.0))
        assert j0.id == 0
        assert j1.id == 1
        assert graph.junctions[0] is j0
        assert graph.junctions[1] is j1

    def test_add_segment_between_existing_junctions(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((1.0, 0.0))
        seg = graph.add_segment(a.id, b.id)
        assert seg.junction_start == a.id
        assert seg.junction_end == b.id
        assert graph.segments[seg.id] is seg

    def test_add_segment_rejects_unknown_junction(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        with pytest.raises(ValueError):
            graph.add_segment(a.id, 99)

    def test_remove_junction_also_removes_incident_segments(self):
        graph = square_room()
        graph.remove_junction(0)
        assert 0 not in graph.junctions
        assert all(0 not in (s.junction_start, s.junction_end) for s in graph.segments.values())
        assert len(graph.segments) == 2

    def test_remove_junction_unknown_raises(self):
        graph = FloorPlanGraph()
        with pytest.raises(KeyError):
            graph.remove_junction(0)

    def test_remove_segment(self):
        graph = square_room()
        graph.remove_segment(0)
        assert 0 not in graph.segments
        assert len(graph.segments) == 3

    def test_move_junction_updates_incident_segment_geometry(self):
        graph = square_room()
        graph.move_junction(0, (5.0, 5.0))
        assert jnp.allclose(graph.junctions[0].position, jnp.array([5.0, 5.0]))
        seg = graph.segments[0]
        assert jnp.allclose(seg.as_array(graph.junctions)[:2], jnp.array([5.0, 5.0]))


class TestMergeAndSplit:
    def test_merge_junctions_moves_to_midpoint(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((2.0, 0.0))
        merged = graph.merge_junctions(a.id, b.id)
        assert jnp.allclose(merged.position, jnp.array([1.0, 0.0]))
        assert b.id not in graph.junctions

    def test_merge_junctions_rewires_segments(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((2.0, 0.0))
        c = graph.add_junction((2.0, 2.0))
        seg = graph.add_segment(b.id, c.id)
        graph.merge_junctions(a.id, b.id)
        assert graph.segments[seg.id].junction_start == a.id

    def test_merge_junctions_drops_degenerate_segment(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((2.0, 0.0))
        seg = graph.add_segment(a.id, b.id)
        graph.merge_junctions(a.id, b.id)
        assert seg.id not in graph.segments

    def test_merge_self_raises(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        with pytest.raises(ValueError):
            graph.merge_junctions(a.id, a.id)

    def test_split_junction_creates_two_junctions_and_connecting_segment(self):
        graph = FloorPlanGraph()
        center = graph.add_junction((0.0, 0.0))
        left = graph.add_junction((-1.0, 0.0))
        right = graph.add_junction((1.0, 0.0))
        graph.add_segment(center.id, left.id)
        graph.add_segment(center.id, right.id)

        junction_a, junction_b = graph.split_junction(center.id, direction=(1.0, 0.0))

        assert center.id not in graph.junctions
        assert junction_a.id in graph.junctions
        assert junction_b.id in graph.junctions
        connecting = [
            s
            for s in graph.segments.values()
            if {s.junction_start, s.junction_end} == {junction_a.id, junction_b.id}
        ]
        assert len(connecting) == 1

    def test_split_junction_partitions_incident_segments_by_direction(self):
        graph = FloorPlanGraph()
        center = graph.add_junction((0.0, 0.0))
        left = graph.add_junction((-1.0, 0.0))
        right = graph.add_junction((1.0, 0.0))
        graph.add_segment(center.id, left.id)
        graph.add_segment(center.id, right.id)

        junction_a, junction_b = graph.split_junction(center.id, direction=(1.0, 0.0))

        left_segment = next(
            s for s in graph.segments.values() if left.id in (s.junction_start, s.junction_end)
        )
        right_segment = next(
            s for s in graph.segments.values() if right.id in (s.junction_start, s.junction_end)
        )
        assert junction_b.id in (left_segment.junction_start, left_segment.junction_end)
        assert junction_a.id in (right_segment.junction_start, right_segment.junction_end)

    def test_split_junction_rejects_zero_direction(self):
        graph = FloorPlanGraph()
        center = graph.add_junction((0.0, 0.0))
        with pytest.raises(ValueError):
            graph.split_junction(center.id, direction=(0.0, 0.0))


class TestGeometricQueries:
    def test_incident_segments(self):
        graph = square_room()
        assert set(graph.incident_segments(0)) == {0, 3}

    def test_to_segment_array_shape_and_padding(self):
        graph = square_room()
        arr = graph.to_segment_array()
        assert arr.shape == (K_MAX, 4)
        assert jnp.allclose(arr[4:], 0.0)
        assert jnp.allclose(arr[0], jnp.array([0.0, 0.0, 1.0, 0.0]))

    def test_segment_mask(self):
        graph = square_room()
        mask = graph.segment_mask()
        assert mask.shape == (K_MAX,)
        assert mask[:4].all()
        assert not mask[4:].any()

    def test_to_junction_array_shape_and_padding(self):
        graph = square_room()
        arr = graph.to_junction_array()
        assert arr.shape == (J_MAX, 2)
        assert jnp.allclose(arr[4:], 0.0)

    def test_junction_mask(self):
        graph = square_room()
        mask = graph.junction_mask()
        assert mask.shape == (J_MAX,)
        assert mask[:4].all()
        assert not mask[4:].any()

    def test_to_segment_array_exceeds_k_max_raises(self):
        graph = FloorPlanGraph()
        graph.segments = {i: object() for i in range(K_MAX + 1)}  # length check precedes access
        with pytest.raises(ValueError):
            graph.to_segment_array()

    def test_to_junction_array_exceeds_j_max_raises(self):
        graph = FloorPlanGraph()
        graph.junctions = {i: object() for i in range(J_MAX + 1)}  # length check precedes access
        with pytest.raises(ValueError):
            graph.to_junction_array()

    def test_nearest_segment_distance(self):
        graph = square_room()
        distance = graph.nearest_segment_distance((0.5, -1.0))
        assert jnp.allclose(distance, 1.0)

    def test_nearest_segment_distance_on_segment_is_zero(self):
        graph = square_room()
        distance = graph.nearest_segment_distance((0.5, 0.0))
        assert jnp.allclose(distance, 0.0)

    def test_nearest_segment_distance_empty_graph_is_inf(self):
        graph = FloorPlanGraph()
        distance = graph.nearest_segment_distance((0.0, 0.0))
        assert jnp.isinf(distance)


class TestGraphTheoreticQueries:
    def test_cycles_detects_the_room(self):
        graph = square_room()
        cycles = graph.cycles()
        assert len(cycles) == 1
        assert set(cycles[0]) == {0, 1, 2, 3}

    def test_cycles_empty_for_open_chain(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((1.0, 0.0))
        graph.add_segment(a.id, b.id)
        assert graph.cycles() == []

    def test_is_connected_true_for_square(self):
        assert square_room().is_connected()

    def test_is_connected_false_for_disjoint_components(self):
        graph = square_room()
        a = graph.add_junction((10.0, 10.0))
        b = graph.add_junction((11.0, 10.0))
        graph.add_segment(a.id, b.id)
        assert not graph.is_connected()

    def test_is_connected_true_for_empty_graph(self):
        assert FloorPlanGraph().is_connected()

    def test_to_networkx_nodes_and_edges(self):
        graph = square_room()
        nx_graph = graph.to_networkx()
        assert set(nx_graph.nodes) == {0, 1, 2, 3}
        assert nx_graph.number_of_edges() == 4


class TestValidation:
    def test_is_planar_true_for_square(self):
        assert square_room().is_planar()

    def test_is_planar_false_for_crossing_segments(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((1.0, 1.0))
        c = graph.add_junction((0.0, 1.0))
        d = graph.add_junction((1.0, 0.0))
        graph.add_segment(a.id, b.id)
        graph.add_segment(c.id, d.id)
        assert not graph.is_planar()

    def test_is_planar_true_for_segments_sharing_only_a_junction(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((1.0, 0.0))
        c = graph.add_junction((0.0, 1.0))
        graph.add_segment(a.id, b.id)
        graph.add_segment(a.id, c.id)
        assert graph.is_planar()

    def test_planarity_penalty_zero_when_planar(self):
        assert square_room().planarity_penalty() == 0.0

    def test_planarity_penalty_counts_crossings(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((1.0, 1.0))
        c = graph.add_junction((0.0, 1.0))
        d = graph.add_junction((1.0, 0.0))
        graph.add_segment(a.id, b.id)
        graph.add_segment(c.id, d.id)
        assert graph.planarity_penalty() == 1.0

    def test_length_penalty_zero_for_valid_lengths(self):
        assert square_room().length_penalty() == 0.0

    def test_length_penalty_positive_for_short_segment(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((0.05, 0.0))
        graph.add_segment(a.id, b.id)
        assert graph.length_penalty() > 0.0

    def test_length_penalty_empty_graph_is_zero(self):
        assert FloorPlanGraph().length_penalty() == 0.0

    def test_is_valid_true_for_square(self):
        assert square_room().is_valid()

    def test_is_valid_false_for_dangling_junction(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((1.0, 0.0))
        graph.add_segment(a.id, b.id)
        assert not graph.is_valid()

    def test_is_valid_false_for_short_segment(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((0.05, 0.0))
        c = graph.add_junction((0.05, 1.0))
        graph.add_segment(a.id, b.id)
        graph.add_segment(b.id, c.id)
        graph.add_segment(c.id, a.id)
        assert not graph.is_valid()
