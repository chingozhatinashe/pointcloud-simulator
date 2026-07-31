"""Tests for the within-model RJMCMC moves: inference.rjmcmc.proposals and
inference.rjmcmc.kernel, per docs/inference/rjmcmc_algorithm.md sections 2 and 6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from inference.config import PriorConfig
from inference.rjmcmc.kernel import (
    MoveType,
    RJMCMCConfig,
    StepResult,
    _sample_move_type,
    log_posterior,
    step,
)
from inference.rjmcmc.proposals import (
    _non_adjacent_pairs,
    propose_junction_perturbation,
    propose_segment_birth,
    propose_segment_death,
    propose_segment_perturbation,
)
from representation.floor_plan import FloorPlanGraph
from simulator.engine import FloorPointCloud

NO_PRIOR = PriorConfig(
    use_complexity=False,
    use_orthogonality=False,
    use_min_length=False,
    use_junction_regularity=False,
    use_room_size=False,
)


def _square_room() -> FloorPlanGraph:
    graph = FloorPlanGraph()
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    junctions = [graph.add_junction(corner) for corner in corners]
    for start, end in zip(junctions, junctions[1:] + junctions[:1]):
        graph.add_segment(start.id, end.id)
    return graph


def _square_missing_one_diagonal() -> tuple[FloorPlanGraph, int, int]:
    """Four junctions with all edges connected except one diagonal -- exactly one
    non-adjacent pair, and connecting it crosses the other (already-present)
    diagonal, so segment birth must reject it for non-planarity.
    """
    graph = FloorPlanGraph()
    a = graph.add_junction((0.0, 0.0))
    c = graph.add_junction((2.0, 0.0))
    b = graph.add_junction((2.0, 2.0))
    d = graph.add_junction((0.0, 2.0))
    for start, end in ((a, c), (c, b), (b, d), (d, a), (c, d)):
        graph.add_segment(start.id, end.id)
    return graph, a.id, b.id


def _empty_cloud() -> FloorPointCloud:
    return FloorPointCloud(
        slices=[],
        points_3d=jnp.zeros((0, 3)),
        scanner_positions=jnp.zeros((1, 2)),
        floor_plan_bounds=jnp.array([0.0, 0.0, 1.0, 1.0]),
    )


class _ConstantLikelihood:
    """Returns `value` for every graph -- for isolating the acceptance-ratio math
    from the real likelihood terms.
    """

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def log_prob(self, floor_plan, point_cloud) -> float:
        return self.value

    def log_prob_batched(self, floor_plans, point_cloud):
        return jnp.array([self.value for _ in floor_plans])


class _RejectsCopiesLikelihood:
    """0 for `original` itself, hugely negative for anything else -- forces every
    proposal (always a copy, per FloorPlanGraph.copy) to be rejected.
    """

    def __init__(self, original: FloorPlanGraph) -> None:
        self.original = original

    def log_prob(self, floor_plan, point_cloud) -> float:
        return 0.0 if floor_plan is self.original else -1e9

    def log_prob_batched(self, floor_plans, point_cloud):
        return jnp.array([self.log_prob(fp, point_cloud) for fp in floor_plans])


# --- proposals.py --------------------------------------------------------------


class TestNonAdjacentPairs:
    def test_square_has_two_diagonals_as_candidates(self):
        room = _square_room()
        pairs = _non_adjacent_pairs(room)
        assert len(pairs) == 2

    def test_graph_missing_one_edge_has_exactly_one_candidate(self):
        graph, _, _ = _square_missing_one_diagonal()
        pairs = _non_adjacent_pairs(graph)
        assert len(pairs) == 1


class TestProposeSegmentBirth:
    def test_returns_none_with_fewer_than_two_junctions(self):
        graph = FloorPlanGraph()
        graph.add_junction((0.0, 0.0))
        assert propose_segment_birth(graph, jax.random.PRNGKey(0)) is None

    def test_adds_one_segment_and_stays_planar(self):
        room = _square_room()
        proposal = propose_segment_birth(room, jax.random.PRNGKey(0))
        assert proposal is not None
        assert len(proposal.graph.segments) == len(room.segments) + 1
        assert proposal.graph.is_planar()

    def test_log_proposal_ratio_matches_candidate_counts(self):
        room = _square_room()
        n_pairs_before = len(_non_adjacent_pairs(room))
        proposal = propose_segment_birth(room, jax.random.PRNGKey(0))
        n_segments_after = len(proposal.graph.segments)
        expected = np.log(n_pairs_before) - np.log(n_segments_after)
        assert proposal.log_proposal_ratio == pytest.approx(expected)

    def test_rejects_when_only_candidate_breaks_planarity(self):
        graph, _, _ = _square_missing_one_diagonal()
        for key in range(10):
            assert propose_segment_birth(graph, jax.random.PRNGKey(key)) is None

    def test_does_not_mutate_original_graph(self):
        room = _square_room()
        n_segments_before = len(room.segments)
        propose_segment_birth(room, jax.random.PRNGKey(0))
        assert len(room.segments) == n_segments_before


class TestProposeSegmentDeath:
    def test_returns_none_with_no_segments(self):
        graph = FloorPlanGraph()
        graph.add_junction((0.0, 0.0))
        assert propose_segment_death(graph, jax.random.PRNGKey(0)) is None

    def test_removes_one_segment(self):
        room = _square_room()
        proposal = propose_segment_death(room, jax.random.PRNGKey(0))
        assert proposal is not None
        assert len(proposal.graph.segments) == len(room.segments) - 1

    def test_log_proposal_ratio_matches_candidate_counts(self):
        room = _square_room()
        n_segments_before = len(room.segments)
        proposal = propose_segment_death(room, jax.random.PRNGKey(0))
        n_pairs_after = len(_non_adjacent_pairs(proposal.graph))
        expected = np.log(n_segments_before) - np.log(n_pairs_after)
        assert proposal.log_proposal_ratio == pytest.approx(expected)

    def test_does_not_mutate_original_graph(self):
        room = _square_room()
        n_segments_before = len(room.segments)
        propose_segment_death(room, jax.random.PRNGKey(0))
        assert len(room.segments) == n_segments_before


class TestProposeJunctionPerturbation:
    def test_returns_none_with_no_junctions(self):
        graph = FloorPlanGraph()
        assert propose_junction_perturbation(graph, 0.05, jax.random.PRNGKey(0)) is None

    def test_moves_exactly_one_junction(self):
        room = _square_room()
        proposal = propose_junction_perturbation(room, 0.05, jax.random.PRNGKey(0))
        moved = [
            jid
            for jid in room.junctions
            if not jnp.allclose(
                room.junctions[jid].position, proposal.graph.junctions[jid].position
            )
        ]
        assert len(moved) == 1

    def test_proposal_ratio_is_symmetric(self):
        room = _square_room()
        proposal = propose_junction_perturbation(room, 0.05, jax.random.PRNGKey(0))
        assert proposal.log_proposal_ratio == 0.0

    def test_does_not_mutate_original_graph(self):
        room = _square_room()
        original_positions = {jid: j.position for jid, j in room.junctions.items()}
        propose_junction_perturbation(room, 0.05, jax.random.PRNGKey(0))
        for jid, position in original_positions.items():
            assert jnp.array_equal(room.junctions[jid].position, position)


class TestProposeSegmentPerturbation:
    def test_returns_none_with_no_segments(self):
        graph = FloorPlanGraph()
        assert propose_segment_perturbation(graph, 0.05, jax.random.PRNGKey(0)) is None

    def test_moves_exactly_one_junction_preserving_pivot_distance(self):
        room = _square_room()
        proposal = propose_segment_perturbation(room, 0.2, jax.random.PRNGKey(0))
        moved = [
            jid
            for jid in room.junctions
            if not jnp.allclose(
                room.junctions[jid].position, proposal.graph.junctions[jid].position
            )
        ]
        assert len(moved) == 1
        free_id = moved[0]

        def radius(graph: FloorPlanGraph, other_id: int) -> float:
            return float(
                jnp.linalg.norm(
                    graph.junctions[free_id].position - graph.junctions[other_id].position
                )
            )

        # The moved junction's true pivot is one of its two neighbours in `room`;
        # a rotation about that pivot preserves its distance to `free_id`.
        neighbour_ids = [
            room.segments[sid].other_junction(free_id) for sid in room.incident_segments(free_id)
        ]
        assert any(
            np.isclose(radius(room, n), radius(proposal.graph, n), atol=1e-5) for n in neighbour_ids
        )

    def test_proposal_ratio_is_symmetric(self):
        room = _square_room()
        proposal = propose_segment_perturbation(room, 0.2, jax.random.PRNGKey(0))
        assert proposal.log_proposal_ratio == 0.0


# --- kernel.py -------------------------------------------------------------------


class TestRJMCMCConfig:
    def test_default_probabilities_sum_to_one(self):
        RJMCMCConfig()

    def test_probabilities_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            RJMCMCConfig(p_segment_birth=0.5, p_segment_death=0.5, p_junction_perturb=0.5)

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RJMCMCConfig(p_segment_birth=-0.1, p_segment_death=0.6, p_junction_perturb=0.5)

    @pytest.mark.parametrize("field", ["junction_perturb_sigma", "segment_perturb_sigma"])
    def test_non_positive_sigma_raises(self, field):
        with pytest.raises(ValueError, match=field):
            RJMCMCConfig(**{field: 0.0})


class TestSampleMoveType:
    def test_forced_move_type_is_always_selected(self):
        config = RJMCMCConfig(
            p_segment_birth=1.0, p_segment_death=0.0, p_junction_perturb=0.0, p_segment_perturb=0.0
        )
        for key in range(10):
            assert _sample_move_type(config, jax.random.PRNGKey(key)) is MoveType.SEGMENT_BIRTH


class TestLogPosterior:
    def test_sums_likelihood_and_prior(self):
        room = _square_room()
        cloud = _empty_cloud()
        value = log_posterior(room, cloud, _ConstantLikelihood(-3.5), NO_PRIOR)
        assert value == pytest.approx(-3.5)


class TestStep:
    def test_unproposable_move_is_an_automatic_reject(self):
        graph = FloorPlanGraph()  # no junctions or segments at all
        config = RJMCMCConfig(
            p_segment_birth=0.0, p_segment_death=1.0, p_junction_perturb=0.0, p_segment_perturb=0.0
        )
        result = step(
            graph, _empty_cloud(), _ConstantLikelihood(0.0), NO_PRIOR, config, jax.random.PRNGKey(0)
        )
        assert isinstance(result, StepResult)
        assert result.accepted is False
        assert result.log_acceptance_ratio == float("-inf")
        assert result.graph is graph

    def test_neutral_posterior_symmetric_move_always_accepted(self):
        room = _square_room()
        config = RJMCMCConfig(
            p_segment_birth=0.0, p_segment_death=0.0, p_junction_perturb=1.0, p_segment_perturb=0.0
        )
        for key in range(10):
            result = step(
                room,
                _empty_cloud(),
                _ConstantLikelihood(0.0),
                NO_PRIOR,
                config,
                jax.random.PRNGKey(key),
            )
            assert result.accepted is True
            assert result.log_acceptance_ratio == pytest.approx(0.0)
            assert result.graph is not room

    def test_proposal_always_worse_is_always_rejected(self):
        room = _square_room()
        config = RJMCMCConfig(
            p_segment_birth=0.0, p_segment_death=0.0, p_junction_perturb=1.0, p_segment_perturb=0.0
        )
        likelihood = _RejectsCopiesLikelihood(room)
        for key in range(10):
            result = step(
                room, _empty_cloud(), likelihood, NO_PRIOR, config, jax.random.PRNGKey(key)
            )
            assert result.accepted is False
            assert result.graph is room
