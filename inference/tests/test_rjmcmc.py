"""Tests for the RJMCMC moves: inference.rjmcmc.proposals and
inference.rjmcmc.kernel, per docs/inference/rjmcmc_algorithm.md sections 2-4 and 6.
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
    _pairs_within_threshold,
    _sample_split_radius,
    _split_radius_log_pdf,
    propose_junction_merge,
    propose_junction_perturbation,
    propose_junction_split,
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


def _room_with_nearby_obstacle() -> tuple[FloorPlanGraph, int]:
    """A unit square plus an unrelated vertical wall just past its left edge (at
    x=-0.05, spanning y in [-1,1]) -- close enough that a large perturbation of
    junction `a` (at the origin) reliably sweeps one of its incident segments
    through the obstacle, breaking planarity.
    """
    graph = FloorPlanGraph()
    a = graph.add_junction((0.0, 0.0))
    b = graph.add_junction((1.0, 0.0))
    c = graph.add_junction((1.0, 1.0))
    d = graph.add_junction((0.0, 1.0))
    for start, end in ((a, b), (b, c), (c, d), (d, a)):
        graph.add_segment(start.id, end.id)
    obstacle_a = graph.add_junction((-0.05, -1.0))
    obstacle_b = graph.add_junction((-0.05, 1.0))
    graph.add_segment(obstacle_a.id, obstacle_b.id)
    return graph, a.id


class TestProposeJunctionPerturbation:
    def test_rejects_a_move_that_crosses_an_unrelated_wall(self):
        graph, _ = _room_with_nearby_obstacle()
        results = [
            propose_junction_perturbation(graph, 1.0, jax.random.PRNGKey(key)) for key in range(15)
        ]
        assert any(result is None for result in results)
        assert all(result.graph.is_planar() for result in results if result is not None)

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
    def test_rejects_a_rotation_that_crosses_an_unrelated_wall(self):
        graph, _ = _room_with_nearby_obstacle()
        results = [
            propose_segment_perturbation(graph, 3.0, jax.random.PRNGKey(key)) for key in range(15)
        ]
        assert any(result is None for result in results)
        assert all(result.graph.is_planar() for result in results if result is not None)

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
            p_segment_birth=1.0,
            p_segment_death=0.0,
            p_junction_perturb=0.0,
            p_segment_perturb=0.0,
            p_junction_split=0.0,
            p_junction_merge=0.0,
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
            p_segment_birth=0.0,
            p_segment_death=1.0,
            p_junction_perturb=0.0,
            p_segment_perturb=0.0,
            p_junction_split=0.0,
            p_junction_merge=0.0,
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
            p_segment_birth=0.0,
            p_segment_death=0.0,
            p_junction_perturb=1.0,
            p_segment_perturb=0.0,
            p_junction_split=0.0,
            p_junction_merge=0.0,
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
            p_segment_birth=0.0,
            p_segment_death=0.0,
            p_junction_perturb=1.0,
            p_segment_perturb=0.0,
            p_junction_split=0.0,
            p_junction_merge=0.0,
        )
        likelihood = _RejectsCopiesLikelihood(room)
        for key in range(10):
            result = step(
                room, _empty_cloud(), likelihood, NO_PRIOR, config, jax.random.PRNGKey(key)
            )
            assert result.accepted is False


# --- proposals.py: junction split / merge -----------------------------------------

SPLIT_RADIUS_MEAN = 0.05
MERGE_DISTANCE_THRESHOLD = 0.2
R_MAX = MERGE_DISTANCE_THRESHOLD / 2.0


def _lone_junction() -> FloorPlanGraph:
    graph = FloorPlanGraph()
    graph.add_junction((0.0, 0.0))
    return graph


class TestPairsWithinThreshold:
    def test_filters_by_distance(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((0.05, 0.0))  # within threshold
        c = graph.add_junction((5.0, 0.0))  # far outside threshold
        pairs = _pairs_within_threshold(graph, MERGE_DISTANCE_THRESHOLD)
        assert pairs == [(a.id, b.id)]
        assert (a.id, c.id) not in pairs
        assert (b.id, c.id) not in pairs

    def test_does_not_require_an_existing_segment(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((0.05, 0.0))
        # no segment added between a and b
        assert _pairs_within_threshold(graph, MERGE_DISTANCE_THRESHOLD) == [(a.id, b.id)]


class TestSplitRadius:
    def test_sampled_radius_is_always_within_support(self):
        for key in range(200):
            r = _sample_split_radius(SPLIT_RADIUS_MEAN, R_MAX, jax.random.PRNGKey(key))
            assert 0.0 < r < R_MAX

    def test_log_pdf_matches_truncated_exponential_formula(self):
        r = 0.03
        rate = 1.0 / SPLIT_RADIUS_MEAN
        max_cdf = 1.0 - np.exp(-rate * R_MAX)
        expected = np.log(rate) - rate * r - np.log(max_cdf)
        assert _split_radius_log_pdf(r, SPLIT_RADIUS_MEAN, R_MAX) == pytest.approx(expected)

    def test_log_pdf_decreases_with_radius(self):
        low = _split_radius_log_pdf(0.01, SPLIT_RADIUS_MEAN, R_MAX)
        high = _split_radius_log_pdf(0.08, SPLIT_RADIUS_MEAN, R_MAX)
        assert low > high


class TestProposeJunctionSplit:
    def test_returns_none_with_no_junctions(self):
        graph = FloorPlanGraph()
        proposal = propose_junction_split(
            graph, SPLIT_RADIUS_MEAN, MERGE_DISTANCE_THRESHOLD, jax.random.PRNGKey(0)
        )
        assert proposal is None

    def test_splits_a_lone_junction_into_two_within_threshold(self):
        graph = _lone_junction()
        for key in range(20):
            proposal = propose_junction_split(
                graph, SPLIT_RADIUS_MEAN, MERGE_DISTANCE_THRESHOLD, jax.random.PRNGKey(key)
            )
            assert proposal is not None  # always planar: at most one segment ever exists
            assert len(proposal.graph.junctions) == 2
            assert len(proposal.graph.segments) == 1

            positions = [j.position for j in proposal.graph.junctions.values()]
            separation = float(jnp.linalg.norm(positions[0] - positions[1]))
            assert 0.0 < separation < MERGE_DISTANCE_THRESHOLD

    def test_does_not_mutate_original_graph(self):
        graph = _lone_junction()
        propose_junction_split(
            graph, SPLIT_RADIUS_MEAN, MERGE_DISTANCE_THRESHOLD, jax.random.PRNGKey(0)
        )
        assert len(graph.junctions) == 1
        assert len(graph.segments) == 0


class TestProposeJunctionMerge:
    def test_returns_none_with_no_candidate_pair(self):
        graph = FloorPlanGraph()
        graph.add_junction((0.0, 0.0))
        graph.add_junction((5.0, 0.0))  # far outside merge_distance_threshold
        proposal = propose_junction_merge(
            graph, SPLIT_RADIUS_MEAN, MERGE_DISTANCE_THRESHOLD, jax.random.PRNGKey(0)
        )
        assert proposal is None

    def test_merges_to_the_midpoint(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        graph.add_junction((0.06, 0.0))
        proposal = propose_junction_merge(
            graph, SPLIT_RADIUS_MEAN, MERGE_DISTANCE_THRESHOLD, jax.random.PRNGKey(0)
        )
        assert proposal is not None
        assert len(proposal.graph.junctions) == 1
        merged = next(iter(proposal.graph.junctions.values()))
        assert jnp.allclose(merged.position, jnp.array([0.03, 0.0]), atol=1e-5)
        assert merged.id == a.id  # merge_junctions keeps id_a, per representation.floor_plan

    def test_does_not_mutate_original_graph(self):
        graph = FloorPlanGraph()
        graph.add_junction((0.0, 0.0))
        graph.add_junction((0.06, 0.0))
        propose_junction_merge(
            graph, SPLIT_RADIUS_MEAN, MERGE_DISTANCE_THRESHOLD, jax.random.PRNGKey(0)
        )
        assert len(graph.junctions) == 2


class TestSplitMergeRoundTrip:
    """The core correctness property from docs/inference/rjmcmc_algorithm.md section 3:
    merge is the exact inverse of split, so their log_proposal_ratio values (which each
    fold in the Jacobian, section 4) must be exact negatives of one another.
    """

    def test_log_proposal_ratios_are_exact_negatives(self):
        graph = _lone_junction()
        split_proposal = propose_junction_split(
            graph, SPLIT_RADIUS_MEAN, MERGE_DISTANCE_THRESHOLD, jax.random.PRNGKey(0)
        )
        assert split_proposal is not None
        assert len(split_proposal.graph.junctions) == 2

        # A different key on purpose: with only one candidate pair to merge, the
        # choice is forced regardless of key -- this is exactly the reverse move.
        merge_proposal = propose_junction_merge(
            split_proposal.graph,
            SPLIT_RADIUS_MEAN,
            MERGE_DISTANCE_THRESHOLD,
            jax.random.PRNGKey(999),
        )
        assert merge_proposal is not None
        assert len(merge_proposal.graph.junctions) == 1

        assert merge_proposal.log_proposal_ratio == pytest.approx(
            -split_proposal.log_proposal_ratio, rel=1e-4
        )

        recovered_position = next(iter(merge_proposal.graph.junctions.values())).position
        assert jnp.allclose(recovered_position, jnp.array([0.0, 0.0]), atol=1e-5)


class TestSplitMergeKernelWiring:
    # Both directions must have positive probability -- if the reverse move's
    # probability were 0, `step` correctly forces log_acceptance_ratio = -inf
    # (accepting a move whose reverse can never be proposed would break
    # reversibility), so a "split-only" config isn't a meaningful test of the
    # split path specifically. Both nonzero lets `_sample_move_type` pick either.
    _SPLIT_ONLY = RJMCMCConfig(
        p_segment_birth=0.0,
        p_segment_death=0.0,
        p_junction_perturb=0.0,
        p_segment_perturb=0.0,
        p_junction_split=1.0,
        p_junction_merge=0.0,
    )
    _SPLIT_AND_MERGE = RJMCMCConfig(
        p_segment_birth=0.0,
        p_segment_death=0.0,
        p_junction_perturb=0.0,
        p_segment_perturb=0.0,
        p_junction_split=0.5,
        p_junction_merge=0.5,
    )

    def test_forced_split_move_type_is_always_selected(self):
        for key in range(10):
            assert (
                _sample_move_type(self._SPLIT_ONLY, jax.random.PRNGKey(key))
                is MoveType.JUNCTION_SPLIT
            )

    def test_step_split_either_accepts_to_two_junctions_or_rejects(self):
        graph = _lone_junction()
        saw_split = False
        for key in range(40):
            result = step(
                graph,
                _empty_cloud(),
                _ConstantLikelihood(0.0),
                NO_PRIOR,
                self._SPLIT_AND_MERGE,
                jax.random.PRNGKey(key),
            )
            assert isinstance(result, StepResult)
            if result.move_type is not MoveType.JUNCTION_SPLIT:
                continue
            saw_split = True
            assert np.isfinite(result.log_acceptance_ratio)
            if result.accepted:
                assert len(result.graph.junctions) == 2
            else:
                assert result.graph is graph
        assert saw_split  # sanity check the loop actually exercised the split path

    def test_step_merge_either_accepts_to_one_junction_or_rejects(self):
        graph = FloorPlanGraph()
        graph.add_junction((0.0, 0.0))
        graph.add_junction((0.06, 0.0))
        saw_merge = False
        for key in range(40):
            result = step(
                graph,
                _empty_cloud(),
                _ConstantLikelihood(0.0),
                NO_PRIOR,
                self._SPLIT_AND_MERGE,
                jax.random.PRNGKey(key),
            )
            assert isinstance(result, StepResult)
            if result.move_type is not MoveType.JUNCTION_MERGE:
                continue
            saw_merge = True
            assert np.isfinite(result.log_acceptance_ratio)
            if result.accepted:
                assert len(result.graph.junctions) == 1
            else:
                assert result.graph is graph
        assert saw_merge  # sanity check the loop actually exercised the merge path
