"""Tests for inference.rjmcmc.gibbs: the Gibbs-within-RJMCMC junction-position
refresh, per docs/inference/rjmcmc_algorithm.md section 5.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from inference.config import PriorConfig
from inference.likelihood import (
    AnalyticalLikelihood,
    AnalyticalLikelihoodParams,
    HeightDependentLikelihoodParams,
    analytical_log_prob,
)
from inference.prior import log_prob as prior_log_prob
from inference.rjmcmc.gibbs import (
    _assign_associations,
    _Association,
    _isotropic_gaussian_log_prob,
    _junction_conditional,
    _log_posterior,
    _nearest_segment,
    gibbs_sweep,
)
from representation.floor_plan import FloorPlanGraph
from simulator.engine import FloorPointCloud, PointCloudSlice

NO_PRIOR = PriorConfig(
    use_complexity=False,
    use_orthogonality=False,
    use_min_length=False,
    use_junction_regularity=False,
    use_room_size=False,
)


def _square_room(size: float = 2.0) -> FloorPlanGraph:
    graph = FloorPlanGraph()
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    junctions = [graph.add_junction(corner) for corner in corners]
    for start, end in zip(junctions, junctions[1:] + junctions[:1]):
        graph.add_segment(start.id, end.id)
    return graph


def _cloud(slices: list[PointCloudSlice], bounds=(0.0, 0.0, 2.0, 2.0)) -> FloorPointCloud:
    all_points = jnp.concatenate([s.points for s in slices]) if slices else jnp.zeros((0, 2))
    return FloorPointCloud(
        slices=slices,
        points_3d=jnp.zeros((all_points.shape[0], 3)),
        scanner_positions=jnp.zeros((1, 2)),
        floor_plan_bounds=jnp.array(bounds),
    )


# --- _nearest_segment --------------------------------------------------------------


class TestNearestSegment:
    def test_picks_the_closer_of_two_valid_segments(self):
        segments = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 5.0, 1.0, 5.0]])
        mask = jnp.array([True, True])
        index, distance = _nearest_segment(jnp.array([0.5, 0.1]), segments, mask)
        assert int(index) == 0
        assert float(distance) == pytest.approx(0.1)

    def test_masked_out_segment_is_ignored(self):
        segments = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 5.0, 1.0, 5.0]])
        mask = jnp.array([False, True])
        index, distance = _nearest_segment(jnp.array([0.5, 0.1]), segments, mask)
        assert int(index) == 1
        assert float(distance) == pytest.approx(4.9)


# --- _assign_associations -----------------------------------------------------------


class TestAssignAssociations:
    def test_empty_floor_plan_returns_empty_dict(self):
        graph = FloorPlanGraph()
        point_cloud = _cloud([PointCloudSlice(height=0.3, points=jnp.array([[0.0, 0.0]]))])
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.05, inlier_weight=0.9)
        assert _assign_associations(graph, point_cloud, params) == {}

    def test_every_segment_key_present_even_with_no_points(self):
        graph = _square_room()
        point_cloud = _cloud([])
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.05, inlier_weight=0.9)
        associations = _assign_associations(graph, point_cloud, params)
        assert set(associations) == set(graph.segments)
        assert all(assoc_list == [] for assoc_list in associations.values())

    def test_near_point_is_an_inlier_far_point_is_clutter(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((2.0, 0.0))
        segment = graph.add_segment(a.id, b.id)

        near = jnp.array([1.0, 0.01])  # distance 0.01 from the wall
        far = jnp.array([1.0, 50.0])  # distance 50 from the wall
        point_cloud_slice = PointCloudSlice(height=0.3, points=jnp.stack([near, far]))
        point_cloud = _cloud([point_cloud_slice], bounds=(0.0, 0.0, 2.0, 2.0))
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.05, inlier_weight=0.9)

        associations = _assign_associations(graph, point_cloud, params)
        assigned_points = [assoc.point for assoc in associations[segment.id]]
        assert len(assigned_points) == 1
        assert jnp.allclose(assigned_points[0], near)

    def test_uses_height_dependent_sigma_per_slice(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((2.0, 0.0))
        segment = graph.add_segment(a.id, b.id)

        low_slice = PointCloudSlice(height=0.3, points=jnp.array([[1.0, 0.01]]))
        high_slice = PointCloudSlice(height=1.8, points=jnp.array([[1.0, 0.01]]))
        point_cloud = _cloud([low_slice, high_slice], bounds=(0.0, 0.0, 2.0, 2.0))
        params = HeightDependentLikelihoodParams(
            wall_distance_sigma=lambda h: 0.05 if h < 1.0 else 0.2,
            inlier_weight=lambda h: 0.9,
        )

        associations = _assign_associations(graph, point_cloud, params)
        sigmas = sorted(assoc.sigma for assoc in associations[segment.id])
        assert sigmas == pytest.approx([0.05, 0.2])


# --- _junction_conditional -----------------------------------------------------------


class TestJunctionConditional:
    def test_matches_hand_computed_precision_and_eta(self):
        # x_j = (0,0); two incident segments, one inlier point on each. See
        # docs/inference/rjmcmc_algorithm.md section 5.4 for the formula this
        # reproduces by hand.
        graph = FloorPlanGraph()
        j0 = graph.add_junction((0.0, 0.0))
        j1 = graph.add_junction((10.0, 0.0))
        j2 = graph.add_junction((0.0, 10.0))
        seg_a = graph.add_segment(j0.id, j1.id)
        seg_b = graph.add_segment(j0.id, j2.id)

        associations = {
            seg_a.id: [_Association(point=jnp.array([3.0, 1.0]), sigma=0.1)],
            seg_b.id: [_Association(point=jnp.array([1.0, 4.0]), sigma=0.2)],
        }

        precision, eta = _junction_conditional(graph, j0.id, associations)

        # seg_a: t=0.3, c=0.7 -> precision 0.49/0.01=49, eta 70*(0,1)=(0,70)
        # seg_b: t=0.4, c=0.6 -> precision 0.36/0.04=9,  eta 15*(1,0)=(15,0)
        assert precision == pytest.approx(58.0, rel=1e-5)
        assert jnp.allclose(eta, jnp.array([15.0, 70.0]), atol=1e-4)

    def test_no_associations_gives_zero_precision_and_eta(self):
        graph = _square_room()
        junction_id = next(iter(graph.junctions))
        associations = {sid: [] for sid in graph.segments}
        precision, eta = _junction_conditional(graph, junction_id, associations)
        assert precision == 0.0
        assert jnp.allclose(eta, jnp.zeros(2))

    def test_zero_length_segment_uses_t_equal_to_zero(self):
        graph = FloorPlanGraph()
        j0 = graph.add_junction((0.0, 0.0))
        j1 = graph.add_junction((0.0, 0.0))  # coincides with j0 -> edge_len_sq == 0
        segment = graph.add_segment(j0.id, j1.id)
        associations = {segment.id: [_Association(point=jnp.array([2.0, 3.0]), sigma=0.5)]}

        precision, eta = _junction_conditional(graph, j0.id, associations)

        # t=0 -> c=1: a direct vote for x_j at the point itself
        assert precision == pytest.approx(1.0 / 0.25)
        assert jnp.allclose(eta, jnp.array([2.0, 3.0]) / 0.25)


# --- _isotropic_gaussian_log_prob -----------------------------------------------------


class TestIsotropicGaussianLogProb:
    def test_matches_formula_at_the_mean(self):
        mean = jnp.array([1.0, 2.0])
        variance = 0.5
        expected = -np.log(2 * np.pi * variance)
        assert _isotropic_gaussian_log_prob(mean, mean, variance) == pytest.approx(expected)

    def test_matches_formula_away_from_the_mean(self):
        x = jnp.array([1.0, 1.0])
        mean = jnp.array([0.0, 0.0])
        variance = 2.0
        expected = -np.log(2 * np.pi * variance) - 0.5 * (1.0 + 1.0) / variance
        assert _isotropic_gaussian_log_prob(x, mean, variance) == pytest.approx(expected)


# --- _log_posterior ------------------------------------------------------------------


class TestLogPosterior:
    def test_non_planar_graph_is_negative_infinity(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((1.0, 1.0))
        c = graph.add_junction((0.0, 1.0))
        d = graph.add_junction((1.0, 0.0))
        graph.add_segment(a.id, b.id)
        graph.add_segment(c.id, d.id)  # crosses the first segment, no shared junction
        assert not graph.is_planar()

        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.1, inlier_weight=0.9)
        assert _log_posterior(graph, _cloud([]), params, NO_PRIOR) == float("-inf")

    def test_prior_violation_is_negative_infinity(self):
        graph = FloorPlanGraph()
        a = graph.add_junction((0.0, 0.0))
        b = graph.add_junction((0.01, 0.0))  # shorter than min_segment_length below
        graph.add_segment(a.id, b.id)
        assert graph.is_planar()

        prior_config = PriorConfig(
            use_complexity=False,
            use_orthogonality=False,
            use_min_length=True,
            min_segment_length=0.1,
            use_junction_regularity=False,
            use_room_size=False,
        )
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.1, inlier_weight=0.9)
        assert _log_posterior(graph, _cloud([]), params, prior_config) == float("-inf")

    def test_matches_sum_of_likelihood_and_prior_for_a_valid_graph(self):
        graph = _square_room()
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.1, inlier_weight=0.9)
        point_cloud = _cloud([PointCloudSlice(height=1.0, points=jnp.array([[0.0, 0.5]]))])

        expected = analytical_log_prob(graph, point_cloud, params) + prior_log_prob(graph, NO_PRIOR)
        assert _log_posterior(graph, point_cloud, params, NO_PRIOR) == pytest.approx(expected)


# --- gibbs_sweep ----------------------------------------------------------------------


class TestGibbsSweep:
    def test_junction_with_no_incident_segments_is_left_unchanged(self):
        graph = FloorPlanGraph()
        graph.add_junction((0.0, 0.0))  # no segments -> no evidence is possible
        likelihood = AnalyticalLikelihood(
            AnalyticalLikelihoodParams(wall_distance_sigma=0.05, inlier_weight=0.9)
        )
        result = gibbs_sweep(graph, _cloud([]), likelihood, NO_PRIOR, jax.random.PRNGKey(0))

        assert len(result.junctions) == 1
        junction_id = next(iter(graph.junctions))
        assert jnp.array_equal(
            result.junctions[junction_id].position, graph.junctions[junction_id].position
        )

    def test_does_not_mutate_the_input_graph(self):
        graph = _square_room()
        original_positions = {jid: j.position for jid, j in graph.junctions.items()}
        likelihood = AnalyticalLikelihood(
            AnalyticalLikelihoodParams(wall_distance_sigma=0.02, inlier_weight=0.999)
        )
        points = jnp.array([[1.0, 0.0], [1.0, 2.0], [0.0, 1.0], [2.0, 1.0]])
        point_cloud = _cloud(
            [PointCloudSlice(height=0.3, points=points)], bounds=(-1.0, -1.0, 3.0, 3.0)
        )
        gibbs_sweep(graph, point_cloud, likelihood, NO_PRIOR, jax.random.PRNGKey(0))

        for jid, position in original_positions.items():
            assert jnp.array_equal(graph.junctions[jid].position, position)

    def test_preserves_junction_and_segment_ids(self):
        graph = _square_room()
        likelihood = AnalyticalLikelihood(
            AnalyticalLikelihoodParams(wall_distance_sigma=0.02, inlier_weight=0.999)
        )
        points = jnp.array([[1.0, 0.0], [1.0, 2.0], [0.0, 1.0], [2.0, 1.0]])
        point_cloud = _cloud(
            [PointCloudSlice(height=0.3, points=points)], bounds=(-1.0, -1.0, 3.0, 3.0)
        )
        result = gibbs_sweep(graph, point_cloud, likelihood, NO_PRIOR, jax.random.PRNGKey(0))

        assert set(result.junctions) == set(graph.junctions)
        assert set(result.segments) == set(graph.segments)

    def test_pulls_a_perturbed_corner_back_toward_strong_wall_evidence(self):
        """A rectangular room with one corner displaced from its true position, and
        dense low-noise evidence along the true walls -- the sweep should move the
        displaced corner closer to the true corner it was perturbed away from.
        """
        true_corners = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        graph = FloorPlanGraph()
        junctions = [graph.add_junction(corner) for corner in true_corners]
        for start, end in zip(junctions, junctions[1:] + junctions[:1]):
            graph.add_segment(start.id, end.id)

        moved_id = junctions[0].id
        displaced_position = jnp.array([0.15, -0.1])
        graph.move_junction(moved_id, displaced_position)
        initial_distance = float(jnp.linalg.norm(displaced_position))

        bottom_wall = jnp.stack([jnp.array([x, 0.0]) for x in jnp.linspace(0.5, 4.5, 40)])
        left_wall = jnp.stack([jnp.array([0.0, y]) for y in jnp.linspace(0.5, 4.5, 40)])
        points = jnp.concatenate([bottom_wall, left_wall])
        point_cloud = _cloud(
            [PointCloudSlice(height=0.3, points=points)], bounds=(-2.0, -2.0, 7.0, 7.0)
        )

        likelihood = AnalyticalLikelihood(
            AnalyticalLikelihoodParams(wall_distance_sigma=0.05, inlier_weight=0.999)
        )

        for seed in range(5):
            result = gibbs_sweep(graph, point_cloud, likelihood, NO_PRIOR, jax.random.PRNGKey(seed))
            final_distance = float(jnp.linalg.norm(result.junctions[moved_id].position))
            assert final_distance < initial_distance
