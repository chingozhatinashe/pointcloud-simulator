"""Tests for inference.likelihood: the analytical and simulator-based likelihoods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from inference.likelihood import (
    AnalyticalLikelihood,
    AnalyticalLikelihoodParams,
    HeightDependentLikelihoodParams,
    SimulatorLikelihood,
    SimulatorLikelihoodConfig,
    analytical_log_prob,
    point_log_prob,
    slice_log_prob,
)
from representation.floor_plan import FloorPlanGraph
from simulator.calibration import chamfer_distance
from simulator.engine import FloorPointCloud, PointCloudSlice, simulate
from simulator.noise_model import NoiseModel

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


def _cloud(slices: list[PointCloudSlice], bounds=(0.0, 0.0, 2.0, 2.0)) -> FloorPointCloud:
    all_points = jnp.concatenate([s.points for s in slices]) if slices else jnp.zeros((0, 2))
    return FloorPointCloud(
        slices=slices,
        points_3d=jnp.zeros((all_points.shape[0], 3)),
        scanner_positions=jnp.zeros((1, 2)),
        floor_plan_bounds=jnp.array(bounds),
    )


class TestAnalyticalLikelihoodParams:
    def test_defaults_construct_successfully(self):
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.05, inlier_weight=0.9)
        assert params.wall_distance_sigma == 0.05
        assert params.inlier_weight == 0.9

    def test_non_positive_sigma_raises(self):
        with pytest.raises(ValueError, match="wall_distance_sigma"):
            AnalyticalLikelihoodParams(wall_distance_sigma=0.0, inlier_weight=0.9)

    @pytest.mark.parametrize("weight", [-0.1, 1.1])
    def test_weight_out_of_range_raises(self, weight):
        with pytest.raises(ValueError, match="inlier_weight"):
            AnalyticalLikelihoodParams(wall_distance_sigma=0.05, inlier_weight=weight)


class TestHeightDependentLikelihoodParams:
    def test_at_height_evaluates_both_callables(self):
        params = HeightDependentLikelihoodParams(
            wall_distance_sigma=lambda h: 0.05 if h < 1.0 else 0.2,
            inlier_weight=lambda h: 0.95 if h < 1.0 else 0.8,
        )
        low = params.at_height(0.3)
        high = params.at_height(1.8)
        assert low == AnalyticalLikelihoodParams(wall_distance_sigma=0.05, inlier_weight=0.95)
        assert high == AnalyticalLikelihoodParams(wall_distance_sigma=0.2, inlier_weight=0.8)


class TestPointLogProb:
    def test_matches_mixture_formula_at_zero_distance(self):
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.1, inlier_weight=0.9)
        clutter_density = 0.25
        expected = jnp.log(0.9 / (0.1 * jnp.sqrt(2 * jnp.pi)) + 0.1 * clutter_density)
        actual = point_log_prob(jnp.array([0.0]), params, clutter_density)
        assert actual[0] == pytest.approx(float(expected), abs=1e-6)

    def test_far_distance_approaches_pure_clutter_term(self):
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.01, inlier_weight=0.9)
        clutter_density = 0.25
        actual = point_log_prob(jnp.array([100.0]), params, clutter_density)
        expected = jnp.log(0.1 * clutter_density)
        assert actual[0] == pytest.approx(float(expected), abs=1e-6)


class TestSliceLogProb:
    def test_empty_slice_is_zero(self):
        floor_plan = _square_room()
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.1, inlier_weight=0.9)
        empty_slice = PointCloudSlice(height=1.0, points=jnp.zeros((0, 2)))
        bounds = jnp.array([0.0, 0.0, 2.0, 2.0])
        assert slice_log_prob(floor_plan, empty_slice, params, bounds) == pytest.approx(0.0)

    def test_matches_sum_of_point_log_prob(self):
        floor_plan = _square_room()
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.1, inlier_weight=0.9)
        bounds = jnp.array([0.0, 0.0, 2.0, 2.0])
        points = jnp.array([[0.0, 0.5], [1.0, 1.0]])  # on the wall, and the room centre
        point_cloud_slice = PointCloudSlice(height=1.0, points=points)

        distances = jax.vmap(floor_plan.nearest_segment_distance)(points)
        clutter_density = 1.0 / ((2.0 - 0.0) * (2.0 - 0.0))
        expected = jnp.sum(point_log_prob(distances, params, clutter_density))

        actual = slice_log_prob(floor_plan, point_cloud_slice, params, bounds)
        assert actual == pytest.approx(float(expected), abs=1e-6)


class TestAnalyticalLogProb:
    def test_sums_across_slices_with_height_dependent_params(self):
        floor_plan = _square_room()
        params = HeightDependentLikelihoodParams(
            wall_distance_sigma=lambda h: 0.1 if h < 1.0 else 0.2,
            inlier_weight=lambda h: 0.9,
        )
        low_slice = PointCloudSlice(height=0.3, points=jnp.array([[0.0, 0.5]]))
        high_slice = PointCloudSlice(height=1.8, points=jnp.array([[1.0, 1.0]]))
        point_cloud = _cloud([low_slice, high_slice])

        expected = float(
            slice_log_prob(
                floor_plan, low_slice, params.at_height(0.3), point_cloud.floor_plan_bounds
            )
        ) + float(
            slice_log_prob(
                floor_plan, high_slice, params.at_height(1.8), point_cloud.floor_plan_bounds
            )
        )
        assert analytical_log_prob(floor_plan, point_cloud, params) == pytest.approx(
            expected, abs=1e-6
        )


class TestAnalyticalLikelihood:
    def test_log_prob_matches_function(self):
        floor_plan = _square_room()
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.1, inlier_weight=0.9)
        point_cloud = _cloud([PointCloudSlice(height=1.0, points=jnp.array([[0.0, 0.5]]))])

        likelihood = AnalyticalLikelihood(params)
        assert likelihood.log_prob(floor_plan, point_cloud) == pytest.approx(
            analytical_log_prob(floor_plan, point_cloud, params)
        )

    def test_log_prob_batched_matches_individual_calls(self):
        params = AnalyticalLikelihoodParams(wall_distance_sigma=0.1, inlier_weight=0.9)
        floor_plans = [_square_room(2.0), _square_room(3.0)]
        point_cloud = _cloud([PointCloudSlice(height=1.0, points=jnp.array([[0.0, 0.5]]))])

        likelihood = AnalyticalLikelihood(params)
        batched = likelihood.log_prob_batched(floor_plans, point_cloud)
        expected = [likelihood.log_prob(fp, point_cloud) for fp in floor_plans]
        assert batched.shape == (2,)
        assert [pytest.approx(v, abs=1e-6) for v in expected] == [float(x) for x in batched]


class TestSimulatorLikelihoodConfig:
    def test_non_positive_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SimulatorLikelihoodConfig(
                scanner_positions=jnp.array([[1.0, 1.0]]),
                noise_model=ZERO_NOISE,
                temperature=0.0,
            )

    def test_empty_scanner_positions_raises(self):
        with pytest.raises(ValueError, match="scanner_positions"):
            SimulatorLikelihoodConfig(
                scanner_positions=jnp.zeros((0, 2)),
                noise_model=ZERO_NOISE,
                temperature=1.0,
            )


class TestSimulatorLikelihood:
    def _config(self, temperature: float = 1.0) -> SimulatorLikelihoodConfig:
        return SimulatorLikelihoodConfig(
            scanner_positions=jnp.array([[1.0, 1.0]]),
            noise_model=ZERO_NOISE,
            temperature=temperature,
        )

    def test_identical_floor_plan_gives_zero_log_prob(self):
        # zero noise makes simulate() deterministic regardless of rng key
        floor_plan = _square_room()
        config = self._config()
        observed = simulate(
            floor_plan, config.scanner_positions, [1.0], config.noise_model, jax.random.PRNGKey(7)
        )

        likelihood = SimulatorLikelihood(config, jax.random.PRNGKey(0))
        assert likelihood.log_prob(floor_plan, observed) == pytest.approx(0.0, abs=1e-4)

    def test_log_prob_matches_temperature_scaled_chamfer_distance(self):
        floor_plan = _square_room(2.0)
        shifted_floor_plan = _square_room(2.5)
        config = self._config(temperature=2.0)
        observed = simulate(
            shifted_floor_plan,
            config.scanner_positions,
            [1.0],
            config.noise_model,
            jax.random.PRNGKey(7),
        )
        expected_distance = chamfer_distance(
            simulate(
                floor_plan,
                config.scanner_positions,
                [1.0],
                config.noise_model,
                jax.random.PRNGKey(9),
            ),
            observed,
        )

        likelihood = SimulatorLikelihood(config, jax.random.PRNGKey(0))
        assert likelihood.log_prob(floor_plan, observed) == pytest.approx(
            -2.0 * expected_distance, abs=1e-4
        )

    def test_no_slices_raises(self):
        floor_plan = _square_room()
        config = self._config()
        likelihood = SimulatorLikelihood(config, jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match="slices"):
            likelihood.log_prob(floor_plan, _cloud([]))

    def test_log_prob_batched_matches_individual_calls(self):
        floor_plans = [_square_room(2.0), _square_room(2.2)]
        config = self._config()
        observed = simulate(
            _square_room(2.0),
            config.scanner_positions,
            [1.0],
            config.noise_model,
            jax.random.PRNGKey(7),
        )

        likelihood = SimulatorLikelihood(config, jax.random.PRNGKey(0))
        batched = likelihood.log_prob_batched(floor_plans, observed)
        assert batched.shape == (2,)
        # zero noise -> deterministic, so this must match a fresh single-graph run exactly
        single = SimulatorLikelihood(config, jax.random.PRNGKey(0))
        assert float(batched[0]) == pytest.approx(single.log_prob(floor_plans[0], observed))
