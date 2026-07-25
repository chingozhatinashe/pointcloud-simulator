"""Tests for simulator.noise_model: the probabilistic noise pipeline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simulator.noise_model import (
    HeightDependentNoiseModel,
    NoiseModel,
    NoiseModelPrior,
    apply_clutter,
    apply_missing_returns,
    apply_noise,
    apply_range_noise,
    perturb_directions,
)


class TestNoiseModel:
    def test_valid_construction(self):
        model = NoiseModel(
            range_sigma=0.01,
            angular_sigma=0.001,
            outlier_rate=0.05,
            missing_rate=0.1,
            height_sigma=0.02,
        )
        assert model.range_sigma == 0.01
        assert model.outlier_rate == 0.05

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError):
            NoiseModel(
                range_sigma=-0.01,
                angular_sigma=0.001,
                outlier_rate=0.0,
                missing_rate=0.0,
                height_sigma=0.0,
            )

    def test_rejects_rate_outside_unit_interval(self):
        with pytest.raises(ValueError):
            NoiseModel(
                range_sigma=0.0,
                angular_sigma=0.0,
                outlier_rate=1.5,
                missing_rate=0.0,
                height_sigma=0.0,
            )
        with pytest.raises(ValueError):
            NoiseModel(
                range_sigma=0.0,
                angular_sigma=0.0,
                outlier_rate=0.0,
                missing_rate=-0.1,
                height_sigma=0.0,
            )


class TestHeightDependentNoiseModel:
    def test_at_height_evaluates_callables(self):
        model = HeightDependentNoiseModel(
            range_sigma=lambda h: 0.01 + 0.001 * h,
            angular_sigma=lambda h: 0.001,
            outlier_rate=lambda h: 0.02,
            missing_rate=lambda h: 0.05 if h > 1.5 else 0.01,
            height_sigma=lambda h: 0.02,
        )
        low = model.at_height(0.3)
        high = model.at_height(1.8)

        assert isinstance(low, NoiseModel)
        np.testing.assert_allclose(low.range_sigma, 0.0103, atol=1e-6)
        assert low.missing_rate == 0.01
        assert high.missing_rate == 0.05

    def test_at_height_propagates_invalid_parameter(self):
        model = HeightDependentNoiseModel(
            range_sigma=lambda h: -1.0,
            angular_sigma=lambda h: 0.0,
            outlier_rate=lambda h: 0.0,
            missing_rate=lambda h: 0.0,
            height_sigma=lambda h: 0.0,
        )
        with pytest.raises(ValueError):
            model.at_height(1.0)


class TestNoiseModelPrior:
    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError):
            NoiseModelPrior(
                range_sigma=(0.1, 0.01),
                angular_sigma=(0.0, 0.01),
                outlier_rate=(0.0, 0.1),
                missing_rate=(0.0, 0.1),
                height_sigma=(0.0, 0.01),
            )

    def test_sample_stays_within_bounds(self):
        prior = NoiseModelPrior(
            range_sigma=(0.01, 0.05),
            angular_sigma=(0.0, 0.01),
            outlier_rate=(0.0, 0.2),
            missing_rate=(0.0, 0.3),
            height_sigma=(0.0, 0.02),
        )
        for seed in range(20):
            model = prior.sample(jax.random.PRNGKey(seed))
            assert 0.01 <= model.range_sigma <= 0.05
            assert 0.0 <= model.angular_sigma <= 0.01
            assert 0.0 <= model.outlier_rate <= 0.2
            assert 0.0 <= model.missing_rate <= 0.3
            assert 0.0 <= model.height_sigma <= 0.02

    def test_sample_is_deterministic_given_key(self):
        prior = NoiseModelPrior(
            range_sigma=(0.01, 0.05),
            angular_sigma=(0.0, 0.01),
            outlier_rate=(0.0, 0.2),
            missing_rate=(0.0, 0.3),
            height_sigma=(0.0, 0.02),
        )
        key = jax.random.PRNGKey(42)
        first = prior.sample(key)
        second = prior.sample(key)
        assert first == second

    def test_sample_degenerate_range_returns_fixed_value(self):
        prior = NoiseModelPrior(
            range_sigma=(0.02, 0.02),
            angular_sigma=(0.0, 0.0),
            outlier_rate=(0.0, 0.0),
            missing_rate=(0.0, 0.0),
            height_sigma=(0.0, 0.0),
        )
        model = prior.sample(jax.random.PRNGKey(0))
        np.testing.assert_allclose(model.range_sigma, 0.02, atol=1e-6)
        assert model.outlier_rate == 0.0


class TestPerturbDirections:
    def test_shape_preserved(self):
        directions = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        jittered = perturb_directions(directions, 0.05, jax.random.PRNGKey(0))
        assert jittered.shape == directions.shape

    def test_zero_sigma_leaves_directions_unchanged(self):
        directions = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
        jittered = perturb_directions(directions, 0.0, jax.random.PRNGKey(0))
        np.testing.assert_allclose(np.asarray(jittered), np.asarray(directions), atol=1e-6)

    def test_preserves_unit_norm(self):
        directions = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-0.6, -0.8]])
        jittered = perturb_directions(directions, 0.3, jax.random.PRNGKey(1))
        norms = jnp.linalg.norm(jittered, axis=-1)
        np.testing.assert_allclose(np.asarray(norms), np.ones(4), atol=1e-5)

    def test_deterministic_given_same_key(self):
        directions = jnp.array([[1.0, 0.0]] * 5)
        key = jax.random.PRNGKey(7)
        first = perturb_directions(directions, 0.1, key)
        second = perturb_directions(directions, 0.1, key)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_different_keys_give_different_jitter(self):
        directions = jnp.array([[1.0, 0.0]] * 5)
        first = perturb_directions(directions, 0.1, jax.random.PRNGKey(1))
        second = perturb_directions(directions, 0.1, jax.random.PRNGKey(2))
        assert not np.allclose(np.asarray(first), np.asarray(second))


class TestApplyRangeNoise:
    def test_zero_sigma_leaves_points_unchanged(self):
        points = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        directions = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        hit_mask = jnp.array([True, True])
        noisy = apply_range_noise(points, directions, hit_mask, 0.0, jax.random.PRNGKey(0))
        np.testing.assert_allclose(np.asarray(noisy), np.asarray(points), atol=1e-6)

    def test_perturbation_is_along_ray_direction(self):
        point = jnp.array([[2.0, 1.0]])
        direction = jnp.array([[1.0, 0.0]])
        hit_mask = jnp.array([True])
        noisy = apply_range_noise(point, direction, hit_mask, 0.1, jax.random.PRNGKey(3))
        delta = np.asarray(noisy - point)[0]
        # displacement should be purely along the x-axis (the ray direction)
        np.testing.assert_allclose(delta[1], 0.0, atol=1e-6)
        assert abs(delta[0]) > 0.0

    def test_missed_rays_are_left_unchanged(self):
        points = jnp.array([[5.0, 5.0]])
        directions = jnp.array([[1.0, 0.0]])
        hit_mask = jnp.array([False])
        noisy = apply_range_noise(points, directions, hit_mask, 1.0, jax.random.PRNGKey(4))
        np.testing.assert_allclose(np.asarray(noisy), np.asarray(points), atol=1e-6)


class TestApplyMissingReturns:
    def test_zero_rate_keeps_all_hits(self):
        hits = jnp.array([True, True, False, True])
        result = apply_missing_returns(hits, 0.0, jax.random.PRNGKey(0))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(hits))

    def test_full_rate_drops_all_hits(self):
        hits = jnp.array([True, True, True, True])
        result = apply_missing_returns(hits, 1.0, jax.random.PRNGKey(0))
        assert not bool(jnp.any(result))

    def test_approximately_matches_missing_rate_statistically(self):
        hits = jnp.ones((10_000,), dtype=bool)
        result = apply_missing_returns(hits, 0.3, jax.random.PRNGKey(123))
        survival_rate = float(jnp.mean(result))
        np.testing.assert_allclose(survival_rate, 0.7, atol=0.02)


class TestApplyClutter:
    def test_zero_rate_leaves_points_and_mask_unchanged(self):
        points = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        hits = jnp.array([True, False])
        new_points, new_hits = apply_clutter(
            points, hits, 0.0, (0.0, 0.0, 5.0, 5.0), jax.random.PRNGKey(0)
        )
        np.testing.assert_allclose(np.asarray(new_points), np.asarray(points))
        np.testing.assert_array_equal(np.asarray(new_hits), np.asarray(hits))

    def test_full_rate_marks_everything_valid_and_within_bounds(self):
        points = jnp.zeros((50, 2))
        hits = jnp.zeros((50,), dtype=bool)
        bounds = (0.0, 0.0, 4.0, 2.0)
        new_points, new_hits = apply_clutter(points, hits, 1.0, bounds, jax.random.PRNGKey(5))
        assert bool(jnp.all(new_hits))
        points_np = np.asarray(new_points)
        assert np.all(points_np[:, 0] >= 0.0) and np.all(points_np[:, 0] <= 4.0)
        assert np.all(points_np[:, 1] >= 0.0) and np.all(points_np[:, 1] <= 2.0)

    def test_can_rescue_a_missed_ray(self):
        points = jnp.array([[0.0, 0.0]])
        hits = jnp.array([False])
        _, new_hits = apply_clutter(points, hits, 1.0, (0.0, 0.0, 1.0, 1.0), jax.random.PRNGKey(9))
        assert bool(new_hits[0])


class TestApplyNoise:
    def test_all_zero_noise_is_a_passthrough(self):
        points = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        directions = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        hit_mask = jnp.array([True, True])
        noise_model = NoiseModel(
            range_sigma=0.0, angular_sigma=0.0, outlier_rate=0.0, missing_rate=0.0, height_sigma=0.0
        )
        out_points, out_hits = apply_noise(
            points, directions, hit_mask, noise_model, (0.0, 0.0, 5.0, 5.0), jax.random.PRNGKey(0)
        )
        np.testing.assert_allclose(np.asarray(out_points), np.asarray(points), atol=1e-6)
        np.testing.assert_array_equal(np.asarray(out_hits), np.asarray(hit_mask))

    def test_output_shapes(self):
        points = jnp.zeros((10, 2))
        directions = jnp.tile(jnp.array([[1.0, 0.0]]), (10, 1))
        hit_mask = jnp.ones((10,), dtype=bool)
        noise_model = NoiseModel(
            range_sigma=0.05,
            angular_sigma=0.01,
            outlier_rate=0.1,
            missing_rate=0.1,
            height_sigma=0.02,
        )
        out_points, out_hits = apply_noise(
            points, directions, hit_mask, noise_model, (0.0, 0.0, 5.0, 5.0), jax.random.PRNGKey(1)
        )
        assert out_points.shape == (10, 2)
        assert out_hits.shape == (10,)

    def test_jit_compilable(self):
        points = jnp.zeros((8, 2))
        directions = jnp.tile(jnp.array([[1.0, 0.0]]), (8, 1))
        hit_mask = jnp.ones((8,), dtype=bool)
        noise_model = NoiseModel(
            range_sigma=0.05,
            angular_sigma=0.01,
            outlier_rate=0.1,
            missing_rate=0.1,
            height_sigma=0.02,
        )
        bounds = (0.0, 0.0, 5.0, 5.0)

        jitted = jax.jit(lambda p, d, h, key: apply_noise(p, d, h, noise_model, bounds, key))
        out_points, out_hits = jitted(points, directions, hit_mask, jax.random.PRNGKey(2))
        assert out_points.shape == (8, 2)
        assert out_hits.shape == (8,)
