"""Tests for experiments.run_inference: the run_inference() CLI entry point."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from experiments.run_inference import (
    _analytical_params_from_noise_model,
    build_likelihood,
    run_inference,
    sample_synthetic_ground_truth,
    save_result,
)
from inference.config import InferenceConfig, PriorConfig
from inference.likelihood import AnalyticalLikelihood, SimulatorLikelihood
from inference.result import InferenceResult
from simulator.noise_model import NoiseModel

_SMALL_PRIOR = PriorConfig(
    room_count_rate=1.0, bounding_box_min_size=2.0, bounding_box_max_size=4.0, max_bsp_depth=1
)


def _noise_model(**overrides) -> NoiseModel:
    defaults = dict(
        range_sigma=0.02,
        angular_sigma=0.01,
        outlier_rate=0.02,
        missing_rate=0.05,
        height_sigma=0.02,
    )
    defaults.update(overrides)
    return NoiseModel(**defaults)


def _config(output_dir: Path, **overrides) -> InferenceConfig:
    defaults = dict(
        method="rjmcmc",
        likelihood="analytical",
        prior_config=_SMALL_PRIOR,
        noise_model=_noise_model(),
        n_iterations=15,
        rng_seed=0,
        output_dir=output_dir,
    )
    defaults.update(overrides)
    return InferenceConfig(**defaults)


class TestAnalyticalParamsFromNoiseModel:
    def test_matches_the_documented_mapping(self):
        params = _analytical_params_from_noise_model(
            _noise_model(range_sigma=0.1, outlier_rate=0.3)
        )
        assert params.wall_distance_sigma == pytest.approx(0.1)
        assert params.inlier_weight == pytest.approx(0.7)

    def test_zero_range_sigma_is_floored(self):
        params = _analytical_params_from_noise_model(_noise_model(range_sigma=0.0))
        assert params.wall_distance_sigma > 0.0

    def test_high_outlier_rate_floors_inlier_weight(self):
        params = _analytical_params_from_noise_model(_noise_model(outlier_rate=0.99))
        assert params.inlier_weight > 0.0


class TestBuildLikelihood:
    def test_analytical_config_builds_analytical_likelihood(self, tmp_path):
        config = _config(tmp_path, likelihood="analytical")
        g_true, point_cloud = sample_synthetic_ground_truth(config, jax.random.PRNGKey(0))
        likelihood = build_likelihood(config, point_cloud, temperature=10.0)
        assert isinstance(likelihood, AnalyticalLikelihood)

    def test_simulator_config_builds_simulator_likelihood(self, tmp_path):
        config = _config(tmp_path, likelihood="simulator")
        g_true, point_cloud = sample_synthetic_ground_truth(config, jax.random.PRNGKey(0))
        likelihood = build_likelihood(config, point_cloud, temperature=7.5)
        assert isinstance(likelihood, SimulatorLikelihood)
        assert likelihood.config.temperature == pytest.approx(7.5)
        assert jnp.array_equal(likelihood.config.scanner_positions, point_cloud.scanner_positions)


class TestSampleSyntheticGroundTruth:
    def test_produces_a_non_empty_floor_plan_and_matching_point_cloud(self, tmp_path):
        config = _config(tmp_path)
        g_true, point_cloud = sample_synthetic_ground_truth(config, jax.random.PRNGKey(1))
        assert g_true.junctions
        assert g_true.segments
        assert len(point_cloud.slices) == 3  # DEFAULT_HEIGHTS
        assert point_cloud.scanner_positions.shape == (1, 2)

    def test_is_reproducible_given_the_same_key(self, tmp_path):
        config = _config(tmp_path)
        g_1, _ = sample_synthetic_ground_truth(config, jax.random.PRNGKey(2))
        g_2, _ = sample_synthetic_ground_truth(config, jax.random.PRNGKey(2))
        assert len(g_1.junctions) == len(g_2.junctions)
        assert len(g_1.segments) == len(g_2.segments)


class TestRunInference:
    def test_unimplemented_method_raises_not_implemented_error(self, tmp_path):
        config = _config(tmp_path, method="smc")
        with pytest.raises(NotImplementedError, match="smc"):
            run_inference(None, config, None, jax.random.PRNGKey(0))

    def test_rjmcmc_returns_a_valid_inference_result(self, tmp_path):
        config = _config(tmp_path)
        g_true, point_cloud = sample_synthetic_ground_truth(config, jax.random.PRNGKey(3))
        likelihood = build_likelihood(config, point_cloud, temperature=10.0)

        result = run_inference(point_cloud, config, likelihood, jax.random.PRNGKey(4))

        assert isinstance(result, InferenceResult)
        assert result.method == "rjmcmc"
        assert result.samples
        assert result.map_estimate.junctions
        assert result.log_posterior_trace.shape[0] == config.n_iterations
        assert result.runtime_seconds >= 0.0

    def test_gibbs_interval_with_simulator_likelihood_raises(self, tmp_path):
        config = _config(tmp_path, likelihood="simulator", gibbs_interval=3)
        g_true, point_cloud = sample_synthetic_ground_truth(config, jax.random.PRNGKey(5))
        likelihood = build_likelihood(config, point_cloud, temperature=10.0)

        with pytest.raises(ValueError, match="gibbs_interval"):
            run_inference(point_cloud, config, likelihood, jax.random.PRNGKey(6))

    def test_gibbs_interval_with_analytical_likelihood_runs(self, tmp_path):
        config = _config(tmp_path, likelihood="analytical", gibbs_interval=3)
        g_true, point_cloud = sample_synthetic_ground_truth(config, jax.random.PRNGKey(7))
        likelihood = build_likelihood(config, point_cloud, temperature=10.0)

        result = run_inference(point_cloud, config, likelihood, jax.random.PRNGKey(8))
        assert result.method == "rjmcmc"


class TestSaveResult:
    def test_writes_config_summary_and_pickled_result(self, tmp_path):
        config = _config(tmp_path)
        g_true, point_cloud = sample_synthetic_ground_truth(config, jax.random.PRNGKey(9))
        likelihood = build_likelihood(config, point_cloud, temperature=10.0)
        result = run_inference(point_cloud, config, likelihood, jax.random.PRNGKey(10))

        from inference.evaluate import evaluate

        evaluation = evaluate(result, g_true)
        save_result(result, config, tmp_path, evaluation=evaluation)

        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "result.pkl").exists()

        restored_config = InferenceConfig.from_json((tmp_path / "config.json").read_text())
        assert restored_config == config

        import json

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["method"] == "rjmcmc"
        assert summary["n_samples"] == len(result.samples)
        assert "evaluation" in summary

        import pickle

        with (tmp_path / "result.pkl").open("rb") as f:
            restored_result = pickle.load(f)
        assert restored_result.method == result.method
        assert len(restored_result.samples) == len(result.samples)

    def test_creates_output_dir_if_missing(self, tmp_path):
        output_dir = tmp_path / "nested" / "run1"
        config = _config(output_dir)
        g_true, point_cloud = sample_synthetic_ground_truth(config, jax.random.PRNGKey(11))
        likelihood = build_likelihood(config, point_cloud, temperature=10.0)
        result = run_inference(point_cloud, config, likelihood, jax.random.PRNGKey(12))

        save_result(result, config, output_dir)
        assert output_dir.exists()
        assert (output_dir / "result.pkl").exists()
