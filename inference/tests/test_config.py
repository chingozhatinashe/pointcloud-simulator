"""Tests for inference.config: InferenceConfig and PriorConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.config import VALID_LIKELIHOODS, VALID_METHODS, InferenceConfig, PriorConfig
from simulator.noise_model import NoiseModel


def _noise_model() -> NoiseModel:
    return NoiseModel(
        range_sigma=0.02, angular_sigma=0.01, outlier_rate=0.05, missing_rate=0.1, height_sigma=0.02
    )


def _inference_config(**overrides) -> InferenceConfig:
    defaults = dict(
        method="rjmcmc",
        likelihood="analytical",
        prior_config=PriorConfig(),
        noise_model=_noise_model(),
        n_iterations=1000,
        rng_seed=0,
        output_dir=Path("/tmp/results"),
    )
    defaults.update(overrides)
    return InferenceConfig(**defaults)


class TestPriorConfig:
    def test_defaults_construct_successfully(self):
        config = PriorConfig()
        assert config.use_complexity
        assert config.lambda_complexity == 0.1

    @pytest.mark.parametrize(
        "field", ["lambda_complexity", "lambda_orthogonality", "lambda_junction"]
    )
    def test_rejects_negative_lambda(self, field):
        with pytest.raises(ValueError):
            PriorConfig(**{field: -1.0})

    def test_rejects_non_positive_min_segment_length(self):
        with pytest.raises(ValueError):
            PriorConfig(min_segment_length=0.0)

    def test_rejects_non_positive_room_area_params(self):
        with pytest.raises(ValueError):
            PriorConfig(room_area_mean=0.0)
        with pytest.raises(ValueError):
            PriorConfig(room_area_std=0.0)

    def test_rejects_non_positive_room_count_rate(self):
        with pytest.raises(ValueError):
            PriorConfig(room_count_rate=0.0)

    def test_rejects_inverted_bounding_box_range(self):
        with pytest.raises(ValueError):
            PriorConfig(bounding_box_min_size=10.0, bounding_box_max_size=5.0)

    def test_rejects_non_positive_max_bsp_depth(self):
        with pytest.raises(ValueError):
            PriorConfig(max_bsp_depth=0)

    def test_rejects_negative_perturbation_sigma(self):
        with pytest.raises(ValueError):
            PriorConfig(orthogonality_perturbation_sigma=-0.1)

    def test_to_dict_from_dict_round_trip(self):
        config = PriorConfig(lambda_complexity=0.5, use_room_size=False)
        restored = PriorConfig.from_dict(config.to_dict())
        assert restored == config


class TestInferenceConfig:
    @pytest.mark.parametrize("method", sorted(VALID_METHODS))
    def test_accepts_every_valid_method(self, method):
        config = _inference_config(method=method)
        assert config.method == method

    @pytest.mark.parametrize("likelihood", sorted(VALID_LIKELIHOODS))
    def test_accepts_every_valid_likelihood(self, likelihood):
        config = _inference_config(likelihood=likelihood)
        assert config.likelihood == likelihood

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError):
            _inference_config(method="bogus")

    def test_rejects_unknown_likelihood(self):
        with pytest.raises(ValueError):
            _inference_config(likelihood="bogus")

    def test_rejects_non_positive_n_iterations(self):
        with pytest.raises(ValueError):
            _inference_config(n_iterations=0)

    def test_rejects_non_positive_n_particles(self):
        with pytest.raises(ValueError):
            _inference_config(n_particles=0)

    def test_n_particles_defaults_to_one(self):
        config = _inference_config()
        assert config.n_particles == 1

    def test_rejects_negative_gibbs_interval(self):
        with pytest.raises(ValueError):
            _inference_config(gibbs_interval=-1)

    def test_gibbs_interval_defaults_to_zero(self):
        config = _inference_config()
        assert config.gibbs_interval == 0

    def test_to_dict_serialises_output_dir_as_string(self):
        config = _inference_config(output_dir=Path("results") / "run1")
        data = config.to_dict()
        assert data["output_dir"] == str(Path("results") / "run1")
        assert isinstance(data["prior_config"], dict)
        assert isinstance(data["noise_model"], dict)

    def test_to_dict_from_dict_round_trip(self):
        config = _inference_config(n_particles=32, method="smc")
        restored = InferenceConfig.from_dict(config.to_dict())
        assert restored == config

    def test_to_json_from_json_round_trip(self):
        config = _inference_config()
        restored = InferenceConfig.from_json(config.to_json())
        assert restored == config
