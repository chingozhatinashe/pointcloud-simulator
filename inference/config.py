"""InferenceConfig and PriorConfig: experiment configuration, per docs/inference.md
sections 1 and 9.

`PriorConfig` holds the weights for each term of `log P(G)` (each individually
switchable for ablation studies, per section 1) plus the hyperparameters of the
procedural floor-plan sampler used for `G ~ P(G)`. `InferenceConfig` fully specifies
one inference run -- method, likelihood, prior, noise model, and iteration budget --
and is JSON-serialisable so it can be logged alongside every result for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from simulator.noise_model import NoiseModel

VALID_METHODS = frozenset({"rjmcmc", "smc", "abc", "sbi", "map"})
VALID_LIKELIHOODS = frozenset({"analytical", "simulator"})


@dataclass(frozen=True)
class PriorConfig:
    """Weights for each term of `log P(G)`, plus the procedural sampler's hyperparameters.

    Each `use_*` flag independently switches its term on or off for ablation studies;
    a disabled term contributes 0 to `log P(G)` regardless of its weight.
    """

    # term toggles
    use_complexity: bool = True
    use_orthogonality: bool = True
    use_min_length: bool = True
    use_junction_regularity: bool = True
    use_room_size: bool = True

    # complexity prior: log P = -lambda_complexity * |V|
    lambda_complexity: float = 0.1

    # orthogonality prior (Manhattan world assumption)
    lambda_orthogonality: float = 1.0

    # minimum length prior: -inf below this threshold, else 0
    min_segment_length: float = 0.1  # metres

    # junction regularity prior: prefers T- and L-junctions
    lambda_junction: float = 1.0

    # room size prior: log N(area(c); room_area_mean, room_area_std^2) per room
    room_area_mean: float = 15.0  # m^2
    room_area_std: float = 8.0  # m^2

    # procedural sampler hyperparameters, for G ~ P(G)
    room_count_rate: float = 4.0  # Poisson rate for number of rooms
    bounding_box_min_size: float = 3.0  # metres
    bounding_box_max_size: float = 15.0  # metres
    max_bsp_depth: int = 4
    orthogonality_perturbation_sigma: float = 0.05  # radians

    def __post_init__(self) -> None:
        for name in ("lambda_complexity", "lambda_orthogonality", "lambda_junction"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.min_segment_length <= 0:
            raise ValueError(f"min_segment_length must be positive, got {self.min_segment_length}")
        if self.room_area_mean <= 0:
            raise ValueError(f"room_area_mean must be positive, got {self.room_area_mean}")
        if self.room_area_std <= 0:
            raise ValueError(f"room_area_std must be positive, got {self.room_area_std}")
        if self.room_count_rate <= 0:
            raise ValueError(f"room_count_rate must be positive, got {self.room_count_rate}")
        if self.bounding_box_min_size <= 0:
            raise ValueError(
                f"bounding_box_min_size must be positive, got {self.bounding_box_min_size}"
            )
        if self.bounding_box_max_size <= self.bounding_box_min_size:
            raise ValueError(
                f"bounding_box_max_size ({self.bounding_box_max_size}) must exceed "
                f"bounding_box_min_size ({self.bounding_box_min_size})"
            )
        if self.max_bsp_depth < 1:
            raise ValueError(f"max_bsp_depth must be at least 1, got {self.max_bsp_depth}")
        if self.orthogonality_perturbation_sigma < 0:
            raise ValueError(
                "orthogonality_perturbation_sigma must be non-negative, got "
                f"{self.orthogonality_perturbation_sigma}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PriorConfig:
        return cls(**data)


@dataclass(frozen=True)
class InferenceConfig:
    """Fully specifies one inference run, per docs/inference.md section 9."""

    method: str  # "rjmcmc" | "smc" | "abc" | "sbi" | "map"
    likelihood: str  # "analytical" | "simulator"
    prior_config: PriorConfig
    noise_model: NoiseModel
    n_iterations: int
    rng_seed: int
    output_dir: Path
    n_particles: int = 1  # SMC only; ignored by other methods
    gibbs_interval: int = 0  # RJMCMC only; ignored by other methods. 0 disables it.

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(f"method must be one of {sorted(VALID_METHODS)}, got {self.method!r}")
        if self.likelihood not in VALID_LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(VALID_LIKELIHOODS)}, got {self.likelihood!r}"
            )
        if self.n_iterations <= 0:
            raise ValueError(f"n_iterations must be positive, got {self.n_iterations}")
        if self.n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {self.n_particles}")
        if self.gibbs_interval < 0:
            raise ValueError(f"gibbs_interval must be non-negative, got {self.gibbs_interval}")

    def to_dict(self) -> dict:
        """A JSON-serialisable representation, for logging alongside every result."""
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> InferenceConfig:
        payload = dict(data)
        payload["output_dir"] = Path(payload["output_dir"])
        payload["prior_config"] = PriorConfig(**payload["prior_config"])
        payload["noise_model"] = NoiseModel(**payload["noise_model"])
        return cls(**payload)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> InferenceConfig:
        return cls.from_dict(json.loads(text))
