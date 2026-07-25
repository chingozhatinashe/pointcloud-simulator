"""NoiseModel: converts clean ray-cast intersections into noisy point cloud observations.

Implements the noise pipeline from docs/simulator.md section 3 as a probabilistic
program -- every stochastic step is an explicit `jax.random.PRNGKey` draw, so results
are fully reproducible and auditable.

`NoiseModel` is a plain fixed-parameter container used by every simulation call.
`HeightDependentNoiseModel` lets those parameters vary with slice height. Sampling a
`NoiseModel` from a prior distribution (for SBI domain randomisation) is a separate
concern, handled by `NoiseModelPrior.sample()` -- `NoiseModel` itself never randomises
its own parameters, since it has no distribution to draw from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

Bounds = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), metres

_NOISE_FIELDS = ("range_sigma", "angular_sigma", "outlier_rate", "missing_rate", "height_sigma")


@dataclass(frozen=True)
class NoiseModel:
    """Fixed noise parameters for one simulation call."""

    range_sigma: float  # Gaussian noise on return distance (metres)
    angular_sigma: float  # perturbation on ray direction (radians)
    outlier_rate: float  # fraction of spurious clutter returns, in [0, 1]
    missing_rate: float  # fraction of valid hits randomly dropped, in [0, 1]
    height_sigma: float  # uncertainty in height slice assignment (metres)

    def __post_init__(self) -> None:
        for name in ("range_sigma", "angular_sigma", "height_sigma"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        for name in ("outlier_rate", "missing_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class HeightDependentNoiseModel:
    """Noise parameters as callables of slice height `h`, rather than fixed scalars."""

    range_sigma: Callable[[float], float]
    angular_sigma: Callable[[float], float]
    outlier_rate: Callable[[float], float]
    missing_rate: Callable[[float], float]
    height_sigma: Callable[[float], float]

    def at_height(self, height: float) -> NoiseModel:
        """Evaluate every parameter callable at `height`, returning a concrete `NoiseModel`."""
        return NoiseModel(**{name: float(getattr(self, name)(height)) for name in _NOISE_FIELDS})


@dataclass(frozen=True)
class NoiseModelPrior:
    """Independent Uniform(low, high) priors over each `NoiseModel` parameter.

    Used only for domain randomisation during SBI training data generation -- most
    simulation calls should build a `NoiseModel` directly rather than sampling one.
    """

    range_sigma: tuple[float, float]
    angular_sigma: tuple[float, float]
    outlier_rate: tuple[float, float]
    missing_rate: tuple[float, float]
    height_sigma: tuple[float, float]

    def __post_init__(self) -> None:
        for name in _NOISE_FIELDS:
            low, high = getattr(self, name)
            if high < low:
                raise ValueError(f"{name} prior range high ({high}) must be >= low ({low})")

    def sample(self, rng_key: jax.random.PRNGKey) -> NoiseModel:
        """Draw one `NoiseModel`, each parameter independently Uniform(low, high)."""
        keys = jax.random.split(rng_key, len(_NOISE_FIELDS))
        values = {}
        for name, key in zip(_NOISE_FIELDS, keys):
            low, high = getattr(self, name)
            values[name] = float(jax.random.uniform(key, minval=low, maxval=high))
        return NoiseModel(**values)


def perturb_directions(
    directions: jnp.ndarray, angular_sigma: float, rng_key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Rotate each unit ray direction by an independent `N(0, angular_sigma^2)` angle.

    Applied before ray casting to model angular pointing jitter of the scanner.
    """
    thetas = jax.random.normal(rng_key, shape=(directions.shape[0],)) * angular_sigma
    cos_t, sin_t = jnp.cos(thetas), jnp.sin(thetas)
    x, y = directions[:, 0], directions[:, 1]
    return jnp.stack([cos_t * x - sin_t * y, sin_t * x + cos_t * y], axis=-1)


def apply_range_noise(
    points: jnp.ndarray,
    directions: jnp.ndarray,
    hit_mask: jnp.ndarray,
    range_sigma: float,
    rng_key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Perturb each hit point by `N(0, range_sigma^2)` along its (unit) ray direction.

    Points without a hit are left unchanged -- there is no return to add range noise to.
    """
    noise = jax.random.normal(rng_key, shape=(points.shape[0],)) * range_sigma
    noisy_points = points + noise[:, None] * directions
    return jnp.where(hit_mask[:, None], noisy_points, points)


def apply_missing_returns(
    hit_mask: jnp.ndarray, missing_rate: float, rng_key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Drop each valid hit independently with probability `missing_rate`."""
    dropped = jax.random.uniform(rng_key, shape=hit_mask.shape) < missing_rate
    return hit_mask & (~dropped)


def apply_clutter(
    points: jnp.ndarray,
    hit_mask: jnp.ndarray,
    outlier_rate: float,
    bounds: Bounds,
    rng_key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """With probability `outlier_rate`, replace a return with a uniform random point in
    `bounds` -- models multi-path returns, dust, and other spurious clutter. Clutter
    returns are always marked valid, including at rays that otherwise missed every wall.
    """
    select_key, x_key, y_key = jax.random.split(rng_key, 3)
    x_min, y_min, x_max, y_max = bounds
    is_clutter = jax.random.uniform(select_key, shape=hit_mask.shape) < outlier_rate
    clutter_x = jax.random.uniform(x_key, shape=hit_mask.shape, minval=x_min, maxval=x_max)
    clutter_y = jax.random.uniform(y_key, shape=hit_mask.shape, minval=y_min, maxval=y_max)
    clutter_points = jnp.stack([clutter_x, clutter_y], axis=-1)
    new_points = jnp.where(is_clutter[:, None], clutter_points, points)
    new_hit_mask = hit_mask | is_clutter
    return new_points, new_hit_mask


def apply_noise(
    points: jnp.ndarray,
    directions: jnp.ndarray,
    hit_mask: jnp.ndarray,
    noise_model: NoiseModel,
    bounds: Bounds,
    rng_key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the post-cast noise pipeline: range noise, then missing returns, then clutter.

    `points`, `directions`, `hit_mask` come from a hard-occlusion ray cast. Angular
    jitter (`perturb_directions`) is a separate step applied before casting, not here.
    """
    range_key, missing_key, clutter_key = jax.random.split(rng_key, 3)
    points = apply_range_noise(points, directions, hit_mask, noise_model.range_sigma, range_key)
    hit_mask = apply_missing_returns(hit_mask, noise_model.missing_rate, missing_key)
    points, hit_mask = apply_clutter(
        points, hit_mask, noise_model.outlier_rate, bounds, clutter_key
    )
    return points, hit_mask
