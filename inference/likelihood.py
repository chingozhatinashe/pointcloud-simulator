"""P(D | G): the analytical and simulator-based likelihood models, per docs/inference.md
section 2.

Both implementations conform to the `LikelihoodModel` protocol from the Public
Interface Summary -- `.log_prob(G, D)` and `.log_prob_batched(Gs, D)` -- so RJMCMC,
SMC, ABC, SBI, and MAP inference can be written against the protocol and have the
concrete likelihood swapped in from outside ("Likelihood is injected, not
hardcoded", per Key Design Constraints).

The analytical likelihood treats each observed point independently as a mixture of
an inlier Gaussian centred on its distance to the nearest wall segment and a uniform
clutter component over the observed point cloud's bounding box:

    P(p | G, h) = alpha(h) * N(d(p, G); 0, sigma(h)^2) + (1 - alpha(h)) * U

`sigma` and `alpha` may be height-dependent (`HeightDependentLikelihoodParams`,
mirroring `simulator.noise_model.HeightDependentNoiseModel`) or fixed
(`AnalyticalLikelihoodParams`). `FloorPlanGraph` is a plain Python object and not a
JAX pytree (see `simulator/engine.py`), so `log_prob_batched` loops over graphs in
Python rather than vmapping -- only the per-slice, per-point distance computation
against a single graph is vectorised, reusing `FloorPlanGraph.nearest_segment_distance`
under an outer `vmap` over points.

The simulator-based likelihood is not a true likelihood but the unnormalised
Chamfer-distance approximation from section 2: it calls the (stochastic) Simulator
and so needs a fresh JAX PRNG key on every evaluation. Since the shared
`LikelihoodModel` protocol has no key parameter, `SimulatorLikelihood` holds one
internally and splits it on each call -- the one deliberate deviation from "explicit
PRNG keys everywhere" needed to conform to the shared interface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from representation.floor_plan import FloorPlanGraph
from simulator.calibration import chamfer_distance
from simulator.engine import FloorPointCloud, PointCloudSlice, simulate
from simulator.noise_model import HeightDependentNoiseModel, NoiseModel


class LikelihoodModel(Protocol):
    """Common interface for P(D | G), evaluated either analytically or via the Simulator."""

    def log_prob(self, floor_plan: FloorPlanGraph, point_cloud: FloorPointCloud) -> float: ...

    def log_prob_batched(
        self, floor_plans: Sequence[FloorPlanGraph], point_cloud: FloorPointCloud
    ) -> jnp.ndarray: ...


# --- analytical likelihood --------------------------------------------------------


@dataclass(frozen=True)
class AnalyticalLikelihoodParams:
    """Fixed sigma(h), alpha(h) for one slice height."""

    wall_distance_sigma: float  # sigma(h), metres -- std of the inlier Gaussian
    inlier_weight: float  # alpha(h) -- mixture weight on the inlier component, in [0, 1]

    def __post_init__(self) -> None:
        if self.wall_distance_sigma <= 0:
            raise ValueError(
                f"wall_distance_sigma must be positive, got {self.wall_distance_sigma}"
            )
        if not 0.0 <= self.inlier_weight <= 1.0:
            raise ValueError(f"inlier_weight must be in [0, 1], got {self.inlier_weight}")


@dataclass(frozen=True)
class HeightDependentLikelihoodParams:
    """sigma(h), alpha(h) as callables of slice height, mirroring
    `simulator.noise_model.HeightDependentNoiseModel`.
    """

    wall_distance_sigma: Callable[[float], float]
    inlier_weight: Callable[[float], float]

    def at_height(self, height: float) -> AnalyticalLikelihoodParams:
        """Evaluate both callables at `height`, returning concrete `AnalyticalLikelihoodParams`."""
        return AnalyticalLikelihoodParams(
            wall_distance_sigma=float(self.wall_distance_sigma(height)),
            inlier_weight=float(self.inlier_weight(height)),
        )


def _concrete_params(
    params: AnalyticalLikelihoodParams | HeightDependentLikelihoodParams, height: float
) -> AnalyticalLikelihoodParams:
    if isinstance(params, HeightDependentLikelihoodParams):
        return params.at_height(height)
    return params


def _gaussian_pdf(distance: jnp.ndarray, sigma: float) -> jnp.ndarray:
    return jnp.exp(-0.5 * (distance / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))


def _bounding_box_area(bounds: jnp.ndarray) -> jnp.ndarray:
    x_min, y_min, x_max, y_max = bounds
    return (x_max - x_min) * (y_max - y_min)


def point_log_prob(
    distances: jnp.ndarray, params: AnalyticalLikelihoodParams, clutter_density: jnp.ndarray
) -> jnp.ndarray:
    """log P(p | G, h) for a batch of nearest-wall `distances`, per the mixture formula."""
    inlier = params.inlier_weight * _gaussian_pdf(distances, params.wall_distance_sigma)
    clutter = (1.0 - params.inlier_weight) * clutter_density
    return jnp.log(inlier + clutter)


def slice_log_prob(
    floor_plan: FloorPlanGraph,
    point_cloud_slice: PointCloudSlice,
    params: AnalyticalLikelihoodParams,
    bounds: jnp.ndarray,
) -> jnp.ndarray:
    """Sum of `point_log_prob` over every point in one slice; zero for an empty slice."""
    points = point_cloud_slice.points
    if points.shape[0] == 0:
        return jnp.array(0.0)
    distances = jax.vmap(floor_plan.nearest_segment_distance)(points)
    clutter_density = 1.0 / _bounding_box_area(bounds)
    return jnp.sum(point_log_prob(distances, params, clutter_density))


def analytical_log_prob(
    floor_plan: FloorPlanGraph,
    point_cloud: FloorPointCloud,
    params: AnalyticalLikelihoodParams | HeightDependentLikelihoodParams,
) -> float:
    """log P(D | G): sum of `slice_log_prob` across every slice, per section 2."""
    total = 0.0
    for point_cloud_slice in point_cloud.slices:
        concrete = _concrete_params(params, point_cloud_slice.height)
        total += float(
            slice_log_prob(floor_plan, point_cloud_slice, concrete, point_cloud.floor_plan_bounds)
        )
    return total


class AnalyticalLikelihood:
    """Closed-form Gaussian-plus-clutter likelihood; no Simulator call, differentiable
    with respect to junction positions via JAX autodiff. The primary likelihood for
    RJMCMC and MAP inference.
    """

    def __init__(
        self, params: AnalyticalLikelihoodParams | HeightDependentLikelihoodParams
    ) -> None:
        self.params = params

    def log_prob(self, floor_plan: FloorPlanGraph, point_cloud: FloorPointCloud) -> float:
        return analytical_log_prob(floor_plan, point_cloud, self.params)

    def log_prob_batched(
        self, floor_plans: Sequence[FloorPlanGraph], point_cloud: FloorPointCloud
    ) -> jnp.ndarray:
        return jnp.array([self.log_prob(floor_plan, point_cloud) for floor_plan in floor_plans])


# --- simulator-based likelihood ---------------------------------------------------


@dataclass(frozen=True)
class SimulatorLikelihoodConfig:
    """Everything the Simulator needs to draw `D_sim`, plus the temperature `gamma`
    controlling how sharply Chamfer distance is converted into a pseudo-log-likelihood.
    Slice heights are not configured here -- each `log_prob` call reuses the observed
    `FloorPointCloud`'s own slice heights, so `D_sim` and `D_obs` are always compared
    slice-for-slice.
    """

    scanner_positions: jnp.ndarray  # (M, 2)
    noise_model: NoiseModel | HeightDependentNoiseModel
    temperature: float  # gamma

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if jnp.asarray(self.scanner_positions).shape[0] == 0:
            raise ValueError("scanner_positions must be non-empty")


class SimulatorLikelihood:
    """Implicit likelihood used by ABC and pseudo-marginal MCMC: approximates
    `P(D_obs | G)` by simulating `D_sim` and comparing to `D_obs` via Chamfer
    distance, per section 2. Not a true (normalised) likelihood.

    Stochastic -- every `log_prob` call draws a fresh simulation. Since the shared
    `LikelihoodModel` protocol takes no PRNG key, this class holds one internally and
    splits it on each call.
    """

    def __init__(self, config: SimulatorLikelihoodConfig, rng_key: jax.random.PRNGKey) -> None:
        self.config = config
        self._rng_key = rng_key

    def _next_key(self) -> jax.random.PRNGKey:
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    def log_prob(self, floor_plan: FloorPlanGraph, point_cloud: FloorPointCloud) -> float:
        heights = [point_cloud_slice.height for point_cloud_slice in point_cloud.slices]
        if not heights:
            raise ValueError("point_cloud has no slices to compare against")
        simulated = simulate(
            floor_plan,
            self.config.scanner_positions,
            heights,
            self.config.noise_model,
            self._next_key(),
        )
        distance = chamfer_distance(simulated, point_cloud)
        return -self.config.temperature * distance

    def log_prob_batched(
        self, floor_plans: Sequence[FloorPlanGraph], point_cloud: FloorPointCloud
    ) -> jnp.ndarray:
        return jnp.array([self.log_prob(floor_plan, point_cloud) for floor_plan in floor_plans])
