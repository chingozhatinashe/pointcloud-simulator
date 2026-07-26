"""Top-level forward model: `simulate(G, scanner_positions, heights, noise_model, rng_key)`.

Assembles the scanner sweep (`scanner.py`), ray casting (`ray_caster.py`), and noise
pipeline (`noise_model.py`) into the full multi-slice `FloorPointCloud` described in
docs/simulator.md section 4.

`simulate()` takes a `FloorPlanGraph`, which is a plain Python object and therefore not
a valid JAX pytree -- it can be constructed, mutated, and validated, but not traced or
differentiated directly. `simulate_segments()` is the jit-/grad-compatible core that
operates on the raw `(K_MAX, 4)` array export instead; gradient-based MAP optimisation
over junction positions should call it directly with `differentiable=True`, rather than
differentiating through `simulate()`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from representation.floor_plan import FloorPlanGraph
from simulator.noise_model import (
    HeightDependentNoiseModel,
    NoiseModel,
    apply_noise,
    apply_range_noise,
    perturb_directions,
)
from simulator.ray_caster import DEFAULT_BETA, cast_rays
from simulator.scanner import DEFAULT_NUM_RAYS, FULL_SWEEP, ScannerConfig, generate_rays_multi

DEFAULT_HEIGHTS: tuple[float, ...] = (0.3, 1.0, 1.8)  # metres — default slice heights
DEFAULT_VOXEL_SIZE = 0.02  # metres — default downsampling voxel size
DEFAULT_SLICE_TOL = 0.05  # metres — default height slice tolerance


@dataclass
class PointCloudSlice:
    height: float  # metres
    points: jnp.ndarray  # shape (N, 2)


@dataclass
class FloorPointCloud:
    slices: list[PointCloudSlice]
    points_3d: jnp.ndarray  # shape (N_total, 3) — all points with z coordinate
    scanner_positions: jnp.ndarray  # shape (M, 2)
    floor_plan_bounds: jnp.ndarray  # shape (4,) — bounding box (x_min, y_min, x_max, y_max)

    def plot(self, floor_plan: FloorPlanGraph, mode: str = "3d", **kwargs):
        """Dispatch to simulator.visualisation: mode is "3d", "slices", or "coverage".

        Imported lazily to avoid a circular import (visualisation.py imports
        `FloorPointCloud` from this module for type hints).
        """
        from simulator import visualisation

        if mode == "3d":
            return visualisation.plot_3d(self, floor_plan, **kwargs)
        if mode == "slices":
            return visualisation.plot_slices(self, floor_plan, **kwargs)
        if mode == "coverage":
            return visualisation.plot_coverage_heatmap(self.points_3d[:, :2], floor_plan, **kwargs)
        raise ValueError(f"Unknown plot mode: {mode!r}")


def _floor_plan_bounds(floor_plan: FloorPlanGraph) -> jnp.ndarray:
    positions = jnp.stack([junction.position for junction in floor_plan.junctions.values()])
    return jnp.concatenate([jnp.min(positions, axis=0), jnp.max(positions, axis=0)])


@partial(jax.jit, static_argnames=("noise_model", "differentiable", "beta"))
def simulate_segments(
    origins: jnp.ndarray,
    directions: jnp.ndarray,
    segments: jnp.ndarray,
    segment_mask: jnp.ndarray,
    noise_model: NoiseModel,
    bounds: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
    *,
    differentiable: bool = False,
    beta: float = DEFAULT_BETA,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cast every ray against `segments` and apply the post-cast noise pipeline.

    `origins`/`directions` (already scanner-swept and angle-jittered) have shape
    `(M, R, 2)` for `M` scanner positions and `R` rays each. Returns `(points, valid)`
    flattened to shape `(M * R, 2)` / `(M * R,)` -- kept at fixed size throughout so
    this stays traceable; compacting `points[valid]` into the final variable-length
    point cloud happens outside this function, in `simulate()`.

    In `differentiable` mode, missing-return and clutter Bernoulli draws are skipped
    (they are not differentiable per docs/simulator.md section 5) -- only range noise
    is applied, and `valid` reports the underlying (soft) hit mask unchanged.
    """
    batched_cast = jax.vmap(
        lambda o, d: cast_rays(
            o, d, segments, segment_mask, differentiable=differentiable, beta=beta
        ),
        in_axes=(0, 0),
    )
    _, hits, points = batched_cast(origins, directions)

    num_scanners, num_rays = hits.shape
    points = points.reshape(num_scanners * num_rays, 2)
    hits = hits.reshape(num_scanners * num_rays)
    directions_flat = directions.reshape(num_scanners * num_rays, 2)

    if differentiable:
        points = apply_range_noise(points, directions_flat, hits, noise_model.range_sigma, rng_key)
        valid = hits
    else:
        points, valid = apply_noise(points, directions_flat, hits, noise_model, bounds, rng_key)
    return points, valid


def simulate(
    floor_plan: FloorPlanGraph,
    scanner_positions: jnp.ndarray,
    heights: Sequence[float],
    noise_model: NoiseModel | HeightDependentNoiseModel,
    rng_key: jax.random.PRNGKey,
    differentiable: bool = False,
    *,
    num_rays: int = DEFAULT_NUM_RAYS,
    angular_range: tuple[float, float] = FULL_SWEEP,
    beta: float = DEFAULT_BETA,
) -> FloorPointCloud:
    """Simulate a multi-slice LiDAR survey of `floor_plan` from `scanner_positions`.

    `scanner_positions` has shape `(M, 2)`. One `PointCloudSlice` is produced per
    entry in `heights`. In hard-occlusion mode (`differentiable=False`, the default),
    each slice's points are compacted to only the surviving hits; in differentiable
    mode they stay at the full `(M * num_rays, 2)` size (see `simulate_segments`).
    """
    if not floor_plan.junctions:
        raise ValueError("Cannot simulate an empty floor plan (no junctions)")
    if not heights:
        raise ValueError("heights must be non-empty")

    scanner_positions = jnp.asarray(scanner_positions, dtype=jnp.float32)
    segments = floor_plan.to_segment_array()
    segment_mask = floor_plan.segment_mask()
    bounds = _floor_plan_bounds(floor_plan)

    slice_keys = jax.random.split(rng_key, len(heights))
    slices: list[PointCloudSlice] = []
    for height, slice_key in zip(heights, slice_keys):
        jitter_key, cast_key = jax.random.split(slice_key)
        concrete_noise_model = (
            noise_model.at_height(height)
            if isinstance(noise_model, HeightDependentNoiseModel)
            else noise_model
        )

        config = ScannerConfig(height=height, angular_range=angular_range, num_rays=num_rays)
        origins, directions = generate_rays_multi(scanner_positions, config)

        num_scanners, rays_per_scanner, _ = directions.shape
        jittered = perturb_directions(
            directions.reshape(-1, 2), concrete_noise_model.angular_sigma, jitter_key
        )
        directions = jittered.reshape(num_scanners, rays_per_scanner, 2)

        points, valid = simulate_segments(
            origins,
            directions,
            segments,
            segment_mask,
            concrete_noise_model,
            bounds,
            cast_key,
            differentiable=differentiable,
            beta=beta,
        )

        slice_points = points if differentiable else points[valid]
        slices.append(PointCloudSlice(height=float(height), points=slice_points))

    points_3d = jnp.concatenate(
        [
            jnp.column_stack(
                [s.points, jnp.full((s.points.shape[0],), s.height, dtype=jnp.float32)]
            )
            for s in slices
        ]
    )

    return FloorPointCloud(
        slices=slices,
        points_3d=points_3d,
        scanner_positions=scanner_positions,
        floor_plan_bounds=bounds,
    )
