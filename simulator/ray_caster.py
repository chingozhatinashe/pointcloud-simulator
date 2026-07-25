"""JAX ray-segment intersection with hard and soft occlusion.

Given a batch of rays (origins, directions) and a padded wall segment array from
`FloorPlanGraph.to_segment_array()`, computes where each ray first strikes a wall.

Hard occlusion (default) takes the nearest intersection via `argmin` -- physically
correct but not differentiable. Soft occlusion approximates the nearest hit with a
softmin over all intersecting segments, differentiable w.r.t. segment geometry.
The two are separate code paths (per docs/simulator.md) and should not be unified.

Every function here is built from `jax.vmap` and array ops only -- no Python loops
over rays or segments -- so `cast_rays` itself is `vmap`-able over an extra leading
batch dimension (e.g. to cast from many scanner positions at once).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

DEFAULT_BETA = 50.0  # softmin temperature; larger -> closer to hard occlusion
_EPS = 1e-12


def _cross2(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a[0] * b[1] - a[1] * b[0]


# point on ray path is defined by origin + t * direction, t > 0 is in front t < 0 is behind source
# point on wall segment is defined by start + s * edge, s in [0, 1]
# this function essentially solves the system of equation : origin + t * direction = start + s * edge
def _intersect_one(
    origin: jnp.ndarray, direction: jnp.ndarray, segment: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Intersect one ray with one segment (x1, y1, x2, y2).

    Returns `(distance, hit, point)`. `distance` is `inf` and `point` is `nan` when
    the ray and segment are parallel, the intersection is behind the origin, or it
    falls outside the segment's extent.
    """
    start, end = segment[:2], segment[2:]
    edge = end - start
    diff = start - origin

    denom = _cross2(direction, edge)
    parallel = jnp.abs(denom) < _EPS
    safe_denom = jnp.where(parallel, 1.0, denom)

    t = _cross2(diff, edge) / safe_denom
    s = _cross2(diff, direction) / safe_denom

    hit = (~parallel) & (t >= 0.0) & (s >= 0.0) & (s <= 1.0)
    distance = jnp.where(hit, t * jnp.linalg.norm(direction), jnp.inf)
    point = jnp.where(hit, origin + t * direction, jnp.nan)
    return distance, hit, point


_intersect_all_segments = jax.vmap(_intersect_one, in_axes=(None, None, 0))


def _cast_one_hard(
    origin: jnp.ndarray, direction: jnp.ndarray, segments: jnp.ndarray, segment_mask: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Nearest-hit distance, hit mask, and point for one ray against all segments."""
    distances, hits, points = _intersect_all_segments(origin, direction, segments)
    distances = jnp.where(segment_mask & hits, distances, jnp.inf)
    nearest = jnp.argmin(distances)
    nearest_distance = distances[nearest]
    hit = jnp.isfinite(nearest_distance)
    point = jnp.where(hit, points[nearest], jnp.nan)
    return jnp.where(hit, nearest_distance, jnp.inf), hit, point


def _cast_one_soft(
    origin: jnp.ndarray,
    direction: jnp.ndarray,
    segments: jnp.ndarray,
    segment_mask: jnp.ndarray,
    beta: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Softmin-weighted distance, hit mask, and point for one ray against all segments.

    Weights are computed relative to the minimum distance (the standard log-sum-exp
    shift): `sum_j(d_j * exp(-beta*(d_j - d_min))) / sum_j(exp(-beta*(d_j - d_min)))`,
    which is algebraically identical to the spec's formula but keeps the dominant
    weight at exactly 1 instead of `exp(-beta*d_min)`, which underflows to zero (and
    NaNs the gradient through 0/0) in float32 for even moderately large `beta`.
    """
    distances, hits, _ = _intersect_all_segments(origin, direction, segments)
    valid = segment_mask & hits
    safe_distances = jnp.where(valid, distances, jnp.inf)

    min_distance = jnp.min(safe_distances)
    hit = jnp.isfinite(min_distance)
    safe_min = jnp.where(hit, min_distance, 0.0)

    shifted = jnp.where(valid, distances - safe_min, 0.0)
    weights = jnp.where(valid, jnp.exp(-beta * shifted), 0.0)
    weight_sum = jnp.sum(weights)
    safe_weight_sum = jnp.where(hit, weight_sum, 1.0)
    soft_distance = safe_min + jnp.sum(weights * shifted) / safe_weight_sum

    unit_direction = direction / jnp.maximum(jnp.linalg.norm(direction), _EPS)
    point = jnp.where(hit, origin + soft_distance * unit_direction, jnp.nan)
    return jnp.where(hit, soft_distance, jnp.inf), hit, point


_cast_batched_hard = jax.vmap(_cast_one_hard, in_axes=(0, 0, None, None))
_cast_batched_soft = jax.vmap(_cast_one_soft, in_axes=(0, 0, None, None, None))


def cast_rays(
    origins: jnp.ndarray,
    directions: jnp.ndarray,
    segments: jnp.ndarray,
    segment_mask: jnp.ndarray | None = None,
    *,
    differentiable: bool = False,
    beta: float = DEFAULT_BETA,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cast `R` rays against `K` wall segments.

    `origins` and `directions` have shape `(R, 2)`; `segments` has shape `(K, 4)`
    as `(x1, y1, x2, y2)` rows, e.g. from `FloorPlanGraph.to_segment_array()`.
    `segment_mask` is a `(K,)` boolean validity mask (all-`True` if omitted), needed
    to ignore zero-padded rows from a fixed-size `K_MAX` export.

    Returns `(distances, hit_mask, points)` of shape `(R,)`, `(R,)`, `(R, 2)`.

    Hard occlusion (default) returns the nearest wall intersection per ray via
    `argmin` -- not differentiable. Soft occlusion (`differentiable=True`) returns
    a `beta`-temperature softmin-weighted distance, differentiable w.r.t. segment
    geometry; it converges to hard occlusion as `beta -> inf`.
    """
    if segment_mask is None:
        segment_mask = jnp.ones((segments.shape[0],), dtype=bool)

    if differentiable:
        distances, hits, points = _cast_batched_soft(
            origins, directions, segments, segment_mask, beta
        )
    else:
        distances, hits, points = _cast_batched_hard(origins, directions, segments, segment_mask)
    return distances, hits, points
