"""Validation and calibration utilities: measuring and reducing the sim-to-real gap.

Implements docs/simulator.md section 6: Chamfer distance as the primary comparison
metric, per-wall point density and cross-height consistency checks, and a simple
noise-model calibration workflow against a real scan of a known floor plan.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from representation.floor_plan import FloorPlanGraph
from simulator.engine import FloorPointCloud, simulate
from simulator.noise_model import NoiseModel, NoiseModelPrior


def _point_to_segment_distance(point: jnp.ndarray, segment: jnp.ndarray) -> jnp.ndarray:
    """Distance from `point` to the segment (x1, y1, x2, y2)."""
    start, end = segment[:2], segment[2:]
    edge = end - start
    edge_len_sq = jnp.dot(edge, edge)
    t = jnp.where(
        edge_len_sq > 0, jnp.dot(point - start, edge) / jnp.maximum(edge_len_sq, 1e-12), 0.0
    )
    t = jnp.clip(t, 0.0, 1.0)
    projection = start + t * edge
    return jnp.linalg.norm(point - projection)


def chamfer_distance(simulated: FloorPointCloud, real: FloorPointCloud) -> float:
    """Symmetric Chamfer distance between two point clouds' `points_3d` arrays.

    The mean nearest-neighbour distance from each simulated point to the real cloud,
    averaged with the same computed in the other direction. This is `O(N_sim *
    N_real)`; fine for per-slice calibration point counts, not intended for huge
    point clouds.
    """
    sim_points = simulated.points_3d
    real_points = real.points_3d
    if sim_points.shape[0] == 0 or real_points.shape[0] == 0:
        raise ValueError("Cannot compute Chamfer distance against an empty point cloud")

    pairwise = jnp.linalg.norm(sim_points[:, None, :] - real_points[None, :, :], axis=-1)
    sim_to_real = jnp.min(pairwise, axis=1)
    real_to_sim = jnp.min(pairwise, axis=0)
    return float(0.5 * (jnp.mean(sim_to_real) + jnp.mean(real_to_sim)))


def point_density_by_wall(points: jnp.ndarray, floor_plan: FloorPlanGraph) -> dict[int, float]:
    """Points-per-metre density along each wall segment in `floor_plan`.

    Assigns every point in `points` (N, 2) to its nearest wall segment (by point-to-
    segment distance), then divides the count landing on each segment by that
    segment's length. Returns a mapping from segment id to density; useful for
    spotting under-sampled walls in simulator output.
    """
    if not floor_plan.segments:
        return {}
    segment_ids = sorted(floor_plan.segments)
    if points.shape[0] == 0:
        return dict.fromkeys(segment_ids, 0.0)

    segments = floor_plan.to_segment_array()
    mask = floor_plan.segment_mask()
    points = jnp.asarray(points, dtype=jnp.float32)

    distances = jax.vmap(
        lambda point: jax.vmap(_point_to_segment_distance, in_axes=(None, 0))(point, segments)
    )(points)
    distances = jnp.where(mask[None, :], distances, jnp.inf)
    nearest = jnp.argmin(distances, axis=1)
    counts = jnp.bincount(nearest, length=len(segment_ids))

    return {
        segment_id: float(counts[i])
        / float(floor_plan.segments[segment_id].length(floor_plan.junctions))
        for i, segment_id in enumerate(segment_ids)
    }


def height_slice_consistency(point_cloud: FloorPointCloud, floor_plan: FloorPlanGraph) -> float:
    """Mean, across walls, of the standard deviation of per-wall density across slices.

    A consistent scan shows similar coverage of each wall regardless of slice height;
    a high value flags occlusion or a badly calibrated noise model at some height.
    Zero means every wall was sampled with identical density at every height.
    """
    if not point_cloud.slices:
        raise ValueError("point_cloud has no slices")
    segment_ids = sorted(floor_plan.segments)
    if not segment_ids:
        return 0.0

    per_slice_density = jnp.stack(
        [
            jnp.array([point_density_by_wall(sl.points, floor_plan)[sid] for sid in segment_ids])
            for sl in point_cloud.slices
        ]
    )
    return float(jnp.mean(jnp.std(per_slice_density, axis=0)))


def calibrate_noise_model(
    known_floor_plan: FloorPlanGraph,
    real_point_cloud: FloorPointCloud,
    scanner_positions: jnp.ndarray,
    noise_prior: NoiseModelPrior,
    rng_key: jax.random.PRNGKey,
    *,
    num_candidates: int = 50,
) -> NoiseModel:
    """Calibrate a `NoiseModel` to minimise Chamfer distance against a real scan.

    Draws `num_candidates` samples from `noise_prior`, simulates `known_floor_plan`
    with each at the same slice heights as `real_point_cloud`, and returns the
    candidate with the lowest Chamfer distance to `real_point_cloud`.

    This is a simple random search rather than gradient-based optimisation, since
    missing-return and clutter rates enter the simulator through non-differentiable
    Bernoulli draws (see `simulator.noise_model`).
    """
    heights = [s.height for s in real_point_cloud.slices]
    if not heights:
        raise ValueError("real_point_cloud has no slices to calibrate against")
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be at least 1, got {num_candidates}")

    best_model: NoiseModel | None = None
    best_distance = float("inf")
    for candidate_key in jax.random.split(rng_key, num_candidates):
        sample_key, sim_key = jax.random.split(candidate_key)
        candidate = noise_prior.sample(sample_key)
        simulated = simulate(known_floor_plan, scanner_positions, heights, candidate, sim_key)
        distance = chamfer_distance(simulated, real_point_cloud)
        if distance < best_distance:
            best_distance = distance
            best_model = candidate

    return best_model
