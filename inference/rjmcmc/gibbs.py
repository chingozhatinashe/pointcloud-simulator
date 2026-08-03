"""Gibbs-within-RJMCMC junction-position refresh, per
docs/inference/rjmcmc_algorithm.md section 5.

Given a fixed topology, junction positions are (locally) a Gaussian Markov Random
Field under the analytical likelihood alone (section 5.5): freezing, for the
duration of one sweep, which mixture component explains each point (`z_p`), which
segment it is an inlier of (`s_p`), and each inlier's projection parameter (`t_p`)
onto its segment makes the log-likelihood exactly quadratic in every junction
position. That per-junction conditional `N(mu_j, Sigma_j)` (section 5.4) is *not*
`p(x_j | x_{-j}, E, D)` -- it omits the (non-quadratic) prior terms and treats the
associations as fixed rather than resampled (section 5.6) -- so it is used as an
independence-sampler Metropolis-Hastings proposal rather than a raw draw. The
accept/reject step, scored against the true `P(D|G)P(G)`, absorbs both
approximations exactly, so the chain still targets the true posterior.

`gibbs_sweep` is `GIBBS_SWEEP` from section 5.7: it visits every junction once,
proposing and accepting/rejecting independently, using the latest position of
every other junction (including ones already updated earlier in the same sweep) --
matching the "current values of the others" single-site Gibbs semantics of section
5.1. Associations, however, are computed once per sweep from the graph passed in
and held fixed across every junction visited (section 5.6's "recomputed fresh every
sweep").
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from inference.config import PriorConfig
from inference.likelihood import (
    AnalyticalLikelihood,
    AnalyticalLikelihoodParams,
    HeightDependentLikelihoodParams,
    _bounding_box_area,
    _concrete_params,
    _gaussian_pdf,
    analytical_log_prob,
)
from inference.prior import log_prob as prior_log_prob
from representation.floor_plan import FloorPlanGraph, _point_to_segment_distance
from simulator.engine import FloorPointCloud

_MIN_PRECISION = 1e-9  # guards against a junction with no inlier evidence this sweep


@dataclass(frozen=True)
class _Association:
    """One inlier point (`z_p = inlier`) frozen against the segment it is nearest to
    (`s_p`), with the sigma(h) that applied at its slice height.
    """

    point: jnp.ndarray  # (2,)
    sigma: float


def _nearest_segment(
    point: jnp.ndarray, segments: jnp.ndarray, mask: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Index into `segments` (row-order matching `sorted(floor_plan.segments)`) and
    distance of the nearest valid segment to `point`.
    """
    distances = jax.vmap(_point_to_segment_distance, in_axes=(None, 0))(point, segments)
    distances = jnp.where(mask, distances, jnp.inf)
    index = jnp.argmin(distances)
    return index, distances[index]


def _assign_associations(
    floor_plan: FloorPlanGraph,
    point_cloud: FloorPointCloud,
    params: AnalyticalLikelihoodParams | HeightDependentLikelihoodParams,
) -> dict[int, list[_Association]]:
    """Freeze `z_p` (inlier vs. clutter) and `s_p` (nearest segment) for every
    observed point, per section 5.2: a point is an inlier of its nearest segment iff
    the inlier mixture component's density there is at least the clutter
    component's -- the same mixture split `point_log_prob` (inference/likelihood.py)
    scores continuously, hardened here into the discrete assignment section 5.2's
    data augmentation needs.
    """
    associations: dict[int, list[_Association]] = {sid: [] for sid in floor_plan.segments}
    if not floor_plan.segments:
        return associations

    segment_ids = sorted(floor_plan.segments)
    segments_array = floor_plan.to_segment_array()
    mask = floor_plan.segment_mask()
    clutter_density = 1.0 / _bounding_box_area(point_cloud.floor_plan_bounds)

    for point_cloud_slice in point_cloud.slices:
        points = point_cloud_slice.points
        if points.shape[0] == 0:
            continue
        concrete = _concrete_params(params, point_cloud_slice.height)

        indices, distances = jax.vmap(_nearest_segment, in_axes=(0, None, None))(
            points, segments_array, mask
        )
        inlier_density = concrete.inlier_weight * _gaussian_pdf(
            distances, concrete.wall_distance_sigma
        )
        clutter = (1.0 - concrete.inlier_weight) * clutter_density
        is_inlier = inlier_density >= clutter

        for point, index, inlier in zip(points, indices, is_inlier):
            if not bool(inlier):
                continue
            segment_id = segment_ids[int(index)]
            associations[segment_id].append(
                _Association(point=point, sigma=concrete.wall_distance_sigma)
            )
    return associations


def _junction_conditional(
    floor_plan: FloorPlanGraph,
    junction_id: int,
    associations: dict[int, list[_Association]],
) -> tuple[float, jnp.ndarray]:
    """`(precision, eta)` of `x_j`'s Gaussian conditional, per section 5.4 --
    `Lambda_j = precision * I_2` (isotropic: every per-point contribution is a
    scalar multiple of `I_2`), `eta_j = eta`. `t_p` is recomputed here from the
    *current* positions of `j` and each incident segment's far junction (which may
    already have moved earlier in this sweep), per section 5.3's linearisation.
    """
    x_j = floor_plan.junctions[junction_id].position
    precision = 0.0
    eta_j = jnp.zeros((2,))

    for segment_id in floor_plan.incident_segments(junction_id):
        segment = floor_plan.segments[segment_id]
        other_id = segment.other_junction(junction_id)
        x_o = floor_plan.junctions[other_id].position
        edge = x_o - x_j
        edge_len_sq = float(jnp.dot(edge, edge))

        for assoc in associations.get(segment_id, []):
            t = (
                float(jnp.clip(jnp.dot(assoc.point - x_j, edge) / edge_len_sq, 0.0, 1.0))
                if edge_len_sq > 0
                else 0.0
            )
            c = 1.0 - t
            sigma2 = assoc.sigma**2
            precision += (c**2) / sigma2
            eta_j = eta_j + (c / sigma2) * (assoc.point - t * x_o)

    return precision, eta_j


def _isotropic_gaussian_log_prob(x: jnp.ndarray, mean: jnp.ndarray, variance: float) -> float:
    diff = x - mean
    return float(-jnp.log(2.0 * jnp.pi * variance) - 0.5 * jnp.dot(diff, diff) / variance)


def _log_posterior(
    floor_plan: FloorPlanGraph,
    point_cloud: FloorPointCloud,
    likelihood_params: AnalyticalLikelihoodParams | HeightDependentLikelihoodParams,
    prior_config: PriorConfig,
) -> float:
    """`log P(D|G) + log P(G)`, `-inf` for a non-planar `G` -- the planarity and
    min-length constraints fall out as `-inf` log-prior on violating proposals
    (section 5.6), the same hard-constraint treatment `proposals.py` gives them.
    """
    if not floor_plan.is_planar():
        return float("-inf")
    prior_value = prior_log_prob(floor_plan, prior_config)
    if prior_value == float("-inf"):
        return float("-inf")
    return analytical_log_prob(floor_plan, point_cloud, likelihood_params) + prior_value


def gibbs_sweep(
    floor_plan: FloorPlanGraph,
    point_cloud: FloorPointCloud,
    likelihood: AnalyticalLikelihood,
    prior_config: PriorConfig,
    rng_key: jax.random.PRNGKey,
) -> FloorPlanGraph:
    """`GIBBS_SWEEP` (section 5.7): visit every junction once, proposing
    `x_j' ~ N(mu_j, Sigma_j)` from its likelihood-only conditional as an
    independence-sampler MH proposal and accepting/rejecting against the true
    `P(D|G)P(G)` (section 5.6). A junction with no inlier evidence this sweep
    (`precision < _MIN_PRECISION`) is left unchanged -- its conditional variance is
    unbounded, so there is nothing to propose from.

    Gibbs is specifically an analytical-likelihood refinement (the conditional in
    section 5.4 is derived from the Gaussian mixture likelihood) -- `likelihood`
    must be an `AnalyticalLikelihood`, not the general `LikelihoodModel` protocol
    `inference.rjmcmc.kernel.step` accepts.
    """
    graph = floor_plan.copy()
    associations = _assign_associations(graph, point_cloud, likelihood.params)

    for junction_id in sorted(graph.junctions):
        rng_key, propose_key, accept_key = jax.random.split(rng_key, 3)

        precision, eta_j = _junction_conditional(graph, junction_id, associations)
        if precision < _MIN_PRECISION:
            continue

        variance = 1.0 / precision
        mu_j = eta_j * variance
        current_position = graph.junctions[junction_id].position
        proposed_position = mu_j + jnp.sqrt(variance) * jax.random.normal(propose_key, (2,))

        current_log_posterior = _log_posterior(graph, point_cloud, likelihood.params, prior_config)
        proposed_graph = graph.copy()
        proposed_graph.move_junction(junction_id, proposed_position)
        proposed_log_posterior = _log_posterior(
            proposed_graph, point_cloud, likelihood.params, prior_config
        )

        log_q_current = _isotropic_gaussian_log_prob(current_position, mu_j, variance)
        log_q_proposed = _isotropic_gaussian_log_prob(proposed_position, mu_j, variance)

        log_alpha = min(
            0.0,
            (proposed_log_posterior - current_log_posterior) + (log_q_current - log_q_proposed),
        )
        if float(jnp.log(jax.random.uniform(accept_key))) < log_alpha:
            graph = proposed_graph

    return graph
