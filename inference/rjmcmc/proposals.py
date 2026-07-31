"""RJMCMC proposals: segment birth/death, junction/segment perturbation (the
within-model moves, section 2), and junction split/merge (the between-model
moves, sections 3-4) -- per docs/inference/rjmcmc_algorithm.md.

Each propose_* function draws the move's own randomness, applies it to a *copy*
of the graph (the original is never mutated), and returns a `Proposal` bundling
the proposed `FloorPlanGraph` with `log_proposal_ratio` -- everything the
Metropolis-Hastings ratio needs from this move's own candidate-selection
probabilities (and, for split/merge, the dimension-matching Jacobian). The
move-type selection probability and the likelihood/prior ratio are added by
`inference.rjmcmc.kernel`, uniformly across all move types.

For the within-model moves (birth/death/perturbation), `log_proposal_ratio =
log q(reverse | proposed) - log q(forward | current)`; no Jacobian is needed
since Nj is unchanged. For the between-model moves (split/merge),
`log_proposal_ratio` also folds in `log|det dM/d(p,r,theta)| = log(4r)` (split)
or its reciprocal `-log(4r)` (merge) -- see section 3's derivation of `M`.

A move returns `None` when nothing of that type is currently proposable (no
segments to remove, no non-adjacent junction pair to connect, no junction pair
within the merge distance threshold, no junctions at all) or when the proposed
graph fails the planarity check -- `docs/inference.md` "reject proposal
immediately if segment crosses existing segments". The kernel treats `None` as
an automatic reject.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from representation.floor_plan import FloorPlanGraph

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Proposal:
    """A proposed graph plus the move-specific piece of the MH proposal ratio."""

    graph: FloorPlanGraph
    log_proposal_ratio: float  # log q(reverse | proposed) - log q(forward | current)


def _non_adjacent_pairs(floor_plan: FloorPlanGraph) -> list[tuple[int, int]]:
    """Every unordered junction pair with no segment directly connecting them --
    the candidate set segment birth samples from.
    """
    existing = {
        frozenset((segment.junction_start, segment.junction_end))
        for segment in floor_plan.segments.values()
    }
    return [
        (a, b)
        for a, b in itertools.combinations(sorted(floor_plan.junctions), 2)
        if frozenset((a, b)) not in existing
    ]


def _choose_uniform(items: list, rng_key: jax.random.PRNGKey):
    index = int(jax.random.randint(rng_key, (), 0, len(items)))
    return items[index]


def propose_segment_birth(
    floor_plan: FloorPlanGraph, rng_key: jax.random.PRNGKey
) -> Proposal | None:
    """Add a segment between a uniformly-chosen non-adjacent junction pair.

    Sampling is restricted to non-adjacent pairs (rather than "any pair,
    resampling on collision") so that the candidate count used here is exactly
    the one the reverse segment-death move needs -- see `propose_segment_death`.
    """
    candidates = _non_adjacent_pairs(floor_plan)
    if not candidates:
        return None

    junction_a, junction_b = _choose_uniform(candidates, rng_key)

    proposed = floor_plan.copy()
    proposed.add_segment(junction_a, junction_b)
    if not proposed.is_planar():
        return None

    n_pairs_before = len(candidates)
    n_segments_after = len(proposed.segments)
    # forward: 1 / n_pairs_before ; reverse (death picks this new segment): 1 / n_segments_after
    log_proposal_ratio = float(np.log(n_pairs_before) - np.log(n_segments_after))
    return Proposal(graph=proposed, log_proposal_ratio=log_proposal_ratio)


def propose_segment_death(
    floor_plan: FloorPlanGraph, rng_key: jax.random.PRNGKey
) -> Proposal | None:
    """Remove a uniformly-chosen existing segment -- the reverse of segment birth."""
    segment_ids = sorted(floor_plan.segments)
    if not segment_ids:
        return None

    segment_id = _choose_uniform(segment_ids, rng_key)

    proposed = floor_plan.copy()
    proposed.remove_segment(segment_id)

    n_segments_before = len(segment_ids)
    n_pairs_after = len(_non_adjacent_pairs(proposed))
    # forward: 1 / n_segments_before ; reverse (birth picks this pair again): 1 / n_pairs_after
    log_proposal_ratio = float(np.log(n_segments_before) - np.log(n_pairs_after))
    return Proposal(graph=proposed, log_proposal_ratio=log_proposal_ratio)


def propose_junction_perturbation(
    floor_plan: FloorPlanGraph, sigma: float, rng_key: jax.random.PRNGKey
) -> Proposal | None:
    """Move a uniformly-chosen junction by `delta ~ N(0, sigma^2 I)`.

    Symmetric random walk in position -- `log_proposal_ratio` is exactly 0.
    """
    junction_ids = sorted(floor_plan.junctions)
    if not junction_ids:
        return None

    id_key, noise_key = jax.random.split(rng_key)
    junction_id = _choose_uniform(junction_ids, id_key)

    noise = jax.random.normal(noise_key, (2,)) * sigma
    proposed = floor_plan.copy()
    proposed.move_junction(junction_id, floor_plan.junctions[junction_id].position + noise)

    return Proposal(graph=proposed, log_proposal_ratio=0.0)


def propose_segment_perturbation(
    floor_plan: FloorPlanGraph, sigma_rot: float, rng_key: jax.random.PRNGKey
) -> Proposal | None:
    """Rotate a uniformly-chosen segment about one of its two junctions (chosen
    uniformly, independently each call) by `delta_phi ~ N(0, sigma_rot^2)`,
    moving only the other ("free") junction, along the circle of the segment's
    current length centred on the pivot.

    Symmetric: rotating by `-delta_phi` from the proposed state exactly reverses
    the move, `delta_phi` and `-delta_phi` are equally likely under
    `N(0, sigma_rot^2)`, and the pivot is chosen identically (uniform over the
    segment's two junctions) forward and reverse -- so `log_proposal_ratio` is 0.
    """
    segment_ids = sorted(floor_plan.segments)
    if not segment_ids:
        return None

    segment_key, pivot_key, angle_key = jax.random.split(rng_key, 3)
    segment_id = _choose_uniform(segment_ids, segment_key)
    segment = floor_plan.segments[segment_id]

    pivot_is_start = bool(jax.random.bernoulli(pivot_key))
    pivot_id = segment.junction_start if pivot_is_start else segment.junction_end
    free_id = segment.junction_end if pivot_is_start else segment.junction_start

    pivot_position = floor_plan.junctions[pivot_id].position
    free_position = floor_plan.junctions[free_id].position

    delta_phi = jax.random.normal(angle_key) * sigma_rot
    cos_d, sin_d = jnp.cos(delta_phi), jnp.sin(delta_phi)
    rotation = jnp.array([[cos_d, -sin_d], [sin_d, cos_d]])
    offset = free_position - pivot_position
    new_free_position = pivot_position + rotation @ offset

    proposed = floor_plan.copy()
    proposed.move_junction(free_id, new_free_position)

    return Proposal(graph=proposed, log_proposal_ratio=0.0)


# --- junction split / merge (between-model) --------------------------------------


def _pairs_within_threshold(
    floor_plan: FloorPlanGraph, distance_threshold: float
) -> list[tuple[int, int]]:
    """Every unordered junction pair strictly closer than `distance_threshold` --
    the candidate set junction merge samples from. Unlike `_non_adjacent_pairs`,
    a segment may or may not already connect the pair; docs/inference.md's merge
    move has no such requirement.
    """
    ids = sorted(floor_plan.junctions)
    pairs = []
    for a, b in itertools.combinations(ids, 2):
        distance = float(
            jnp.linalg.norm(floor_plan.junctions[a].position - floor_plan.junctions[b].position)
        )
        if distance < distance_threshold:
            pairs.append((a, b))
    return pairs


def _sample_split_radius(mean: float, r_max: float, rng_key: jax.random.PRNGKey) -> float:
    """`r ~ Exponential(mean)` truncated to `(0, r_max)`, via inverse-CDF sampling.

    Truncated (rather than plain Exponential) so that the split offset always
    satisfies `2r < r_max * 2`, i.e. lands within the merge move's own distance
    threshold -- guaranteeing the reverse merge can always recover this exact
    split as one of its candidates. See `_split_radius_log_pdf` for the matching
    density, and section 3 of docs/inference/rjmcmc_algorithm.md for `M(p,r,theta)`.
    """
    rate = 1.0 / mean
    max_cdf = 1.0 - float(jnp.exp(-rate * r_max))
    u = float(jax.random.uniform(rng_key, minval=0.0, maxval=max_cdf))
    return -np.log(1.0 - u) / rate


def _split_radius_log_pdf(r: float, mean: float, r_max: float) -> float:
    """log q_r(r) for the truncated-Exponential proposal `_sample_split_radius` draws from."""
    rate = 1.0 / mean
    max_cdf = 1.0 - np.exp(-rate * r_max)
    return float(np.log(rate) - rate * r - np.log(max_cdf))


def propose_junction_split(
    floor_plan: FloorPlanGraph,
    split_radius_mean: float,
    merge_distance_threshold: float,
    rng_key: jax.random.PRNGKey,
) -> Proposal | None:
    """Split a uniformly-chosen junction into two via `M(p, r, theta)` (section 3):
    `r` drawn from `_sample_split_radius` (truncated to keep the new pair within
    `merge_distance_threshold` of each other) and `theta ~ Uniform(0, 2*pi)`.

    Rejected (returns `None`) if there are no junctions, or if the split breaks
    planarity.
    """
    junction_ids = sorted(floor_plan.junctions)
    if not junction_ids:
        return None

    id_key, radius_key, angle_key = jax.random.split(rng_key, 3)
    junction_id = _choose_uniform(junction_ids, id_key)

    r_max = merge_distance_threshold / 2.0
    r = _sample_split_radius(split_radius_mean, r_max, radius_key)
    theta = float(jax.random.uniform(angle_key, minval=0.0, maxval=_TWO_PI))
    offset = jnp.array([r * np.cos(theta), r * np.sin(theta)])

    proposed = floor_plan.copy()
    proposed.split_junction(junction_id, offset)
    if not proposed.is_planar():
        return None

    k = len(junction_ids)  # Nj before the split
    n_merge_candidates_after = len(_pairs_within_threshold(proposed, merge_distance_threshold))

    # forward: 1/k (junction) * q_r(r) * 1/(2*pi) (theta) ; reverse: 1/n_merge_candidates_after
    log_forward = -np.log(k) + _split_radius_log_pdf(r, split_radius_mean, r_max) - np.log(_TWO_PI)
    log_reverse = -np.log(n_merge_candidates_after)
    log_jacobian = np.log(4.0 * r)

    log_proposal_ratio = float(log_reverse - log_forward + log_jacobian)
    return Proposal(graph=proposed, log_proposal_ratio=log_proposal_ratio)


def propose_junction_merge(
    floor_plan: FloorPlanGraph,
    split_radius_mean: float,
    merge_distance_threshold: float,
    rng_key: jax.random.PRNGKey,
) -> Proposal | None:
    """Merge a uniformly-chosen junction pair within `merge_distance_threshold` of
    each other into one, at their midpoint -- the exact reverse of
    `propose_junction_split`, recovering `(r, theta) = M^{-1}(position_a, position_b)`
    (section 3) to evaluate the split proposal density it would need to reproduce
    this exact merge.

    Rejected (returns `None`) if there is no such pair, or if the merge breaks
    planarity.
    """
    candidates = _pairs_within_threshold(floor_plan, merge_distance_threshold)
    if not candidates:
        return None

    junction_a, junction_b = _choose_uniform(candidates, rng_key)
    position_a = floor_plan.junctions[junction_a].position
    position_b = floor_plan.junctions[junction_b].position
    r = float(jnp.linalg.norm(position_a - position_b)) / 2.0

    proposed = floor_plan.copy()
    proposed.merge_junctions(junction_a, junction_b)
    if not proposed.is_planar():
        return None

    k_after = len(proposed.junctions)  # Nj after the merge (= "k" in section 4's split notation)
    n_merge_candidates_before = len(candidates)
    r_max = merge_distance_threshold / 2.0

    # forward: 1/n_merge_candidates_before ; reverse: 1/k_after * q_r(r) * 1/(2*pi)
    log_forward = -np.log(n_merge_candidates_before)
    log_reverse = (
        -np.log(k_after) + _split_radius_log_pdf(r, split_radius_mean, r_max) - np.log(_TWO_PI)
    )
    log_jacobian = -np.log(4.0 * r)

    log_proposal_ratio = float(log_reverse - log_forward + log_jacobian)
    return Proposal(graph=proposed, log_proposal_ratio=log_proposal_ratio)
