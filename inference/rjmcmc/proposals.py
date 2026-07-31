"""Within-model RJMCMC proposals: segment birth/death and junction/segment
perturbation, per docs/inference/rjmcmc_algorithm.md sections 2 and 6.

Each propose_* function draws the move's own randomness, applies it to a *copy*
of the graph (the original is never mutated), and returns a `Proposal` bundling
the proposed `FloorPlanGraph` with `log_proposal_ratio = log q(reverse | proposed)
- log q(forward | current)` -- the piece of the Metropolis-Hastings ratio that
comes from this move's own candidate-selection probabilities. The move-type
selection probability and the likelihood/prior ratio are added by
`inference.rjmcmc.kernel`, uniformly across all move types; none of these moves
needs a dimension-matching Jacobian since Nj is unchanged (section 2).

A move returns `None` when nothing of that type is currently proposable (no
segments to remove, no non-adjacent junction pair to connect, no junctions at
all) or when the birth candidate fails the planarity check -- `docs/inference.md`
"reject proposal immediately if segment crosses existing segments". The kernel
treats `None` as an automatic reject.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from representation.floor_plan import FloorPlanGraph


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
