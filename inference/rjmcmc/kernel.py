"""RJMCMC Markov kernel: dispatches to `inference.rjmcmc.proposals`, computes the
Metropolis-Hastings acceptance ratio, and accepts or rejects one move per `step`
call, per docs/inference/rjmcmc_algorithm.md section 6.

Implements all six moves: segment birth/death and junction/segment perturbation
(within-model, section 2) plus junction split/merge (between-model, sections
3-4). The acceptance ratio is uniformly

    [likelihood ratio] x [prior ratio] x [move-type-selection ratio]
        x [Proposal.log_proposal_ratio from the move itself]

for every move type -- the dimension-matching Jacobian split/merge need is
folded into their own `log_proposal_ratio` by `inference.rjmcmc.proposals`
(section 3), so `step` itself needs no special-casing per move.

The Gibbs junction-position refresh (section 5) is implemented separately in
`inference.rjmcmc.gibbs.gibbs_sweep` -- it is not part of `step` itself, since
section 6's main loop interleaves it between `step` calls (`t mod
config.gibbs_interval == 0`) rather than folding it into the per-move
acceptance ratio above.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from inference.config import PriorConfig
from inference.likelihood import LikelihoodModel
from inference.prior import log_prob as prior_log_prob
from inference.rjmcmc.proposals import (
    Proposal,
    propose_junction_merge,
    propose_junction_perturbation,
    propose_junction_split,
    propose_segment_birth,
    propose_segment_death,
    propose_segment_perturbation,
)
from representation.floor_plan import FloorPlanGraph
from simulator.engine import FloorPointCloud


class MoveType(enum.Enum):
    SEGMENT_BIRTH = "segment_birth"
    SEGMENT_DEATH = "segment_death"
    JUNCTION_PERTURB = "junction_perturb"
    SEGMENT_PERTURB = "segment_perturb"
    JUNCTION_SPLIT = "junction_split"
    JUNCTION_MERGE = "junction_merge"


_REVERSE_MOVE = {
    MoveType.SEGMENT_BIRTH: MoveType.SEGMENT_DEATH,
    MoveType.SEGMENT_DEATH: MoveType.SEGMENT_BIRTH,
    MoveType.JUNCTION_PERTURB: MoveType.JUNCTION_PERTURB,
    MoveType.SEGMENT_PERTURB: MoveType.SEGMENT_PERTURB,
    MoveType.JUNCTION_SPLIT: MoveType.JUNCTION_MERGE,
    MoveType.JUNCTION_MERGE: MoveType.JUNCTION_SPLIT,
}


@dataclass(frozen=True)
class RJMCMCConfig:
    """Move-selection probabilities (must sum to 1 across all six move types)
    and step sizes for every move.
    """

    p_segment_birth: float = 0.2
    p_segment_death: float = 0.2
    p_junction_perturb: float = 0.2
    p_segment_perturb: float = 0.2
    p_junction_split: float = 0.1
    p_junction_merge: float = 0.1

    junction_perturb_sigma: float = 0.05  # metres
    segment_perturb_sigma: float = 0.05  # radians

    split_radius_mean: float = 0.05  # metres -- mean of q_r, the split-offset proposal
    merge_distance_threshold: float = 0.2  # metres -- max distance for a mergeable pair

    def __post_init__(self) -> None:
        for name in (
            "p_segment_birth",
            "p_segment_death",
            "p_junction_perturb",
            "p_segment_perturb",
            "p_junction_split",
            "p_junction_merge",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        total = (
            self.p_segment_birth
            + self.p_segment_death
            + self.p_junction_perturb
            + self.p_segment_perturb
            + self.p_junction_split
            + self.p_junction_merge
        )
        if not np.isclose(total, 1.0):
            raise ValueError(f"move probabilities must sum to 1, got {total}")
        if self.junction_perturb_sigma <= 0:
            raise ValueError(
                f"junction_perturb_sigma must be positive, got {self.junction_perturb_sigma}"
            )
        if self.segment_perturb_sigma <= 0:
            raise ValueError(
                f"segment_perturb_sigma must be positive, got {self.segment_perturb_sigma}"
            )
        if self.split_radius_mean <= 0:
            raise ValueError(f"split_radius_mean must be positive, got {self.split_radius_mean}")
        if self.merge_distance_threshold <= 0:
            raise ValueError(
                f"merge_distance_threshold must be positive, got {self.merge_distance_threshold}"
            )

    def move_probability(self, move_type: MoveType) -> float:
        return {
            MoveType.SEGMENT_BIRTH: self.p_segment_birth,
            MoveType.SEGMENT_DEATH: self.p_segment_death,
            MoveType.JUNCTION_PERTURB: self.p_junction_perturb,
            MoveType.SEGMENT_PERTURB: self.p_segment_perturb,
            MoveType.JUNCTION_SPLIT: self.p_junction_split,
            MoveType.JUNCTION_MERGE: self.p_junction_merge,
        }[move_type]


@dataclass(frozen=True)
class StepResult:
    graph: FloorPlanGraph
    move_type: MoveType
    accepted: bool
    log_acceptance_ratio: float  # log alpha, i.e. log min(1, ratio); -inf if unproposable


_PROPOSE = {
    MoveType.SEGMENT_BIRTH: lambda graph, config, key: propose_segment_birth(graph, key),
    MoveType.SEGMENT_DEATH: lambda graph, config, key: propose_segment_death(graph, key),
    MoveType.JUNCTION_PERTURB: lambda graph, config, key: propose_junction_perturbation(
        graph, config.junction_perturb_sigma, key
    ),
    MoveType.SEGMENT_PERTURB: lambda graph, config, key: propose_segment_perturbation(
        graph, config.segment_perturb_sigma, key
    ),
    MoveType.JUNCTION_SPLIT: lambda graph, config, key: propose_junction_split(
        graph, config.split_radius_mean, config.merge_distance_threshold, key
    ),
    MoveType.JUNCTION_MERGE: lambda graph, config, key: propose_junction_merge(
        graph, config.split_radius_mean, config.merge_distance_threshold, key
    ),
}


def _sample_move_type(config: RJMCMCConfig, rng_key: jax.random.PRNGKey) -> MoveType:
    move_types = list(MoveType)
    probabilities = jnp.array([config.move_probability(move) for move in move_types])
    index = int(jax.random.choice(rng_key, len(move_types), p=probabilities))
    return move_types[index]


def log_posterior(
    floor_plan: FloorPlanGraph,
    point_cloud: FloorPointCloud,
    likelihood: LikelihoodModel,
    prior_config: PriorConfig,
) -> float:
    """log P(D|G) + log P(G), the unnormalised log posterior the acceptance ratio compares."""
    return likelihood.log_prob(floor_plan, point_cloud) + prior_log_prob(floor_plan, prior_config)


def step(
    floor_plan: FloorPlanGraph,
    point_cloud: FloorPointCloud,
    likelihood: LikelihoodModel,
    prior_config: PriorConfig,
    config: RJMCMCConfig,
    rng_key: jax.random.PRNGKey,
) -> StepResult:
    """One RJMCMC iteration: propose one move, compute the Metropolis-Hastings
    acceptance ratio, and accept or reject it.
    """
    move_key, propose_key, accept_key = jax.random.split(rng_key, 3)
    move_type = _sample_move_type(config, move_key)

    proposal: Proposal | None = _PROPOSE[move_type](floor_plan, config, propose_key)
    if proposal is None:
        # Nothing was proposable (e.g. no segments to remove) -- an automatic
        # reject, not a rejected evaluation of the acceptance ratio.
        return StepResult(
            graph=floor_plan,
            move_type=move_type,
            accepted=False,
            log_acceptance_ratio=float("-inf"),
        )

    current_log_posterior = log_posterior(floor_plan, point_cloud, likelihood, prior_config)
    proposed_log_posterior = log_posterior(proposal.graph, point_cloud, likelihood, prior_config)

    log_move_selection_ratio = float(
        np.log(config.move_probability(_REVERSE_MOVE[move_type]))
        - np.log(config.move_probability(move_type))
    )

    log_acceptance_ratio = float(
        min(
            0.0,
            (proposed_log_posterior - current_log_posterior)
            + log_move_selection_ratio
            + proposal.log_proposal_ratio,
        )
    )

    accepted = bool(float(jnp.log(jax.random.uniform(accept_key))) < log_acceptance_ratio)
    return StepResult(
        graph=proposal.graph if accepted else floor_plan,
        move_type=move_type,
        accepted=accepted,
        log_acceptance_ratio=log_acceptance_ratio,
    )
