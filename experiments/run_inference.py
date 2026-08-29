"""CLI entry point: loads an `InferenceConfig`, runs inference, saves the result.

Implements `run_inference(D, config, likelihood, rng_key) -> InferenceResult`, the
top-level entry point described in docs/inference.md's Public Interface Summary --
the notebooks drove the RJMCMC loop directly because this didn't exist yet
(`notebooks/04_rjmcmc_synthetic.ipynb`'s "RJMCMC driver" section). It lives here
rather than inside `inference/` because it is orchestration, not a component API:
it loads data and saves files, both of which the Inference component's own "No data
loading here" constraint (docs/inference.md Key Design Constraints) forbids it from
doing itself.

Per CLAUDE.md's Development Order, only RJMCMC is implemented so far -- `method`
values "smc", "abc", "sbi", "map" are accepted by `InferenceConfig` but raise
`NotImplementedError` here until their components exist.

Without `--dataset`, a fresh `(G_true, D)` pair is sampled from the floor-plan prior
and the Simulator, mirroring `docs/inference/rjmcmc_algorithm.md` section 6's
`G <- config.init_graph ?? prior.sample(rng_key)` fallback -- there is no
`config.init_graph` field (a `FloorPlanGraph` is not JSON-serialisable, so it can't
live in `InferenceConfig`), so the chain always cold-starts from an independent
prior sample, never from `G_true`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from data.synthetic import SyntheticDataset
from inference.config import InferenceConfig
from inference.evaluate import evaluate
from inference.likelihood import (
    AnalyticalLikelihood,
    AnalyticalLikelihoodParams,
    LikelihoodModel,
    SimulatorLikelihood,
    SimulatorLikelihoodConfig,
)
from inference.prior import sample as prior_sample
from inference.result import EvaluationReport, InferenceResult
from inference.rjmcmc.gibbs import gibbs_sweep
from inference.rjmcmc.kernel import MoveType, RJMCMCConfig, log_posterior, step
from representation.floor_plan import FloorPlanGraph
from simulator.engine import DEFAULT_HEIGHTS, FloorPointCloud, simulate
from simulator.noise_model import NoiseModel

# Floors for the analytical likelihood's mixture parameters, so a noiseless (or
# near-total-outlier) NoiseModel still yields a valid, non-degenerate likelihood --
# AnalyticalLikelihoodParams requires wall_distance_sigma > 0 and rejects an
# inlier_weight of 0 would make every point equally explained by clutter.
_MIN_WALL_DISTANCE_SIGMA = 0.01  # metres
_MIN_INLIER_WEIGHT = 0.05

_MAX_STORED_SAMPLES = 200  # caps memory for large n_iterations; see sample_every below


def _analytical_params_from_noise_model(noise_model: NoiseModel) -> AnalyticalLikelihoodParams:
    """Heuristic mapping from the Simulator's noise model to the analytical
    likelihood's mixture parameters: `range_sigma` directly sets the inlier
    Gaussian's std, and `outlier_rate` sets the clutter mixture weight. There is no
    principled inverse here (the two likelihoods model the noise process
    differently) -- this only needs to be a reasonable default when a config
    specifies `likelihood: "analytical"` without hand-tuned parameters of its own.
    """
    return AnalyticalLikelihoodParams(
        wall_distance_sigma=max(float(noise_model.range_sigma), _MIN_WALL_DISTANCE_SIGMA),
        inlier_weight=float(np.clip(1.0 - noise_model.outlier_rate, _MIN_INLIER_WEIGHT, 1.0)),
    )


def build_likelihood(
    config: InferenceConfig, point_cloud: FloorPointCloud, *, temperature: float
) -> LikelihoodModel:
    """Constructs the `LikelihoodModel` `config` specifies, per its `likelihood`
    field -- this is deliberately outside `run_inference` itself, since "Likelihood
    is injected, not hardcoded" (docs/inference.md Key Design Constraints) means
    `run_inference` must accept an already-built one, not construct it from config.
    """
    if config.likelihood == "analytical":
        return AnalyticalLikelihood(_analytical_params_from_noise_model(config.noise_model))
    return SimulatorLikelihood(
        SimulatorLikelihoodConfig(
            scanner_positions=point_cloud.scanner_positions,
            noise_model=config.noise_model,
            temperature=temperature,
        ),
        jax.random.PRNGKey(config.rng_seed),
    )


def _junction_position_std(
    samples: list[FloorPlanGraph], map_estimate: FloorPlanGraph
) -> jnp.ndarray:
    """Rough per-junction std across stored samples sharing the MAP's junction
    count -- approximate, since junction identity isn't tracked across a
    transdimensional chain; good enough as a diagnostic, not a rigorous estimate.
    """
    target_nj = len(map_estimate.junctions)
    matching = [s for s in samples if len(s.junctions) == target_nj]
    if len(matching) < 2:
        return jnp.zeros((target_nj, 2))
    arrays = jnp.stack([s.to_junction_array()[:target_nj] for s in matching])
    return jnp.std(arrays, axis=0)


def _run_rjmcmc(
    point_cloud: FloorPointCloud,
    config: InferenceConfig,
    likelihood: LikelihoodModel,
    rng_key: jax.random.PRNGKey,
) -> InferenceResult:
    if config.gibbs_interval and not isinstance(likelihood, AnalyticalLikelihood):
        raise ValueError(
            "gibbs_interval > 0 requires an AnalyticalLikelihood -- gibbs_sweep derives "
            "its proposal from the analytical mixture likelihood specifically "
            "(docs/inference/rjmcmc_algorithm.md section 5)"
        )

    rjmcmc_config = RJMCMCConfig()
    init_key, chain_key = jax.random.split(rng_key)
    graph = prior_sample(init_key, config.prior_config)
    sample_every = max(1, config.n_iterations // _MAX_STORED_SAMPLES)

    log_posterior_trace = []
    proposed_counts = {move: 0 for move in MoveType}
    accepted_counts = {move: 0 for move in MoveType}
    samples: list[FloorPlanGraph] = []
    n_likelihood_evaluations = 0

    map_estimate = graph
    map_log_posterior = log_posterior(graph, point_cloud, likelihood, config.prior_config)

    start = time.time()
    for i in range(config.n_iterations):
        chain_key, step_key = jax.random.split(chain_key)
        result = step(graph, point_cloud, likelihood, config.prior_config, rjmcmc_config, step_key)
        graph = result.graph
        proposed_counts[result.move_type] += 1
        if result.accepted:
            accepted_counts[result.move_type] += 1
        if result.log_acceptance_ratio > float("-inf"):
            n_likelihood_evaluations += 2  # current + proposed

        if config.gibbs_interval and (i + 1) % config.gibbs_interval == 0:
            chain_key, gibbs_key = jax.random.split(chain_key)
            graph = gibbs_sweep(graph, point_cloud, likelihood, config.prior_config, gibbs_key)
            # upper bound: a junction with no inlier evidence this sweep skips its
            # proposal (inference.rjmcmc.gibbs._MIN_PRECISION), so this overcounts
            n_likelihood_evaluations += 2 * len(graph.junctions)

        current_lp = log_posterior(graph, point_cloud, likelihood, config.prior_config)
        log_posterior_trace.append(current_lp)
        if current_lp > map_log_posterior:
            map_log_posterior = current_lp
            map_estimate = graph
        if i % sample_every == 0:
            samples.append(graph)
    runtime_seconds = time.time() - start

    acceptance_rates = {
        move.value: (
            accepted_counts[move] / proposed_counts[move] if proposed_counts[move] else 0.0
        )
        for move in MoveType
    }
    segment_counts = [len(s.segments) for s in samples] if samples else [len(graph.segments)]

    return InferenceResult(
        samples=samples,
        log_weights=jnp.zeros(len(samples)),
        map_estimate=map_estimate,
        junction_position_std=_junction_position_std(samples, map_estimate),
        n_segments_distribution=jnp.array(np.bincount(segment_counts)),
        method="rjmcmc",
        n_likelihood_evaluations=n_likelihood_evaluations,
        runtime_seconds=runtime_seconds,
        acceptance_rates=acceptance_rates,
        ess=float(len(samples)),  # naive upper bound -- no autocorrelation-based ESS yet
        log_posterior_trace=jnp.array(log_posterior_trace),
    )


_IMPLEMENTED_METHODS = {"rjmcmc": _run_rjmcmc}


def run_inference(
    point_cloud: FloorPointCloud,
    config: InferenceConfig,
    likelihood: LikelihoodModel,
    rng_key: jax.random.PRNGKey,
) -> InferenceResult:
    """Runs `config.method` against `point_cloud`, per docs/inference.md's Public
    Interface Summary. Only `"rjmcmc"` is implemented so far.
    """
    if config.method not in _IMPLEMENTED_METHODS:
        raise NotImplementedError(
            f"method {config.method!r} is not implemented yet -- only "
            f"{sorted(_IMPLEMENTED_METHODS)} are available so far. See CLAUDE.md's "
            "Development Order (Phase 4) for the remaining build order: MAP, SMC, "
            "ABC, then SBI."
        )
    return _IMPLEMENTED_METHODS[config.method](point_cloud, config, likelihood, rng_key)


def sample_synthetic_ground_truth(
    config: InferenceConfig, rng_key: jax.random.PRNGKey
) -> tuple[FloorPlanGraph, FloorPointCloud]:
    """Generates a fresh `(G_true, D)` pair from `config.prior_config` and
    `config.noise_model`, for a self-contained run when no `--dataset` is given.
    A single scanner is placed at the sampled floor plan's centroid.
    """
    prior_key, sim_key = jax.random.split(rng_key)
    g_true = prior_sample(prior_key, config.prior_config)
    positions = jnp.stack([junction.position for junction in g_true.junctions.values()])
    centre = jnp.mean(positions, axis=0, keepdims=True)
    point_cloud = simulate(g_true, centre, list(DEFAULT_HEIGHTS), config.noise_model, sim_key)
    return g_true, point_cloud


def save_result(
    result: InferenceResult,
    config: InferenceConfig,
    output_dir: Path,
    *,
    evaluation: EvaluationReport | None = None,
) -> None:
    """Writes `config.json` (reproducibility, per docs/inference.md section 9),
    `summary.json` (scalar diagnostics, human-readable without unpickling), and
    `result.pkl` (the full `InferenceResult`, including posterior samples).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    summary = {
        "method": result.method,
        "n_samples": len(result.samples),
        "ess": result.ess,
        "runtime_seconds": result.runtime_seconds,
        "n_likelihood_evaluations": result.n_likelihood_evaluations,
        "acceptance_rates": result.acceptance_rates,
        "map_n_junctions": len(result.map_estimate.junctions),
        "map_n_segments": len(result.map_estimate.segments),
    }
    if evaluation is not None:
        summary["evaluation"] = evaluation.to_dict()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with (output_dir / "result.pkl").open("wb") as f:
        pickle.dump(result, f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to an InferenceConfig JSON file"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to a SyntheticDataset HDF5 file; omit to sample a fresh (G_true, D) "
        "pair from the config's prior and noise model instead",
    )
    parser.add_argument("--index", type=int, default=0, help="Which sample to load from --dataset")
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help='Chamfer-to-log-likelihood temperature, likelihood: "simulator" only',
    )
    args = parser.parse_args()

    config = InferenceConfig.from_json(args.config.read_text())
    rng_key = jax.random.PRNGKey(config.rng_seed)

    if args.dataset is not None:
        dataset = SyntheticDataset.from_file(args.dataset)
        g_true, point_cloud = dataset[args.index]
    else:
        rng_key, data_key = jax.random.split(rng_key)
        g_true, point_cloud = sample_synthetic_ground_truth(config, data_key)

    likelihood = build_likelihood(config, point_cloud, temperature=args.temperature)

    rng_key, run_key = jax.random.split(rng_key)
    result = run_inference(point_cloud, config, likelihood, run_key)

    evaluation = evaluate(result, g_true)
    save_result(result, config, config.output_dir, evaluation=evaluation)

    print(f"method: {result.method}")
    print(f"runtime: {result.runtime_seconds:.1f}s")
    print(
        f"MAP: {len(result.map_estimate.junctions)} junctions, "
        f"{len(result.map_estimate.segments)} segments"
    )
    print(f"junction localisation error: {evaluation.junction_localisation_error:.3f} m")
    print(f"segment Chamfer distance:    {evaluation.chamfer_distance_segments:.3f} m")
    print(f"results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
