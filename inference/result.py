"""InferenceResult and EvaluationReport dataclasses, per docs/inference.md section 8.

Every inference algorithm returns a common `InferenceResult`, enabling direct
comparison between methods. `EvaluationReport` is the schema for metrics against
ground truth (populated by `evaluate()` in `inference/evaluate.py`, not here).

Plotting methods reuse `representation.visualisation.plot_point_cloud_overlay`
directly; there is no dedicated `inference/visualisation.py` module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from representation.floor_plan import FloorPlanGraph
from representation.visualisation import plot_point_cloud_overlay
from simulator.engine import FloorPointCloud


@dataclass
class InferenceResult:
    # posterior representation
    samples: list[FloorPlanGraph]  # posterior samples (empty for MAP)
    log_weights: jnp.ndarray  # importance weights (uniform for MCMC)
    map_estimate: FloorPlanGraph  # highest posterior sample or MAP optimum

    # uncertainty summaries
    junction_position_std: jnp.ndarray  # per-junction position uncertainty
    n_segments_distribution: jnp.ndarray  # histogram over number of segments

    # diagnostics
    method: str  # name of inference algorithm used
    n_likelihood_evaluations: int  # total simulator/likelihood calls
    runtime_seconds: float
    acceptance_rates: dict[str, float]  # per move type, for RJMCMC
    ess: float  # effective sample size
    log_posterior_trace: jnp.ndarray  # for convergence assessment

    def __post_init__(self) -> None:
        if self.samples and self.log_weights.shape[0] != len(self.samples):
            raise ValueError(
                f"log_weights has {self.log_weights.shape[0]} entries but there are "
                f"{len(self.samples)} samples"
            )
        if self.n_likelihood_evaluations < 0:
            raise ValueError(
                f"n_likelihood_evaluations must be non-negative, got {self.n_likelihood_evaluations}"
            )
        if self.runtime_seconds < 0:
            raise ValueError(f"runtime_seconds must be non-negative, got {self.runtime_seconds}")
        if self.ess < 0:
            raise ValueError(f"ess must be non-negative, got {self.ess}")
        for name, rate in self.acceptance_rates.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"acceptance rate for {name!r} must be in [0, 1], got {rate}")

    def plot_samples(self, point_cloud: FloorPointCloud, n: int = 10) -> None:
        """Plot up to `n` posterior samples, each overlaid with `point_cloud`."""
        if not self.samples:
            raise ValueError("No posterior samples to plot (samples is empty for MAP results)")

        chosen = self.samples[:n]
        cols = min(len(chosen), 5)
        rows = -(-len(chosen) // cols)  # ceil division
        _, axes = plt.subplots(rows, cols, squeeze=False)
        flat_axes = [ax for row in axes for ax in row]

        points_2d = np.asarray(point_cloud.points_3d)[:, :2]
        for ax, floor_plan in zip(flat_axes, chosen):
            plot_point_cloud_overlay(floor_plan, points_2d, ax=ax)
        for ax in flat_axes[len(chosen) :]:
            ax.axis("off")

    def plot_map(self, point_cloud: FloorPointCloud) -> None:
        """Plot the MAP estimate overlaid with `point_cloud`."""
        points_2d = np.asarray(point_cloud.points_3d)[:, :2]
        ax = plot_point_cloud_overlay(self.map_estimate, points_2d)
        ax.set_title(f"MAP estimate ({self.method})")

    def plot_diagnostics(self) -> None:
        """Plot the log posterior trace, acceptance rates, and segment-count distribution."""
        num_panels = (
            1 + bool(self.acceptance_rates) + int(self.n_segments_distribution.shape[0] > 0)
        )
        _, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 4), squeeze=False)
        axes = axes[0]

        panel = 0
        axes[panel].plot(np.asarray(self.log_posterior_trace))
        axes[panel].set_title("Log posterior trace")
        axes[panel].set_xlabel("iteration")
        panel += 1

        if self.acceptance_rates:
            names = list(self.acceptance_rates)
            values = [self.acceptance_rates[name] for name in names]
            axes[panel].bar(names, values)
            axes[panel].set_title("Acceptance rate by move type")
            axes[panel].set_ylim(0, 1)
            panel += 1

        if self.n_segments_distribution.shape[0] > 0:
            counts = np.asarray(self.n_segments_distribution)
            axes[panel].bar(range(len(counts)), counts)
            axes[panel].set_title("Segment count distribution")


@dataclass(frozen=True)
class EvaluationReport:
    """Metrics comparing an `InferenceResult` against ground truth, per section 8."""

    # mean nearest-neighbour distance, inferred vs true segment midpoints
    chamfer_distance_segments: float
    junction_localisation_error: float  # mean distance between matched junctions
    room_count_accuracy: float  # fraction of samples with the correct number of rooms
    topology_accuracy: float  # fraction of samples with the correct graph topology

    def __post_init__(self) -> None:
        if self.chamfer_distance_segments < 0:
            raise ValueError(
                f"chamfer_distance_segments must be non-negative, got {self.chamfer_distance_segments}"
            )
        if self.junction_localisation_error < 0:
            raise ValueError(
                f"junction_localisation_error must be non-negative, got {self.junction_localisation_error}"
            )
        if not 0.0 <= self.room_count_accuracy <= 1.0:
            raise ValueError(
                f"room_count_accuracy must be in [0, 1], got {self.room_count_accuracy}"
            )
        if not 0.0 <= self.topology_accuracy <= 1.0:
            raise ValueError(f"topology_accuracy must be in [0, 1], got {self.topology_accuracy}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationReport:
        return cls(**data)
