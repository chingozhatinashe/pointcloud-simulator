"""evaluate(): metrics against ground truth, per docs/inference.md section 8.

Junction matching uses a simple greedy nearest-neighbour assignment rather than an
optimal (e.g. Hungarian) algorithm, since scipy isn't a declared dependency of any
component per CLAUDE.md -- greedy matching is a standard, adequate approximation for
an evaluation metric, as opposed to something that needs to be exactly optimal.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from inference.result import EvaluationReport, InferenceResult
from representation.floor_plan import FloorPlanGraph


def _segment_midpoints(floor_plan: FloorPlanGraph) -> np.ndarray:
    """`(K, 2)` array of segment midpoints."""
    midpoints = []
    for segment in floor_plan.segments.values():
        start = np.asarray(floor_plan.junctions[segment.junction_start].position)
        end = np.asarray(floor_plan.junctions[segment.junction_end].position)
        midpoints.append((start + end) / 2)
    return np.array(midpoints) if midpoints else np.zeros((0, 2))


def segment_chamfer_distance(inferred: FloorPlanGraph, true: FloorPlanGraph) -> float:
    """Symmetric mean nearest-neighbour distance between inferred and true segment
    midpoints.
    """
    inferred_midpoints = _segment_midpoints(inferred)
    true_midpoints = _segment_midpoints(true)
    if inferred_midpoints.shape[0] == 0 or true_midpoints.shape[0] == 0:
        raise ValueError("Cannot compute segment Chamfer distance with no segments on either side")

    pairwise = np.linalg.norm(inferred_midpoints[:, None, :] - true_midpoints[None, :, :], axis=-1)
    inferred_to_true = pairwise.min(axis=1).mean()
    true_to_inferred = pairwise.min(axis=0).mean()
    return float(0.5 * (inferred_to_true + true_to_inferred))


def junction_localisation_error(inferred: FloorPlanGraph, true: FloorPlanGraph) -> float:
    """Mean distance between matched junctions, via greedy nearest-neighbour matching:
    each true junction is matched, without replacement, to its nearest remaining
    inferred junction.
    """
    inferred_positions = np.array([np.asarray(j.position) for j in inferred.junctions.values()])
    true_positions = np.array([np.asarray(j.position) for j in true.junctions.values()])
    if inferred_positions.shape[0] == 0 or true_positions.shape[0] == 0:
        raise ValueError("Cannot match junctions with none present on either side")

    available = list(range(inferred_positions.shape[0]))
    distances = []
    for true_position in true_positions:
        if not available:
            break
        candidates = inferred_positions[available]
        deltas = np.linalg.norm(candidates - true_position, axis=1)
        nearest = int(np.argmin(deltas))
        distances.append(float(deltas[nearest]))
        available.pop(nearest)
    return float(np.mean(distances))


def room_count_accuracy(result: InferenceResult, true: FloorPlanGraph) -> float:
    """Fraction of posterior samples (or just the MAP estimate, if `result.samples`
    is empty) with the correct number of rooms (cycles).
    """
    graphs = result.samples if result.samples else [result.map_estimate]
    true_room_count = len(true.cycles())
    correct = sum(1 for graph in graphs if len(graph.cycles()) == true_room_count)
    return correct / len(graphs)


def topology_accuracy(result: InferenceResult, true: FloorPlanGraph) -> float:
    """Fraction of posterior samples (or just the MAP estimate, if `result.samples`
    is empty) isomorphic to the true graph's topology.
    """
    graphs = result.samples if result.samples else [result.map_estimate]
    true_graph = true.to_networkx()
    correct = sum(1 for graph in graphs if nx.is_isomorphic(graph.to_networkx(), true_graph))
    return correct / len(graphs)


def evaluate(result: InferenceResult, g_true: FloorPlanGraph) -> EvaluationReport:
    """Compute all four evaluation metrics for `result` against ground truth `g_true`."""
    if not g_true.junctions or not g_true.segments:
        raise ValueError("g_true has no junctions/segments to evaluate against")
    if not result.map_estimate.junctions or not result.map_estimate.segments:
        raise ValueError("result.map_estimate has no junctions/segments")

    return EvaluationReport(
        chamfer_distance_segments=segment_chamfer_distance(result.map_estimate, g_true),
        junction_localisation_error=junction_localisation_error(result.map_estimate, g_true),
        room_count_accuracy=room_count_accuracy(result, g_true),
        topology_accuracy=topology_accuracy(result, g_true),
    )
