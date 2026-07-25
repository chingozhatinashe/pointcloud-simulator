"""Hard validators and soft penalty functions for FloorPlanGraph physical constraints.

Each constraint from docs/representation.md is exposed twice: as a hard validator that
raises `ConstraintViolation` on violation, and as a pure soft penalty function for use
as a prior term in inference. The underlying geometry is computed by `FloorPlanGraph`;
this module only interprets and enforces it.
"""

from __future__ import annotations

from representation.floor_plan import MIN_SEGMENT_LENGTH, FloorPlanGraph

DEFAULT_MIN_VALENCE = 2

Bounds = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), metres


class ConstraintViolation(ValueError):
    """Raised when a FloorPlanGraph violates a hard physical constraint."""


# --- planarity ---------------------------------------------------------------


def planarity_penalty(graph: FloorPlanGraph) -> float:
    """Soft penalty: count of crossing (non-adjacent) segment pairs; zero iff planar."""
    return graph.planarity_penalty()


def validate_planarity(graph: FloorPlanGraph) -> None:
    """Hard validator: raise if any two segments cross without sharing a junction."""
    if not graph.is_planar():
        raise ConstraintViolation(
            "Floor plan is not planar: segments cross without sharing a junction"
        )


# --- minimum segment length ---------------------------------------------------


def length_penalty(graph: FloorPlanGraph, min_length: float = MIN_SEGMENT_LENGTH) -> float:
    """Soft penalty: total shortfall below `min_length` across all segments."""
    return float(graph.length_penalty(min_length))


def validate_min_segment_length(
    graph: FloorPlanGraph, min_length: float = MIN_SEGMENT_LENGTH
) -> None:
    """Hard validator: raise if any segment is shorter than `min_length`."""
    if graph.length_penalty(min_length) > 0:
        raise ConstraintViolation(f"Floor plan has segments shorter than {min_length}m")


# --- junction valence ----------------------------------------------------------


def valence_penalty(graph: FloorPlanGraph, min_valence: int = DEFAULT_MIN_VALENCE) -> float:
    """Soft penalty: total valence shortfall below `min_valence` across all junctions."""
    return float(
        sum(
            max(min_valence - len(graph.incident_segments(junction_id)), 0)
            for junction_id in graph.junctions
        )
    )


def validate_junction_valence(
    graph: FloorPlanGraph, min_valence: int = DEFAULT_MIN_VALENCE
) -> None:
    """Hard validator: raise if any junction has fewer than `min_valence` incident segments."""
    dangling = [
        junction_id
        for junction_id in graph.junctions
        if len(graph.incident_segments(junction_id)) < min_valence
    ]
    if dangling:
        raise ConstraintViolation(
            f"Junctions {sorted(dangling)} have fewer than {min_valence} incident segments"
        )


# --- coordinate bounds -----------------------------------------------------------


def bounds_penalty(graph: FloorPlanGraph, bounds: Bounds | None) -> float:
    """Soft penalty: total distance junctions fall outside `bounds`; zero if `bounds` is None."""
    if bounds is None:
        return 0.0
    x_min, y_min, x_max, y_max = bounds
    total = 0.0
    for junction in graph.junctions.values():
        x, y = float(junction.position[0]), float(junction.position[1])
        total += max(x_min - x, 0.0) + max(x - x_max, 0.0)
        total += max(y_min - y, 0.0) + max(y - y_max, 0.0)
    return total


def validate_bounds(graph: FloorPlanGraph, bounds: Bounds | None) -> None:
    """Hard validator: raise if any junction lies outside `bounds`; no-op if `bounds` is None."""
    if bounds is None:
        return
    x_min, y_min, x_max, y_max = bounds
    outside = [
        junction_id
        for junction_id, junction in graph.junctions.items()
        if not (x_min <= junction.position[0] <= x_max and y_min <= junction.position[1] <= y_max)
    ]
    if outside:
        raise ConstraintViolation(f"Junctions {sorted(outside)} lie outside bounds {bounds}")


# --- composite ---------------------------------------------------------------------


def validate_all(
    graph: FloorPlanGraph,
    *,
    min_length: float = MIN_SEGMENT_LENGTH,
    min_valence: int = DEFAULT_MIN_VALENCE,
    bounds: Bounds | None = None,
) -> None:
    """Hard validator: raise `ConstraintViolation` on the first constraint that fails."""
    validate_planarity(graph)
    validate_min_segment_length(graph, min_length)
    validate_junction_valence(graph, min_valence)
    validate_bounds(graph, bounds)
