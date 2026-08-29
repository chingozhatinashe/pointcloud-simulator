"""Deriving a `FloorPlanGraph` from semantically-labelled wall points.

Used where a benchmark dataset ships point-wise semantic labels (e.g. 3DSES, S3DIS)
rather than an explicit vector floor plan or CAD/BIM model -- the ground truth graph
has to be reconstructed from the raw geometry instead of read directly off disk.

Three independent steps, each usable on its own:

1. `infer_wall_label` -- figure out which integer label means "wall" for a given
   dataset variant, since that mapping is frequently undocumented (or, as with 3DSES's
   Gold/Silver tiers, differs between variants of the same dataset).
2. `fit_wall_segments` -- iterative RANSAC line-fitting on the 2D wall points at a
   chosen slice height, producing straight-line wall segments.
3. `build_floor_plan_graph` -- snap segment endpoints into shared junctions and
   assemble a `FloorPlanGraph`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from representation.floor_plan import MIN_SEGMENT_LENGTH, FloorPlanGraph

DEFAULT_MIN_LABEL_POINTS = 50

DEFAULT_RANSAC_ITERATIONS = 200
DEFAULT_INLIER_THRESHOLD = 0.05  # metres
DEFAULT_MIN_INLIERS = 30
DEFAULT_MIN_REMAINING_FRACTION = 0.02


def _plane_normal_and_eigenvalues(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit plane normal (smallest-variance direction) and ascending eigenvalues."""
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[:, 0], eigenvalues


def infer_wall_label(
    points: np.ndarray, labels: np.ndarray, *, min_points: int = DEFAULT_MIN_LABEL_POINTS
) -> int:
    """Pick the label whose points most resemble the room's walls.

    Scores each label by `point_count * flatness * verticality`: wall points form a
    large, planar cluster (high flatness -- the fitted plane's normal-direction
    eigenvalue is small relative to its in-plane eigenvalue) whose surface is vertical
    (the plane normal lies close to horizontal, i.e. its Z-component is near zero --
    the opposite of floor/ceiling labels, whose normal is close to +-Z). Labels with
    fewer than `min_points` points are skipped as too sparse to fit a reliable plane.
    """
    best_label: int | None = None
    best_score = -np.inf

    for label in np.unique(labels):
        mask = labels == label
        count = int(mask.sum())
        if count < min_points:
            continue

        normal, eigenvalues = _plane_normal_and_eigenvalues(points[mask])
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        verticality = 1.0 - abs(float(normal[2]))
        flatness = 1.0 - eigenvalues[0] / (eigenvalues[1] + 1e-12)
        score = count * flatness * verticality

        if score > best_score:
            best_score = score
            best_label = int(label)

    if best_label is None:
        raise ValueError(f"No label has at least {min_points} points; cannot infer which is 'wall'")
    return best_label


def _fit_line_ransac(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    iterations: int,
    inlier_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Best 2-point line hypothesis: `(point_on_line, unit_direction, inlier_mask)`."""
    n = points.shape[0]
    if n < 2:
        return None

    best_inliers: np.ndarray | None = None
    best_count = -1
    best_point = best_direction = None

    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False)
        p1, p2 = points[i], points[j]
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]])
        distances = np.abs((points - p1) @ normal)
        inliers = distances < inlier_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_point, best_direction = p1, direction

    if best_inliers is None:
        return None
    return best_point, best_direction, best_inliers


def fit_wall_segments(
    points_2d: np.ndarray,
    *,
    inlier_threshold: float = DEFAULT_INLIER_THRESHOLD,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    iterations: int = DEFAULT_RANSAC_ITERATIONS,
    min_remaining_fraction: float = DEFAULT_MIN_REMAINING_FRACTION,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Iteratively RANSAC-fit straight wall segments from 2D wall-labelled points.

    Repeatedly fits the dominant line among the remaining points, removes its inliers
    (a finite segment spanning the inliers' projected extent along the fitted line),
    and repeats on what's left. Stops once too few points remain to reliably fit
    another wall (below `min_inliers`, or below `min_remaining_fraction` of the
    original point count).
    """
    rng = rng if rng is not None else np.random.default_rng()
    remaining = points_2d
    total = points_2d.shape[0]
    segments: list[np.ndarray] = []

    while remaining.shape[0] >= min_inliers and remaining.shape[0] > min_remaining_fraction * total:
        result = _fit_line_ransac(
            remaining, rng, iterations=iterations, inlier_threshold=inlier_threshold
        )
        if result is None:
            break
        point, direction, inliers = result
        if int(inliers.sum()) < min_inliers:
            break

        inlier_points = remaining[inliers]
        projections = (inlier_points - point) @ direction
        start = point + projections.min() * direction
        end = point + projections.max() * direction
        segments.append(np.concatenate([start, end]))
        remaining = remaining[~inliers]

    return segments


def build_floor_plan_graph(
    segments: Sequence[np.ndarray], *, junction_tolerance: float = MIN_SEGMENT_LENGTH
) -> FloorPlanGraph:
    """Assemble a `FloorPlanGraph` from `(x1, y1, x2, y2)` segments, snapping endpoints
    within `junction_tolerance` of each other into a shared `Junction`.

    Segments whose two endpoints snap to the same junction (degenerate after snapping)
    are dropped rather than raising, since that's an expected outcome of noisy fits.
    """
    graph = FloorPlanGraph()
    positions: list[np.ndarray] = []
    junction_ids: list[int] = []

    def _junction_for(point: np.ndarray) -> int:
        for position, junction_id in zip(positions, junction_ids):
            if np.linalg.norm(position - point) < junction_tolerance:
                return junction_id
        junction = graph.add_junction(point)
        positions.append(np.asarray(junction.position))
        junction_ids.append(junction.id)
        return junction.id

    for segment in segments:
        segment = np.asarray(segment, dtype=np.float64)
        start_id = _junction_for(segment[:2])
        end_id = _junction_for(segment[2:])
        if start_id != end_id:
            graph.add_segment(start_id, end_id)

    return graph
