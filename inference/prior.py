"""P(G): the five prior terms, their combination, and the procedural floor-plan
sampler, per docs/inference.md section 1.

Each term is a plain function of a `FloorPlanGraph` and its `PriorConfig` weight(s),
individually switchable via the config's `use_*` flags. `Prior` bundles a `PriorConfig`
into the `.log_prob(G)` / `.sample(rng_key)` interface from the Public Interface
Summary.

The junction-regularity term's exact formula isn't specified in docs/inference.md
(only "prefers T- and L-junctions... deviation_from_right_angle(j)" conceptually) --
here it sums, over each junction's incident directions sorted circularly, the
deviation of each consecutive angular gap from the nearest multiple of 90 degrees.
A perfect L-junction (two segments 90 degrees apart) or T-junction (three segments at
0/90/180 degrees) scores zero.

The procedural sampler need not draw from the exact prior distribution (the spec is
explicit about this) -- it produces diverse, axis-aligned floor plans via BSP
subdivision of a random bounding box, then jitters junctions slightly. The jitter
scale is derived from `orthogonality_perturbation_sigma` (radians) times each
junction's mean incident segment length, approximating "rotate each wall by a small
random angle" (arc length ~= radius * angle for small angles) without the complexity
of rotating shared walls consistently around a shared junction.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from inference.config import PriorConfig
from representation.floor_plan import FloorPlanGraph

_HALF_PI = np.pi / 2
_TWO_PI = 2 * np.pi


# --- individual prior terms -----------------------------------------------------


def complexity_log_prob(floor_plan: FloorPlanGraph, lambda_complexity: float) -> float:
    """log P_complexity(G) = -lambda_complexity * |segments|."""
    return -lambda_complexity * len(floor_plan.segments)


def _segment_angle(floor_plan: FloorPlanGraph, segment) -> float:
    start = np.asarray(floor_plan.junctions[segment.junction_start].position)
    end = np.asarray(floor_plan.junctions[segment.junction_end].position)
    delta = end - start
    return float(np.arctan2(delta[1], delta[0]))


def orthogonality_log_prob(floor_plan: FloorPlanGraph, lambda_orthogonality: float) -> float:
    """log P_ortho(G) = -lambda_orthogonality * sum_l min(theta_l mod 90, 90 - theta_l mod 90)."""
    total_deviation = 0.0
    for segment in floor_plan.segments.values():
        angle = _segment_angle(floor_plan, segment)
        remainder = angle % _HALF_PI
        total_deviation += min(remainder, _HALF_PI - remainder)
    return -lambda_orthogonality * total_deviation


def min_length_log_prob(floor_plan: FloorPlanGraph, min_segment_length: float) -> float:
    """0 if every segment is at least `min_segment_length`, else -inf."""
    for segment in floor_plan.segments.values():
        if float(segment.length(floor_plan.junctions)) < min_segment_length:
            return float("-inf")
    return 0.0


def _incident_directions(floor_plan: FloorPlanGraph, junction_id: int) -> list[float]:
    origin = np.asarray(floor_plan.junctions[junction_id].position)
    directions = []
    for segment_id in floor_plan.incident_segments(junction_id):
        segment = floor_plan.segments[segment_id]
        other = np.asarray(floor_plan.junctions[segment.other_junction(junction_id)].position)
        delta = other - origin
        directions.append(float(np.arctan2(delta[1], delta[0])))
    return directions


def _junction_deviation(directions: list[float]) -> float:
    """Sum, over consecutive (circularly sorted) incident directions, of the
    deviation of the angular gap between them from the nearest multiple of 90 degrees.
    """
    if len(directions) < 2:
        return 0.0
    sorted_directions = sorted(d % _TWO_PI for d in directions)
    n = len(sorted_directions)
    total = 0.0
    for i in range(n):
        gap = (sorted_directions[(i + 1) % n] - sorted_directions[i]) % _TWO_PI
        total += abs(gap - _HALF_PI * round(gap / _HALF_PI))
    return total


def junction_regularity_log_prob(floor_plan: FloorPlanGraph, lambda_junction: float) -> float:
    """log P_junction(G) = -lambda_junction * sum_j deviation_from_right_angle(j)."""
    total = sum(
        _junction_deviation(_incident_directions(floor_plan, junction_id))
        for junction_id in floor_plan.junctions
    )
    return -lambda_junction * total


def _polygon_area(positions: np.ndarray) -> float:
    """Shoelace formula; `positions` is `(V, 2)`, vertices in cycle order."""
    x, y = positions[:, 0], positions[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _gaussian_log_pdf(x: float, mean: float, std: float) -> float:
    return float(-0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((x - mean) / std) ** 2)


def room_size_log_prob(
    floor_plan: FloorPlanGraph, room_area_mean: float, room_area_std: float
) -> float:
    """log P_room(G) = sum_c log N(area(c); room_area_mean, room_area_std^2)."""
    total = 0.0
    for cycle in floor_plan.cycles():
        positions = np.array([np.asarray(floor_plan.junctions[jid].position) for jid in cycle])
        total += _gaussian_log_pdf(_polygon_area(positions), room_area_mean, room_area_std)
    return total


def log_prob(floor_plan: FloorPlanGraph, config: PriorConfig) -> float:
    """log P(G): the sum of every term enabled by `config`'s `use_*` flags."""
    if config.use_min_length and min_length_log_prob(
        floor_plan, config.min_segment_length
    ) == float("-inf"):
        return float("-inf")

    total = 0.0
    if config.use_complexity:
        total += complexity_log_prob(floor_plan, config.lambda_complexity)
    if config.use_orthogonality:
        total += orthogonality_log_prob(floor_plan, config.lambda_orthogonality)
    if config.use_junction_regularity:
        total += junction_regularity_log_prob(floor_plan, config.lambda_junction)
    if config.use_room_size:
        total += room_size_log_prob(floor_plan, config.room_area_mean, config.room_area_std)
    return total


# --- procedural sampler ----------------------------------------------------------


@dataclass
class _Rect:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _bsp_subdivide(
    rect: _Rect, target_count: int, max_splits: int, rng_key: jax.random.PRNGKey
) -> list[_Rect]:
    """Split `rect` along its longer axis, at a random point, repeatedly splitting
    the largest current leaf, until `target_count` leaves exist or `max_splits`
    splits have been performed.
    """
    leaves = [rect]
    splits = 0
    while len(leaves) < target_count and splits < max_splits:
        areas = [(r.x_max - r.x_min) * (r.y_max - r.y_min) for r in leaves]
        rect_to_split = leaves.pop(int(np.argmax(areas)))

        rng_key, frac_key = jax.random.split(rng_key)
        width = rect_to_split.x_max - rect_to_split.x_min
        height = rect_to_split.y_max - rect_to_split.y_min
        frac = float(jax.random.uniform(frac_key, minval=0.3, maxval=0.7))

        if width >= height:
            cut = rect_to_split.x_min + frac * width
            leaves.append(_Rect(rect_to_split.x_min, rect_to_split.y_min, cut, rect_to_split.y_max))
            leaves.append(_Rect(cut, rect_to_split.y_min, rect_to_split.x_max, rect_to_split.y_max))
        else:
            cut = rect_to_split.y_min + frac * height
            leaves.append(_Rect(rect_to_split.x_min, rect_to_split.y_min, rect_to_split.x_max, cut))
            leaves.append(_Rect(rect_to_split.x_min, cut, rect_to_split.x_max, rect_to_split.y_max))
        splits += 1
    return leaves


def _rects_to_floor_plan(rects: list[_Rect], position_tolerance: float = 1e-6) -> FloorPlanGraph:
    floor_plan = FloorPlanGraph()
    junction_lookup: dict[tuple[float, float], int] = {}

    def get_junction(x: float, y: float) -> int:
        key = (round(x / position_tolerance), round(y / position_tolerance))
        if key not in junction_lookup:
            junction_lookup[key] = floor_plan.add_junction((x, y)).id
        return junction_lookup[key]

    seen_segments: set[frozenset[int]] = set()
    for rect in rects:
        corners = [
            (rect.x_min, rect.y_min),
            (rect.x_max, rect.y_min),
            (rect.x_max, rect.y_max),
            (rect.x_min, rect.y_max),
        ]
        corner_ids = [get_junction(x, y) for x, y in corners]
        for start_id, end_id in zip(corner_ids, corner_ids[1:] + corner_ids[:1]):
            key = frozenset((start_id, end_id))
            if key not in seen_segments:
                floor_plan.add_segment(start_id, end_id)
                seen_segments.add(key)

    _split_segments_at_interior_junctions(floor_plan)
    return floor_plan


def _split_segments_at_interior_junctions(floor_plan: FloorPlanGraph, tol: float = 1e-6) -> None:
    """Break any segment that has another junction lying exactly on its interior
    into two sub-segments there.

    BSP subdivision can produce a "T-junction": one rectangle's edge spans two (or
    more) smaller adjacent rectangles stacked along it, so a neighbour's corner lands
    partway along that edge rather than at one of its endpoints. Left alone this
    produces a non-planar graph (the long segment and the neighbour's edge overlap
    without declaring a shared junction), so every segment is checked against every
    junction and split where one lies on its interior, repeating until stable.
    """
    changed = True
    while changed:
        changed = False
        existing_pairs = {
            frozenset((s.junction_start, s.junction_end)) for s in floor_plan.segments.values()
        }
        for segment_id, segment in list(floor_plan.segments.items()):
            start = np.asarray(floor_plan.junctions[segment.junction_start].position)
            end = np.asarray(floor_plan.junctions[segment.junction_end].position)
            edge = end - start
            edge_length_sq = float(np.dot(edge, edge))
            if edge_length_sq < tol:
                continue

            for junction_id, junction in floor_plan.junctions.items():
                if junction_id in (segment.junction_start, segment.junction_end):
                    continue
                point = np.asarray(junction.position)
                t = float(np.dot(point - start, edge) / edge_length_sq)
                if t <= tol or t >= 1 - tol:
                    continue
                projection = start + t * edge
                if np.linalg.norm(point - projection) < tol:
                    floor_plan.remove_segment(segment_id)
                    existing_pairs.discard(
                        frozenset((segment.junction_start, segment.junction_end))
                    )
                    for a, b in (
                        (segment.junction_start, junction_id),
                        (junction_id, segment.junction_end),
                    ):
                        pair = frozenset((a, b))
                        if pair not in existing_pairs:
                            floor_plan.add_segment(a, b)
                            existing_pairs.add(pair)
                    changed = True
                    break
            if changed:
                break


def _perturb_junctions(
    floor_plan: FloorPlanGraph, angular_sigma: float, rng_key: jax.random.PRNGKey
) -> None:
    """Jitter every junction, then revert entirely if doing so broke planarity.

    Independent per-junction jitter can, rarely, push a junction across a nearby
    wall it isn't connected to (more likely for small rooms relative to the jitter
    scale) -- reverting to the unperturbed (always-planar) layout in that case is
    simpler and safer than a partial, per-junction rejection scheme.
    """
    if angular_sigma <= 0:
        return

    original_positions = {jid: j.position for jid, j in floor_plan.junctions.items()}
    for junction_id in list(floor_plan.junctions):
        incident_ids = floor_plan.incident_segments(junction_id)
        if not incident_ids:
            continue
        lengths = [
            float(floor_plan.segments[sid].length(floor_plan.junctions)) for sid in incident_ids
        ]
        scale = angular_sigma * (sum(lengths) / len(lengths))
        rng_key, subkey = jax.random.split(rng_key)
        noise = jax.random.normal(subkey, (2,)) * scale
        floor_plan.move_junction(junction_id, floor_plan.junctions[junction_id].position + noise)

    if not floor_plan.is_planar():
        for junction_id, position in original_positions.items():
            floor_plan.move_junction(junction_id, position)


def sample(rng_key: jax.random.PRNGKey, config: PriorConfig) -> FloorPlanGraph:
    """Procedurally sample a diverse, architecturally plausible floor plan.

    Not a draw from the exact `log_prob` distribution -- docs/inference.md is
    explicit that the sampler need only cover the range of structures inference
    will encounter, which this BSP-subdivision approach does.
    """
    room_key, width_key, height_key, split_key, perturb_key = jax.random.split(rng_key, 5)

    n_rooms = max(int(jax.random.poisson(room_key, config.room_count_rate)), 1)
    width = float(
        jax.random.uniform(
            width_key, minval=config.bounding_box_min_size, maxval=config.bounding_box_max_size
        )
    )
    height = float(
        jax.random.uniform(
            height_key, minval=config.bounding_box_min_size, maxval=config.bounding_box_max_size
        )
    )

    leaves = _bsp_subdivide(
        _Rect(0.0, 0.0, width, height), n_rooms, config.max_bsp_depth, split_key
    )
    floor_plan = _rects_to_floor_plan(leaves)
    _perturb_junctions(floor_plan, config.orthogonality_perturbation_sigma, perturb_key)
    return floor_plan


class Prior:
    """Bundles a `PriorConfig` into the `.log_prob(G)` / `.sample(rng_key)` interface."""

    def __init__(self, config: PriorConfig) -> None:
        self.config = config

    def log_prob(self, floor_plan: FloorPlanGraph) -> float:
        return log_prob(floor_plan, self.config)

    def sample(self, rng_key: jax.random.PRNGKey) -> FloorPlanGraph:
        return sample(rng_key, self.config)
