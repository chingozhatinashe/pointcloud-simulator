"""FloorPlanGraph: the central planar-graph representation of a floor plan.

Junctions are nodes; wall segments are edges connecting them by ID. Rooms are
not stored explicitly -- they emerge as cycles in the graph.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import networkx as nx
from shapely.geometry import LineString

from representation.primitives import Junction, WallSegment

K_MAX = 128  # Maximum number of wall segments per floor plan
J_MAX = 256  # Maximum number of junctions per floor plan

MIN_SEGMENT_LENGTH = 0.1  # metres
SPLIT_OFFSET = 0.05  # metres -- default junction separation on split


def _point_to_segment_distance(point: jnp.ndarray, segment: jnp.ndarray) -> jnp.ndarray:
    """Distance from `point` to the segment (x1, y1, x2, y2)."""
    start = segment[:2]
    end = segment[2:]
    edge = end - start
    edge_len_sq = jnp.dot(edge, edge)
    t = jnp.where(
        edge_len_sq > 0, jnp.dot(point - start, edge) / jnp.maximum(edge_len_sq, 1e-12), 0.0
    )
    t = jnp.clip(t, 0.0, 1.0)
    projection = start + t * edge
    return jnp.linalg.norm(point - projection)


class FloorPlanGraph:
    """A planar graph of junctions and wall segments representing a floor plan."""

    def __init__(self) -> None:
        self.junctions: dict[int, Junction] = {}
        self.segments: dict[int, WallSegment] = {}
        self._next_junction_id = 0
        self._next_segment_id = 0

    def copy(self) -> FloorPlanGraph:
        """A copy safe for non-destructive trial mutation (e.g. RJMCMC proposals,
        which must be able to discard a rejected candidate and keep the original).

        Every mutating method on this class replaces `Junction`/`WallSegment`
        entries wholesale rather than mutating them in place, so copying the two
        id-keyed dicts is sufficient -- the entries themselves never need a deep
        copy.
        """
        new = FloorPlanGraph()
        new.junctions = dict(self.junctions)
        new.segments = dict(self.segments)
        new._next_junction_id = self._next_junction_id
        new._next_segment_id = self._next_segment_id
        return new

    # --- construction and mutation -----------------------------------

    def add_junction(self, position) -> Junction:
        junction = Junction(id=self._next_junction_id, position=position)
        self.junctions[junction.id] = junction
        self._next_junction_id += 1
        return junction

    # possibility of creating duplicate segments?
    def add_segment(self, junction_start: int, junction_end: int) -> WallSegment:
        for junction_id in (junction_start, junction_end):
            if junction_id not in self.junctions:
                raise ValueError(f"Junction {junction_id} does not exist")
        segment = WallSegment(
            id=self._next_segment_id, junction_start=junction_start, junction_end=junction_end
        )
        self.segments[segment.id] = segment
        self._next_segment_id += 1
        return segment

    def remove_junction(self, junction_id: int) -> None:
        if junction_id not in self.junctions:
            raise KeyError(f"Junction {junction_id} does not exist")
        for segment_id in self.incident_segments(junction_id):
            del self.segments[segment_id]
        del self.junctions[junction_id]

    def remove_segment(self, segment_id: int) -> None:
        if segment_id not in self.segments:
            raise KeyError(f"Segment {segment_id} does not exist")
        del self.segments[segment_id]

    def move_junction(self, junction_id: int, new_position) -> None:
        if junction_id not in self.junctions:
            raise KeyError(f"Junction {junction_id} does not exist")
        self.junctions[junction_id] = Junction(id=junction_id, position=new_position)

    def merge_junctions(self, id_a: int, id_b: int) -> Junction:
        """Merge junction b into junction a at their midpoint; rewire b's segments to a."""
        if id_a == id_b:
            raise ValueError("Cannot merge a junction with itself")
        junction_a = self.junctions[id_a]
        junction_b = self.junctions[id_b]
        midpoint = (junction_a.position + junction_b.position) / 2
        self.junctions[id_a] = Junction(id=id_a, position=midpoint)

        for segment_id in self.incident_segments(id_b):
            segment = self.segments[segment_id]
            new_start = id_a if segment.junction_start == id_b else segment.junction_start
            new_end = id_a if segment.junction_end == id_b else segment.junction_end
            if new_start == new_end:
                # Merge collapsed this segment onto itself -- drop it.
                del self.segments[segment_id]
            else:
                self.segments[segment_id] = WallSegment(
                    id=segment_id, junction_start=new_start, junction_end=new_end
                )

        del self.junctions[id_b]
        return self.junctions[id_a]

    def split_junction(self, junction_id: int, direction) -> tuple[Junction, Junction]:
        """Split a junction into two along `direction`, partitioning incident segments by
        which side of the split their far endpoint falls on, and reconnecting the two new
        junctions with a new segment."""
        if junction_id not in self.junctions:
            raise KeyError(f"Junction {junction_id} does not exist")
        origin = self.junctions[junction_id]
        direction = jnp.asarray(direction, dtype=jnp.float32)
        norm = jnp.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Split direction must be non-zero")
        unit = direction / norm
        offset = SPLIT_OFFSET * unit

        incident_ids = self.incident_segments(junction_id)

        junction_a = self.add_junction(origin.position + offset)
        junction_b = self.add_junction(origin.position - offset)

        for segment_id in incident_ids:
            segment = self.segments[segment_id]
            other_id = segment.other_junction(junction_id)
            to_other = self.junctions[other_id].position - origin.position
            target = junction_a.id if jnp.dot(to_other, unit) >= 0 else junction_b.id
            new_start = target if segment.junction_start == junction_id else segment.junction_start
            new_end = target if segment.junction_end == junction_id else segment.junction_end
            self.segments[segment_id] = WallSegment(
                id=segment_id, junction_start=new_start, junction_end=new_end
            )

        del self.junctions[junction_id]
        self.add_segment(junction_a.id, junction_b.id)

        return junction_a, junction_b

    # --- geometric queries ----------------------------------------------

    def incident_segments(self, junction_id: int) -> list[int]:
        return [
            segment_id
            for segment_id, segment in self.segments.items()
            if junction_id in (segment.junction_start, segment.junction_end)
        ]

    def to_segment_array(self) -> jnp.ndarray:
        """Segment geometry padded to (K_MAX, 4); rows beyond the segment count are zero."""
        if len(self.segments) > K_MAX:
            raise ValueError(f"{len(self.segments)} segments exceeds K_MAX={K_MAX}")
        rows = [
            self.segments[segment_id].as_array(self.junctions)
            for segment_id in sorted(self.segments)
        ]
        padded = jnp.zeros((K_MAX, 4), dtype=jnp.float32)
        if rows:
            padded = padded.at[: len(rows)].set(jnp.stack(rows))
        return padded

    def segment_mask(self) -> jnp.ndarray:
        mask = jnp.zeros((K_MAX,), dtype=bool)
        return mask.at[: len(self.segments)].set(True)

    def to_junction_array(self) -> jnp.ndarray:
        """Junction positions padded to (J_MAX, 2); rows beyond the junction count are zero."""
        if len(self.junctions) > J_MAX:
            raise ValueError(f"{len(self.junctions)} junctions exceeds J_MAX={J_MAX}")
        rows = [self.junctions[junction_id].position for junction_id in sorted(self.junctions)]
        padded = jnp.zeros((J_MAX, 2), dtype=jnp.float32)
        if rows:
            padded = padded.at[: len(rows)].set(jnp.stack(rows))
        return padded

    def junction_mask(self) -> jnp.ndarray:
        mask = jnp.zeros((J_MAX,), dtype=bool)
        return mask.at[: len(self.junctions)].set(True)

    def nearest_segment_distance(self, point) -> jnp.ndarray:
        """Distance from `point` to the closest wall segment. Infinite if there are none."""
        point = jnp.asarray(point, dtype=jnp.float32)
        segments = self.to_segment_array()
        mask = self.segment_mask()
        distances = jax.vmap(_point_to_segment_distance, in_axes=(None, 0))(point, segments)
        distances = jnp.where(mask, distances, jnp.inf)
        return jnp.min(distances)

    # --- graph-theoretic queries -----------------------------------------

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        for junction_id, junction in self.junctions.items():
            graph.add_node(junction_id, pos=junction.position)
        for segment_id, segment in self.segments.items():
            graph.add_edge(segment.junction_start, segment.junction_end, segment_id=segment_id)
        return graph

    def cycles(self) -> list[list[int]]:
        """Minimal cycle basis of the graph -- the rooms."""
        return nx.cycle_basis(self.to_networkx())

    def is_connected(self) -> bool:
        if not self.junctions:
            return True
        return nx.is_connected(self.to_networkx())

    # --- validation -------------------------------------------------------

    def is_planar(self) -> bool:
        """True if no two segments intersect except at a shared junction."""
        for segment_a, segment_b in itertools.combinations(self.segments.values(), 2):
            shared = {segment_a.junction_start, segment_a.junction_end} & {
                segment_b.junction_start,
                segment_b.junction_end,
            }
            line_a = LineString(
                [
                    tuple(self.junctions[segment_a.junction_start].position.tolist()),
                    tuple(self.junctions[segment_a.junction_end].position.tolist()),
                ]
            )
            line_b = LineString(
                [
                    tuple(self.junctions[segment_b.junction_start].position.tolist()),
                    tuple(self.junctions[segment_b.junction_end].position.tolist()),
                ]
            )
            intersection = line_a.intersection(line_b)
            if shared:
                shared_junction = self.junctions[next(iter(shared))]
                if intersection.geom_type != "Point" or not jnp.allclose(
                    jnp.array([intersection.x, intersection.y]),
                    shared_junction.position,
                    atol=1e-9,
                ):
                    return False
            elif not intersection.is_empty:
                return False
        return True

    def is_valid(self) -> bool:
        """All constraints hold: planar, minimum segment length, junction valence >= 2."""
        if not self.is_planar():
            return False
        if self.length_penalty() > 0:
            return False
        return all(len(self.incident_segments(jid)) >= 2 for jid in self.junctions)

    def planarity_penalty(self) -> float:
        """Number of non-adjacent segment pairs that intersect; zero iff planar."""
        violations = 0
        for segment_a, segment_b in itertools.combinations(self.segments.values(), 2):
            shared = {segment_a.junction_start, segment_a.junction_end} & {
                segment_b.junction_start,
                segment_b.junction_end,
            }
            if shared:
                continue
            line_a = LineString(
                [
                    tuple(self.junctions[segment_a.junction_start].position.tolist()),
                    tuple(self.junctions[segment_a.junction_end].position.tolist()),
                ]
            )
            line_b = LineString(
                [
                    tuple(self.junctions[segment_b.junction_start].position.tolist()),
                    tuple(self.junctions[segment_b.junction_end].position.tolist()),
                ]
            )
            if line_a.intersects(line_b):
                violations += 1
        return float(violations)

    def length_penalty(self, min_length: float = MIN_SEGMENT_LENGTH) -> jnp.ndarray:
        """Sum of shortfalls below `min_length` across all segments; zero if all are long enough."""
        if not self.segments:
            return jnp.array(0.0)
        lengths = jnp.array([segment.length(self.junctions) for segment in self.segments.values()])
        return jnp.sum(jnp.maximum(min_length - lengths, 0.0))
