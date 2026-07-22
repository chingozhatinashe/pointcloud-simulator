"""Junction and WallSegment: the atomic geometric primitives of a floor plan."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass
class Junction:
    """A 2D point where wall segments meet, in metres."""

    id: int
    position: jnp.ndarray

    def __post_init__(self) -> None:
        self.position = jnp.asarray(self.position, dtype=jnp.float32)

    @property
    def x(self) -> jnp.ndarray:
        return self.position[0]

    @property
    def y(self) -> jnp.ndarray:
        return self.position[1]

    def tree_flatten(self):
        return (self.position,), self.id

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (position,) = children
        return cls(id=aux_data, position=position)


@dataclass(frozen=True)
class WallSegment:
    """A straight wall between two junctions, referenced by ID (not coordinates)."""

    id: int
    junction_start: int
    junction_end: int

    def __post_init__(self) -> None:
        if self.junction_start == self.junction_end:
            raise ValueError(
                f"WallSegment {self.id} cannot connect junction {self.junction_start} to itself"
            )

    def as_array(self, junctions: dict[int, Junction]) -> jnp.ndarray:
        """Geometry as (x1, y1, x2, y2), resolved from junction positions."""
        start = junctions[self.junction_start].position
        end = junctions[self.junction_end].position
        return jnp.concatenate([start, end])

    def length(self, junctions: dict[int, Junction]) -> jnp.ndarray:
        start = junctions[self.junction_start].position
        end = junctions[self.junction_end].position
        return jnp.linalg.norm(end - start)

    def other_junction(self, junction_id: int) -> int:
        """The ID at the opposite end from the given junction ID."""
        if junction_id == self.junction_start:
            return self.junction_end
        if junction_id == self.junction_end:
            return self.junction_start
        raise ValueError(f"Junction {junction_id} is not incident to segment {self.id}")
