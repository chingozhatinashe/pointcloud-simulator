"""ScannerConfig and ray sweep generation for the LiDAR forward model.

Scanner positions are always explicit inputs to the functions here (per
docs/simulator.md) -- `ScannerConfig` captures only the sweep geometry (angular
range, resolution) and the height of the horizontal slice being scanned, never a
position. This keeps the simulator a pure function of its inputs and lets a single
config be reused across many scanner positions via `vmap`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

DEFAULT_NUM_RAYS = 360
FULL_SWEEP: tuple[float, float] = (0.0, 2 * jnp.pi)


@dataclass(frozen=True)
class ScannerConfig:
    """Configuration for a single LiDAR sweep at a given height.

    `angular_range` is `(start, end)` in radians, default a full 360-degree sweep.
    `num_rays` is the number of rays per sweep, default 360 (1-degree spacing).
    `height` is the height of the horizontal scan slice, in metres above the floor.
    """

    height: float
    angular_range: tuple[float, float] = FULL_SWEEP
    num_rays: int = DEFAULT_NUM_RAYS

    def __post_init__(self) -> None:
        if self.num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {self.num_rays}")
        start, end = self.angular_range
        if end <= start:
            raise ValueError(f"angular_range end ({end}) must exceed start ({start})")

    def ray_angles(self) -> jnp.ndarray:
        """Angles, in radians, of each ray in the sweep, evenly spaced over `angular_range`."""
        start, end = self.angular_range
        # endpoint=False avoids double-casting the 0/2*pi ray on a full sweep.
        return jnp.linspace(start, end, self.num_rays, endpoint=False)


def generate_rays(
    scanner_position: jnp.ndarray, config: ScannerConfig
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ray origins and unit directions for one scanner position.

    Returns `(origins, directions)`, each of shape `(config.num_rays, 2)`.
    """
    scanner_position = jnp.asarray(scanner_position, dtype=jnp.float32)
    angles = config.ray_angles()
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    origins = jnp.broadcast_to(scanner_position, directions.shape)
    return origins, directions


def generate_rays_multi(
    scanner_positions: jnp.ndarray, config: ScannerConfig
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ray origins and directions for `M` scanner positions, vmapped over `generate_rays`.

    Returns `(origins, directions)`, each of shape `(M, config.num_rays, 2)`.
    """
    scanner_positions = jnp.asarray(scanner_positions, dtype=jnp.float32)
    return jax.vmap(generate_rays, in_axes=(0, None))(scanner_positions, config)
