"""Coordinate frame normalisation and transform storage, per docs/data.md section 3.

Real LiDAR scans arrive in arbitrary coordinate frames -- GPS coordinates,
scanner-local frames, differing units, differing "up" conventions. `normalise()`
converts raw points to the pipeline's frame (metres, Z-up, origin at the bounding
box minimum) via the documented pipeline: unit conversion, rotation to Z-up,
translation to origin.

Unit and up-axis are explicit caller-supplied parameters rather than auto-detected
from geometry: a heuristic guess (e.g. "the largest-extent axis is up") is actively
wrong for typical building interior scans, which are usually far wider than they are
tall, so a wrong silent guess would be worse than requiring the caller state it.

The returned `NormalisationTransform` is invertible, so inferred floor plans in
pipeline coordinates can be mapped back to the original survey frame, and is
JSON-serialisable so it can be stored alongside the data it was computed from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_UNIT = "m"
DEFAULT_UP_AXIS = "z"

UNIT_SCALES: dict[str, float] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "km": 1000.0,
    "ft": 0.3048,
    "in": 0.0254,
}

# Rotation matrices mapping the named "up" axis of the raw scan onto the pipeline's
# +Z axis. Each is a proper rotation (orthogonal, determinant +1) rather than a
# reflection, so the other two axes are also remapped to preserve handedness.
_UP_AXIS_ROTATIONS: dict[str, np.ndarray] = {
    "z": np.eye(3),
    "-z": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
    "y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "-y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "x": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "-x": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
}


@dataclass(frozen=True)
class NormalisationTransform:
    """The scale, rotation, and translation applied to reach pipeline coordinates.

    Forward: `p_norm = (scale * p_raw) @ rotation.T + translation`.
    """

    scale: float
    rotation: np.ndarray  # (3, 3), orthogonal, determinant +1
    translation: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Raw points -> pipeline (normalised) coordinates."""
        return (points * self.scale) @ self.rotation.T + self.translation

    def invert(self, points: np.ndarray) -> np.ndarray:
        """Pipeline (normalised) coordinates -> the original raw coordinate frame."""
        return ((points - self.translation) @ self.rotation) / self.scale

    def to_dict(self) -> dict:
        """A JSON-serialisable representation, for storing alongside the data."""
        return {
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> NormalisationTransform:
        return cls(
            scale=float(data["scale"]),
            rotation=np.asarray(data["rotation"], dtype=np.float64),
            translation=np.asarray(data["translation"], dtype=np.float64),
        )


def to_metres(points: np.ndarray, unit: str = DEFAULT_UNIT) -> tuple[np.ndarray, float]:
    """Scale `points` from `unit` to metres. Returns `(scaled_points, scale_factor)`."""
    if unit not in UNIT_SCALES:
        raise ValueError(f"Unknown unit {unit!r}; expected one of {sorted(UNIT_SCALES)}")
    scale = UNIT_SCALES[unit]
    return points * scale, scale


def rotate_to_z_up(
    points: np.ndarray, up_axis: str = DEFAULT_UP_AXIS
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate `points` so that `up_axis` (in the raw frame) becomes +Z.

    `up_axis` is one of `"x"`, `"y"`, `"z"`, `"-x"`, `"-y"`, `"-z"`. Returns
    `(rotated_points, rotation_matrix)`.
    """
    if up_axis not in _UP_AXIS_ROTATIONS:
        raise ValueError(
            f"Unknown up_axis {up_axis!r}; expected one of {sorted(_UP_AXIS_ROTATIONS)}"
        )
    rotation = _UP_AXIS_ROTATIONS[up_axis]
    return points @ rotation.T, rotation


def translate_to_origin(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate `points` so their bounding box minimum corner sits at the origin.

    Returns `(translated_points, translation_vector)`.
    """
    translation = -points.min(axis=0)
    return points + translation, translation


def normalise(
    points: np.ndarray, *, unit: str = DEFAULT_UNIT, up_axis: str = DEFAULT_UP_AXIS
) -> tuple[np.ndarray, NormalisationTransform]:
    """Normalise raw `(N, 3)` points to the pipeline's coordinate frame.

    Applies, in order: unit conversion to metres, rotation so `up_axis` becomes
    +Z, and translation so the bounding box minimum sits at the origin (which also
    satisfies "storey alignment" -- Z above the lowest storey's floor -- as long as
    the floor is the lowest surface in the scan, the normal case).

    Returns `(normalised_points, transform)`; `transform.invert(normalised_points)`
    recovers the original raw points.
    """
    if points.shape[0] == 0:
        raise ValueError("Cannot normalise an empty point cloud")

    scaled, scale = to_metres(points, unit)
    rotated, rotation = rotate_to_z_up(scaled, up_axis)
    normalised, translation = translate_to_origin(rotated)

    transform = NormalisationTransform(scale=scale, rotation=rotation, translation=translation)
    return normalised, transform
