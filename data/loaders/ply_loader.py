"""`.ply` / `.pcd` file ingestion via open3d.

Reads raw 3D points and any available colour/normal attributes. Neither format has a
standard convention for encoding scanner position, unlike `.e57`'s pose or some
vendor `.las` extensions, so no scanner position extraction is attempted here.

Coordinate normalisation (data.normalisation) and slicing into a FloorPointCloud
(data.preprocessing) happen downstream -- this module's only job is getting bytes off
disk into numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class PlyScan:
    """Raw contents of a `.ply` / `.pcd` file, before coordinate normalisation."""

    points: np.ndarray  # (N, 3), file's native coordinate frame and units
    colors: np.ndarray | None  # (N, 3) in [0, 1], or None if the file has no colour
    normals: np.ndarray | None  # (N, 3), or None if the file has no normals


def load_ply(filepath: Path) -> PlyScan:
    """Read a `.ply` or `.pcd` file, returning raw points and available attributes."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    cloud = o3d.io.read_point_cloud(str(filepath))
    points = np.asarray(cloud.points)
    if points.shape[0] == 0:
        raise ValueError(f"{filepath} contains no points")

    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    normals = np.asarray(cloud.normals) if cloud.has_normals() else None
    return PlyScan(points=points, colors=colors, normals=normals)
