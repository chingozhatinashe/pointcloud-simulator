"""`.las` / `.laz` file ingestion via laspy.

Reads raw 3D points and available per-point metadata (intensity, return number) from
a LAS/LAZ file in its native coordinate frame and units. Coordinate normalisation
(data.normalisation) and slicing into a FloorPointCloud (data.preprocessing) happen
downstream -- this module's only job is getting bytes off disk into numpy arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np

logger = logging.getLogger(__name__)

# Vendor exports sometimes carry per-point scanner setup coordinates as LAS 1.4 extra
# dimensions under these names. There is no universal LAS standard for encoding
# scanner position, so this is a best-effort, documented convention rather than a
# guaranteed extraction.
_SCANNER_POSITION_DIMENSION_NAMES = ("scanner_x", "scanner_y", "scanner_z")


@dataclass
class LasScan:
    """Raw contents of a `.las` / `.laz` file, before coordinate normalisation."""

    points: np.ndarray  # (N, 3), file's native coordinate frame and units
    intensity: np.ndarray  # (N,)
    return_number: np.ndarray  # (N,)
    scanner_positions: np.ndarray | None  # (M, 3), or None if not encoded in the file
    point_format_id: int
    scale: tuple[float, float, float]  # LAS header scale factors
    offset: tuple[float, float, float]  # LAS header offsets


def _extract_scanner_positions(las: laspy.LasData) -> np.ndarray | None:
    dimension_names = set(las.point_format.dimension_names)
    if not set(_SCANNER_POSITION_DIMENSION_NAMES).issubset(dimension_names):
        return None
    x, y, z = (
        np.asarray(las[name], dtype=np.float64) for name in _SCANNER_POSITION_DIMENSION_NAMES
    )
    return np.unique(np.column_stack([x, y, z]), axis=0)


def load_las(filepath: Path) -> LasScan:
    """Read a `.las` or `.laz` file, returning raw points and available metadata.

    Points are returned in the file's native coordinate frame and units (laspy
    applies the header's scale/offset internally to produce `las.x/.y/.z`) --
    normalisation to the pipeline's metre, Z-up, origin-at-zero convention happens
    in `data.normalisation`, not here.

    `scanner_positions` is `None`, with a logged warning, unless the file carries
    `scanner_x`/`scanner_y`/`scanner_z` extra dimensions.
    """
    filepath = Path(filepath)
    with laspy.open(filepath) as reader:
        las = reader.read()

    points = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    intensity = np.asarray(las.intensity)
    return_number = np.asarray(las.return_number)

    scanner_positions = _extract_scanner_positions(las)
    if scanner_positions is None:
        logger.warning(
            "%s does not encode scanner positions; FloorPointCloud.scanner_positions "
            "will need to be provided separately",
            filepath,
        )

    header = las.header
    return LasScan(
        points=points,
        intensity=intensity,
        return_number=return_number,
        scanner_positions=scanner_positions,
        point_format_id=las.point_format.id,
        scale=tuple(header.scales),
        offset=tuple(header.offsets),
    )
