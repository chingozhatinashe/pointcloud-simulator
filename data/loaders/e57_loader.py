"""`.e57` file ingestion via pye57 (wraps libE57Format).

Reads every scan in an E57 file -- a file can contain multiple scanner setups from a
multi-station survey -- concatenating their points and collecting each scan's pose as
a scanner position. Unlike LAS, per-scan pose is a genuine part of the E57 standard,
so scanner position extraction here is reliable rather than a best-effort convention.

Coordinate normalisation (data.normalisation) and slicing into a FloorPointCloud
(data.preprocessing) happen downstream -- this module's only job is getting bytes off
disk into numpy arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pye57

logger = logging.getLogger(__name__)


@dataclass
class E57Scan:
    """Raw contents of an `.e57` file, before coordinate normalisation."""

    points: np.ndarray  # (N, 3), global coordinate frame (each scan's pose applied)
    intensity: np.ndarray | None  # (N,), or None if not every scan carries intensity
    scanner_positions: np.ndarray | None  # (M, 3), one row per posed scan, or None


def load_e57(filepath: Path) -> E57Scan:
    """Read every scan in an `.e57` file, concatenating points across scans.

    Points are returned already transformed into the file's global coordinate frame
    (each scan's stored pose is applied automatically for scans that have one).
    `intensity` is populated only if every scan in the file carries an intensity
    field; otherwise it is `None`, since a per-point array can't be aligned across a
    mix of scans with and without it. `scanner_positions` collects one row per scan
    that has a pose, or `None` (with a logged warning) if no scan does.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    with pye57.E57(str(filepath)) as reader:
        scan_count = reader.scan_count
        if scan_count == 0:
            raise ValueError(f"{filepath} contains no scans")

        headers = [reader.get_header(index) for index in range(scan_count)]
        has_intensity = all("intensity" in header.point_fields for header in headers)

        all_points = []
        all_intensity = [] if has_intensity else None
        scanner_positions = []
        for index, header in enumerate(headers):
            data = reader.read_scan(index, intensity=has_intensity, ignore_missing_fields=True)
            all_points.append(
                np.column_stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]])
            )
            if has_intensity:
                all_intensity.append(np.asarray(data["intensity"]))
            if header.has_pose():
                scanner_positions.append(reader.scan_position(index)[0])

    points = np.concatenate(all_points, axis=0)
    intensity = np.concatenate(all_intensity, axis=0) if has_intensity else None

    if scanner_positions:
        positions = np.stack(scanner_positions, axis=0)
    else:
        positions = None
        logger.warning(
            "%s does not encode any scan poses; FloorPointCloud.scanner_positions "
            "will need to be provided separately",
            filepath,
        )

    return E57Scan(points=points, intensity=intensity, scanner_positions=positions)
