"""Storey segmentation, height slice extraction, outlier removal, and downsampling.

Transforms a normalised raw 3D point cloud into the `FloorPointCloud` format used
throughout the pipeline, per docs/data.md section 4.

`FloorPointCloud` and `PointCloudSlice` are defined in `simulator/engine.py`, not
here -- per the note in CLAUDE.md's Key Data Structures section, they are the shared
canonical data-model schema (not simulator *behaviour*), and the Data component
produces instances of them without defining them. `DEFAULT_HEIGHTS`,
`DEFAULT_VOXEL_SIZE`, and `DEFAULT_SLICE_TOL` are likewise defined once in
`simulator/engine.py` and reused here rather than duplicated.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import open3d as o3d

from simulator.engine import DEFAULT_SLICE_TOL, DEFAULT_VOXEL_SIZE, FloorPointCloud, PointCloudSlice

DEFAULT_OUTLIER_NEIGHBOURS = 10
DEFAULT_OUTLIER_STD_RATIO = 2.0
DEFAULT_STOREY_BIN_SIZE = 0.1  # metres
DEFAULT_STOREY_VALLEY_FRACTION = 0.1


def segment_storeys(
    points: np.ndarray,
    *,
    bin_size: float = DEFAULT_STOREY_BIN_SIZE,
    valley_fraction: float = DEFAULT_STOREY_VALLEY_FRACTION,
) -> list[np.ndarray]:
    """Partition a full building point cloud into per-storey point arrays.

    Builds a histogram of Z values; peaks correspond to floor and ceiling surfaces,
    and a "valley" -- a contiguous run of bins whose count falls below
    `valley_fraction` of the histogram's peak count, bounded by occupied bins on
    both sides -- marks a storey boundary. Storey boundaries are placed at the
    z-midpoint of each qualifying valley. Returns a single-element list (all points)
    when no valley is found, e.g. for a single-storey building.
    """
    z = points[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    num_bins = max(int(np.ceil((z_max - z_min) / bin_size)), 1)
    counts, edges = np.histogram(z, bins=num_bins, range=(z_min, z_max))

    threshold = valley_fraction * counts.max()
    occupied = counts > threshold

    boundaries = []
    index = 0
    num_edges = len(counts)
    while index < num_edges:
        if occupied[index]:
            index += 1
            continue
        start = index
        while index < num_edges and not occupied[index]:
            index += 1
        end = index  # exclusive
        if start > 0 and end < num_edges:  # a genuine valley has occupied bins on both sides
            boundaries.append((edges[start] + edges[end]) / 2)

    if not boundaries:
        return [points]

    storeys = []
    lower = -np.inf
    for boundary in [*boundaries, np.inf]:
        mask = (z >= lower) & (z < boundary)
        storeys.append(points[mask])
        lower = boundary
    return storeys


def extract_height_slice(
    points: np.ndarray, height: float, slice_tolerance: float = DEFAULT_SLICE_TOL
) -> np.ndarray:
    """2D `(x, y)` points within `slice_tolerance` of `height` along Z."""
    mask = np.abs(points[:, 2] - height) < slice_tolerance
    return points[mask, :2]


def remove_outliers(
    points: np.ndarray,
    *,
    k: int = DEFAULT_OUTLIER_NEIGHBOURS,
    std_ratio: float = DEFAULT_OUTLIER_STD_RATIO,
) -> np.ndarray:
    """Remove statistical outliers: points whose mean distance to their `k` nearest
    neighbours exceeds `std_ratio` standard deviations above the mean.

    Operates on 2D points, padded to 3D for open3d and projected back afterwards.
    Returns `points` unchanged if there aren't more than `k` of them.
    """
    if points.shape[0] <= k:
        return points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.column_stack([points, np.zeros(points.shape[0])]))
    cleaned, _ = cloud.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
    return np.asarray(cleaned.points)[:, :2]


def downsample_voxel(points: np.ndarray, voxel_size: float = DEFAULT_VOXEL_SIZE) -> np.ndarray:
    """Voxel-grid downsample 2D points to `voxel_size`, one point per occupied voxel."""
    if points.shape[0] == 0:
        return points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.column_stack([points, np.zeros(points.shape[0])]))
    downsampled = cloud.voxel_down_sample(voxel_size)
    return np.asarray(downsampled.points)[:, :2]


def compute_bounds(points: np.ndarray) -> jnp.ndarray:
    """2D bounding box `(x_min, y_min, x_max, y_max)` of a storey's full point cloud."""
    xy = points[:, :2]
    return jnp.array(
        [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]
    )


def preprocess(
    points: jnp.ndarray,
    heights: Sequence[float],
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    slice_tolerance: float = DEFAULT_SLICE_TOL,
    outlier_neighbours: int = DEFAULT_OUTLIER_NEIGHBOURS,
) -> FloorPointCloud:
    """Turn a normalised `(N, 3)` single-storey point cloud into a `FloorPointCloud`.

    For each height in `heights`: extracts the slice within `slice_tolerance`,
    removes statistical outliers (`k=outlier_neighbours`), and voxel-grid downsamples
    to `voxel_size`. To process a full multi-storey building, call `segment_storeys`
    first and preprocess each storey's points independently.

    `scanner_positions` is left as an empty `(0, 2)` array -- this function only
    cleans geometry; scanner position extraction happens in the per-format loaders
    and is attached to the result by the caller.
    """
    if not heights:
        raise ValueError("heights must be non-empty")

    points_np = np.asarray(points, dtype=np.float64)
    if points_np.shape[0] == 0:
        raise ValueError("Cannot preprocess an empty point cloud")

    bounds = compute_bounds(points_np)

    slices = []
    for height in heights:
        raw_slice = extract_height_slice(points_np, height, slice_tolerance)
        cleaned = remove_outliers(raw_slice, k=outlier_neighbours)
        downsampled = downsample_voxel(cleaned, voxel_size)
        slices.append(
            PointCloudSlice(
                height=float(height), points=jnp.asarray(downsampled, dtype=jnp.float32)
            )
        )

    points_3d = jnp.concatenate(
        [
            jnp.column_stack(
                [sl.points, jnp.full((sl.points.shape[0],), sl.height, dtype=jnp.float32)]
            )
            for sl in slices
        ]
    )

    return FloorPointCloud(
        slices=slices,
        points_3d=points_3d,
        scanner_positions=jnp.zeros((0, 2)),
        floor_plan_bounds=bounds,
    )
