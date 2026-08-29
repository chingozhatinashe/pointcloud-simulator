"""Aligning IFC reference wall geometry into a point cloud's coordinate frame.

Investigation across all 6 ISPRS Indoor Modelling Benchmark case studies (see the project plan
for `data/datasets/isprs.py`) found that `ifcopenshell`'s placement-resolved world coordinates
already share the released point clouds' X/Y frame -- comparing bounding boxes case study by
case study showed no evidence of any rotation or X/Y translation anywhere. Z can differ by a
real, non-trivial offset (confirmed ~7m on one case study), most likely a site/sea-level
elevation datum baked into the IFC file versus a locally-referenced point cloud. So registration
here is deliberately narrow: correct a Z offset by matching floor-level percentiles, then polish
with a tightly-constrained ICP pass to absorb whatever small residual noise remains (including
any X/Y noise, without ICP having to recover from a large blind offset) -- and reject rather than
silently return a bad transform if the residual after that stays large, so a case study where the
"X/Y already matches" assumption doesn't hold produces `floor_plan=None` instead of a wrong graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

DEFAULT_FLOOR_PERCENTILE = 5.0
DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE = 0.3  # metres
DEFAULT_MAX_REGISTRATION_RESIDUAL = 0.15  # metres


@dataclass(frozen=True)
class RegistrationResult:
    z_offset: float
    icp_transform: np.ndarray  # (4, 4)
    residual: float
    success: bool

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the full solved transform (Z offset, then ICP) to `(N, 3)` points."""
        shifted = np.asarray(points, dtype=np.float64) + np.array([0.0, 0.0, self.z_offset])
        homogeneous = np.column_stack([shifted, np.ones(len(shifted))])
        return (homogeneous @ self.icp_transform.T)[:, :3]


def _estimate_z_offset(
    reference_points: np.ndarray, scan_points: np.ndarray, *, percentile: float
) -> float:
    """Best-guess Z translation: align the low-percentile ("floor") Z of each point set.

    Percentile rather than min() to be robust to a handful of stray low outliers on either side.
    """
    reference_floor = np.percentile(reference_points[:, 2], percentile)
    scan_floor = np.percentile(scan_points[:, 2], percentile)
    return float(scan_floor - reference_floor)


def _residual_distance(
    reference_points: np.ndarray, scan_points: np.ndarray, icp_transform: np.ndarray
) -> float:
    """Median nearest-neighbour distance from the registered reference points to the scan."""
    homogeneous = np.column_stack([reference_points, np.ones(len(reference_points))])
    transformed = (homogeneous @ icp_transform.T)[:, :3]

    transformed_cloud = o3d.geometry.PointCloud()
    transformed_cloud.points = o3d.utility.Vector3dVector(transformed)
    scan_cloud = o3d.geometry.PointCloud()
    scan_cloud.points = o3d.utility.Vector3dVector(scan_points)

    distances = np.asarray(transformed_cloud.compute_point_cloud_distance(scan_cloud))
    return float(np.median(distances))


def register_reference_to_scan(
    reference_points: np.ndarray,
    scan_points: np.ndarray,
    *,
    floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
    icp_max_correspondence_distance: float = DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE,
    max_residual: float = DEFAULT_MAX_REGISTRATION_RESIDUAL,
) -> RegistrationResult:
    """Align `reference_points` (e.g. IFC wall vertices) onto `scan_points` (a point cloud).

    Callers should pass a reasonably-sized (e.g. voxel-downsampled) `scan_points` -- this
    function does no internal downsampling, so ICP/KDTree cost scales with whatever is passed in.

    If the residual after Z-correction and ICP still exceeds `max_residual`, `success` is False
    and the returned transform should not be trusted (the caller should treat ground truth as
    unavailable for this item rather than use it).
    """
    reference_points = np.asarray(reference_points, dtype=np.float64)
    scan_points = np.asarray(scan_points, dtype=np.float64)

    z_offset = _estimate_z_offset(reference_points, scan_points, percentile=floor_percentile)
    shifted_reference = reference_points + np.array([0.0, 0.0, z_offset])

    reference_cloud = o3d.geometry.PointCloud()
    reference_cloud.points = o3d.utility.Vector3dVector(shifted_reference)
    scan_cloud = o3d.geometry.PointCloud()
    scan_cloud.points = o3d.utility.Vector3dVector(scan_points)

    icp_result = o3d.pipelines.registration.registration_icp(
        reference_cloud,
        scan_cloud,
        icp_max_correspondence_distance,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    icp_transform = np.asarray(icp_result.transformation)

    residual = _residual_distance(shifted_reference, scan_points, icp_transform)
    return RegistrationResult(
        z_offset=z_offset,
        icp_transform=icp_transform,
        residual=residual,
        success=residual <= max_residual,
    )
