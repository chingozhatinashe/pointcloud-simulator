"""Data quality validation, per docs/data.md section 6.

Checks are advisory: `validate()` returns a `ValidationReport` listing warnings and
errors rather than raising, so the caller decides whether to proceed with imperfect
data. Most checks are warnings; coordinate frame sanity escalates to an error when a
large fraction of points fall outside the declared bounds, since that indicates the
bounds themselves are likely wrong rather than ordinary residual noise.

"Slice consistency" is checked without a known FloorPlanGraph -- that's what
inference produces, not something available at validation time -- via each slice's
centroid distance from the median of all slices' centroids (robust to a single
misaligned slice skewing the reference), as a simple proxy for "the same dominant
walls appear at consistent positions across height slices".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from simulator.engine import FloorPointCloud

DEFAULT_MIN_POINTS_PER_SLICE = 50
DEFAULT_MAX_OUT_OF_BOUNDS_FRACTION = 0.1
DEFAULT_MAX_CENTROID_OFFSET_FRACTION = 0.5


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "warning" or "error"
    check: str
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def is_valid(self) -> bool:
        """True iff there are no error-level issues; warnings alone don't invalidate."""
        return not self.errors


def check_minimum_points_per_slice(
    point_cloud: FloorPointCloud, min_points_per_slice: int = DEFAULT_MIN_POINTS_PER_SLICE
) -> list[ValidationIssue]:
    """Warn on any slice with fewer than `min_points_per_slice` points."""
    issues = []
    for sl in point_cloud.slices:
        count = sl.points.shape[0]
        if count < min_points_per_slice:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    check="minimum_points_per_slice",
                    message=(
                        f"slice at height {sl.height}m has {count} points, fewer than "
                        f"the minimum {min_points_per_slice}"
                    ),
                )
            )
    return issues


def check_coordinate_frame_sanity(
    point_cloud: FloorPointCloud,
    max_out_of_bounds_fraction: float = DEFAULT_MAX_OUT_OF_BOUNDS_FRACTION,
) -> list[ValidationIssue]:
    """Flag points lying outside `floor_plan_bounds`; escalate to an error if too many do."""
    total = point_cloud.points_3d.shape[0]
    if total == 0:
        return []

    x_min, y_min, x_max, y_max = (float(v) for v in point_cloud.floor_plan_bounds)
    xy = point_cloud.points_3d[:, :2]
    outside = (xy[:, 0] < x_min) | (xy[:, 0] > x_max) | (xy[:, 1] < y_min) | (xy[:, 1] > y_max)
    count = int(jnp.sum(outside))
    if count == 0:
        return []

    fraction = count / total
    severity = "error" if fraction > max_out_of_bounds_fraction else "warning"
    bounds = (round(x_min, 3), round(y_min, 3), round(x_max, 3), round(y_max, 3))
    return [
        ValidationIssue(
            severity=severity,
            check="coordinate_frame_sanity",
            message=(
                f"{count} of {total} points ({fraction:.1%}) lie outside floor_plan_bounds {bounds}"
            ),
        )
    ]


def check_slice_consistency(
    point_cloud: FloorPointCloud,
    max_centroid_offset_fraction: float = DEFAULT_MAX_CENTROID_OFFSET_FRACTION,
) -> list[ValidationIssue]:
    """Warn on slices whose centroid sits far from the other slices' centroids.

    The reference is the *median* of per-slice centroids rather than the pooled
    centroid of all points, so that a single badly misaligned slice doesn't skew
    the reference and cause every other (correctly aligned) slice to be flagged too.
    """
    non_empty_slices = [sl for sl in point_cloud.slices if sl.points.shape[0] > 0]
    if len(non_empty_slices) < 2:
        return []

    centroids = jnp.stack([jnp.mean(sl.points, axis=0) for sl in non_empty_slices])
    reference_centroid = jnp.median(centroids, axis=0)

    x_min, y_min, x_max, y_max = point_cloud.floor_plan_bounds
    diagonal = float(jnp.linalg.norm(jnp.array([x_max - x_min, y_max - y_min])))
    if diagonal == 0:
        return []

    issues = []
    for sl, centroid in zip(non_empty_slices, centroids):
        offset = float(jnp.linalg.norm(centroid - reference_centroid))
        if offset > max_centroid_offset_fraction * diagonal:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    check="slice_consistency",
                    message=(
                        f"slice at height {sl.height}m has a centroid {offset:.2f}m from "
                        f"the other slices' centroids, more than "
                        f"{max_centroid_offset_fraction:.0%} of the bounding box diagonal "
                        f"({diagonal:.2f}m) -- possible coordinate frame error or storey "
                        f"misalignment"
                    ),
                )
            )
    return issues


def check_scanner_position_plausibility(point_cloud: FloorPointCloud) -> list[ValidationIssue]:
    """Warn on any scanner position lying outside `floor_plan_bounds`."""
    positions = point_cloud.scanner_positions
    if positions.shape[0] == 0:
        return []

    x_min, y_min, x_max, y_max = (float(v) for v in point_cloud.floor_plan_bounds)
    issues = []
    for i in range(positions.shape[0]):
        x, y = float(positions[i, 0]), float(positions[i, 1])
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    check="scanner_position_plausibility",
                    message=(
                        f"scanner position {i} at ({x:.3f}, {y:.3f}) lies outside "
                        f"floor_plan_bounds ({x_min:.3f}, {y_min:.3f}, {x_max:.3f}, {y_max:.3f})"
                    ),
                )
            )
    return issues


def validate(
    point_cloud: FloorPointCloud, min_points_per_slice: int = DEFAULT_MIN_POINTS_PER_SLICE
) -> ValidationReport:
    """Run all data quality checks against `point_cloud`, returning a `ValidationReport`."""
    issues = [
        *check_minimum_points_per_slice(point_cloud, min_points_per_slice),
        *check_coordinate_frame_sanity(point_cloud),
        *check_slice_consistency(point_cloud),
        *check_scanner_position_plausibility(point_cloud),
    ]
    return ValidationReport(issues=issues)
