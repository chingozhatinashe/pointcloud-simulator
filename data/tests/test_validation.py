"""Tests for data.validation: advisory data quality checks and ValidationReport."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from data.validation import (
    ValidationIssue,
    ValidationReport,
    check_coordinate_frame_sanity,
    check_minimum_points_per_slice,
    check_scanner_position_plausibility,
    check_slice_consistency,
    validate,
)
from simulator.engine import FloorPointCloud, PointCloudSlice


def _make_point_cloud(
    slices: list[PointCloudSlice],
    bounds=(0.0, 0.0, 4.0, 4.0),
    scanner_positions=None,
) -> FloorPointCloud:
    if scanner_positions is None:
        scanner_positions = jnp.zeros((0, 2))
    points_3d = (
        jnp.concatenate(
            [
                jnp.column_stack(
                    [sl.points, jnp.full((sl.points.shape[0],), sl.height, dtype=jnp.float32)]
                )
                for sl in slices
            ]
        )
        if slices
        else jnp.zeros((0, 3))
    )
    return FloorPointCloud(
        slices=slices,
        points_3d=points_3d,
        scanner_positions=jnp.asarray(scanner_positions, dtype=jnp.float32),
        floor_plan_bounds=jnp.array(bounds),
    )


def _uniform_slice(height: float, n: int, low=(0.5, 0.5), high=(3.5, 3.5)) -> PointCloudSlice:
    rng = np.random.default_rng(0)
    points = rng.uniform(low, high, size=(n, 2))
    return PointCloudSlice(height=height, points=jnp.asarray(points, dtype=jnp.float32))


class TestValidationReport:
    def test_is_valid_true_when_no_errors(self):
        report = ValidationReport(
            issues=[ValidationIssue(severity="warning", check="x", message="m")]
        )
        assert report.is_valid
        assert len(report.warnings) == 1
        assert len(report.errors) == 0

    def test_is_valid_false_when_any_error(self):
        report = ValidationReport(
            issues=[
                ValidationIssue(severity="warning", check="x", message="m"),
                ValidationIssue(severity="error", check="y", message="n"),
            ]
        )
        assert not report.is_valid
        assert len(report.warnings) == 1
        assert len(report.errors) == 1

    def test_empty_report_is_valid(self):
        assert ValidationReport().is_valid


class TestCheckMinimumPointsPerSlice:
    def test_no_issue_when_all_slices_meet_minimum(self):
        cloud = _make_point_cloud([_uniform_slice(0.3, 100), _uniform_slice(1.0, 100)])
        assert check_minimum_points_per_slice(cloud, min_points_per_slice=50) == []

    def test_warns_on_sparse_slice(self):
        cloud = _make_point_cloud([_uniform_slice(0.3, 10), _uniform_slice(1.0, 100)])
        issues = check_minimum_points_per_slice(cloud, min_points_per_slice=50)
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].check == "minimum_points_per_slice"


class TestCheckCoordinateFrameSanity:
    def test_no_issue_when_all_points_within_bounds(self):
        cloud = _make_point_cloud([_uniform_slice(0.3, 100)], bounds=(0.0, 0.0, 4.0, 4.0))
        assert check_coordinate_frame_sanity(cloud) == []

    def test_warning_for_small_out_of_bounds_fraction(self):
        sl = PointCloudSlice(
            height=0.3,
            points=jnp.array(
                [[1.0, 1.0]] * 95 + [[10.0, 10.0]] * 5, dtype=jnp.float32
            ),  # 5% outside
        )
        cloud = _make_point_cloud([sl], bounds=(0.0, 0.0, 4.0, 4.0))
        issues = check_coordinate_frame_sanity(cloud, max_out_of_bounds_fraction=0.1)
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_error_for_large_out_of_bounds_fraction(self):
        sl = PointCloudSlice(
            height=0.3,
            points=jnp.array(
                [[1.0, 1.0]] * 50 + [[10.0, 10.0]] * 50, dtype=jnp.float32
            ),  # 50% outside
        )
        cloud = _make_point_cloud([sl], bounds=(0.0, 0.0, 4.0, 4.0))
        issues = check_coordinate_frame_sanity(cloud, max_out_of_bounds_fraction=0.1)
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_empty_point_cloud_has_no_issues(self):
        cloud = _make_point_cloud([])
        assert check_coordinate_frame_sanity(cloud) == []


class TestCheckSliceConsistency:
    def test_single_slice_has_no_issues(self):
        cloud = _make_point_cloud([_uniform_slice(0.3, 100)])
        assert check_slice_consistency(cloud) == []

    def test_aligned_slices_have_no_issues(self):
        cloud = _make_point_cloud([_uniform_slice(0.3, 100), _uniform_slice(1.0, 100)])
        assert check_slice_consistency(cloud) == []

    def test_misaligned_slice_is_flagged(self):
        aligned_a = _uniform_slice(0.3, 100, low=(0.5, 0.5), high=(1.0, 1.0))
        aligned_b = _uniform_slice(1.0, 100, low=(0.5, 0.5), high=(1.0, 1.0))
        misaligned = PointCloudSlice(
            height=1.8, points=jnp.full((100, 2), 100.0, dtype=jnp.float32)
        )
        cloud = _make_point_cloud([aligned_a, aligned_b, misaligned], bounds=(0.0, 0.0, 4.0, 4.0))
        issues = check_slice_consistency(cloud)
        assert len(issues) == 1
        assert issues[0].check == "slice_consistency"
        assert "1.8" in issues[0].message


class TestCheckScannerPositionPlausibility:
    def test_no_scanner_positions_has_no_issues(self):
        cloud = _make_point_cloud([_uniform_slice(0.3, 10)])
        assert check_scanner_position_plausibility(cloud) == []

    def test_positions_within_bounds_have_no_issues(self):
        cloud = _make_point_cloud(
            [_uniform_slice(0.3, 10)],
            bounds=(0.0, 0.0, 4.0, 4.0),
            scanner_positions=[[1.0, 1.0], [3.0, 3.0]],
        )
        assert check_scanner_position_plausibility(cloud) == []

    def test_position_outside_bounds_is_flagged(self):
        cloud = _make_point_cloud(
            [_uniform_slice(0.3, 10)],
            bounds=(0.0, 0.0, 4.0, 4.0),
            scanner_positions=[[1.0, 1.0], [10.0, 10.0]],
        )
        issues = check_scanner_position_plausibility(cloud)
        assert len(issues) == 1
        assert "10.000" in issues[0].message


class TestValidate:
    def test_clean_point_cloud_has_no_issues(self):
        cloud = _make_point_cloud(
            [_uniform_slice(0.3, 100), _uniform_slice(1.0, 100)],
            bounds=(0.0, 0.0, 4.0, 4.0),
            scanner_positions=[[2.0, 2.0]],
        )
        report = validate(cloud)
        assert report.is_valid
        assert report.issues == []

    def test_aggregates_issues_from_multiple_checks(self):
        sparse_and_misaligned = PointCloudSlice(
            height=1.8, points=jnp.full((5, 2), 100.0, dtype=jnp.float32)
        )
        cloud = _make_point_cloud(
            [_uniform_slice(0.3, 100), sparse_and_misaligned],
            bounds=(0.0, 0.0, 4.0, 4.0),
            scanner_positions=[[20.0, 20.0]],
        )
        report = validate(cloud, min_points_per_slice=50)

        checks = {issue.check for issue in report.issues}
        assert "minimum_points_per_slice" in checks
        assert "slice_consistency" in checks
        assert "scanner_position_plausibility" in checks

    def test_rejects_nothing_raises_instead_returns_report(self):
        # even a badly out-of-bounds cloud should return a report, not raise
        sl = PointCloudSlice(height=0.3, points=jnp.full((10, 2), 1000.0, dtype=jnp.float32))
        cloud = _make_point_cloud([sl], bounds=(0.0, 0.0, 4.0, 4.0))
        report = validate(cloud)
        assert isinstance(report, ValidationReport)
        assert not report.is_valid
