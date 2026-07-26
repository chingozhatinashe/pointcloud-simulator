"""Tests for data.loaders.las_loader: raw .las/.laz ingestion via laspy."""

from __future__ import annotations

import laspy
import numpy as np
import pytest

from data.loaders.las_loader import LasScan, load_las


def _write_las(
    path,
    *,
    points: np.ndarray,
    intensity: np.ndarray,
    return_number: np.ndarray,
    scanner_positions: np.ndarray | None = None,
    point_format: int = 3,
) -> None:
    header = laspy.LasHeader(point_format=point_format, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    if scanner_positions is not None:
        for name in ("scanner_x", "scanner_y", "scanner_z"):
            header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float64))

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.intensity = intensity
    las.return_number = return_number
    if scanner_positions is not None:
        las.scanner_x = scanner_positions[:, 0]
        las.scanner_y = scanner_positions[:, 1]
        las.scanner_z = scanner_positions[:, 2]
    las.write(str(path))


class TestLoadLas:
    def test_reads_points_intensity_and_return_number(self, tmp_path):
        points = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 1.0], [5.0, 6.0, 1.5]])
        intensity = np.array([10, 20, 30], dtype=np.uint16)
        return_number = np.array([1, 1, 2], dtype=np.uint8)
        path = tmp_path / "scan.las"
        _write_las(path, points=points, intensity=intensity, return_number=return_number)

        scan = load_las(path)

        assert isinstance(scan, LasScan)
        np.testing.assert_allclose(scan.points, points, atol=1e-6)
        np.testing.assert_array_equal(scan.intensity, intensity)
        np.testing.assert_array_equal(scan.return_number, return_number)

    def test_scanner_positions_none_when_not_encoded(self, tmp_path, caplog):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        intensity = np.array([1, 2], dtype=np.uint16)
        return_number = np.array([1, 1], dtype=np.uint8)
        path = tmp_path / "scan.las"
        _write_las(path, points=points, intensity=intensity, return_number=return_number)

        with caplog.at_level("WARNING"):
            scan = load_las(path)

        assert scan.scanner_positions is None
        assert any("scanner positions" in record.message for record in caplog.records)

    def test_scanner_positions_extracted_when_encoded(self, tmp_path):
        points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 5.0, 0.0]])
        intensity = np.array([1, 2, 3], dtype=np.uint16)
        return_number = np.array([1, 1, 1], dtype=np.uint8)
        scanner_positions = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 1.5], [10.0, 10.0, 1.5]])
        path = tmp_path / "scan.las"
        _write_las(
            path,
            points=points,
            intensity=intensity,
            return_number=return_number,
            scanner_positions=scanner_positions,
        )

        scan = load_las(path)

        assert scan.scanner_positions is not None
        assert scan.scanner_positions.shape == (2, 3)  # deduplicated
        np.testing.assert_allclose(
            sorted(scan.scanner_positions.tolist()),
            sorted([[0.0, 0.0, 1.5], [10.0, 10.0, 1.5]]),
            atol=1e-6,
        )

    def test_header_metadata_is_captured(self, tmp_path):
        points = np.array([[0.0, 0.0, 0.0]])
        intensity = np.array([1], dtype=np.uint16)
        return_number = np.array([1], dtype=np.uint8)
        path = tmp_path / "scan.las"
        _write_las(
            path, points=points, intensity=intensity, return_number=return_number, point_format=3
        )

        scan = load_las(path)

        assert scan.point_format_id == 3
        np.testing.assert_allclose(scan.scale, (0.001, 0.001, 0.001))
        np.testing.assert_allclose(scan.offset, (0.0, 0.0, 0.0))

    def test_accepts_string_path(self, tmp_path):
        points = np.array([[0.0, 0.0, 0.0]])
        intensity = np.array([1], dtype=np.uint16)
        return_number = np.array([1], dtype=np.uint8)
        path = tmp_path / "scan.las"
        _write_las(path, points=points, intensity=intensity, return_number=return_number)

        scan = load_las(str(path))
        np.testing.assert_allclose(scan.points, points, atol=1e-6)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_las(tmp_path / "does_not_exist.las")
