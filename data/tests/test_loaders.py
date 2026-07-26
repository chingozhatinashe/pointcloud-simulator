"""Tests for data.loaders: raw .las/.laz, .ply/.pcd, and .e57 ingestion."""

from __future__ import annotations

from unittest import mock

import laspy
import numpy as np
import open3d as o3d
import pye57
import pytest

from data.loaders.e57_loader import E57Scan, load_e57
from data.loaders.las_loader import LasScan, load_las
from data.loaders.ply_loader import PlyScan, load_ply


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


def _write_ply(path, *, points: np.ndarray, colors: np.ndarray | None = None, normals=None) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(str(path), cloud)


class TestLoadPly:
    def test_reads_points(self, tmp_path):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
        path = tmp_path / "cloud.ply"
        _write_ply(path, points=points)

        scan = load_ply(path)

        assert isinstance(scan, PlyScan)
        np.testing.assert_allclose(scan.points, points, atol=1e-6)
        assert scan.colors is None
        assert scan.normals is None

    def test_reads_colors_and_normals_when_present(self, tmp_path):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        path = tmp_path / "cloud.ply"
        _write_ply(path, points=points, colors=colors, normals=normals)

        scan = load_ply(path)

        np.testing.assert_allclose(scan.colors, colors, atol=1e-6)
        np.testing.assert_allclose(scan.normals, normals, atol=1e-6)

    def test_works_for_pcd_extension(self, tmp_path):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        path = tmp_path / "cloud.pcd"
        _write_ply(path, points=points)

        scan = load_ply(path)
        np.testing.assert_allclose(scan.points, points, atol=1e-6)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ply(tmp_path / "does_not_exist.ply")

    def test_empty_point_cloud_raises(self, tmp_path):
        # open3d refuses to write a 0-point cloud, so build a minimal empty PLY by hand
        path = tmp_path / "empty.ply"
        path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        with pytest.raises(ValueError):
            load_ply(path)


def _write_e57_scan(writer: pye57.E57, *, points: np.ndarray, translation, intensity=None) -> None:
    data = {
        "cartesianX": points[:, 0],
        "cartesianY": points[:, 1],
        "cartesianZ": points[:, 2],
    }
    if intensity is not None:
        data["intensity"] = intensity
    writer.write_scan_raw(
        data, rotation=np.array([1.0, 0.0, 0.0, 0.0]), translation=np.array(translation)
    )


class TestLoadE57:
    def test_reads_points_transformed_to_global_frame(self, tmp_path):
        points = np.array([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]])
        path = tmp_path / "scan.e57"
        with pye57.E57(str(path), mode="w") as writer:
            _write_e57_scan(writer, points=points, translation=[10.0, 20.0, 1.5])

        scan = load_e57(path)

        assert isinstance(scan, E57Scan)
        expected = points + np.array([10.0, 20.0, 1.5])
        np.testing.assert_allclose(scan.points, expected, atol=1e-4)

    def test_scanner_position_matches_scan_translation(self, tmp_path):
        points = np.array([[1.0, 0.0, 0.5]])
        path = tmp_path / "scan.e57"
        with pye57.E57(str(path), mode="w") as writer:
            _write_e57_scan(writer, points=points, translation=[10.0, 20.0, 1.5])

        scan = load_e57(path)

        assert scan.scanner_positions is not None
        assert scan.scanner_positions.shape == (1, 3)
        np.testing.assert_allclose(scan.scanner_positions[0], [10.0, 20.0, 1.5], atol=1e-4)

    def test_multiple_scans_are_concatenated(self, tmp_path):
        path = tmp_path / "scan.e57"
        with pye57.E57(str(path), mode="w") as writer:
            _write_e57_scan(
                writer,
                points=np.array([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]]),
                translation=[0.0, 0.0, 1.5],
            )
            _write_e57_scan(
                writer,
                points=np.array([[5.0, 1.0, 0.5], [6.0, 1.0, 0.5]]),
                translation=[10.0, 10.0, 1.5],
            )

        scan = load_e57(path)

        assert scan.points.shape == (4, 3)
        assert scan.scanner_positions.shape == (2, 3)
        np.testing.assert_allclose(
            sorted(scan.scanner_positions.tolist()),
            sorted([[0.0, 0.0, 1.5], [10.0, 10.0, 1.5]]),
            atol=1e-4,
        )

    def test_intensity_present_when_every_scan_has_it(self, tmp_path):
        points = np.array([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]])
        intensity = np.array([0.1, 0.2], dtype=np.float32)
        path = tmp_path / "scan.e57"
        with pye57.E57(str(path), mode="w") as writer:
            _write_e57_scan(writer, points=points, translation=[0.0, 0.0, 1.5], intensity=intensity)

        scan = load_e57(path)

        assert scan.intensity is not None
        np.testing.assert_allclose(scan.intensity, intensity, atol=1e-6)

    def test_intensity_none_when_scans_disagree(self, tmp_path):
        path = tmp_path / "scan.e57"
        with pye57.E57(str(path), mode="w") as writer:
            _write_e57_scan(
                writer,
                points=np.array([[1.0, 0.0, 0.5]]),
                translation=[0.0, 0.0, 1.5],
                intensity=np.array([0.5], dtype=np.float32),
            )
            _write_e57_scan(
                writer, points=np.array([[2.0, 0.0, 0.5]]), translation=[10.0, 0.0, 1.5]
            )

        scan = load_e57(path)
        assert scan.intensity is None

    def test_scanner_positions_none_and_warns_when_no_scan_has_a_pose(self, tmp_path, caplog):
        points = np.array([[1.0, 0.0, 0.5]])
        path = tmp_path / "scan.e57"
        with pye57.E57(str(path), mode="w") as writer:
            _write_e57_scan(writer, points=points, translation=[0.0, 0.0, 0.0])

        with mock.patch("pye57.scan_header.ScanHeader.has_pose", return_value=False):
            with caplog.at_level("WARNING"):
                scan = load_e57(path)

        assert scan.scanner_positions is None
        assert any("scan poses" in record.message for record in caplog.records)

    def test_no_scans_raises(self, tmp_path):
        path = tmp_path / "empty.e57"
        with pye57.E57(str(path), mode="w"):
            pass

        with pytest.raises(ValueError):
            load_e57(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_e57(tmp_path / "does_not_exist.e57")
