"""ISPRS Indoor Modelling Benchmark loader, per docs/data.md section 5.

Six real case studies (`data/ISPRS/CaseStudy1..6`), each a TLS/MLS point cloud from a
different sensor/institution paired with a reference IFC2X3 BIM model. Point cloud formats
differ per case study: CS1/CS2/CS3/CS4/CS5 are whitespace-delimited ASCII text (differing
column layouts -- only XYZ is used here, so one reader covers all five), CS6 is standard
binary `.las` (read directly via `laspy` from the zip member stream, no extraction needed).

Ground truth comes from each case study's IFC reference model, not point-wise semantic labels
(none exist in this dataset) and not RANSAC line-fitting (unlike 3DSES/S3DIS): IFC gives every
wall as its own entity with clean geometry, so each wall's centerline is fit directly via PCA on
its footprint, one wall at a time -- no re-clustering needed.

The IFC model and the point cloud are *not* known to share a coordinate frame, so the extracted
walls must be registered before they're usable as ground truth (`data.datasets.registration`).
Investigation across all 6 case studies (see the project plan for this module) found IFC world
coordinates already share the point clouds' X/Y frame -- no rotation or X/Y translation needed --
but Z can be offset by a real amount (confirmed ~7m on CaseStudy3, plausibly a site/sea-level
elevation datum). Registration corrects that Z offset and polishes the residual with a tightly
constrained ICP pass; if the residual stays large, every item for that case study gets
`floor_plan=None` rather than a silently wrong graph.

Note on `ifcopenshell.geom`: `create_shape()` already returns world-coordinate vertices in
metres regardless of the file's declared length unit -- do not also multiply by
`ifcopenshell.util.unit.calculate_unit_scale()`, which applies only to *raw attribute values*
still expressed in the file's native unit (e.g. `IfcBuildingStorey.Elevation`). Getting this
backwards silently shrinks every wall ~1000x (confirmed empirically while building this loader).
"""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.unit
import laspy
import numpy as np
import open3d as o3d

from data.datasets.registration import RegistrationResult, register_reference_to_scan
from data.datasets.wall_fitting import build_floor_plan_graph
from data.normalisation import normalise
from data.preprocessing import preprocess, segment_storeys
from representation.floor_plan import FloorPlanGraph
from simulator.engine import DEFAULT_HEIGHTS, FloorPointCloud

logger = logging.getLogger(__name__)

DEFAULT_REGISTRATION_VOXEL_SIZE = 0.1  # metres -- downsample the scan before registration
_STOREY_MATCH_TOLERANCE = 1.0  # metres of slack around a point-cloud storey's Z range


# --- IFC wall extraction --------------------------------------------------------------


def _containing_storey(wall):
    for rel in wall.ContainedInStructure:
        structure = rel.RelatingStructure
        if structure.is_a("IfcBuildingStorey"):
            return structure
    return None


def _fit_centerline_2d(xy: np.ndarray) -> np.ndarray:
    """PCA-based centerline `(x1, y1, x2, y2)` through a wall's projected footprint.

    A wall is a vertical extrusion, so its XY footprint fully determines its centerline: the
    principal axis (largest-variance direction) is the wall's length direction, and projecting
    every footprint point onto that axis gives the two endpoints as the extremes.
    """
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    projections = centered @ direction
    start = centroid + projections.min() * direction
    end = centroid + projections.max() * direction
    return np.concatenate([start, end])


def extract_walls_by_storey(ifc_path: Path) -> dict[str, dict]:
    """Extract every wall as a centerline segment, grouped by containing storey.

    Returns `{storey_global_id: {"elevation": float, "walls": [wall, ...]}}`, where each wall
    is `{"vertices": (N, 3), "centerline": (2, 3)}` -- `vertices` is the wall's full mesh (used
    for registration, which needs dense coverage), `centerline` is its two fitted endpoints at
    the wall's own mean Z (used for the final floor plan graph once registered). Everything is
    in the IFC file's own world frame, in metres, not yet registered against any point cloud.
    """
    model = ifcopenshell.open(str(ifc_path))

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    unit_scale = ifcopenshell.util.unit.calculate_unit_scale(model)
    storeys: dict[str, dict] = {
        storey.GlobalId: {
            "elevation": float(storey.Elevation) * unit_scale
            if storey.Elevation is not None
            else 0.0,
            "walls": [],
        }
        for storey in model.by_type("IfcBuildingStorey")
    }

    for wall in model.by_type("IfcWall"):
        storey = _containing_storey(wall)
        if storey is None or storey.GlobalId not in storeys:
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, wall)
        except Exception:
            logger.warning("Failed to extract geometry for wall %s; skipping", wall.GlobalId)
            continue

        # create_shape() already returns metres -- no unit_scale here, see module docstring.
        vertices = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        centerline_2d = _fit_centerline_2d(vertices[:, :2])
        mean_z = float(vertices[:, 2].mean())
        centerline_3d = np.array(
            [
                [centerline_2d[0], centerline_2d[1], mean_z],
                [centerline_2d[2], centerline_2d[3], mean_z],
            ]
        )
        storeys[storey.GlobalId]["walls"].append(
            {"vertices": vertices, "centerline": centerline_3d}
        )

    return storeys


# --- per-case-study point cloud ingestion ---------------------------------------------


@dataclass(frozen=True)
class _CaseStudyConfig:
    archive_name: str
    point_cloud_member: str
    point_cloud_format: str  # "ascii" | "las"
    ifc_filename: str


_CASE_STUDIES: dict[str, _CaseStudyConfig] = {
    "cs1": _CaseStudyConfig(
        "CaseStudy1_TUB1.zip",
        "CaseStudy1/PointCloud_cs1_dec_x-y-z-localtime6precision.txt",
        "ascii",
        "CaseStudy1_TUB1_IFC.ifc",
    ),
    "cs2": _CaseStudyConfig(
        "CaseStudy2_TUB2.zip",
        "CaseStudy2/PointCloud_cs2_dec_x-y-z-unixtime12precision.txt",
        "ascii",
        "CaseStudy2_TUB2_IFC.ifc",
    ),
    "cs3": _CaseStudyConfig(
        "CaseStudy3_Fire-Brigade.zip",
        "CaseStudy3/pointcloud_cs3.txt",
        "ascii",
        "CaseStudy3_Fire-Brigade_IFC.ifc",
    ),
    "cs4": _CaseStudyConfig(
        "CaseStudy4_UVigo.zip",
        "CaseStudy4/PointCloud_cs4_dec_x-y-z-unixtime12precision_v2.txt",
        "ascii",
        "CaseStudy4_UVigo_IFC.ifc",
    ),
    "cs5": _CaseStudyConfig(
        "CaseStudy5_UoM.zip",
        "CaseStudy5/pointcloud_cs5.txt",
        "ascii",
        "CaseStudy5_UoM_IFC.ifc",
    ),
    "cs6": _CaseStudyConfig(
        "CaseStudy6_GM.zip",
        "CaseStudy6/GraingerMuseum.las",
        "las",
        "CaseStudy6_GM_IFC.ifc",
    ),
}


def _load_ascii_xyz(zip_path: Path, member: str) -> np.ndarray:
    """First 3 whitespace-delimited columns of every row, as `(N, 3)` XYZ.

    Covers every ISPRS case study's ASCII point cloud uniformly (they differ in trailing
    columns -- time, RGB, intensity -- which aren't needed here) without per-case-study parsing.
    """
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as fh:
        return np.loadtxt(fh, usecols=(0, 1, 2), dtype=np.float64)


def _load_las_xyz(zip_path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as fh, laspy.open(fh) as reader:
        las = reader.read()
    return np.column_stack([las.x, las.y, las.z]).astype(np.float64)


# --- dataset ---------------------------------------------------------------------------


class ISPRSDataset:
    """`BenchmarkDataset` wrapper for the ISPRS Indoor Modelling Benchmark.

    `case_study` selects one of `"cs1"`..`"cs6"`. `root` is the local directory containing the
    case study archives and `.ifc` files (`data/ISPRS/` in this repo).

    Registration between the IFC reference model and the point cloud is computed once per case
    study at construction time (not per item) -- it's a case-study-level property, not a
    per-storey one. If it fails, every item for this case study returns `floor_plan=None`.
    """

    def __init__(
        self,
        case_study: str,
        root: Path,
        *,
        heights: Sequence[float] = DEFAULT_HEIGHTS,
    ) -> None:
        case_study = case_study.lower()
        if case_study not in _CASE_STUDIES:
            raise ValueError(
                f"Unknown ISPRS case study {case_study!r}; expected one of {sorted(_CASE_STUDIES)}"
            )
        self._case_study = case_study
        self._config = _CASE_STUDIES[case_study]
        self._root = Path(root)
        self._heights = heights

        self._raw_points = self._load_raw_points()
        self._pc_storeys = segment_storeys(self._raw_points)
        self._ifc_storeys = extract_walls_by_storey(self._root / self._config.ifc_filename)
        self._registration = self._register()
        self._matched_segments = self._match_storeys()

    def _load_raw_points(self) -> np.ndarray:
        archive_path = self._root / self._config.archive_name
        if self._config.point_cloud_format == "ascii":
            return _load_ascii_xyz(archive_path, self._config.point_cloud_member)
        return _load_las_xyz(archive_path, self._config.point_cloud_member)

    def _register(self) -> RegistrationResult | None:
        reference_points = np.concatenate(
            [wall["vertices"] for storey in self._ifc_storeys.values() for wall in storey["walls"]],
            axis=0,
        )
        if reference_points.shape[0] == 0:
            logger.warning("%s: no wall geometry extracted from IFC model", self._case_study)
            return None

        scan_cloud = o3d.geometry.PointCloud()
        scan_cloud.points = o3d.utility.Vector3dVector(self._raw_points)
        downsampled_scan = np.asarray(
            scan_cloud.voxel_down_sample(DEFAULT_REGISTRATION_VOXEL_SIZE).points
        )

        result = register_reference_to_scan(reference_points, downsampled_scan)
        if not result.success:
            logger.warning(
                "%s: registration residual %.3fm exceeds threshold; ground truth unavailable",
                self._case_study,
                result.residual,
            )
            return None
        return result

    def _match_storeys(self) -> list[list[np.ndarray] | None]:
        """For each detected point-cloud storey, the matching IFC storey's registered wall
        segments (2D, in the point cloud's raw frame), or None if unmatched/unregistered."""
        if self._registration is None:
            return [None] * len(self._pc_storeys)

        registered_storeys = []
        for storey in self._ifc_storeys.values():
            registered_elevation = self._registration.apply(
                np.array([[0.0, 0.0, storey["elevation"]]])
            )[0, 2]
            segments = [
                self._registration.apply(wall["centerline"])[:, :2].reshape(-1)
                for wall in storey["walls"]
            ]
            registered_storeys.append((registered_elevation, segments))

        matches: list[list[np.ndarray] | None] = []
        for pc_storey in self._pc_storeys:
            z_min, z_max = pc_storey[:, 2].min(), pc_storey[:, 2].max()
            candidates = [
                (elevation, segments)
                for elevation, segments in registered_storeys
                if z_min - _STOREY_MATCH_TOLERANCE <= elevation <= z_max + _STOREY_MATCH_TOLERANCE
            ]
            if not candidates:
                matches.append(None)
                continue
            midpoint = (z_min + z_max) / 2
            _, segments = min(candidates, key=lambda c: abs(c[0] - midpoint))
            matches.append(segments)
        return matches

    def split(self, ratio: Sequence[float]) -> tuple[_ISPRSSubset, ...]:
        """Split into `len(ratio)` fixed sub-views over disjoint index ranges."""
        if not np.isclose(sum(ratio), 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got {ratio}")

        n = len(self)
        boundaries = np.round(np.cumsum(ratio) * n).astype(int)
        splits = []
        start = 0
        for boundary in boundaries:
            splits.append(_ISPRSSubset(self, list(range(start, boundary))))
            start = boundary
        return tuple(splits)

    def __len__(self) -> int:
        return len(self._pc_storeys)

    def __getitem__(self, idx: int) -> tuple[FloorPointCloud, FloorPlanGraph | None]:
        pc_storey = self._pc_storeys[idx]
        normalised_xyz, transform = normalise(pc_storey)
        point_cloud = preprocess(normalised_xyz, heights=self._heights)

        segments_raw = self._matched_segments[idx]
        floor_plan: FloorPlanGraph | None = None
        if segments_raw:
            normalised_segments = []
            for segment in segments_raw:
                x1, y1, x2, y2 = segment
                points_3d = np.array([[x1, y1, 0.0], [x2, y2, 0.0]])
                normalised = transform.apply(points_3d)
                normalised_segments.append(normalised[:, :2].reshape(-1))
            floor_plan = build_floor_plan_graph(normalised_segments)

        return point_cloud, floor_plan

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class _ISPRSSubset:
    """A fixed index-range view over an `ISPRSDataset`, returned by `.split()`.

    Delegates to the parent so registration is never recomputed per split.
    """

    def __init__(self, parent: ISPRSDataset, indices: Sequence[int]) -> None:
        self._parent = parent
        self._indices = list(indices)

    def split(self, ratio: Sequence[float]) -> tuple[_ISPRSSubset, ...]:
        if not np.isclose(sum(ratio), 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got {ratio}")

        n = len(self._indices)
        boundaries = np.round(np.cumsum(ratio) * n).astype(int)
        splits = []
        start = 0
        for boundary in boundaries:
            splits.append(_ISPRSSubset(self._parent, self._indices[start:boundary]))
            start = boundary
        return tuple(splits)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[FloorPointCloud, FloorPlanGraph | None]:
        return self._parent[self._indices[idx]]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
