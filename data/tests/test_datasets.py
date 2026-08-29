"""Tests for data.datasets: wall_fitting utilities and the ThreeDSESDataset wrapper.

All fixtures are small and synthetic, built on the fly -- never the real downloaded
3DSES archives (multi-GB, gitignored, not available in CI). See data/tests/test_loaders.py
for the same convention applied to the raw file-format loaders.
"""

from __future__ import annotations

import io
import zipfile

import jax.numpy as jnp
import numpy as np
import pytest

from data.datasets.cache import CachedBenchmarkDataset, cache_benchmark_dataset
from data.datasets.three_d_ses import ThreeDSESDataset
from data.datasets.wall_fitting import build_floor_plan_graph, fit_wall_segments, infer_wall_label
from representation.floor_plan import FloorPlanGraph
from simulator.engine import FloorPointCloud, PointCloudSlice


def _rectangle_perimeter_points(
    corners: list[tuple[float, float]],
    *,
    n_per_side: int,
    z_min: float,
    z_max: float,
    rng: np.random.Generator,
    noise: float = 0.005,
) -> np.ndarray:
    """(N, 3) points scattered on the vertical faces of a closed polygon's perimeter."""
    n = len(corners)
    points = []
    for i in range(n):
        p1 = np.array(corners[i])
        p2 = np.array(corners[(i + 1) % n])
        t = rng.uniform(0.05, 0.95, size=n_per_side)
        xy = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
        xy += rng.normal(scale=noise, size=xy.shape)
        z = rng.uniform(z_min, z_max, size=n_per_side)
        points.append(np.column_stack([xy, z]))
    return np.concatenate(points, axis=0)


def _horizontal_plane_points(
    *, x_range, y_range, z: float, n: int, rng: np.random.Generator, noise: float = 0.005
) -> np.ndarray:
    x = rng.uniform(*x_range, size=n)
    y = rng.uniform(*y_range, size=n)
    zz = np.full(n, z) + rng.normal(scale=noise, size=n)
    return np.column_stack([x, y, zz])


_ROOM_CORNERS = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]


def _make_labeled_room(
    rng: np.random.Generator, *, z_min: float = 0.0, z_max: float = 2.5
) -> tuple[np.ndarray, np.ndarray]:
    """(points_xyz, labels) for a synthetic rectangular room -- wall=5, floor=2, ceiling=7."""
    wall_points = _rectangle_perimeter_points(
        _ROOM_CORNERS, n_per_side=800, z_min=z_min, z_max=z_max, rng=rng
    )
    floor_points = _horizontal_plane_points(x_range=(0, 4), y_range=(0, 3), z=z_min, n=100, rng=rng)
    ceiling_points = _horizontal_plane_points(
        x_range=(0, 4), y_range=(0, 3), z=z_max, n=100, rng=rng
    )
    points = np.concatenate([wall_points, floor_points, ceiling_points], axis=0)
    labels = np.concatenate(
        [
            np.full(len(wall_points), 5),
            np.full(len(floor_points), 2),
            np.full(len(ceiling_points), 7),
        ]
    )
    return points, labels


class TestInferWallLabel:
    def test_identifies_the_wall_label(self):
        rng = np.random.default_rng(0)
        points, labels = _make_labeled_room(rng)

        assert infer_wall_label(points, labels) == 5

    def test_ignores_labels_below_min_points(self):
        rng = np.random.default_rng(1)
        points, labels = _make_labeled_room(rng)
        # a sparse decoy label, planar and vertical like a wall, but too few points to count
        decoy = _rectangle_perimeter_points(
            [(10, 10), (10.01, 10), (10.01, 10.01), (10, 10.01)],
            n_per_side=2,
            z_min=0,
            z_max=1,
            rng=rng,
        )
        points = np.concatenate([points, decoy], axis=0)
        labels = np.concatenate([labels, np.full(len(decoy), 9)])

        assert infer_wall_label(points, labels, min_points=50) == 5

    def test_raises_when_no_label_has_enough_points(self):
        rng = np.random.default_rng(2)
        points = rng.normal(size=(10, 3))
        labels = np.arange(10)

        with pytest.raises(ValueError):
            infer_wall_label(points, labels, min_points=50)


class TestFitWallSegments:
    def test_recovers_four_walls_from_a_rectangle(self):
        rng = np.random.default_rng(3)
        points_3d = _rectangle_perimeter_points(
            _ROOM_CORNERS, n_per_side=200, z_min=1.0, z_max=1.0, rng=rng, noise=0.01
        )
        points_2d = points_3d[:, :2]

        segments = fit_wall_segments(points_2d, rng=np.random.default_rng(42))

        assert len(segments) == 4
        # every fitted segment should lie close to one of the rectangle's four sides
        for segment in segments:
            start, end = segment[:2], segment[2:]
            midpoint = (start + end) / 2
            distances_to_sides = []
            for i in range(4):
                p1 = np.array(_ROOM_CORNERS[i])
                p2 = np.array(_ROOM_CORNERS[(i + 1) % 4])
                edge = p2 - p1
                t = np.clip(np.dot(midpoint - p1, edge) / np.dot(edge, edge), 0, 1)
                projection = p1 + t * edge
                distances_to_sides.append(np.linalg.norm(midpoint - projection))
            assert min(distances_to_sides) < 0.1

    def test_returns_no_segments_for_sparse_noise(self):
        rng = np.random.default_rng(4)
        points_2d = rng.uniform(0, 4, size=(10, 2))

        segments = fit_wall_segments(points_2d, min_inliers=30, rng=rng)

        assert segments == []


class TestBuildFloorPlanGraph:
    def test_snaps_nearby_endpoints_into_shared_junctions(self):
        # a rectangle whose corners are slightly mismatched between adjacent segments
        segments = [
            np.array([0.0, 0.0, 4.0, 0.02]),
            np.array([4.01, 0.0, 4.0, 3.0]),
            np.array([4.0, 3.01, 0.0, 3.0]),
            np.array([0.02, 3.0, 0.0, 0.0]),
        ]

        graph = build_floor_plan_graph(segments, junction_tolerance=0.1)

        assert isinstance(graph, FloorPlanGraph)
        assert len(graph.junctions) == 4
        assert len(graph.segments) == 4
        assert graph.is_connected()

    def test_drops_segments_that_collapse_to_a_single_junction(self):
        segments = [np.array([0.0, 0.0, 0.01, 0.01])]

        graph = build_floor_plan_graph(segments, junction_tolerance=0.1)

        assert len(graph.segments) == 0


def _write_npy_zip(path, members: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, array in members.items():
            buffer = io.BytesIO()
            np.save(buffer, array)
            archive.writestr(name, buffer.getvalue())


def _make_gold_zip(tmp_path, rng) -> None:
    points, labels = _make_labeled_room(rng)
    single_storey = np.column_stack(
        [points, np.zeros((len(points), 3)), np.ones(len(points)), labels]
    )

    lower_points, lower_labels = _make_labeled_room(rng, z_min=0.0, z_max=2.5)
    upper_points, upper_labels = _make_labeled_room(rng, z_min=3.2, z_max=5.7)
    two_storey_points = np.concatenate([lower_points, upper_points], axis=0)
    two_storey_labels = np.concatenate([lower_labels, upper_labels])
    two_storey = np.column_stack(
        [
            two_storey_points,
            np.zeros((len(two_storey_points), 3)),
            np.ones(len(two_storey_points)),
            two_storey_labels,
        ]
    )

    _write_npy_zip(
        tmp_path / "3DSES.zip",
        {"Gold/S1.npy": single_storey, "Gold/S2.npy": two_storey},
    )


def _make_test_zip(tmp_path, rng) -> None:
    points, _labels = _make_labeled_room(rng)
    unlabeled = np.column_stack([points, np.zeros((len(points), 3)), np.ones(len(points))])
    _write_npy_zip(tmp_path / "3DSES_test_area.zip", {"S170.npy": unlabeled})


class TestThreeDSESDataset:
    def test_indexes_one_item_per_storey(self, tmp_path):
        rng = np.random.default_rng(5)
        _make_gold_zip(tmp_path, rng)

        dataset = ThreeDSESDataset("gold", tmp_path)

        # S1 (single storey) + S2 (two storeys) = 3 items
        assert len(dataset) == 3

    def test_getitem_returns_point_cloud_and_floor_plan(self, tmp_path):
        rng = np.random.default_rng(6)
        _make_gold_zip(tmp_path, rng)
        dataset = ThreeDSESDataset("gold", tmp_path)

        point_cloud, floor_plan = dataset[0]

        assert isinstance(point_cloud, FloorPointCloud)
        assert point_cloud.points_3d.shape[0] > 0
        assert isinstance(floor_plan, FloorPlanGraph)
        assert len(floor_plan.segments) >= 3

    def test_unlabeled_test_variant_has_no_ground_truth(self, tmp_path):
        rng = np.random.default_rng(7)
        _make_test_zip(tmp_path, rng)
        dataset = ThreeDSESDataset("test", tmp_path)

        point_cloud, floor_plan = dataset[0]

        assert isinstance(point_cloud, FloorPointCloud)
        assert floor_plan is None

    def test_split_partitions_items(self, tmp_path):
        rng = np.random.default_rng(8)
        _make_gold_zip(tmp_path, rng)
        dataset = ThreeDSESDataset("gold", tmp_path)

        train, test = dataset.split((2 / 3, 1 / 3))

        assert len(train) + len(test) == len(dataset)

    def test_unknown_variant_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ThreeDSESDataset("platinum", tmp_path)


def _make_point_cloud(seed: int, size: float = 3.0) -> FloorPointCloud:
    rng = np.random.default_rng(seed)
    heights = [0.3, 1.0]
    slices = [
        PointCloudSlice(
            height=height,
            points=jnp.asarray(rng.uniform(0.0, size, size=(5, 2)), dtype=jnp.float32),
        )
        for height in heights
    ]
    points_3d = jnp.concatenate(
        [jnp.column_stack([sl.points, jnp.full((sl.points.shape[0],), sl.height)]) for sl in slices]
    )
    return FloorPointCloud(
        slices=slices,
        points_3d=points_3d,
        scanner_positions=jnp.zeros((0, 2)),
        floor_plan_bounds=jnp.array([0.0, 0.0, size, size]),
    )


def _make_square_graph(size: float) -> FloorPlanGraph:
    graph = FloorPlanGraph()
    corners = [graph.add_junction((x, y)) for x, y in [(0, 0), (size, 0), (size, size), (0, size)]]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        graph.add_segment(start.id, end.id)
    return graph


class _FakeBenchmarkDataset:
    """A minimal in-memory BenchmarkDataset, for exercising the cache in isolation from
    ThreeDSESDataset (and tracking how many times each item is actually derived)."""

    def __init__(self, items: list[tuple[FloorPointCloud, FloorPlanGraph | None]]) -> None:
        self._items = items
        self.access_counts = [0] * len(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[FloorPointCloud, FloorPlanGraph | None]:
        self.access_counts[idx] += 1
        return self._items[idx]


class TestCacheBenchmarkDataset:
    def test_round_trip_preserves_point_cloud_and_floor_plan(self, tmp_path):
        items = [(_make_point_cloud(i), _make_square_graph(2.0 + i)) for i in range(3)]
        dataset = _FakeBenchmarkDataset(items)
        cache_path = tmp_path / "cache.h5"

        cache_benchmark_dataset(dataset, cache_path)
        cached = CachedBenchmarkDataset(cache_path)

        assert len(cached) == 3
        for idx, (expected_point_cloud, expected_graph) in enumerate(items):
            point_cloud, floor_plan = cached[idx]
            np.testing.assert_allclose(
                np.asarray(point_cloud.points_3d), np.asarray(expected_point_cloud.points_3d)
            )
            assert isinstance(floor_plan, FloorPlanGraph)
            np.testing.assert_allclose(
                np.asarray(floor_plan.to_segment_array()),
                np.asarray(expected_graph.to_segment_array()),
            )

    def test_missing_ground_truth_round_trips_as_none(self, tmp_path):
        items = [(_make_point_cloud(0), _make_square_graph(2.0)), (_make_point_cloud(1), None)]
        dataset = _FakeBenchmarkDataset(items)
        cache_path = tmp_path / "cache.h5"

        cache_benchmark_dataset(dataset, cache_path)
        cached = CachedBenchmarkDataset(cache_path)

        _, floor_plan_0 = cached[0]
        _, floor_plan_1 = cached[1]
        assert floor_plan_0 is not None
        assert floor_plan_1 is None

    def test_resumes_without_re_deriving_already_cached_items(self, tmp_path):
        items = [(_make_point_cloud(i), _make_square_graph(2.0 + i)) for i in range(4)]
        dataset = _FakeBenchmarkDataset(items)
        cache_path = tmp_path / "cache.h5"

        # simulate an interrupted first run that only got through the first two items
        partial_dataset = _FakeBenchmarkDataset(items[:2])
        cache_benchmark_dataset(partial_dataset, cache_path)
        assert partial_dataset.access_counts == [1, 1]

        cache_benchmark_dataset(dataset, cache_path)

        assert dataset.access_counts == [0, 0, 1, 1]  # 0, 1 not re-derived; only 2, 3 were
        cached = CachedBenchmarkDataset(cache_path)
        assert len(cached) == 4

    def test_split_partitions_items(self, tmp_path):
        items = [(_make_point_cloud(i), _make_square_graph(2.0 + i)) for i in range(6)]
        dataset = _FakeBenchmarkDataset(items)
        cache_path = tmp_path / "cache.h5"
        cache_benchmark_dataset(dataset, cache_path)
        cached = CachedBenchmarkDataset(cache_path)

        train, test = cached.split((2 / 3, 1 / 3))

        assert len(train) + len(test) == len(cached)

    def test_caches_a_real_three_d_ses_dataset(self, tmp_path):
        rng = np.random.default_rng(9)
        gold_dir = tmp_path / "source"
        gold_dir.mkdir()
        _make_gold_zip(gold_dir, rng)
        dataset = ThreeDSESDataset("gold", gold_dir)
        cache_path = tmp_path / "cache.h5"

        cache_benchmark_dataset(dataset, cache_path)
        cached = CachedBenchmarkDataset(cache_path)

        assert len(cached) == len(dataset)
        point_cloud, floor_plan = cached[0]
        assert isinstance(point_cloud, FloorPointCloud)
        assert isinstance(floor_plan, FloorPlanGraph)
