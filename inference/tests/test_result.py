"""Tests for inference.result: InferenceResult and EvaluationReport."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from inference.result import EvaluationReport, InferenceResult
from representation.floor_plan import FloorPlanGraph
from simulator.engine import FloorPointCloud, PointCloudSlice


def _square_room(size: float = 2.0) -> FloorPlanGraph:
    graph = FloorPlanGraph()
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    junctions = [graph.add_junction(corner) for corner in corners]
    for start, end in zip(junctions, junctions[1:] + junctions[:1]):
        graph.add_segment(start.id, end.id)
    return graph


def _point_cloud() -> FloorPointCloud:
    points = jnp.array([[0.5, 0.5], [1.0, 1.0], [1.5, 0.5]], dtype=jnp.float32)
    sl = PointCloudSlice(height=1.0, points=points)
    points_3d = jnp.column_stack([points, jnp.full((3,), 1.0, dtype=jnp.float32)])
    return FloorPointCloud(
        slices=[sl],
        points_3d=points_3d,
        scanner_positions=jnp.zeros((0, 2)),
        floor_plan_bounds=jnp.array([0.0, 0.0, 2.0, 2.0]),
    )


def _make_result(**overrides) -> InferenceResult:
    samples = overrides.pop("samples", [_square_room(), _square_room(2.5)])
    defaults = dict(
        samples=samples,
        log_weights=jnp.zeros((len(samples),)),
        map_estimate=_square_room(),
        junction_position_std=jnp.zeros((4, 2)),
        n_segments_distribution=jnp.array([0, 0, 0, 0, 2]),
        method="rjmcmc",
        n_likelihood_evaluations=100,
        runtime_seconds=1.5,
        acceptance_rates={"birth": 0.3, "death": 0.25},
        ess=50.0,
        log_posterior_trace=jnp.array([-10.0, -8.0, -6.0, -5.0]),
    )
    defaults.update(overrides)
    return InferenceResult(**defaults)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestInferenceResultValidation:
    def test_valid_construction(self):
        result = _make_result()
        assert result.method == "rjmcmc"

    def test_empty_samples_and_log_weights_is_valid_for_map(self):
        result = _make_result(samples=[], log_weights=jnp.zeros((0,)), method="map")
        assert result.samples == []

    def test_log_weights_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _make_result(log_weights=jnp.zeros((5,)))

    def test_negative_n_likelihood_evaluations_raises(self):
        with pytest.raises(ValueError):
            _make_result(n_likelihood_evaluations=-1)

    def test_negative_runtime_raises(self):
        with pytest.raises(ValueError):
            _make_result(runtime_seconds=-1.0)

    def test_negative_ess_raises(self):
        with pytest.raises(ValueError):
            _make_result(ess=-1.0)

    def test_acceptance_rate_outside_unit_interval_raises(self):
        with pytest.raises(ValueError):
            _make_result(acceptance_rates={"birth": 1.5})
        with pytest.raises(ValueError):
            _make_result(acceptance_rates={"birth": -0.1})


class TestPlotSamples:
    def test_raises_when_no_samples(self):
        result = _make_result(samples=[], log_weights=jnp.zeros((0,)), method="map")
        with pytest.raises(ValueError):
            result.plot_samples(_point_cloud())

    def test_creates_one_axes_per_sample(self):
        result = _make_result()
        result.plot_samples(_point_cloud())
        assert len(plt.gcf().axes) == 2

    def test_limits_to_n_samples(self):
        result = _make_result(
            samples=[_square_room(), _square_room(2.5), _square_room(3.0)],
            log_weights=jnp.zeros((3,)),
        )
        result.plot_samples(_point_cloud(), n=2)
        # 2 requested samples fit in a 1x2 grid -- no turned-off placeholder axes
        assert len(plt.gcf().axes) == 2


class TestPlotMap:
    def test_creates_a_single_axes(self):
        result = _make_result()
        result.plot_map(_point_cloud())
        assert len(plt.gcf().axes) == 1


class TestPlotDiagnostics:
    def test_creates_three_panels_when_everything_present(self):
        result = _make_result()
        result.plot_diagnostics()
        assert len(plt.gcf().axes) == 3

    def test_creates_one_panel_when_only_log_posterior_present(self):
        result = _make_result(acceptance_rates={}, n_segments_distribution=jnp.zeros((0,)))
        result.plot_diagnostics()
        assert len(plt.gcf().axes) == 1


class TestEvaluationReport:
    def test_valid_construction(self):
        report = EvaluationReport(
            chamfer_distance_segments=0.1,
            junction_localisation_error=0.05,
            room_count_accuracy=0.9,
            topology_accuracy=0.8,
        )
        assert report.room_count_accuracy == 0.9

    def test_negative_chamfer_distance_raises(self):
        with pytest.raises(ValueError):
            EvaluationReport(
                chamfer_distance_segments=-0.1,
                junction_localisation_error=0.05,
                room_count_accuracy=0.9,
                topology_accuracy=0.8,
            )

    def test_negative_junction_localisation_error_raises(self):
        with pytest.raises(ValueError):
            EvaluationReport(
                chamfer_distance_segments=0.1,
                junction_localisation_error=-0.05,
                room_count_accuracy=0.9,
                topology_accuracy=0.8,
            )

    @pytest.mark.parametrize("field", ["room_count_accuracy", "topology_accuracy"])
    def test_accuracy_outside_unit_interval_raises(self, field):
        base = dict(
            chamfer_distance_segments=0.1,
            junction_localisation_error=0.05,
            room_count_accuracy=0.9,
            topology_accuracy=0.8,
        )
        base[field] = 1.5
        with pytest.raises(ValueError):
            EvaluationReport(**base)

    def test_to_dict_from_dict_round_trip(self):
        report = EvaluationReport(
            chamfer_distance_segments=0.1,
            junction_localisation_error=0.05,
            room_count_accuracy=0.9,
            topology_accuracy=0.8,
        )
        restored = EvaluationReport.from_dict(report.to_dict())
        assert restored == report
