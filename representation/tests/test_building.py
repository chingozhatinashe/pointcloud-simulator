"""Tests for representation.building: Building and Storey containers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from representation.building import Building, Storey
from representation.floor_plan import FloorPlanGraph


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestStorey:
    def test_holds_floor_plan_and_height_range(self):
        floor_plan = FloorPlanGraph()
        storey = Storey(z_min=0.0, z_max=3.0, floor_plan=floor_plan)
        assert storey.floor_plan is floor_plan
        assert storey.z_min == 0.0
        assert storey.z_max == 3.0

    def test_defaults_to_empty_floor_plan(self):
        storey = Storey(z_min=0.0, z_max=3.0)
        assert isinstance(storey.floor_plan, FloorPlanGraph)
        assert storey.floor_plan.junctions == {}
        assert storey.floor_plan.segments == {}

    def test_distinct_default_floor_plans_are_independent(self):
        a = Storey(z_min=0.0, z_max=3.0)
        b = Storey(z_min=3.0, z_max=6.0)
        a.floor_plan.add_junction((0.0, 0.0))
        assert a.floor_plan is not b.floor_plan
        assert b.floor_plan.junctions == {}

    def test_rejects_non_positive_height_range(self):
        with pytest.raises(ValueError):
            Storey(z_min=3.0, z_max=3.0)
        with pytest.raises(ValueError):
            Storey(z_min=3.0, z_max=0.0)


class TestBuilding:
    def test_starts_with_no_storeys(self):
        building = Building()
        assert building.storeys == []

    def test_add_storey_returns_storey_with_given_range(self):
        building = Building()
        storey = building.add_storey(0.0, 3.0)
        assert isinstance(storey, Storey)
        assert storey.z_min == 0.0
        assert storey.z_max == 3.0

    def test_add_storey_appends_in_order(self):
        building = Building()
        first = building.add_storey(0.0, 3.0)
        second = building.add_storey(3.0, 6.0)
        assert building.storeys == [first, second]

    def test_add_storey_creates_independent_floor_plans(self):
        building = Building()
        ground = building.add_storey(0.0, 3.0)
        first = building.add_storey(3.0, 6.0)
        ground.floor_plan.add_junction((0.0, 0.0))
        assert first.floor_plan.junctions == {}

    def test_plot_returns_figure_with_one_axes_per_storey(self):
        building = Building()
        building.add_storey(0.0, 3.0)
        building.add_storey(3.0, 6.0)
        fig = building.plot()
        assert len(fig.axes) == 2

    def test_plot_raises_when_no_storeys(self):
        building = Building()
        with pytest.raises(ValueError):
            building.plot()
