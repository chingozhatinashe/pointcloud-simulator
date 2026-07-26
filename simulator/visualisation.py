"""matplotlib visualisation utilities combining simulator output with the
FloorPlanGraph that generated it. Development and debugging tooling only -- no
dependency on the Inference component.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projection

from representation.floor_plan import FloorPlanGraph
from representation.visualisation import plot_floor_plan, plot_point_cloud_overlay
from simulator.calibration import point_density_by_wall
from simulator.engine import FloorPointCloud

DEFAULT_SIM_COLOR = "tab:green"
DEFAULT_REAL_COLOR = "tab:orange"


def plot_3d(
    point_cloud: FloorPointCloud,
    floor_plan: FloorPlanGraph,
    storey_height: float | None = None,
    ax: Axes3D | None = None,
    *,
    point_color: str = "tab:green",
    point_size: float = 2.0,
    wall_color: str = "black",
) -> Axes3D:
    """3D scatter of `point_cloud.points_3d` with `floor_plan`'s walls extruded to
    `storey_height` as wireframe geometry -- the primary tool for assessing
    simulator realism at a glance.

    If `storey_height` is omitted, it defaults to a small margin above the highest
    slice height in `point_cloud`.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    if storey_height is None:
        storey_height = max((sl.height for sl in point_cloud.slices), default=0.0) + 0.5

    points = np.asarray(point_cloud.points_3d)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=point_color, s=point_size)

    for segment in floor_plan.segments.values():
        start = np.asarray(floor_plan.junctions[segment.junction_start].position)
        end = np.asarray(floor_plan.junctions[segment.junction_end].position)
        ax.plot([start[0], start[0]], [start[1], start[1]], [0, storey_height], color=wall_color)
        ax.plot([end[0], end[0]], [end[1], end[1]], [0, storey_height], color=wall_color)
        ax.plot([start[0], end[0]], [start[1], end[1]], [0, 0], color=wall_color)
        ax.plot(
            [start[0], end[0]], [start[1], end[1]], [storey_height, storey_height], color=wall_color
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    return ax


def plot_slices(
    point_cloud: FloorPointCloud,
    floor_plan: FloorPlanGraph,
    *,
    cols: int = 1,
    **overlay_kwargs,
) -> Figure:
    """One 2D subplot per height slice: `floor_plan` with that slice's points
    overlaid, for diagnosing occlusion and coverage at specific heights.
    """
    slices = point_cloud.slices
    if not slices:
        raise ValueError("point_cloud has no slices to plot")

    rows = -(-len(slices) // cols)  # ceil division
    fig, axes = plt.subplots(rows, cols, squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for ax, sl in zip(flat_axes, slices):
        plot_point_cloud_overlay(floor_plan, sl.points, ax=ax, **overlay_kwargs)
        ax.set_title(f"h = {sl.height:.2f} m")

    for ax in flat_axes[len(slices) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_coverage_heatmap(
    points: Any,
    floor_plan: FloorPlanGraph,
    ax: Axes | None = None,
    *,
    cmap: str = "viridis",
) -> Axes:
    """Colour each wall segment by its point density (points per metre), highlighting
    under-sampled walls.
    """
    if ax is None:
        _, ax = plt.subplots()

    density = point_density_by_wall(points, floor_plan)
    if not density:
        ax.set_aspect("equal")
        return ax

    values = np.array(list(density.values()))
    vmin, vmax = float(values.min()), float(values.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
    colormap = plt.get_cmap(cmap)

    for segment_id, value in density.items():
        segment = floor_plan.segments[segment_id]
        start = floor_plan.junctions[segment.junction_start].position
        end = floor_plan.junctions[segment.junction_end].position
        ax.plot(
            [float(start[0]), float(end[0])],
            [float(start[1]), float(end[1])],
            color=colormap(norm(value)),
            linewidth=4,
        )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    plt.colorbar(mappable, ax=ax, label="points / m")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def plot_sim_to_real_overlay(
    simulated: FloorPointCloud,
    real: FloorPointCloud,
    floor_plan: FloorPlanGraph,
    ax: Axes | None = None,
    *,
    sim_color: str = DEFAULT_SIM_COLOR,
    real_color: str = DEFAULT_REAL_COLOR,
    point_size: float = 4.0,
    point_alpha: float = 0.6,
) -> Axes:
    """Overlay simulated and real point clouds (projected to the floor plane) on
    `floor_plan` in contrasting colours, to visualise the sim-to-real gap.
    """
    ax = plot_floor_plan(floor_plan, ax=ax)
    sim_points = np.asarray(simulated.points_3d)
    real_points = np.asarray(real.points_3d)
    ax.scatter(
        sim_points[:, 0],
        sim_points[:, 1],
        color=sim_color,
        s=point_size,
        alpha=point_alpha,
        label="simulated",
        zorder=3,
    )
    ax.scatter(
        real_points[:, 0],
        real_points[:, 1],
        color=real_color,
        s=point_size,
        alpha=point_alpha,
        label="real",
        zorder=4,
    )
    ax.legend()
    return ax
