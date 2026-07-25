"""Lightweight matplotlib plotting utilities for FloorPlanGraph and its containers.

Development and debugging tooling only -- not part of the inference pipeline.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from representation.floor_plan import FloorPlanGraph

DEFAULT_SEGMENT_COLOR = "black"
DEFAULT_JUNCTION_COLOR = "tab:red"
DEFAULT_ROOM_COLOR = "tab:blue"
DEFAULT_ROOM_ALPHA = 0.2
DEFAULT_POINT_COLOR = "tab:green"


def plot_floor_plan(
    graph: FloorPlanGraph,
    ax: Axes | None = None,
    *,
    show_rooms: bool = True,
    show_junction_ids: bool = False,
    segment_color: str = DEFAULT_SEGMENT_COLOR,
    junction_color: str = DEFAULT_JUNCTION_COLOR,
    room_color: str = DEFAULT_ROOM_COLOR,
    room_alpha: float = DEFAULT_ROOM_ALPHA,
) -> Axes:
    """Plot a FloorPlanGraph: segments as lines, junctions as points, cycles as shaded polygons."""
    if ax is None:
        _, ax = plt.subplots()

    if show_rooms:
        for cycle in graph.cycles():
            polygon_points = np.array([graph.junctions[jid].position for jid in cycle])
            ax.add_patch(
                Polygon(
                    polygon_points,
                    closed=True,
                    facecolor=room_color,
                    alpha=room_alpha,
                    edgecolor="none",
                    zorder=0,
                )
            )

    for segment in graph.segments.values():
        start = graph.junctions[segment.junction_start].position
        end = graph.junctions[segment.junction_end].position
        ax.plot([start[0], end[0]], [start[1], end[1]], color=segment_color, linewidth=2, zorder=1)

    if graph.junctions:
        positions = np.array([junction.position for junction in graph.junctions.values()])
        ax.scatter(positions[:, 0], positions[:, 1], color=junction_color, zorder=2)

    if show_junction_ids:
        for junction in graph.junctions.values():
            ax.annotate(
                str(junction.id), (float(junction.position[0]), float(junction.position[1]))
            )

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def plot_building(storeys: Sequence[Any], *, cols: int = 1, **floor_plan_kwargs) -> Figure:
    """Plot a stack of floor plans, one subplot per storey.

    `storeys` is any sequence of objects exposing `.floor_plan` (a `FloorPlanGraph`) and
    a height range via `.z_min` / `.z_max` -- the `Building.storeys` interface described
    in docs/representation.md. Accepting duck-typed storeys here (rather than importing
    `representation.building`) keeps this module usable ahead of that container's landing.
    """
    storeys = list(storeys)
    if not storeys:
        raise ValueError("Cannot plot a building with no storeys")

    rows = -(-len(storeys) // cols)  # ceil division
    fig, axes = plt.subplots(rows, cols, squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for ax, storey in zip(flat_axes, storeys):
        plot_floor_plan(storey.floor_plan, ax=ax, **floor_plan_kwargs)
        ax.set_title(f"z ∈ [{storey.z_min:.2f}, {storey.z_max:.2f}] m")

    for ax in flat_axes[len(storeys) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_point_cloud_overlay(
    graph: FloorPlanGraph,
    points: Any,
    ax: Axes | None = None,
    *,
    point_color: str = DEFAULT_POINT_COLOR,
    point_size: float = 4.0,
    point_alpha: float = 0.6,
    **floor_plan_kwargs,
) -> Axes:
    """Overlay 2D points on a FloorPlanGraph plot, for simulator validation.

    `points` is an (N, 2) (or (N, 3), where the third column is ignored) array-like of
    scanned point coordinates -- e.g. a `FloorPointCloud` slice projected to the floor
    plane. Accepting a raw array rather than importing `simulator.engine` types respects
    the representation -> simulator dependency direction in CLAUDE.md.
    """
    ax = plot_floor_plan(graph, ax=ax, **floor_plan_kwargs)
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected points of shape (N, >=2), got {points.shape}")
    ax.scatter(
        points[:, 0], points[:, 1], color=point_color, s=point_size, alpha=point_alpha, zorder=3
    )
    return ax
