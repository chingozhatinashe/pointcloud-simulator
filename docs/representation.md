# Representation Component

## Overview

The Representation component is the foundational layer of the BIM inference pipeline. It defines the core data structures and geometric primitives used to represent a building floor plan as a mathematical object. Every other component — the Simulator, Inference, and Data — depends on this component, so correctness and clean interfaces here are critical.

The central object is the **FloorPlanGraph**: a planar graph where nodes are wall segments and connectivity is encoded via shared junction points. Rooms are not explicitly represented — they emerge implicitly as cycles in the graph.

---

## Language and Libraries

**Primary language:** Python 3.11+

**Core libraries:**

- `JAX` — array representation of graph geometry; all geometric arrays should be JAX-compatible pytrees to support downstream differentiation and vectorisation via `vmap`
- `numpy` — fallback for non-differentiable operations and interoperability
- `networkx` — graph-theoretic operations (cycle detection, connectivity queries, planarity checking) where JAX is not appropriate
- `dataclasses` — clean, typed primitive definitions using Python's standard library `@dataclass` decorator
- `shapely` — 2D geometric predicates (intersection testing, planarity enforcement) during graph construction and validation

---

## Core Responsibilities

### 1. Define Geometric Primitives

The component must define the atomic building blocks from which floor plans are composed.

**Junction**

A point in 2D space where two or more wall segments meet. Junctions are the shared endpoints that give the graph its connectivity structure — moving a junction moves all incident wall segments simultaneously.

- Identified by a unique integer ID
- Position represented as a 2D coordinate array `(x, y)` in metres
- Must be JAX-compatible to support gradient-based junction position refinement

**WallSegment**

A straight line segment connecting two junctions, representing a single wall in the floor plan.

- Identified by a unique integer ID
- Defined by references to two junction IDs (start and end) — **not** by raw coordinates. Coordinates are derived from the junction positions
- Carries no thickness at this stage — walls are modelled as zero-thickness line segments
- Must expose its geometry as a length-4 array `(x1, y1, x2, y2)` for use by the Simulator

**Building and Storey**

Container structures that organise floor plans into a complete building representation.

- A `Storey` holds a single `FloorPlanGraph` and a height range `(z_min, z_max)`
- A `Building` holds an ordered list of `Storey` objects
- These are lightweight containers — the geometric content lives in `FloorPlanGraph`

---

### 2. Implement the FloorPlanGraph

The `FloorPlanGraph` is the central data structure of the entire project. It must support:

**Construction and mutation**

- Add and remove junctions
- Add and remove wall segments between existing junctions
- Move a junction position (updating all incident segments implicitly)
- Merge two nearby junctions into one (dimensionality-reducing operation needed by RJMCMC)
- Split one junction into two (dimensionality-increasing operation needed by RJMCMC)

**Geometric queries**

- Return all segments as a batched array of shape `(K, 4)` — required by the Simulator for vectorised ray casting
- Return all junction positions as an array of shape `(J, 2)`
- Compute the length of a segment
- Compute the distance from an arbitrary 2D point to the nearest segment — required by the likelihood model
- Find all segments incident to a given junction

**Graph-theoretic queries**

- Detect all cycles in the graph — these are the rooms
- Check graph connectivity
- Return the adjacency matrix or edge list for use with `networkx`

**Validation**

- Check planarity — no two segments intersect except at shared junctions
- Check that all segment endpoint references point to existing junctions
- Check minimum segment length constraint

**JAX array export**

The graph must be exportable as a fixed-size padded JAX array representation for use inside `jit`-compiled and `vmap`-mapped functions:

- Segments represented as `(K_max, 4)` float array with a boolean validity mask of shape `(K_max,)`
- Junctions represented as `(J_max, 2)` float array with a boolean validity mask of shape `(J_max,)`
- `K_max` and `J_max` are configuration constants defining the maximum supported floor plan complexity

This padded representation is what the Simulator and Inference components consume. The richer Python graph object is for construction, mutation, and validation only.

---

### 3. Enforce Physical Constraints

The representation layer is responsible for knowing what constitutes a **valid** floor plan, even if inference components choose to enforce this softly. Constraints to implement:

- **Planarity** — no crossing segments (except at shared junctions)
- **Minimum segment length** — configurable threshold, default 0.1m
- **Junction valence** — junctions should have at least 2 incident segments (no dangling endpoints)
- **Coordinate bounds** — all geometry must lie within a configurable bounding box

Each constraint should be exposed both as a hard validator (raises on violation) and as a soft scalar penalty (for use as a prior term in inference).

---

### 4. Visualisation Utilities

For development and debugging, the component should provide lightweight 2D visualisation:

- Plot a `FloorPlanGraph` using `matplotlib` — segments as lines, junctions as points, cycles (rooms) as shaded polygons
- Plot a `Building` as a stack of floor plan plots
- Overlay a `FloorPointCloud` on a `FloorPlanGraph` for simulator validation

These are development utilities, not production outputs.

---

## Key Design Constraints

**JAX pytree compatibility.** The padded array export of `FloorPlanGraph` must be registered as a JAX pytree so that `jit`, `vmap`, and `grad` can operate on it transparently. This is a hard requirement — retrofitting it later is expensive.

**Immutability of the JAX representation.** The padded array form should be treated as immutable. All mutation happens on the Python graph object; a fresh export is generated when the Simulator or Inference components need it.

**No inference logic here.** The representation component defines what a floor plan *is* — it does not implement priors, likelihoods, or proposals. Those belong in the Inference component. The soft constraint penalties are the only concession to inference concerns, and they are exposed as pure functions of the graph geometry.

**Unit convention.** All coordinates are in metres. All angles are in radians. This must be enforced consistently — do not allow mixed units.

---

## Public Interface Summary

```
FloorPlanGraph
    .add_junction(position) → Junction
    .add_segment(j_start, j_end) → WallSegment
    .remove_junction(id)
    .remove_segment(id)
    .move_junction(id, new_position)
    .merge_junctions(id_a, id_b) → Junction
    .split_junction(id, direction) → (Junction, Junction)
    .to_segment_array() → jnp.ndarray          # (K_max, 4)
    .to_junction_array() → jnp.ndarray         # (J_max, 2)
    .nearest_segment_distance(point) → float
    .cycles() → list[list[int]]
    .is_planar() → bool
    .is_valid() → bool
    .planarity_penalty() → float
    .length_penalty() → float

Building
    .add_storey(z_min, z_max) → Storey
    .plot()
```

---

## Out of Scope

- Doors, windows, and openings — handled in Layer 4 of the pipeline
- Wall thickness — walls are zero-thickness line segments at this stage
- Curved walls — only straight segments are supported
- 3D geometry beyond storey height ranges — walls are extruded implicitly by the Simulator
- IFC export or import of any kind
