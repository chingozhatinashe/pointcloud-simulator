# Simulator Component

## Overview

The Simulator component implements the forward model of the BIM inference pipeline. It takes a `FloorPlanGraph` from the Representation component and generates a synthetic LiDAR point cloud — the data that would be observed if a real scanner were placed inside that floor plan.

Formally the simulator implements:

```
f(G, S, η) → D
```

Where `G` is a `FloorPlanGraph`, `S` is a set of scanner positions, `η` is a noise model parametrisation, and `D` is a `FloorPointCloud` — a collection of 2D point cloud slices at different heights.

This forward model is the most leveraged piece of the entire codebase. It serves multiple roles simultaneously:

- **Likelihood evaluation** — implicitly defines `P(D | G)` for ABC and pseudo-marginal MCMC
- **Training data generation** — produces unlimited `(G, D)` pairs for neural likelihood and posterior estimation (SBI)
- **Validation tool** — comparing simulator output against real scans calibrates the noise model and quantifies the sim-to-real gap
- **Gradient source** — if implemented differentiably, enables gradient-based MAP inference and differentiable fine-tuning

Because the simulator underpins every inference approach, its correctness and physical realism directly bound the quality of all downstream results.

---

## Language and Libraries

**Primary language:** Python 3.11+

**Core libraries:**

- `JAX` — primary implementation language for all geometric computations. Ray casting, intersection testing, noise sampling, and point cloud assembly must all be implemented as JAX operations to enable `jit` compilation, `vmap` vectorisation over rays and scanner positions, and `grad` for differentiable simulation
- `numpy` — interoperability and non-JIT utility code only
- `dataclasses` — `ScannerConfig` and `NoiseModel` parameter containers

**JAX-specific features in active use:**

- `jax.vmap` — vectorise ray casting over all rays and all scanner positions simultaneously, avoiding Python loops
- `jax.jit` — compile the full simulation forward pass for significant speed gains
- `jax.grad` / `jax.jacobian` — differentiate simulator outputs with respect to junction positions for gradient-based inference
- `jax.random` — explicit PRNG key management for reproducible stochastic simulation

---

## Core Responsibilities

### 1. Ray Casting Engine

The geometric core of the simulator. Given a set of rays emitted from a scanner position and a `FloorPlanGraph`, compute where each ray intersects the wall segments.

**Inputs:**
- Ray origins and directions — arrays of shape `(R, 2)` and `(R, 2)` for `R` rays
- Wall segments from `FloorPlanGraph.to_segment_array()` — shape `(K, 4)`

**Outputs:**
- Intersection distances along each ray — shape `(R,)`
- Hit mask — boolean array of shape `(R,)` indicating which rays struck a wall
- Intersection coordinates — shape `(R, 2)`

**Occlusion handling** is the key design decision here. Two modes must be supported, controlled by a flag:

**Hard occlusion (default):** Each ray returns only the nearest wall intersection — all walls behind the first hit are invisible to that ray. This is physically correct but involves an `argmin` over intersections which is not differentiable. Used for ABC, pseudo-marginal MCMC, and dataset generation.

**Soft occlusion (differentiable):** The nearest intersection is approximated via a softmin over all intersections:

```
d_return = sum_j(d_j * exp(-β * d_j)) / sum_j(exp(-β * d_j))
```

As `β → ∞` this converges to hard occlusion. Differentiable with respect to segment endpoint positions. Used when gradients are required. `β` is a configurable temperature parameter.

The entire ray casting pass must be implemented as a single `vmap`-able function so that casting all rays from all scanner positions in a scene can be expressed as a batched JAX operation with no Python-level loops.

---

### 2. Scanner Model

Defines how a physical LiDAR scanner emits rays into the scene.

**Scanner configuration** (parametrised via `ScannerConfig` dataclass):

- Position in 2D floor plan coordinates `(x, y)`
- Angular sweep range — default full 360°
- Angular resolution — number of rays per sweep, default 360 (1° spacing) but configurable up to realistic densities (~3600 rays)
- Height of the scan slice `h` — determines which horizontal plane is being simulated

**Multi-scanner support:**

A realistic LiDAR survey uses multiple scanner positions per floor to achieve full coverage. The simulator must accept a list of scanner positions and aggregate their returns into a single point cloud. `vmap` over scanner positions makes this efficient.

**Scanner position as an explicit model input:**

Scanner positions are passed in explicitly at every simulator call — they are never hardcoded. This is important because:

- In real surveys scanner positions are known (from registration)
- In synthetic generation they can be randomised for domain randomisation
- In future work they could be treated as latent variables and inferred jointly with `G`

---

### 3. Noise Model

Converts clean geometric ray intersections into realistic noisy point cloud observations. Implemented as a `NoiseModel` dataclass with explicit, controllable parameters:

```python
@dataclass
class NoiseModel:
    range_sigma: float        # Gaussian noise on return distance (metres)
    angular_sigma: float      # Perturbation on ray direction (radians)
    outlier_rate: float       # Fraction of spurious clutter returns [0, 1]
    missing_rate: float       # Fraction of valid hits randomly dropped [0, 1]
    height_sigma: float       # Uncertainty in height slice assignment (metres)
```

**Noise pipeline per ray:**

1. Cast ray geometrically to get clean intersection `d_clean`
2. Apply angular jitter: perturb ray direction by `N(0, angular_sigma²)` before casting
3. Apply range noise: `d_noisy = d_clean + N(0, range_sigma²)`
4. Apply missing return: drop the point with probability `missing_rate` (models glass, absorption, specular surfaces)
5. Apply clutter: with probability `outlier_rate` replace with a uniformly random point in the bounding box (models multi-path returns, dust, equipment)

The noise model must be implemented as a **probabilistic program** — each stochastic element is an explicit JAX random draw keyed by a PRNG key. This ensures reproducibility and makes the noise model auditable.

**Height-dependent noise:**

The noise parameters can optionally be functions of slice height `h` rather than fixed scalars, reflecting the physical reality that some heights are noisier than others. A `HeightDependentNoiseModel` subclass should support this by accepting callable parameters `range_sigma(h)`, `missing_rate(h)` etc.

---

### 4. Height Slice Generation

Assembles the full multi-slice output `D = {(h_i, P_i)}` from individual height-level simulations.

**Process per height slice `h_i`:**

1. Configure scanner at height `h_i`
2. Cast rays from all scanner positions at that height
3. Apply noise model
4. Collect surviving points into `P_i` — a 2D point array of shape `(N_i, 2)`

**Height schedule:**

The set of heights `{h_i}` to simulate is configurable. Defaults should reflect typical survey practice:

- Low slice at ~0.3m — captures wall bases cleanly
- Mid slice at ~1.0m — main wall surface, some furniture occlusion
- Upper slice at ~1.8m — above most furniture, below door lintels

Additional heights can be added freely. The simulator should accept an arbitrary list of heights.

**Output format:**

```python
@dataclass
class PointCloudSlice:
    height: float                    # metres
    points: jnp.ndarray             # shape (N, 2)

@dataclass
class FloorPointCloud:
    slices: list[PointCloudSlice]
    points_3d: jnp.ndarray          # shape (N_total, 3) — all points with z coordinate
    scanner_positions: jnp.ndarray  # shape (M, 2)
    floor_plan_bounds: jnp.ndarray  # shape (4,) — bounding box (x_min, y_min, x_max, y_max)
```

`points_3d` is assembled by concatenating all slices and appending the slice height as the Z coordinate:

```python
points_3d = jnp.concatenate([
    jnp.column_stack([s.points, jnp.full((len(s.points),), s.height)])
    for s in slices
])
```

This field is redundant with `slices` but stored explicitly for two reasons. First, it is the natural format for 3D visualisation — a single scatter plot of `points_3d` overlaid against the `Building` geometry gives an immediate picture of simulator coverage and noise. Second, real LiDAR data loaded by the Data component is natively 3D; storing `points_3d` keeps the real and synthetic formats fully aligned so inference code never needs to distinguish between them.

This is the canonical data format consumed by the Inference component and produced by the Data component when loading real scans.

---

### 5. Differentiable Mode

When inference requires gradients — specifically for differentiable MAP optimisation and neural likelihood training — the simulator must be end-to-end differentiable with respect to junction positions.

**What is differentiable:**
- Junction positions `(x, y)` in `FloorPlanGraph.to_junction_array()`
- Segment geometry derived from junction positions
- Soft occlusion ray returns
- Gaussian range noise (via the reparametrisation trick)

**What is not differentiable:**
- Graph topology — the number and connectivity of segments is discrete
- Hard occlusion `argmin`
- Clutter and missing return Bernoulli draws (though these can be handled via straight-through estimators if needed)

The differentiable mode should be activated by passing `differentiable=True` to the engine, which internally switches to soft occlusion and ensures all operations are JAX-traceable. Gradients flow from point cloud outputs back to junction positions via `jax.grad`.

---

### 6. Validation and Calibration Utilities

The simulator is only useful if it produces point clouds that resemble real scans. This component must provide tools to measure and reduce the sim-to-real gap.

**Comparison metrics:**

- **Chamfer distance** — mean nearest-neighbour distance between simulated and real point clouds; primary calibration metric
- **Point density distributions** — histogram of point counts per unit wall length; checks that coverage is realistic
- **Height slice consistency** — verify that the same wall appears at consistent positions across height slices

**Calibration workflow:**

Given a real scan of a known floor plan (ground truth `G` available):

1. Run simulator with ground truth `G` and a candidate `NoiseModel`
2. Compute Chamfer distance between simulated and real `FloorPointCloud`
3. Optimise `NoiseModel` parameters to minimise Chamfer distance — this is a small continuous optimisation over `[range_sigma, angular_sigma, outlier_rate, missing_rate]`
4. Report final Chamfer distance as the sim-to-real gap estimate

**Domain randomisation support:**

For SBI training, the simulator must support randomising `NoiseModel` parameters over a prior distribution per simulation call. This produces training data that is robust to variation in scan quality. A `NoiseModelPrior` dataclass specifies the sampling distribution over each noise parameter.

---

### 7. Visualisation Utilities

For development and debugging, the simulator provides visualisation tools that combine the `FloorPointCloud` with the `Building` or `FloorPlanGraph` that generated it.

- **3D scatter plot** — render `points_3d` as a 3D scatter using `matplotlib` with the wall segments extruded to storey height overlaid as wireframe geometry. The primary tool for assessing simulator realism at a glance
- **Per-slice 2D plot** — render each `PointCloudSlice` as a 2D scatter overlaid on the `FloorPlanGraph`, one subplot per height. Useful for diagnosing occlusion and coverage at specific heights
- **Coverage heatmap** — render point density per unit wall length across the floor plan, highlighting under-sampled walls
- **Sim-to-real overlay** — given both a simulated and a real `FloorPointCloud` for the same floor, render them in contrasting colours to visualise the sim-to-real gap

These are development utilities only. They live in `simulator/visualisation.py` and have no dependencies on the Inference component.

---

## Key Design Constraints

**JAX-first implementation.** Every geometric operation in the ray caster must be a JAX array operation. No Python loops over rays or segments — use `vmap`. This is non-negotiable: the simulator will be called millions of times during SBI training and must be fast.

**Explicit PRNG key management.** JAX requires explicit random keys — there is no global random state. Every stochastic call in the simulator takes a `jax.random.PRNGKey` and returns a new key. The engine manages key splitting internally and exposes a single `rng_key` parameter at the top level.

**Hard and soft occlusion are separate code paths.** Do not try to unify them. Hard occlusion should be the default and is used for all non-differentiable inference. Soft occlusion is activated explicitly. The two modes should produce nearly identical outputs for large `β` — this can be tested.

**Scanner positions are always explicit inputs.** Never hardcode or infer scanner positions inside the simulator. They are always passed in from outside, keeping the simulator a pure function of its inputs.

**The simulator is a pure function.** `simulate(G, scanner_config, noise_model, rng_key) → FloorPointCloud` has no side effects and no internal state. This is required for `jit` compilation and for safe parallel execution during SMC.

---

## Public Interface Summary

```
simulate(
    G: FloorPlanGraph,
    scanner_positions: jnp.ndarray,     # (M, 2)
    heights: list[float],
    noise_model: NoiseModel,
    rng_key: jax.random.PRNGKey,
    differentiable: bool = False,
) → FloorPointCloud

chamfer_distance(
    D_sim: FloorPointCloud,
    D_real: FloorPointCloud,
) → float

calibrate_noise_model(
    G_known: FloorPlanGraph,
    D_real: FloorPointCloud,
    scanner_positions: jnp.ndarray,
    noise_prior: NoiseModelPrior,
) → NoiseModel

FloorPointCloud
    .slices: list[PointCloudSlice]
    .points_3d: jnp.ndarray                         # (N_total, 3)
    .scanner_positions: jnp.ndarray                 # (M, 2)
    .floor_plan_bounds: jnp.ndarray                 # (4,)
    .plot(G: FloorPlanGraph, mode: str = "3d")      # "3d" | "slices" | "coverage"

NoiseModel
    .range_sigma: float
    .angular_sigma: float
    .outlier_rate: float
    .missing_rate: float
    .height_sigma: float
    .randomise(rng_key) → NoiseModel    # sample from NoiseModelPrior
```

---

## Out of Scope

- 3D volumetric ray casting — simulation is 2D per height slice only
- Multiple return LiDAR — only single return (nearest hit) is modelled
- Reflectance / intensity modelling — points carry position only, no intensity value
- Scanner position optimisation — scanner positions are inputs, not inferred here
- Doors and openings — walls are solid with no gaps at this stage; openings belong to Layer 4
- Real-time simulation — the simulator is designed for batch offline use, not interactive rendering
