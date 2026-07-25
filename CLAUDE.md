# CLAUDE.md

## Project Overview

This repository implements a Bayesian approach to reconstructing Building Information Models (BIM) from LiDAR point cloud data. The core research question is whether Bayesian inference can recover structured architectural representations — specifically floor plans as planar graphs — from raw 3D point cloud scans.

The pipeline targets floor plan inference as its primary research problem: given a set of 2D point cloud slices at different heights for a single building storey, infer the posterior distribution over floor plan graphs `P(G | D)`.

The codebase is structured as four components with a strict dependency order:

```
Representation  ←  Simulator  ←  Inference
      ↑                               ↑
      └──────────  Data  ─────────────┘
```

---

## Repository Structure

```
bim_inference/
│
├── CLAUDE.md                          # This file
├── README.md                          # Project overview and quickstart
├── pyproject.toml                     # Project metadata and dependencies
├── requirements.txt                   # Pinned dependency versions
│
├── representation/                    # Component 1 — floor plan data structures
│   ├── __init__.py
│   ├── primitives.py                  # Junction, WallSegment dataclasses
│   ├── floor_plan.py                  # FloorPlanGraph — core graph structure
│   ├── building.py                    # Building, Storey container structures
│   ├── constraints.py                 # Hard validators and soft penalty functions
│   ├── visualisation.py              # matplotlib plot utilities for graphs
│   └── tests/
│       ├── __init__.py
│       ├── test_primitives.py
│       ├── test_floor_plan.py
│       ├── test_building.py
│       └── test_constraints.py
│
├── simulator/                         # Component 2 — forward model
│   ├── __init__.py
│   ├── ray_caster.py                  # JAX ray-segment intersection; hard and soft occlusion
│   ├── scanner.py                     # ScannerConfig; ray sweep generation
│   ├── noise_model.py                 # NoiseModel, HeightDependentNoiseModel, NoiseModelPrior
│   ├── engine.py                      # Top-level simulate() function
│   ├── calibration.py                 # chamfer_distance(), calibrate_noise_model()
│   ├── visualisation.py              # 3D scatter, per-slice, coverage, sim-to-real plots
│   └── tests/
│       ├── __init__.py
│       ├── test_ray_caster.py
│       ├── test_scanner.py
│       ├── test_noise_model.py
│       ├── test_engine.py
│       └── test_calibration.py
│
├── data/                              # Component 3 — data ingestion and management
│   ├── __init__.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── las_loader.py              # .las / .laz ingestion via laspy
│   │   ├── ply_loader.py              # .ply / .pcd ingestion via open3d
│   │   └── e57_loader.py             # .e57 ingestion
│   ├── preprocessing.py               # Storey segmentation, slicing, outlier removal, downsampling
│   ├── normalisation.py               # Coordinate frame normalisation and transform storage
│   ├── synthetic.py                   # DatasetGenerator, SyntheticDataset (HDF5-backed)
│   ├── validation.py                  # ValidationReport and data quality checks
│   ├── datasets/                      # Benchmark dataset wrappers
│   │   ├── __init__.py
│   │   ├── scannet.py                 # ScanNet loader and ground truth extractor
│   │   ├── matterport.py             # Matterport3D loader
│   │   └── s3dis.py                  # S3DIS loader
│   └── tests/
│       ├── __init__.py
│       ├── test_loaders.py
│       ├── test_preprocessing.py
│       ├── test_normalisation.py
│       ├── test_synthetic.py
│       └── test_datasets.py
│
├── inference/                         # Component 4 — posterior inference
│   ├── __init__.py
│   ├── prior.py                       # P(G) — all prior terms and floor plan sampler
│   ├── likelihood.py                  # LikelihoodModel protocol; analytical and simulator-based
│   ├── result.py                      # InferenceResult, EvaluationReport dataclasses
│   ├── config.py                      # InferenceConfig, PriorConfig dataclasses
│   ├── evaluate.py                    # evaluate() — metrics against ground truth G
│   ├── rjmcmc/
│   │   ├── __init__.py
│   │   ├── kernel.py                  # RJMCMC Markov kernel; MH acceptance ratios
│   │   ├── proposals.py               # Birth, death, perturbation, merge, split moves
│   │   ├── gibbs.py                   # Gibbs sampler for junction positions given topology
│   │   └── diagnostics.py            # Acceptance rates, trace plots, convergence checks
│   ├── smc/
│   │   ├── __init__.py
│   │   ├── sampler.py                 # SMC main loop; reweighting and resampling
│   │   └── resampling.py             # Multinomial and systematic resampling; ESS computation
│   ├── abc/
│   │   ├── __init__.py
│   │   ├── distance.py                # Chamfer distance and summary statistics
│   │   └── sampler.py                 # ABC-MCMC; adaptive threshold schedule
│   ├── sbi/
│   │   ├── __init__.py
│   │   ├── encoder.py                 # PointNet-style point cloud encoder (Flax)
│   │   ├── decoder.py                 # Floor plan distribution decoder (Flax)
│   │   ├── network.py                 # Combined NPE network
│   │   └── training.py               # Training loop, checkpointing, validation
│   ├── map/
│   │   ├── __init__.py
│   │   └── optimiser.py              # Gradient-based junction position optimisation via optax
│   └── tests/
│       ├── __init__.py
│       ├── test_prior.py
│       ├── test_likelihood.py
│       ├── test_rjmcmc.py
│       ├── test_smc.py
│       ├── test_abc.py
│       ├── test_sbi.py
│       └── test_map.py
│
├── experiments/                       # Experiment scripts and configuration
│   ├── configs/                       # InferenceConfig JSON files per experiment
│   │   ├── rjmcmc_analytical.json
│   │   ├── rjmcmc_simulator.json
│   │   ├── smc_analytical.json
│   │   ├── abc_chamfer.json
│   │   ├── sbi_npe.json
│   │   └── map_gradient.json
│   ├── run_inference.py               # CLI entry point: loads config, runs inference, saves result
│   ├── run_sbi_training.py            # CLI entry point: generates dataset, trains SBI network
│   ├── compare_methods.py             # Runs all methods on same data; produces comparison report
│   └── calibrate_simulator.py        # Runs noise model calibration against real scans
│
├── docs/                              # Component design specifications
│   ├── representation.md              # Representation component spec
│   ├── simulator.md                   # Simulator component spec
│   ├── data.md                        # Data component spec
│   └── inference.md                   # Inference component spec
│
├── notebooks/                         # Jupyter notebooks for exploration and figures
│   ├── 01_representation_demo.ipynb   # FloorPlanGraph construction and visualisation
│   ├── 02_simulator_demo.ipynb        # Forward model demo; sim-to-real comparison
│   ├── 03_prior_samples.ipynb         # Sampling from P(G); prior visualisation
│   ├── 04_rjmcmc_synthetic.ipynb      # RJMCMC on synthetic data; recovery experiment
│   ├── 05_smc_comparison.ipynb        # SMC vs RJMCMC comparison
│   ├── 06_abc_demo.ipynb              # ABC inference walkthrough
│   ├── 07_sbi_training.ipynb          # SBI training curves and posterior quality
│   └── 08_real_data.ipynb             # Full pipeline on real LiDAR scan
│
└── assets/                            # Static assets
    └── test_data/                     # Small real and synthetic scans for unit tests
        ├── simple_room.las            # Single rectangular room scan
        ├── two_room.las               # Two connected rooms
        └── synthetic_reference.h5     # Small pre-generated synthetic dataset
```

---

## Component Dependency Map

| Component | Depends On | Do Not Import From |
|---|---|---|
| `representation` | stdlib, JAX, numpy, networkx, shapely | simulator, data, inference |
| `simulator` | representation | data, inference |
| `data` | representation | simulator, inference |
| `inference` | representation, simulator, data | — |

Violating these dependency directions introduces circular imports and couples components that should be independently testable. The `inference` component is the only one permitted to import from all others.

---

## Component Specifications

Detailed design specifications for each component live in `docs/`. These cover responsibilities, public interfaces, key design constraints, and out-of-scope items for each component. They are the authoritative reference for what each component should do and why.

| Spec | Path | Covers |
|---|---|---|
| Representation | `docs/representation.md` | `FloorPlanGraph`, `Junction`, `WallSegment`, `Building`, `Storey`, constraints, visualisation |
| Simulator | `docs/simulator.md` | Ray casting, noise model, `FloorPointCloud`, calibration, differentiable mode |
| Data | `docs/data.md` | File loaders, preprocessing, coordinate normalisation, synthetic dataset management, benchmark wrappers |
| Inference | `docs/inference.md` | Prior, likelihood, RJMCMC, SMC, ABC, SBI, MAP, `InferenceResult`, experiment management |

Read the relevant spec before implementing or modifying any component. If implementation diverges from the spec, update the spec first and treat it as the source of truth.

---

## Key Data Structures

The following types are the primary interface between components. All four components must agree on these definitions — changes require updates across all components.

```
FloorPlanGraph          representation.floor_plan
    .junctions          dict[int, Junction]
    .segments           dict[int, WallSegment]
    .to_segment_array() → jnp.ndarray      # (K_max, 4)
    .to_junction_array() → jnp.ndarray     # (J_max, 2)

Junction                representation.primitives
    .id                 int
    .position           jnp.ndarray        # (2,)

WallSegment             representation.primitives
    .id                 int
    .junction_start     int
    .junction_end       int

PointCloudSlice         simulator.engine   (also produced by data.preprocessing)
    .height             float
    .points             jnp.ndarray        # (N, 2)

FloorPointCloud         simulator.engine   (also produced by data loaders)
    .slices             list[PointCloudSlice]
    .points_3d          jnp.ndarray        # (N_total, 3)
    .scanner_positions  jnp.ndarray        # (M, 2)
    .floor_plan_bounds  jnp.ndarray        # (4,)

NoiseModel              simulator.noise_model
    .range_sigma        float
    .angular_sigma      float
    .outlier_rate       float
    .missing_rate       float
    .height_sigma       float

InferenceResult         inference.result
    .samples            list[FloorPlanGraph]
    .log_weights        jnp.ndarray
    .map_estimate       FloorPlanGraph
    .ess                float
    .log_posterior_trace jnp.ndarray
```

`PointCloudSlice` and `FloorPointCloud` are defined in `simulator/engine.py` and imported by `data/` — not the other way around. The Data component produces these types but does not define them.

---

## Language and Environment

**Python version:** 3.11+

**Package manager:** `uv`

---

### Package Management with uv

All dependency management, virtual environment creation, and script execution uses [`uv`](https://docs.astral.sh/uv/). Do not use `pip`, `conda`, or `poetry` directly.

**Initial setup:**

```bash
# Install uv (once, system-wide)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo and create the virtual environment
git clone <repo-url>
cd bim_inference
uv sync                        # creates .venv and installs all dependencies from uv.lock
```

**Running code:**

```bash
# Run a script inside the managed environment
uv run python experiments/run_inference.py --config experiments/configs/rjmcmc_analytical.json

# Launch a notebook
uv run jupyter lab

# Run the test suite
uv run pytest
```

**Adding dependencies:**

```bash
# Add a core dependency
uv add jax

# Add an optional dependency (e.g. for benchmark datasets only)
uv add --optional benchmarks trimesh

# Add a development-only dependency
uv add --dev pytest matplotlib
```

**Dependency groups:**

Dependencies are declared in `pyproject.toml` with the following groups:

```toml
[project]
dependencies = [
    # Core — required by all components
    "jax",
    "jaxlib",
    "numpy",
    "networkx",
    "shapely",
    "matplotlib",
    "laspy",
    "open3d",
    "h5py",
    "optax",
    "flax",
    "blackjax",
    "tqdm",
]

[project.optional-dependencies]
benchmarks = [
    "plyfile",
    "trimesh",
]
research = [
    "sbi",
    "optuna",
]
dev = [
    "pytest",
    "pytest-cov",
    "jupyter",
    "ipykernel",
]
```

Install optional groups as needed:

```bash
uv sync --extra benchmarks     # adds plyfile, trimesh
uv sync --extra research       # adds sbi, optuna
uv sync --extra dev            # adds pytest, jupyter
uv sync --all-extras           # installs everything
```

**Lockfile:**

`uv.lock` is committed to the repository. This pins all transitive dependencies to exact versions, ensuring reproducible environments across machines. Always commit `uv.lock` after adding or updating dependencies.

```bash
# After editing pyproject.toml manually, regenerate the lockfile
uv lock

# Upgrade a specific package and update the lockfile
uv lock --upgrade-package jax
```

**Note on JAX installation:**

JAX requires platform-specific installation for GPU/TPU support. The default `uv sync` installs the CPU-only build. For GPU support, override after syncing:

```bash
# CPU only (default)
uv sync

# CUDA 12 (after uv sync, override jaxlib)
uv run pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

### Dependency Summary by Component

| Library | Component(s) | Purpose |
|---|---|---|
| `jax` | all | Array operations, autodiff, JIT, vmap |
| `numpy` | all | Interop and non-JIT utilities |
| `networkx` | representation, inference | Graph topology operations |
| `shapely` | representation | 2D geometric predicates |
| `matplotlib` | representation, simulator, inference | Visualisation |
| `dataclasses` | all | Typed parameter containers (stdlib) |
| `laspy` | data | `.las` / `.laz` file ingestion |
| `open3d` | data | `.ply` / `.pcd` ingestion and downsampling |
| `h5py` | data | HDF5 synthetic dataset storage |
| `optax` | inference | Gradient-based optimisers |
| `flax` | inference | Neural network definitions (SBI) |
| `blackjax` | inference | JAX-native MCMC kernels |
| `tqdm` | inference | Progress tracking |

**Optional dependencies:**

| Library | Group | Purpose |
|---|---|---|
| `sbi` | `research` | Reference NPE/NLE/NRE implementations for cross-validation |
| `optuna` | `research` | Hyperparameter search |
| `plyfile` | `benchmarks` | Low-level `.ply` parsing for benchmark datasets |
| `trimesh` | `benchmarks` | Mesh-to-point-cloud conversion |
| `pytest` | `dev` | Test runner |
| `jupyter` | `dev` | Notebook environment |

---

## Unit Conventions

These conventions are enforced across the entire codebase. Mixed units are a silent failure mode.

- **Coordinates** — metres throughout; no millimetres, centimetres, or feet anywhere
- **Angles** — radians throughout; degrees only in human-readable config files and must be converted on load
- **Heights** — metres above the floor level of the current storey (Z=0 at floor)
- **Bounding boxes** — always `(x_min, y_min, x_max, y_max)` in that order
- **Segment arrays** — always `(x1, y1, x2, y2)` in that order
- **Point arrays** — always `(x, y)` for 2D, `(x, y, z)` for 3D

---

## JAX Conventions

All numerical code follows these JAX-specific rules:

- **No Python loops over arrays** — use `jax.vmap` for batched operations over rays, segments, particles, or scanner positions
- **Explicit PRNG keys everywhere** — no `numpy.random` or Python `random` calls inside any JAX-compiled function; all stochastic operations receive a `jax.random.PRNGKey`
- **Padded fixed-size arrays for graphs** — `FloorPlanGraph` exports as `(K_max, 4)` segments and `(J_max, 2)` junctions with boolean validity masks; `K_max` and `J_max` are global constants defined in `representation/floor_plan.py`
- **Immutable JAX arrays** — mutation happens on the Python `FloorPlanGraph` object; JAX array exports are generated fresh and treated as immutable
- **`jit` boundaries** — the simulator `engine.simulate()` and inference likelihood evaluation are the primary `jit` targets; do not `jit` functions that perform Python-level graph mutation

---

## Constants

Defined in `representation/floor_plan.py` and imported where needed:

```python
K_MAX = 128        # Maximum number of wall segments per floor plan
J_MAX = 256        # Maximum number of junctions per floor plan
```

Defined in `simulator/engine.py`:

```python
DEFAULT_HEIGHTS    = [0.3, 1.0, 1.8]   # metres — default slice heights
DEFAULT_VOXEL_SIZE = 0.02               # metres — default downsampling voxel size
DEFAULT_SLICE_TOL  = 0.05              # metres — default height slice tolerance
```

---

## Testing Strategy

Each component has its own `tests/` subdirectory. The most critical tests are:

**Representation**
- Cycle detection returns correct rooms for known graphs
- Planarity check correctly rejects crossing segments
- JAX array export round-trips correctly through `to_segment_array()` / `to_junction_array()`

**Simulator**
- A known rectangular room produces points only on the four wall segments
- Hard and soft occlusion agree for large `β`
- `simulate()` is `jit`-compilable and `vmap`-able without errors
- Chamfer distance between simulator output and itself is zero

**Data**
- Loader round-trips: load a `.las` file, check point count and coordinate bounds
- Preprocessing produces slices with points within `δh` of target heights
- `SyntheticDataset` loads correct `(G, D)` pairs by index

**Inference**
- Prior `log_prob` and `sample` are consistent (prior samples should have high prior probability)
- RJMCMC acceptance rates per move type are in a healthy range (10–40%) on synthetic data
- End-to-end synthetic recovery: generate `G_true`, simulate `D`, run inference, verify MAP estimate is close to `G_true` by Chamfer distance on segments

The end-to-end synthetic recovery test is the single most important test in the codebase and should be run as a nightly integration test once inference is implemented.

---

## Development Order

Components must be built in dependency order. Do not begin a component until its dependencies pass their unit tests.

```
Phase 1 — Representation
    Primitives, FloorPlanGraph, constraints, visualisation
    Exit criterion: all representation tests pass

Phase 2 — Simulator
    Ray caster, noise model, engine, calibration utilities
    Exit criterion: known geometry produces expected point density;
                   simulate() is jit-compilable

Phase 3 — Data
    Loaders, preprocessing, synthetic dataset management
    Exit criterion: real scan loads to FloorPointCloud;
                   SyntheticDataset generates and retrieves (G, D) pairs

Phase 4 — Inference
    Build in this order within the component:
        Prior and likelihood (analytical)  →  MAP  →  RJMCMC  →  SMC  →  ABC  →  SBI
    Exit criterion per method: synthetic recovery test passes
```

Do not begin Inference until the Simulator is validated against real scans. Debugging inference and simulator simultaneously is intractable.
