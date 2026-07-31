# Inference Component

## Overview

The Inference component implements the inverse problem at the heart of the research programme: given an observed `FloorPointCloud` produced by the Data component, recover a posterior distribution over `FloorPlanGraph` objects consistent with that observation.

Formally it targets:

```
P(G | D) ∝ P(D | G) · P(G)
```

Where `P(D | G)` is the likelihood — evaluated analytically or via the Simulator — and `P(G)` is the prior over floor plan graphs encoding architectural knowledge.

The Inference component is deliberately designed as a **research platform** rather than a single algorithm. Multiple inference approaches are implemented behind a common interface so that they can be run on identical data and their outputs compared directly. The Simulator is the load-bearing dependency — most inference approaches use it either to evaluate the likelihood implicitly (ABC, pseudo-marginal MCMC) or to generate training data for learned approximations (SBI, domain randomisation).

The component depends on all three other components: Representation (for `FloorPlanGraph` and `FloorPointCloud`), Simulator (for the forward model and likelihood), and Data (for `SyntheticDataset` used in SBI training).

---

## Language and Libraries

**Primary language:** Python 3.11+

**Core libraries:**

- `JAX` — all numerical computation; gradient-based optimisation, likelihood evaluation, and array operations throughout
- `optax` — gradient-based optimisers (Adam, SGD, schedule) for MAP inference and neural network training
- `flax` — neural network definitions for SBI likelihood and posterior networks
- `blackjax` — JAX-native MCMC kernels; used as a backend for standard MCMC moves within the RJMCMC and SMC implementations
- `networkx` — graph topology operations during RJMCMC proposals (cycle detection, planarity checking)
- `dataclasses` — `InferenceConfig`, `InferenceResult`, prior parameter containers
- `numpy` — non-JAX utility operations and result post-processing
- `matplotlib` — diagnostic plots and posterior visualisation
- `tqdm` — progress tracking for long-running MCMC chains and SBI training

**Optional libraries (activated per inference method):**

- `sbi` (Python SBI library) — reference implementations of NPE, NLE, NRE as cross-validation against the custom SBI implementation
- `optuna` — hyperparameter tuning for noise model calibration and SBI training

---

## Core Responsibilities

### 1. Prior Distribution `P(G)`

The prior encodes architectural knowledge about what a valid floor plan looks like. It must be evaluable as a log probability for any proposed `FloorPlanGraph` and must support sampling — drawing random floor plans from the prior for SBI dataset generation and RJMCMC initialisation.

**Prior terms to implement:**

**Complexity prior** — penalises unnecessary segments; prefer the simplest graph consistent with the data:

```
log P_complexity(G) = -λ_k · |V|
```

Where `|V|` is the number of wall segments and `λ_k` is a configurable regularisation weight.

**Orthogonality prior** — encodes the Manhattan world assumption; walls should meet at right angles:

```
log P_ortho(G) = -λ_ortho · Σ_ℓ min(θ_ℓ mod 90°, 90° - θ_ℓ mod 90°)
```

Where `θ_ℓ` is the orientation of segment `ℓ`. Segments aligned to the dominant axis incur zero penalty.

**Minimum length prior** — assigns zero probability to segments below a minimum length threshold `ℓ_min`:

```
log P_length(G) = -∞  if any |ℓ| < ℓ_min, else 0
```

**Junction regularity prior** — prefers T-junctions and L-junctions over arbitrary-angle meetings:

```
log P_junction(G) = -λ_junc · Σ_j deviation_from_right_angle(j)
```

**Room size prior** — penalises rooms (cycles) whose area falls outside a plausible range for the building type:

```
log P_room(G) = Σ_c log N(area(c); μ_room, σ_room²)
```

Where the sum is over detected cycles `c` and `(μ_room, σ_room)` are configurable per building type.

**Combined prior:**

```
log P(G) = log P_complexity + log P_ortho + log P_length + log P_junction + log P_room
```

Each term is individually switchable — any subset can be activated for ablation studies. All `λ` parameters are configurable.

**Sampling from the prior:**

The prior must support forward sampling `G ~ P(G)` for SBI dataset generation. This is implemented as a procedural floor plan generator:

1. Sample number of rooms from a Poisson distribution
2. Sample a bounding box
3. Recursively subdivide the bounding box into rooms using axis-aligned splits (BSP tree approach)
4. Convert room boundaries to wall segments and junctions
5. Apply random perturbations within the orthogonality prior

The sampler need not produce samples from the exact prior distribution — it must produce diverse, architecturally plausible floor plans covering the range of structures the inference algorithm will encounter in real data.

---

### 2. Likelihood Model `P(D | G)`

The likelihood specifies how well a proposed floor plan `G` explains the observed point cloud `D`. Two implementations are provided, both conforming to the same interface.

**Analytical likelihood:**

The closed-form Gaussian likelihood discussed in the problem formulation. For each observed point `p` in each slice at height `h`:

```
P(p | G, h) = α(h) · N(d(p, G); 0, σ²(h)) + (1 - α(h)) · U
```

Where `d(p, G)` is the distance from point `p` to the nearest wall segment in `G`, `σ²(h)` is a height-dependent variance, `α(h)` is a height-dependent mixture weight, and `U` is a uniform clutter distribution over the floor plan bounding box.

The full likelihood across all slices, assuming independence between points and slices:

```
log P(D | G) = Σ_i Σ_{p ∈ P_i} log P(p | G, h_i)
```

This is the primary likelihood for RJMCMC and MAP inference. It is fast to evaluate — no Simulator call required — and differentiable with respect to junction positions via JAX autodiff.

**Simulator-based likelihood (implicit):**

Used by ABC and pseudo-marginal MCMC. Rather than evaluating `P(D | G)` analytically, the Simulator is called to generate `D_sim = f(G, S, η)` and the likelihood is approximated by comparing `D_sim` to `D_obs` via Chamfer distance:

```
P(D_obs | G) ≈ exp(-γ · ChamferDistance(D_sim, D_obs))
```

Where `γ` is a temperature parameter controlling the sharpness of the approximation. This is not a true likelihood but an unnormalised approximation sufficient for MCMC acceptance decisions and ABC thresholding.

**Likelihood interface:**

Both implementations conform to a common protocol so inference algorithms are agnostic to which likelihood is in use:

```python
class LikelihoodModel(Protocol):
    def log_prob(self, G: FloorPlanGraph, D: FloorPointCloud) -> float: ...
    def log_prob_batched(self, Gs: list[FloorPlanGraph], D: FloorPointCloud) -> jnp.ndarray: ...
```

---

### 3. Reversible-Jump MCMC (RJMCMC)

The reference Bayesian inference implementation. RJMCMC targets the posterior `P(G | D)` directly via a Markov chain that moves between floor plan graphs of different dimensionality, making it the most principled approach for this variable-dimension problem.

**Proposal moves:**

The RJMCMC kernel implements the following move types, each proposed with a configurable probability. A `WallSegment` stores only the IDs of the two junctions it connects — it has no free parameters of its own — so the continuous model dimension is `2 · Nj` (junction positions) regardless of how many segments exist. Moves therefore split into two kinds: **within-model** moves, which hold `Nj` fixed and so leave the continuous dimension unchanged (no Jacobian needed), and **between-model** moves, which change `Nj` and require the dimension-matching Jacobian.

*Within-model moves (`Nj` fixed, standard MH ratio, no Jacobian):*

**Segment birth** — propose adding a new wall segment between two existing junctions:
- Sample a pair of junctions `(j_a, j_b)` from the current junction set
- Propose segment `ℓ_new = (j_a, j_b)`
- Check planarity — reject proposal immediately if segment crosses existing segments
- `Nj` is unchanged; only the discrete edge set `E` gains a member, so the acceptance ratio is a plain likelihood × prior × proposal ratio — no Jacobian term

**Segment death** — propose removing an existing wall segment:
- Sample segment `ℓ` uniformly from current segments
- Reverse of birth — removes an edge from `E`; `Nj` unchanged, no Jacobian term

**Junction perturbation** — move a junction's position, updating all incident segments:
- Sample junction `j` uniformly from current junctions
- Propose new position `j' = j + δ` where `δ ~ N(0, σ_perturb²I)`
- Standard MH acceptance ratio — no dimension change
- This is the primary move for refining geometric precision given fixed topology

**Segment perturbation** — perturb a segment's orientation:
- Sample segment `ℓ` and propose a small rotation of its orientation
- Update its junction positions to maintain connectivity
- `Nj` and `E` are both unchanged — used for fine-tuning wall angles

*Between-model moves (`Nj` changes, dimension-matching Jacobian required):*

**Junction merge** — collapse two nearby junctions into one:
- Sample a pair of junctions `(j_a, j_b)` within a distance threshold
- Replace both with a single junction at their midpoint
- Update all incident segments
- `Nj → Nj - 1`; dimensionality-reducing transdimensional move

**Junction split** — split one junction into two:
- Sample a junction `j` and a split direction
- Replace with two junctions displaced along the split direction
- `Nj → Nj + 1`; dimensionality-increasing transdimensional move; reverse of merge

**Gibbs-within-RJMCMC structure:**

Junction positions are sampled via a Gibbs step conditioned on the current graph topology. Given fixed topology, junction positions have a conditionally Gaussian posterior (under the analytical likelihood) that can be sampled efficiently without MH. This dramatically improves mixing compared to proposing junction positions via random walk.

**Implementation details:**

- All MH acceptance ratio computations must include the correct Jacobian terms for transdimensional moves — this is the most error-prone aspect of RJMCMC and must be carefully validated
- The chain state is a `FloorPlanGraph`; the full chain history is stored as a list of `FloorPlanGraph` snapshots at a configurable thinning interval
- Burn-in samples are discarded; the number of burn-in steps is configurable
- Convergence diagnostics: acceptance rate per move type, log posterior trace, number of segments over iterations

---

### 4. Sequential Monte Carlo (SMC)

A population-based alternative to RJMCMC. Maintains `S` particles (candidate floor plans) and refines them by sequentially incorporating evidence from each height slice, reweighting and resampling at each step.

**Algorithm:**

```
Initialise:
    {G^(s)}_{s=1}^S ~ P(G)     — sample particles from prior
    {w^(s)} = 1/S               — uniform weights

For each height slice (h_i, P_i):
    For each particle s:
        w^(s) ← w^(s) · P(P_i | G^(s), h_i)   — reweight by slice likelihood
    Normalise weights
    If ESS < S/2:
        Resample particles by weights                — multinomial or systematic resampling
        Apply MCMC rejuvenation moves               — RJMCMC kernel steps to diversify
        Reset weights to uniform

Output:
    Weighted particle population {(G^(s), w^(s))} approximating P(G | D)
```

**Advantages over RJMCMC:**

- Naturally parallelisable — particles are independent between rejuvenation steps; `vmap` over particles where the graph representation allows
- No burn-in phase — the particle population is a valid approximation at every step
- Multi-modal posterior coverage — different particles can maintain different topological hypotheses simultaneously, avoiding the mode-collapse risk of a single MCMC chain

**Effective Sample Size (ESS) monitoring:**

Track ESS = `(Σ w^(s))² / Σ (w^(s))²` at each slice. Resampling is triggered when ESS falls below `S/2`. Log ESS per slice as a diagnostic — rapidly falling ESS indicates the likelihood is very sharp or particles have diverged from the true posterior.

**Rejuvenation kernel:**

After resampling, particles are diversified by running a short RJMCMC chain (configurable number of steps) from each particle's current state. This prevents particle collapse. The RJMCMC kernel is reused directly from §3.

---

### 5. Approximate Bayesian Computation (ABC)

The simplest simulator-based inference approach. Uses the Simulator as a black-box likelihood and accepts proposals based on similarity between simulated and observed point clouds.

**ABC-MCMC algorithm:**

```
Initialise G_0 from prior

For each step t:
    Propose G* from proposal q(G* | G_t)    — RJMCMC move
    Simulate D* = f(G*, S, η)               — call Simulator
    Compute ρ = ChamferDistance(D*, D_obs)
    If ρ < ε:
        Accept: G_{t+1} = G*
    Else:
        Reject: G_{t+1} = G_t
```

**Distance function:**

Chamfer distance computed on the full `points_3d` arrays or per-slice, configurable. Per-slice Chamfer distance is preferred as it gives a finer-grained comparison that is more sensitive to wall position errors.

**Threshold schedule:**

Rather than a fixed `ε`, implement an adaptive threshold schedule that tightens `ε` over the course of the chain as the chain converges. Start with a permissive `ε` for fast early exploration; tighten to a strict `ε` for precision in later iterations. This is the SMC-ABC variant and significantly improves efficiency.

**Summary statistics:**

Raw point clouds are high-dimensional; Chamfer distance is a lossy summary. Optionally support learned summary statistics — a neural network trained to compress `FloorPointCloud` into a fixed-dimensional vector that is more informative for distinguishing floor plans. This summary statistic network can be trained using the synthetic dataset from the Data component.

---

### 6. Simulation-Based Inference (SBI) / Neural Posterior Estimation

Trains a neural network to amortise posterior inference — once trained, the network produces an approximate posterior `q_φ(G | D) ≈ P(G | D)` with a single forward pass, eliminating the need for MCMC at test time.

**Training data:**

Generated by the Data component's `DatasetGenerator`:

```
{(G^(n), D^(n))}_{n=1}^N    where G^(n) ~ P(G), D^(n) = f(G^(n), S, η^(n))
```

With `η^(n) ~ NoiseModelPrior` for domain randomisation.

**Neural Posterior Estimation (NPE):**

Train a conditional density estimator `q_φ(G | D)` by minimising the expected KL divergence:

```
L(φ) = E_{G ~ P(G), D ~ P(D|G)} [-log q_φ(G | D)]
```

This is equivalent to maximum likelihood training on the synthetic dataset. At test time, query `q_φ(G | D_obs)` to obtain the approximate posterior.

**Architecture:**

The network has two components:

**Point cloud encoder** — maps `FloorPointCloud` to a fixed-dimensional embedding:
- Input: `points_3d` of shape `(N, 3)`
- Architecture: PointNet-style set encoder — per-point MLP followed by global max pooling
- Output: embedding vector of shape `(d_embed,)`, default `d_embed = 256`
- Must be permutation-invariant with respect to point ordering

**Floor plan decoder** — maps the embedding to a distribution over `FloorPlanGraph`:
- The variable-dimension structure of `G` makes this the hard part
- Implemented as a distribution over the padded array representation `(K_max, 4)` with a predicted validity mask
- Architecture: MLP from embedding to parameters of a normalising flow over the padded segment array
- Validity mask predicted as independent Bernoulli logits — segments with low predicted validity are excluded from the output graph

**Training loop:**

- Batch size configurable, default 64
- Optimiser: Adam via `optax` with cosine learning rate schedule
- Data loading from `SyntheticDataset` with random access batching
- Validation loss monitored on held-out split; early stopping on validation loss plateau
- Checkpointing via `flax` train state serialisation

**Inference at test time:**

```
D_obs → encoder → embedding → decoder → q_φ(G | D_obs)
```

Draw samples from `q_φ` to obtain an approximate posterior sample set. Optionally refine with a short RJMCMC chain initialised from the SBI samples — the hybrid amortised + MCMC approach.

---

### 7. MAP Inference via Differentiable Optimisation

Finds the maximum a posteriori floor plan:

```
G* = argmax_G  log P(D | G) + log P(G)
```

Using gradient-based optimisation through the differentiable Simulator and analytical likelihood.

**What is optimised:**

Given fixed graph topology (fixed number of segments and connectivity), optimise junction positions continuously via gradient descent:

```
J* = argmin_J  -log P(D | G(J)) - log P(G(J))
```

Where `J` is the `(J_max, 2)` junction position array and `G(J)` is the floor plan graph with those junction positions.

**Optimisation loop:**

```
Initialise J_0 (from prior sample or SBI output)
For each step:
    Compute log P(D | G(J_t)) via analytical likelihood or differentiable Simulator
    Compute log P(G(J_t)) via prior terms
    Compute gradient ∇_J [-log P(D | G) - log P(G)] via jax.grad
    J_{t+1} = J_t - η · gradient    (Adam update via optax)
Until convergence
```

**Topology search:**

Gradient descent cannot change the graph topology — it can only refine junction positions. Topology search is handled by an outer loop:

1. Propose a topology (number and connectivity of segments) via a discrete search or RJMCMC proposal
2. Optimise junction positions for that topology via gradient descent
3. Record the MAP value at convergence
4. Accept the topology with the best MAP value

This makes MAP inference a hybrid of discrete topology search and continuous geometry optimisation.

**Use as initialiser:**

MAP inference is most valuable as a fast initialiser for RJMCMC and SMC — running gradient descent to a good junction configuration before launching a full Bayesian sampler dramatically reduces burn-in.

---

### 8. Inference Result and Diagnostics

All inference algorithms return a common `InferenceResult` object enabling direct comparison between methods.

```python
@dataclass
class InferenceResult:
    # Posterior representation
    samples: list[FloorPlanGraph]        # posterior samples (empty for MAP)
    log_weights: jnp.ndarray            # importance weights (uniform for MCMC)
    map_estimate: FloorPlanGraph        # highest posterior sample or MAP optimum

    # Uncertainty summaries
    junction_position_std: jnp.ndarray  # per-junction position uncertainty
    n_segments_distribution: jnp.ndarray # histogram over number of segments

    # Diagnostics
    method: str                          # name of inference algorithm used
    n_likelihood_evaluations: int        # total simulator/likelihood calls
    runtime_seconds: float
    acceptance_rates: dict[str, float]   # per move type for RJMCMC
    ess: float                           # effective sample size
    log_posterior_trace: jnp.ndarray    # for convergence assessment

    # Visualisation
    def plot_samples(self, D: FloorPointCloud, n: int = 10) → None: ...
    def plot_map(self, D: FloorPointCloud) → None: ...
    def plot_diagnostics() → None: ...
```

**Evaluation metrics:**

When ground truth `G_true` is available (synthetic data or benchmark datasets):

- **Chamfer distance on segments** — mean nearest-neighbour distance between inferred and true segment midpoints
- **Junction localisation error** — mean distance between matched junctions
- **Room count accuracy** — fraction of samples with the correct number of rooms (cycles)
- **Topology accuracy** — fraction of samples with the correct graph topology (isomorphism check)

These metrics are computed by an `evaluate(result, G_true)` function and logged to a structured report.

---

### 9. Experiment Management

The Inference component is the locus of experimental runs — different algorithms, likelihood models, prior configurations, and datasets. Lightweight experiment management is needed to keep results organised and reproducible.

**Configuration:**

Every inference run is fully specified by an `InferenceConfig` dataclass:

```python
@dataclass
class InferenceConfig:
    method: str                     # "rjmcmc" | "smc" | "abc" | "sbi" | "map"
    likelihood: str                 # "analytical" | "simulator"
    prior_config: PriorConfig
    noise_model: NoiseModel
    n_iterations: int
    n_particles: int                # SMC only
    rng_seed: int
    output_dir: Path
```

**Reproducibility:**

- All randomness is seeded through JAX PRNG keys derived from `rng_seed`
- The full `InferenceConfig` is serialised to JSON alongside every result
- Simulator and Representation component versions are logged by git hash

---

## Key Design Constraints

**Common interface across all algorithms.** Every inference method accepts the same inputs — `FloorPointCloud`, `LikelihoodModel`, `PriorConfig`, `InferenceConfig` — and returns an `InferenceResult`. This is the non-negotiable design constraint that makes this a research platform rather than a collection of scripts.

**Likelihood is injected, not hardcoded.** No inference algorithm should import or instantiate a likelihood directly. The `LikelihoodModel` is always passed in from outside. This allows the analytical and simulator-based likelihoods to be swapped without touching inference code.

**RJMCMC Jacobian terms must be correct.** Errors in the MH acceptance ratio for transdimensional moves produce a biased chain that targets the wrong distribution. These terms must be derived carefully and validated against known distributions before use on real data.

**JAX PRNG discipline.** All stochastic operations use explicit JAX PRNG keys. No `numpy.random` or Python `random` calls inside any inference algorithm — this breaks reproducibility under `jit`.

**No data loading here.** The Inference component receives `FloorPointCloud` objects — it never reads files. Data loading is the Data component's responsibility.

---

## Public Interface Summary

```
run_inference(
    D: FloorPointCloud,
    config: InferenceConfig,
    likelihood: LikelihoodModel,
    rng_key: jax.random.PRNGKey,
) → InferenceResult

evaluate(
    result: InferenceResult,
    G_true: FloorPlanGraph,
) → EvaluationReport

InferenceResult
    .samples: list[FloorPlanGraph]
    .map_estimate: FloorPlanGraph
    .ess: float
    .log_posterior_trace: jnp.ndarray
    .plot_samples(D, n)
    .plot_map(D)
    .plot_diagnostics()

LikelihoodModel                          # Protocol — both implementations conform
    .log_prob(G, D) → float
    .log_prob_batched(Gs, D) → jnp.ndarray

Prior
    .log_prob(G) → float
    .sample(rng_key) → FloorPlanGraph
```

---

## Out of Scope

- Layer 1 (storey detection) — the Inference component operates on a single pre-segmented `FloorPointCloud` per storey; storey detection is a preprocessing step in the Data component
- Layer 3 (wall dimensioning) and Layer 4 (door/window detection) — the Inference component infers floor plan topology and junction positions only; downstream BIM element inference is future work
- Online or incremental inference — the component operates in batch mode on a complete `FloorPointCloud`; streaming updates as new scan data arrives are out of scope
- Multi-storey joint inference — each storey is inferred independently; joint inference across storeys is future work
- Hyperparameter optimisation — prior weights `λ` and likelihood parameters `σ(h)`, `α(h)` are set by configuration; automated tuning is out of scope beyond the noise model calibration in the Simulator component
