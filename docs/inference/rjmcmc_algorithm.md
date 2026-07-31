# RJMCMC Algorithm for Floor Plan Graph Inference

This document works out the model dimension, move classification, and dimension-matching
mathematics for the Reversible-Jump MCMC (RJMCMC) sampler described in
[`docs/inference.md`](../inference.md). It is a companion derivation, not a replacement —
`docs/inference.md` remains the source of truth for the prior, likelihood, and the
high-level move list; this document derives *why* the moves are classified the way they
are and the exact acceptance ratios for the transdimensional moves.

---

## 1. Model dimension

A `FloorPlanGraph` (see [`representation/floor_plan.py`](../../representation/floor_plan.py)
and [`representation/primitives.py`](../../representation/primitives.py)) consists of:

- `Nj` junctions, each a free 2D position `p ∈ ℝ²`
- `Ns` wall segments, each a `WallSegment(id, junction_start, junction_end)` — **a reference
  to two existing junction IDs, with no free parameters of its own**

Because a segment's geometry (`as_array()` → `(x1,y1,x2,y2)`) is fully resolved from the
junction positions it references, it is not an independent continuous parameter. The
continuous model dimension is therefore

```
dim(θ) = 2 · Nj
```

and does **not** grow with `Ns`. `Nj×2 + Ns×4` double-counts junction coordinates once per
incident segment.

### Model hierarchy

For a fixed `Nj`, there are `C(Nj,2) = Nj(Nj-1)/2` possible segments (unordered junction
pairs), and hence `2^(Nj(Nj-1)/2)` possible segment configurations (labelled graphs on `Nj`
junctions) — each potential edge independently present or absent. `Ns = |E|` is just the
size of whichever edge subset `E` is realised; it indexes a *discrete* choice nested inside
the continuous space fixed by `Nj`, not an independent dimension.

The full state therefore decomposes as

```
(Nj, E, θ)      θ ∈ ℝ^{2·Nj},   E ⊆ {possible junction pairs},   |E| = Ns
```

subject to the validity constraints in `FloorPlanGraph.is_valid()` (planarity, minimum
segment length, junction valence ≥ 2), which prune the reachable subset of the
`2^(Nj(Nj-1)/2)` possible graphs.

---

## 2. Move classification

Only moves that change `Nj` change `dim(θ)`. This gives a clean split of the six moves in
`docs/inference.md` into within-model and between-model moves.

### Within-model moves (`Nj` fixed — standard MH ratio, no Jacobian)

| Move | Effect |
|---|---|
| **Segment birth** | Adds an edge between two *existing* junctions. `Nj` unchanged; only the discrete edge set `E` gains a member. |
| **Segment death** | Removes an edge. `Nj` unchanged. |
| **Junction perturbation** | `p → p + δ`, `δ ~ N(0, σ_perturb² I)`. `Nj` and `E` unchanged. |
| **Segment perturbation** | Rotates a segment's orientation, adjusting its junctions' positions. `Nj` and `E` unchanged. |

These reduce to the standard Metropolis–Hastings ratio (likelihood ratio × prior ratio ×
proposal ratio) — no dimension-matching Jacobian is needed, since the continuous space
`ℝ^{2Nj}` is identical before and after the move.

### Between-model moves (`Nj` changes — Jacobian required)

| Move | Effect |
|---|---|
| **Junction split** | One junction → two. `Nj → Nj + 1`. |
| **Junction merge** | Two junctions → one. `Nj → Nj - 1`. |

These are the only moves requiring the Green (1995) dimension-matching machinery, derived
below.

---

## 3. The split/merge mapping

`split_junction` and `merge_junctions`
([`floor_plan.py:109-142`](../../representation/floor_plan.py) and
[`floor_plan.py:85-106`](../../representation/floor_plan.py)) are written as an exact
forward/reverse pair. This section derives the mapping `M` between them and its Jacobian.

### Setup

Splitting junction `j` at position `p ∈ ℝ²` into junctions `a, b` at `p_a, p_b ∈ ℝ²` is a
2 → 4 dimensional change (all other junctions and edges are untouched by the map and
contribute an identity Jacobian block).

`split_junction` draws a `direction`, normalises it to a unit vector, and offsets by a
**fixed** magnitude `SPLIT_OFFSET`. For the map to be a proper bijection between a
2-dimensional space and a 4-dimensional space, the auxiliary variable introduced by the
forward move must itself be 2-dimensional (Green's requirement:
`dim(x) + dim(u) = dim(x') + dim(u')`, and since merge is deterministic, `u' = ∅`, forcing
`dim(u) = 2`). A fixed offset magnitude only supplies 1 auxiliary dimension (the angle),
which collapses the image of the map onto a measure-zero slice of `ℝ⁴` and breaks the
Jacobian formalism.

**Consequence:** `SPLIT_OFFSET` needs to become the parameter of a proposal density `q_r(·)`
over the offset radius (e.g. `r ~ Exponential(mean = SPLIT_OFFSET)` or a truncated normal),
not a literal constant. This is flagged as an open change — not yet implemented.

### The mapping `M`

With `r ~ q_r(·)` and `θ ~ q_θ(·) = Uniform(0, 2π)`:

```
M(p, r, θ) = (p_a, p_b) = ( p + r·(cos θ, sin θ),  p − r·(cos θ, sin θ) )
```

This matches `split_junction` exactly once `r` is drawn rather than fixed.

### The inverse `M⁻¹` — confirmed against `merge_junctions`

```
p = (p_a + p_b) / 2          (midpoint)
v = (p_a − p_b) / 2
r = |v|
θ = atan2(v_y, v_x)
```

`merge_junctions` computes exactly `p = (p_a + p_b) / 2` and discards `(r, θ)` — the correct
reverse move, since the auxiliary is not retained (`u' = ∅`).

The hinge segment `add_segment(a, b)` created at the end of `split_junction` is exactly the
segment that becomes a self-loop and is dropped inside `merge_junctions`'s rewiring loop
(the `new_start == new_end` branch). The topology bookkeeping in the two functions is
already a consistent exact inverse pair — the only correction needed is the fixed-vs-random
offset radius above.

### Jacobian

Decompose `M` as:

1. Polar → Cartesian, `(r, θ) ↦ v`, with `p` held fixed: Jacobian `= r`.
2. Linear map `(p, v) ↦ (p+v, p−v)`: block matrix `[[I, I], [I, −I]]`, `det = det(AD − BC) =
   det(−2I₂) = 4`.

```
|det ∂M/∂(p, r, θ)| = 4r
```

independent of `p` and `θ`. The reverse (merge) Jacobian is `1 / (4r)` at the same `r`,
recovered via `M⁻¹`.

---

## 4. Full acceptance ratios

### Notation

- `k = Nj` — junction count of the current state `G`
- `G'` — proposed state after split (`k+1` junctions) or before merge
- `b_k` — probability the kernel proposes a split when at `k` junctions
- `d_{k+1}` — probability the kernel proposes a merge when at `k+1` junctions
- Junction `j` chosen uniformly among `k` candidates for splitting: probability `1/k`
- Pair `(a,b)` chosen uniformly among `M_{k+1}` eligible pairs (within the merge distance
  threshold) in `G'`: probability `1/M_{k+1}`
- `P(G)` — full prior (complexity + orthogonality + length + junction-regularity +
  room-size terms, §1 of `docs/inference.md`)
- `P(D|G)` — likelihood

### Proposal densities

Forward (split, `G → G'`):

```
q(G'|G) = b_k · (1/k) · q_r(r) · q_θ(θ)
```

Reverse (merge, `G' → G`):

```
q(G|G') = d_{k+1} · (1/M_{k+1})
```

### Split

```
α_split(G→G') = min{ 1,
    P(D|G')P(G')       8π · r · k · d_{k+1}
   ─────────────  ×  ─────────────────────────
    P(D|G)P(G)         b_k · M_{k+1} · q_r(r)
}
```

(obtained from Green's `α = min{1, likelihood ratio × prior ratio × proposal ratio ×
|Jacobian|}` with `|det J_M| = 4r` and `q_θ(θ) = 1/(2π)` substituted in.)

### Merge

Since merge is the exact inverse map, its ratio is the reciprocal of the split ratio's
argument, with `r` recovered from the pair being merged via `M⁻¹`:

```
α_merge(G'→G) = min{ 1,
    P(D|G)P(G)         b_k · M_{k+1} · q_r(r)
   ─────────────  ×  ─────────────────────────
    P(D|G')P(G')       8π · r · k · d_{k+1}
}
```

---

## 5. Gibbs refresh of junction positions

Section 2 classified junction perturbation as a within-model move that mixes slowly as a
plain random walk. This section derives a much faster single-site conditional draw for it,
and is explicit about what that draw approximates. Notation local to this section: `x_j`
denotes a junction *position* (to avoid clashing with an observed data point `p`); `E` is
the current edge set (topology fixed throughout this section).

### 5.1 What is actually being sampled

The target is the joint `p(θ | E, D)` over all junction positions at once, `θ = (x_1, …,
x_{Nj}) ∈ ℝ^{2Nj}`. A single Gibbs step does **not** draw this joint. It draws one
junction's position from its full conditional, holding every other junction fixed:

```
x_j | x_{-j}, E, D
```

A **sweep** (`for j in G.junctions: …`) updates every junction once, each conditioned on
the current values of the others. Repeated sweeps form a Markov chain over `θ`; it is that
chain's *stationary* distribution — not any individual step — that should equal
`p(θ|E,D)`.

### 5.2 Data augmentation needed for tractability

The raw likelihood (`docs/inference.md:114-122`) is a Gaussian/uniform mixture with a
`min`-over-segments distance, neither of which is Gaussian in a junction's position. Both
are resolved by treating two per-point quantities as fixed for the duration of a sweep
(standard data augmentation for a robust/mixture model, resampled elsewhere in the
algorithm):

- `z_p ∈ {inlier, clutter}` — which mixture component explains point `p`
- `s_p` — which segment `p` is an inlier of (resolves the `min`)

### 5.3 Per-point Gaussian contribution

For segment `ℓ = (x_j, x_o)` (incident to `j`, far endpoint `x_o`) and an observed point
`p`, `_point_to_segment_distance` (`floor_plan.py:25-36`) computes

```
t_p  = clip( (p − x_j)·(x_o − x_j) / |x_o − x_j|² , 0, 1 )
proj = (1 − t_p) x_j + t_p x_o
```

`t_p` depends on `x_j`, which is the only source of nonlinearity. Freezing `t_p` at its
current value (the same linearization ICP-style point-to-line registration uses) makes
`proj` exactly linear in `x_j`, so the inlier log-likelihood term is exactly quadratic:

```
log N(d(p,ℓ); 0, σ_p²) = −|p − t_p x_o − (1−t_p) x_j|² / (2σ_p²) + const
```

Writing `y_p = p − t_p x_o` and `c_p = 1 − t_p`, point `p` is a Gaussian "vote" for `x_j`
centred at `y_p / c_p` with precision `(c_p² / σ_p²) I₂`. `t_p=0` (foot lands at `j`)
recovers a direct vote at `p`; `t_p=1` (foot lands at the far endpoint) contributes zero
information about `x_j` (`c_p=0`) — both fall out of the formula with no special-casing.

### 5.4 Closed-form single-site conditional

Summing precisions and information vectors over every incident segment `ℓ ∈ N(j)` and
every inlier point `p` assigned to it:

```
Λ_j = [ Σ_{ℓ∈N(j)} Σ_{p∈I_ℓ}  (1−t_p)² / σ_p² ]  I₂
η_j =   Σ_{ℓ∈N(j)} Σ_{p∈I_ℓ}  [(1−t_p)/σ_p²] (p − t_p x_o(ℓ))

x_j | x_{-j}, E, D, {z_p,s_p,t_p}  ~  N( μ_j, Σ_j ),   Σ_j = Λ_j⁻¹,   μ_j = Σ_j η_j
```

`μ_j` is exactly the weighted least-squares point-to-segment fit for `j` given the frozen
correspondences; `Σ_j` is the resulting positional uncertainty.

### 5.5 The joint is a Gaussian Markov Random Field

The full (likelihood-only, associations-frozen) log-likelihood is a sum of squared linear
functions of pairs of junction positions, hence a single quadratic form in the *stacked*
`θ`:

```
log P(D | θ, E) = −½ θᵀ Λ θ + ηᵀθ + const        θ | E, D  ~  N(Λ⁻¹η, Λ⁻¹)
```

`Λ ∈ ℝ^{2Nj×2Nj}` is assembled from exactly the per-segment terms of §5.4, and critically
`Λ_{jk} = 0` whenever `j, k` are not connected by a segment — a junction only couples to its
graph neighbours. `θ` is therefore a **Gaussian Markov Random Field** whose
conditional-independence structure is exactly the floor-plan graph `G`. The standard
conditioning formula for one block of a multivariate Gaussian, specialised to this sparse
`Λ`, reduces to a sum over `j`'s neighbours only — which is exactly the §5.4 formula. That
per-junction conditional is therefore not a separate heuristic; it *is* the true full
conditional of `N(Λ⁻¹η, Λ⁻¹)`, and by the standard Gibbs-sampling guarantee (a consistent
set of full conditionals determines a unique joint; cycling through them is ergodic for it)
sweeping single-site draws from it converges exactly to that Gaussian.

### 5.6 What `N(Λ⁻¹η, Λ⁻¹)` approximates

`N(Λ⁻¹η, Λ⁻¹)` is **not** `p(θ|E,D)`. It differs by exactly the two approximations
introduced to make it tractable:

1. **Priors dropped.** `Λ, η` contain only likelihood terms; orthogonality,
   junction-regularity, and room-size priors (non-quadratic in `θ`) are excluded.
2. **Associations frozen.** `z_p, s_p, t_p` are held at their current values rather than
   resampled as functions of `θ` — a local linearization, not a global one.

So `N(Λ⁻¹η, Λ⁻¹)` is a local quadratic (Laplace-style) approximation to `p(θ|E,D)`,
recomputed fresh every sweep. Gibbs-sampling it exactly targets *that approximation's*
stationary distribution, not the true posterior — treating it as a literal always-accept
step biases the chain.

**Resolution:** use `N(μ_j, Σ_j)` as an independence-sampler MH **proposal** rather than a
raw draw. `μ_j, Σ_j` are computed once from the state at the start of the update (not
recomputed at the proposed point), so the Hastings correction only needs the fixed
proposal density evaluated at the two points:

```
α = min{ 1,
    P(D|G')P(G')      N(x_j ; μ_j, Σ_j)
   ────────────── × ─────────────────────
    P(D|G)P(G)        N(x_j'; μ_j, Σ_j)
}
```

This keeps almost all of the mixing benefit (the proposal is already close to the true
conditional's mode) while the accept/reject step absorbs the dropped priors and any
association mismatch exactly, so the chain provably targets `p(θ|E,D)` rather than the
Gaussian stand-in. The min-length/planarity constraints fall out for free as `−∞` log-prior
on violating proposals, with no need for a bespoke truncated-Gaussian sampler.

### 5.7 Pseudo-code

```
function GIBBS_SWEEP(G, D, rng_key):
    for j in G.junctions:
        x_j ← j.position
        Λ_j ← 0 (2×2);  η_j ← 0 (2-vector)

        for ℓ in G.incident_segments(j):
            x_o ← position of the other junction on ℓ
            for p in inliers_of(ℓ, D):                  # frozen z_p = inlier, s_p = ℓ
                t_p  ← projection_parameter(p, x_j, x_o) # frozen at current positions
                c_p  ← 1 − t_p
                σ_p² ← σ²(height(p))
                Λ_j  ← Λ_j + (c_p² / σ_p²) · I₂
                η_j  ← η_j + (c_p / σ_p²) · (p − t_p · x_o)

        Σ_j ← Λ_j⁻¹;  μ_j ← Σ_j η_j             # §5.4 closed form

        x_j' ← sample N(μ_j, Σ_j)                # independence-sampler proposal
        G' ← G.move_junction(j, x_j')

        α ← min(1,
              [P(D|G')P(G')] / [P(D|G)P(G)]
            × [N(x_j; μ_j, Σ_j)] / [N(x_j'; μ_j, Σ_j)]
        )
        u ← Uniform(0, 1)
        if u < α:
            G ← G'                               # else G unchanged, next junction

    return G
```

This is what the `Gibbs-within-RJMCMC` block in §6's main loop calls.

---

## 6. Full algorithm pseudo-code

```
function RJMCMC(D, prior, likelihood, config, rng_key):
    G ← config.init_graph ?? prior.sample(rng_key)      # e.g. MAP/SBI warm-start, or prior draw
    samples ← []
    diagnostics ← { move: {proposed: 0, accepted: 0} for move in MOVE_TYPES }

    for t in 1 .. config.n_iterations:
        move ← sample_categorical(rng_key, config.move_probs)
            # MOVE_TYPES = {SEGMENT_BIRTH, SEGMENT_DEATH, JUNCTION_PERTURB,
            #               SEGMENT_PERTURB, JUNCTION_SPLIT, JUNCTION_MERGE}
            # probabilities configurable, sum to 1
        diagnostics[move].proposed += 1
        k ← |G.junctions|                                # current Nj

        # ---- within-model moves: Nj fixed, no Jacobian ----------------------

        case SEGMENT_BIRTH:
            (j_a, j_b) ← sample_uniform_pair(G.junctions)
            G' ← G.add_segment(j_a, j_b)
            if not G'.is_planar(): reject; continue      # planarity check, docs/inference.md:165
            α ← min(1, [P(D|G')P(G')]/[P(D|G)P(G)] · [d_birth/b_death] · [1/N_pairs(G)])
                # N_pairs(G): number of eligible (non-adjacent, planar) junction pairs

        case SEGMENT_DEATH:
            ℓ ← sample_uniform(G.segments)
            G' ← G.remove_segment(ℓ)
            α ← min(1, [P(D|G')P(G')]/[P(D|G)P(G)] · [b_death/d_birth] · [N_pairs(G')/1])
                # reciprocal of birth; standard MH, no Jacobian

        case JUNCTION_PERTURB:
            j ← sample_uniform(G.junctions)
            δ ← N(0, σ_perturb² I)
            G' ← G.move_junction(j, j.position + δ)
            α ← min(1, [P(D|G')P(G')]/[P(D|G)P(G)])       # symmetric proposal, cancels

        case SEGMENT_PERTURB:
            ℓ ← sample_uniform(G.segments)
            Δφ ← N(0, σ_rot²)
            G' ← rotate_segment(G, ℓ, Δφ)                 # updates ℓ's junction positions
            α ← min(1, [P(D|G')P(G')]/[P(D|G)P(G)] · proposal_ratio)

        # ---- between-model moves: Nj changes, Jacobian required --------------

        case JUNCTION_SPLIT:
            j ← sample_uniform(G.junctions)               # probability 1/k
            r ← q_r(rng_key); θ ← Uniform(0, 2π)
            G' ← G.split_junction(j, direction = r·(cos θ, sin θ))
                # realises M(p, r, θ) = (p + r(cosθ,sinθ), p - r(cosθ,sinθ)); §3
            α ← min(1, [P(D|G')P(G')]/[P(D|G)P(G)] · [8π·r·k·d_merge]/[b_split·M_{k+1}·q_r(r)])

        case JUNCTION_MERGE:
            (a, b) ← sample_pair_within_threshold(G.junctions)   # probability 1/M_k
            (r, θ) ← M⁻¹(a.position, b.position)                 # recovered, §3
            G' ← G.merge_junctions(a, b)
            α ← min(1, [P(D|G)P(G)]/[P(D|G')P(G')] · [b_split·M_k·q_r(r)]/[8π·r·(k-1)·d_merge])

        # ---- accept/reject -----------------------------------------------------

        u ← Uniform(0, 1)
        if u < α:
            G ← G'
            diagnostics[move].accepted += 1

        # ---- Gibbs-within-RJMCMC: refresh junction positions given fixed topology --

        if t mod config.gibbs_interval == 0:
            G ← GIBBS_SWEEP(G, D, rng_key)                # §5

        # ---- record ----------------------------------------------------------------

        record log_posterior(G), n_segments(G) to diagnostics
        if t > config.burn_in and t mod config.thinning == 0:
            samples.append(copy(G))

    return samples, diagnostics
```

This directly instantiates the move classification (§2) and the split/merge ratios (§4);
`d_birth`, `b_death`, `b_split`, `d_merge` are the configurable per-move proposal
probabilities from `config.move_probs`, and `N_pairs(G)`, `M_k` are the eligible-candidate
counts used to make the birth/death and split/merge proposals uniform over their
respective candidate sets.

---

## 7. Open items before implementation

- **`q_r(·)` must be chosen and implemented.** `SPLIT_OFFSET` currently hard-codes the
  offset radius in `split_junction`; it should instead parameterise a proposal distribution
  whose density is evaluable (needed on the right-hand side of both ratios above).
- **`merge_junctions` needs to expose the recovered `(r, θ)`** (or equivalently the
  pre-merge positions `p_a, p_b`) so the acceptance-ratio code can evaluate `q_r(r)` for the
  reverse move.
- **`docs/inference.md`'s move descriptions for split/merge** should be updated once the
  above is implemented, to reference `q_r` instead of the fixed `SPLIT_OFFSET`, mirroring
  the fix already made to the segment birth/death descriptions.
- **The junction-position Gibbs refresh (§5) needs to be implemented as an
  independence-sampler MH step**, not a raw always-accept draw — `GIBBS_SWEEP` in §5.7
  requires evaluating `P(D|G)`, `P(G)`, and the fixed Gaussian proposal density at both the
  current and proposed positions for each junction.

---

## 8. References

Green, P. J. (1995). Reversible jump Markov chain Monte Carlo computation and Bayesian
model determination. *Biometrika*, 82(4), 711–732. https://doi.org/10.1093/biomet/82.4.711

Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*.
Chapman & Hall/CRC.
