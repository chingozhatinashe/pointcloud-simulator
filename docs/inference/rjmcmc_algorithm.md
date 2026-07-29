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

## 5. Full algorithm pseudo-code

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
            for j in G.junctions:
                G.junctions[j].position ← sample from conditional Gaussian posterior
                    (closed form under the analytical likelihood, topology E held fixed)

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

## 6. Open items before implementation

- **`q_r(·)` must be chosen and implemented.** `SPLIT_OFFSET` currently hard-codes the
  offset radius in `split_junction`; it should instead parameterise a proposal distribution
  whose density is evaluable (needed on the right-hand side of both ratios above).
- **`merge_junctions` needs to expose the recovered `(r, θ)`** (or equivalently the
  pre-merge positions `p_a, p_b`) so the acceptance-ratio code can evaluate `q_r(r)` for the
  reverse move.
- **`docs/inference.md`'s move descriptions for split/merge** should be updated once the
  above is implemented, to reference `q_r` instead of the fixed `SPLIT_OFFSET`, mirroring
  the fix already made to the segment birth/death descriptions.

---

## 7. References

Green, P. J. (1995). Reversible jump Markov chain Monte Carlo computation and Bayesian
model determination. *Biometrika*, 82(4), 711–732. https://doi.org/10.1093/biomet/82.4.711
