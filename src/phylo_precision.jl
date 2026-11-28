# PrecisionPhy — a labeled sparse phylogenetic precision in R's convention.
#
# `gllvmTMB` (the R twin) canonicalises every phylo/animal/kernel user input
# (ape tree, sparse Ainv, pedigree, dense vcv=) into ONE bundle: a sparse
# augmented precision `Ainv_phy_rr`, its `log_det_A_phy_rr`, `n_aug_phy`
# (= 2p − 2), and a `species_aug_id` tip → row map
# (`fit-multi.R:3796-3860`, `phylo-tree-precision.R:183-249`; see
# docs/dev-log/core070/phylo-transport-design.md §1.1, §2). This file gives
# GLLVModels.jl a consumer for exactly that bundle, so the bridge (a later slice,
# S3) can hand it across without either side re-deriving or re-inverting
# anything.
#
# R conventions this struct follows (differ from `AugmentedPhy`,
# `src/sparse_phy.jl`, which is leaves-first with the root INCLUDED):
#   * tree: root already dropped        -> n_aug = 2p − 2, full rank
#   * animal / sparse Ainv: tip-only    -> n_aug = p, identity tip map
#     or ancestors kept                 -> n_aug > p, tips mapped in
#   * tree node order: internal first, tips last (MCMCglmm::inverseA)
#   * optional unit-height ("correlation") scaling folded into Q already
#
# The raw-triplet constructor does not enforce 2p − 2. Admission
# (`admit_phylo_precision_payload`) requires only n_aug ≥ n_leaves.
#
# No inversion happens in this file. The `PrecisionPhy(phy::AugmentedPhy)`
# constructor only drops a row/col and permutes; the raw-triplet
# constructor only wraps what the caller (eventually the bridge) already
# built. AugmentedPhy is untouched.

using SparseArrays
using LinearAlgebra

"""
    PrecisionPhy{T}

Labeled sparse phylogenetic precision, in R's (`gllvmTMB`) convention:
root dropped, internal nodes first, tips last.

Fields
------
* `n_leaves::Int`               – number of tip / phenotyped units (p).
* `n_aug::Int`                  – rows of `Q`. Tree payloads are
  `2p − 2` (root already dropped). Animal / sparse `Ainv` payloads
  are `p` when every individual is phenotyped, or `> p` when
  unphenotyped ancestors are kept in the precision.
* `Q::SparseMatrixCSC{T,Int}`   – (n_aug × n_aug) precision. The actual
  phylogenetic precision is `Q / σ²_phy`, matching `AugmentedPhy`'s
  `Q_topology` convention. Already includes any unit-height ("correlation")
  scaling.
* `log_det::Float64`            – log-determinant of `Q` (the negative of R's
  covariance `log_det_A_phy_rr`; recompute-and-compare with
  [`precision_logdet_check`](@ref)).
* `scale::Float64`              – the height actually applied to `Q`
  (1.0 when `correlation = false`, matching R's un-scaled convention).
* `species_aug_id::Vector{Int}` – length-`n_leaves` map from tip index
  t ∈ 1:n_leaves to its row/col in `Q`.
* `node_labels::Vector{String}` – length-`n_aug` labels aligned with `Q`'s
  rows (internal-node placeholders first, then leaf names, per convention).
"""
struct PrecisionPhy{T}
    n_leaves::Int
    n_aug::Int
    Q::SparseMatrixCSC{T,Int}
    log_det::Float64
    scale::Float64
    species_aug_id::Vector{Int}
    node_labels::Vector{String}
end

#     _phylo_root_to_tip_heights(phy::AugmentedPhy) :: Vector{Float64}
#
# Root-to-every-node path length, by BFS over `phy.Q_topology`'s explicit
# off-diagonal entries (each stores `-1/branch_length` for both directions
# of an edge, so `-1/Q[i,j]` recovers the branch length). O(p).
function _phylo_root_to_tip_heights(phy::AugmentedPhy)
    n = phy.n_total
    Q = phy.Q_topology
    rows = rowvals(Q)
    vals = nonzeros(Q)
    dist = fill(NaN, n)
    dist[phy.root_index] = 0.0
    visited = falses(n)
    visited[phy.root_index] = true
    queue = Int[phy.root_index]
    head = 1
    while head <= length(queue)
        i = queue[head]
        head += 1
        for idx in nzrange(Q, i)
            j = rows[idx]
            (j == i || visited[j]) && continue
            b = -1.0 / vals[idx]
            dist[j] = dist[i] + b
            visited[j] = true
            push!(queue, j)
        end
    end
    return dist
end

#     _phylo_check_ultrametric_height(phy::AugmentedPhy) :: Float64
#
# Root-to-tip height of `phy`, after checking every leaf agrees within
# `sqrt(eps()) * height` (mirrors `phylo-tree-precision.R:140-146`). Raises
# `GJL-GATE-PHYLO-NONULTRAMETRIC` otherwise.
function _phylo_check_ultrametric_height(phy::AugmentedPhy)
    dist = _phylo_root_to_tip_heights(phy)
    leaf_heights = dist[phy.leaf_indices]
    h = leaf_heights[1]
    # Matches R's tolerance scale exactly (phylo-tree-precision.R:137-141):
    # `scale <- max(1, abs(height), abs(tip_depths))`.
    tol_scale = max(1.0, abs(h), maximum(abs.(leaf_heights)))
    tol = sqrt(eps()) * tol_scale
    maximum(abs.(leaf_heights .- h)) <= tol ||
        throw(ArgumentError("GJL-GATE-PHYLO-NONULTRAMETRIC: tree is not " *
            "ultrametric within sqrt(eps())*height (root-to-tip heights " *
            "range over $(extrema(leaf_heights))); correlation = true " *
            "requires an ultrametric tree."))
    return h
end

"""
    PrecisionPhy(phy::AugmentedPhy; correlation::Bool = false) :: PrecisionPhy{Float64}

Build the R-convention precision bundle from a native `AugmentedPhy`:
drop the root row/col, reorder to internal-first/tips-last, optionally
rescale to unit root-to-tip height (`correlation = true`, gated on
ultrametricity — see [`_phylo_check_ultrametric_height`](@ref)), and
compute `log_det`.

This feeds the same sparse-phylo Gaussian marginal likelihood kernel
(`gaussian_marginal_loglik_sparse_phy`) as `AugmentedPhy` itself, after its
own root-row deletion — the two paths must agree to machine precision
(Workflow Q check 2; see `test/test_phylo_precision.jl`).

If `phy` was itself already built with `correlation = true`
(`augmented_phy`/`make_phy`, phylo transport S2 — `phy.scale != 1.0`), the
height scaling is already baked into `phy.Q_topology`: passing
`correlation = true` here is then a no-op re-use of `phy.scale` (no double
scaling), and `correlation = false` is rejected as contradictory.
"""
function PrecisionPhy(phy::AugmentedPhy{T}; correlation::Bool = false) where {T}
    p = phy.n_leaves
    n_total = phy.n_total
    keep = filter(i -> i != phy.root_index, 1:n_total)
    Q_cond = phy.Q_topology[keep, keep]     # Julia order here: leaves 1:p, internals p+1:n_aug
    n_aug = n_total - 1
    n_aug == 2p - 2 ||
        error("internal indexing error: expected n_aug = 2p - 2 = $(2p - 2), got $n_aug")

    scale = phy.scale
    if phy.scale == 1.0
        if correlation
            scale = _phylo_check_ultrametric_height(phy)
            Q_cond = scale .* Q_cond
        end
    else
        correlation ||
            throw(ArgumentError("phy already carries scale=$(phy.scale) " *
                "(built with correlation=true); cannot build a PrecisionPhy " *
                "with correlation=false from an already unit-height-scaled " *
                "AugmentedPhy"))
        # phy.Q_topology already has phy.scale baked in — nothing to do.
    end

    # Reorder Julia's leaves-first Q_cond to R's internal-first/tips-last.
    perm = vcat(collect((p + 1):n_aug), collect(1:p))
    Q_R = SparseMatrixCSC{Float64,Int}(Q_cond[perm, perm])

    species_aug_id = collect((n_aug - p + 1):n_aug)
    internal_labels = ["internal_$(i)" for i in 1:(n_aug - p)]
    node_labels = vcat(internal_labels, phy.leaf_names)

    log_det = logdet(cholesky(Symmetric(Q_R)))

    return PrecisionPhy{Float64}(p, n_aug, Q_R, log_det, scale, species_aug_id, node_labels)
end

"""
    PrecisionPhy(I, J, V, n_aug, n_leaves, node_labels, log_det, scale, species_aug_id)
        :: PrecisionPhy{Float64}

Build a `PrecisionPhy` from raw sparse triplets `(I, J, V)` of the
precision `Q` plus the accompanying labels, checksum, scale, and tip map
— the shape a bridge (later slice) will ship across from R's
`Ainv_phy_rr` / `log_det_A_phy_rr` / `species_aug_id` bundle. Accepts
tree (`n_aug = 2p − 2`) and animal / sparse `Ainv` (`n_aug ≥ p`)
sizes. No inversion happens here; this constructor only wraps what the
caller already built.
"""
function PrecisionPhy(I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer},
                       V::AbstractVector{<:Real}, n_aug::Integer, n_leaves::Integer,
                       node_labels::AbstractVector{<:AbstractString},
                       log_det::Real, scale::Real,
                       species_aug_id::AbstractVector{<:Integer})
    length(node_labels) == n_aug ||
        throw(ArgumentError("node_labels length ($(length(node_labels))) must equal n_aug ($n_aug)"))
    length(species_aug_id) == n_leaves ||
        throw(ArgumentError("species_aug_id length ($(length(species_aug_id))) must equal n_leaves ($n_leaves)"))
    Q = sparse(collect(Int, I), collect(Int, J), Float64.(V), Int(n_aug), Int(n_aug))
    return PrecisionPhy{Float64}(Int(n_leaves), Int(n_aug), Q, Float64(log_det),
                                  Float64(scale), collect(Int, species_aug_id),
                                  collect(String, node_labels))
end

"""
    precision_logdet_check(pp::PrecisionPhy) :: (recomputed, shipped, abs_diff)

Recompute `logdet(pp.Q)` independently via a fresh sparse Cholesky factor
of the stored precision (no densification of `Q`) and compare against the
shipped/recorded `pp.log_det`. A per-fit sanity checksum for a
bridge-shipped bundle: a mismatch here means the two engines built
different Q's from what was meant to be one canonical precision.
"""
function precision_logdet_check(pp::PrecisionPhy)
    F = cholesky(Symmetric(pp.Q); check = false)
    recomputed = logdet(F)
    shipped = pp.log_det
    return recomputed, shipped, abs(recomputed - shipped)
end

# S3a payload assembly/validation lives in `src/bridge.jl`
# (`phylo_precision_payload` / `admit_phylo_precision_payload`). This file
# keeps the PrecisionPhy constructors and the independent log-det checksum.

# ---------------------------------------------------------------------------
# Dispatch into the sparse-phylo likelihood kernel: PrecisionPhy is already
# root-dropped, so this method is just field access — the AugmentedPhy
# method (in likelihood_sparse_phy.jl) does the root-row deletion PrecisionPhy
# has already performed at construction time.
# ---------------------------------------------------------------------------
_phy_cond_and_leafpos(phy::PrecisionPhy) = (phy.Q, phy.species_aug_id)
