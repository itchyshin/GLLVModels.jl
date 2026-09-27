# `fit_phylo_latent_gllvm`: the Julia twin of gllvmTMB's bare Gaussian
# `phylo_latent(species, d = K)` at gllvmTMB P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9, version 0.7.1).
#
# The model is R's `phylo_rr` block: per-trait intercepts, a rank-K
# lower-triangular trait loading matrix on K independent phylogenetic fields
# with prior precision `A^{-1}` over the augmented (root-dropped) nodes, and
# one shared Gaussian residual. It is fitted by the existing
# `fit_precision_multivariate` path; this file only admits R's inputs, matches
# species by label and ports R's refusals. See
# `docs/design/phylo-latent-port-spec.md` (draft PR #545) for the R citations.
#
# Two P1 facts differ from the spec and are recorded in the decisions note:
#  * R's in-keyword `phylo_latent(Ainv = )` is rewritten to
#    `vcv = solve(as.matrix(Ainv))` (R/brms-sugar.R:3311-3319), i.e. the dense
#    ridged route, while a global sparse `phylo_vcv` takes the sparse direct
#    route. Which one the Julia `Ainv` keyword twins is a maintainer question,
#    so `Ainv` is refused here for now (GJL-GATE-PHYLO-LATENT-AINV).
#  * R's tree validator (R/phylo-tree-precision.R:70-160) does not refuse a
#    node of out-degree one, so no unary-node refusal is added.

using LinearAlgebra
using SparseArrays

const _PHYLO_LATENT_RIDGE = 1e-8          # R/fit-multi.R:4741, replicated exactly
const _PHYLO_LATENT_KAPPA_WARN = 1e8       # Q2 condition-number warning (never a refusal)

@noinline function _phylo_latent_refuse(sentence::AbstractString, tag::AbstractString,
        hint::AbstractString = "")
    msg = isempty(hint) ? "$(sentence) ($(tag))" : "$(sentence) ($(tag)) $(hint)"
    throw(ArgumentError(msg))
end

# A general rooted tree as parent/child/edge-length arrays, tips first in
# tip-label order. Accepts a Newick string (polytomies and unary nodes are
# represented as R represents them) or an `AugmentedPhy`.
function _phylo_latent_tree_edges(tree::AbstractString)
    s = filter(!isspace, String(tree))
    parsed = try
        endswith(s, ";") || error("no terminating ';'")
        body = s[1:prevind(s, lastindex(s))]
        c = _NewickCursor(body, firstindex(body))
        node_parent = Int[]; node_is_leaf = Bool[]; node_name = String[]
        node_length = Float64[]; leaf_indices = Int[]; leaf_names = String[]
        root = _parse_node!(c, node_parent, node_is_leaf, node_name, node_length,
            leaf_indices, leaf_names)
        c.i <= lastindex(c.s) && error("extra characters after the tree")
        (node_parent, node_is_leaf, node_length, leaf_indices, leaf_names, root)
    catch err
        err isa InterruptException && rethrow()
        _phylo_latent_refuse("tree must be a phylogeny object.",
            "GJL-GATE-PHYLO-LATENT-TREE",
            "The Newick string could not be parsed: $(sprint(showerror, err)).")
    end
    node_parent, node_is_leaf, node_length, leaf_indices, leaf_names, root = parsed
    n_total = length(node_parent)
    n_tip = length(leaf_indices)
    # Renumber: tips 1:n_tip in encounter order, internal nodes after.
    new_id = zeros(Int, n_total)
    for (k, old) in enumerate(leaf_indices)
        new_id[old] = k
    end
    next = n_tip
    for old in 1:n_total
        node_is_leaf[old] && continue
        next += 1
        new_id[old] = next
    end
    parent = Int[]; child = Int[]; len = Float64[]
    for old in 1:n_total
        node_parent[old] == 0 && continue
        push!(parent, new_id[node_parent[old]])
        push!(child, new_id[old])
        push!(len, node_length[old])
    end
    return (n_tip = n_tip, n_total = n_total, tip_labels = collect(String, leaf_names),
            parent = parent, child = child, edge_length = len, root = new_id[root])
end

function _phylo_latent_tree_edges(tree::AugmentedPhy)
    p = tree.n_leaves
    n_total = tree.n_total
    Q = tree.Q_topology
    rows = rowvals(Q); vals = nonzeros(Q)
    # Tips first (tip-label order), internal nodes after.
    order = vcat(tree.leaf_indices, setdiff(1:n_total, tree.leaf_indices))
    new_id = zeros(Int, n_total)
    new_id[order] = 1:n_total
    parent = Int[]; child = Int[]; len = Float64[]
    visited = falses(n_total)
    visited[tree.root_index] = true
    queue = [tree.root_index]
    head = 1
    while head <= length(queue)
        i = queue[head]; head += 1
        for idx in nzrange(Q, i)
            j = rows[idx]
            (j == i || visited[j]) && continue
            visited[j] = true
            push!(queue, j)
            push!(parent, new_id[i]); push!(child, new_id[j])
            # Off-diagonal entries store -scale / branch_length.
            push!(len, -tree.scale / vals[idx])
        end
    end
    return (n_tip = p, n_total = n_total, tip_labels = copy(tree.leaf_names),
            parent = parent, child = child, edge_length = len,
            root = new_id[tree.root_index])
end

_phylo_latent_tree_edges(::Any) = _phylo_latent_refuse("tree must be a phylogeny object.",
    "GJL-GATE-PHYLO-LATENT-TREE",
    "Pass a Newick string or an AugmentedPhy (augmented_phy / make_phy).")

"""
    _phylo_latent_tree_precision(tree) -> (phy::PrecisionPhy, tip_labels)

R's `.gllvm_phylo_tree_precision(tree, correlation = TRUE)`
(`R/phylo-tree-precision.R:183-249`): validate, keep every non-root node
(internal nodes first, tips last in tip-label order), add `1/edge_length`
blocks with the root row dropped, scale by the root-to-tip height, and record
`log_det = n_aug * log(height) - sum(log(edge_length))`. Polytomies give
`n_aug = n_tip + Nnode - 1`, exactly as in R.
"""
function _phylo_latent_tree_precision(tree)
    e = _phylo_latent_tree_edges(tree)
    n_tip, n_total = e.n_tip, e.n_total
    tag = "GJL-GATE-PHYLO-LATENT-TREE"
    labels = e.tip_labels
    (n_tip >= 2 && all(!isempty, labels)) ||
        _phylo_latent_refuse("tree must contain at least two non-missing tip labels.", tag)
    length(unique(labels)) == n_tip ||
        _phylo_latent_refuse("tree tip labels must be unique.", tag)
    (all(isfinite, e.edge_length) && length(e.edge_length) == n_total - 1) ||
        _phylo_latent_refuse("tree must contain finite branch lengths for every edge.", tag)
    all(>(0), e.edge_length) ||
        _phylo_latent_refuse("tree branch lengths must be positive to build sparse precision.", tag)
    # Root-to-node depths.
    depth = fill(NaN, n_total)
    depth[e.root] = 0.0
    children = [Int[] for _ in 1:n_total]
    elen = zeros(n_total)
    for k in eachindex(e.child)
        push!(children[e.parent[k]], e.child[k])
        elen[e.child[k]] = e.edge_length[k]
    end
    stack = [e.root]
    while !isempty(stack)
        i = pop!(stack)
        for c in children[i]
            depth[c] = depth[i] + elen[c]
            push!(stack, c)
        end
    end
    tip_depths = depth[1:n_tip]
    height = tip_depths[1]
    tol = sqrt(eps(Float64)) * max(1.0, abs(height), maximum(abs, tip_depths))
    maximum(abs.(tip_depths .- height)) <= tol ||
        _phylo_latent_refuse("tree must be ultrametric. Root-to-tip distances differ by more than $(sqrt(eps(Float64))).",
            "GJL-GATE-PHYLO-NONULTRAMETRIC")
    all(>(0), tip_depths) ||
        _phylo_latent_refuse("tree must have positive root-to-tip height.", tag)
    # R's node order: internal (non-root) nodes first, tips last.
    internal = [i for i in (n_tip + 1):n_total if i != e.root]
    included = vcat(internal, collect(1:n_tip))
    n_aug = length(included)
    index = zeros(Int, n_total)
    index[included] = 1:n_aug
    I = Int[]; J = Int[]; V = Float64[]
    for k in eachindex(e.child)
        c = index[e.child[k]]
        w = 1.0 / e.edge_length[k]
        push!(I, c); push!(J, c); push!(V, w)
        if e.parent[k] != e.root
            pa = index[e.parent[k]]
            append!(I, (pa, pa, c)); append!(J, (pa, c, pa)); append!(V, (w, -w, -w))
        end
    end
    scale = height
    V .*= scale
    log_det = n_aug * log(scale) - sum(log, e.edge_length)
    node_labels = vcat(["node$(i)" for i in internal], labels)
    species_aug_id = index[1:n_tip]
    phy = PrecisionPhy(I, J, V, n_aug, n_tip, node_labels, log_det, scale, species_aug_id)
    return phy, labels
end

"""
    _phylo_latent_dense_precision(vcv, tip_labels, levels) -> PrecisionPhy

R's legacy dense route (`R/fit-multi.R:4724-4755`): reorder the tip covariance
to the species levels, add the `1e-8` diagonal ridge, invert, and record
`log_det = -logdet(A + 1e-8 I)`. Tip-only precision with the identity map.
"""
function _phylo_latent_dense_precision(vcv::AbstractMatrix, labels::Vector{String},
        levels::Vector{String})
    idx = [findfirst(==(l), labels) for l in levels]
    Aphy = Matrix{Float64}(vcv[idx, idx]) + _PHYLO_LATENT_RIDGE * I
    F = try
        cholesky(Symmetric(Aphy))
    catch err
        err isa PosDefException || rethrow()
        _phylo_latent_refuse("phylo_vcv must be a positive-definite covariance matrix.",
            "GJL-GATE-PHYLO-LATENT-VCV")
    end
    kappa = cond(Aphy)
    kappa > _PHYLO_LATENT_KAPPA_WARN &&
        @warn "phylo_vcv is ill-conditioned (condition number $(round(kappa; sigdigits = 3)) > 1e8 after the 1e-8 ridge); estimates may be numerically fragile."
    Qd = inv(F)
    Qd = (Qd + Qd') / 2
    Qs = sparse(Qd)
    I_, J_, V_ = findnz(Qs)
    p = length(levels)
    return PrecisionPhy(I_, J_, V_, p, p, levels, -logdet(F), 1.0, collect(1:p))
end

function _phylo_latent_coverage(levels::Vector{String}, covered::Vector{String},
        observed::Vector{String}, what::AbstractString)
    missing_levels = setdiff(levels, covered)
    isempty(missing_levels) && return nothing
    unused = intersect(missing_levels, setdiff(levels, observed))
    observed_missing = intersect(missing_levels, observed)
    parts = String["$(what) do not cover all species levels."]
    if !isempty(unused)
        n = length(unused)
        push!(parts, "$(n) declared level$(n > 1 ? "s" : "") of `species` " *
            "$(n > 1 ? "have" : "has") no observations and $(n > 1 ? "are" : "is") " *
            "not covered: $(join(unused, ", ")).")
        push!(parts, "Call `droplevels()` on `species` (in Julia: drop the unused " *
            "labels from `species_levels`) so its levels match the species actually being fit.")
    end
    if !isempty(observed_missing)
        n = length(observed_missing)
        push!(parts, "$(n) observed species level$(n > 1 ? "s" : "") of `species` not " *
            "covered: $(join(observed_missing, ", ")). This is a genuine mismatch and is " *
            "NOT fixed by droplevels() -- supply a tree/vcv that covers these species.")
    end
    _phylo_latent_refuse(join(parts, " "), "GJL-GATE-PHYLO-LATENT-COVERAGE")
end

"""
    fit_phylo_latent_gllvm(Y, species; d = 1, tree = nothing, vcv = nothing,
                           A = nothing, Ainv = nothing, tip_labels = nothing,
                           species_levels = nothing, unique = false, rho = 1,
                           X = nothing, start = nothing, g_tol = 1e-5,
                           iterations = 400) -> PrecisionMultivariateFit

Fit the Gaussian phylogenetic latent-factor model of gllvmTMB's
`phylo_latent(species, d = K)` (the twin of R at gllvmTMB 0.7.1):

    y[t, o] = b[t] + sum_k Lambda[t, k] * g_k[species(o)] + eps[t, o],
    g_k ~ N(0, A) independently over k,  eps ~ N(0, sigma_eps^2),

with `A` the unit-height (correlation-form) phylogenetic covariance, a
lower-triangular `T x K` loading matrix `Lambda` with a signed diagonal (only
`Sigma_phy = Lambda * Lambda'` is identified), one intercept per trait and one
shared residual variance. The phylogenetic scale is fixed at one and absorbed
by `Lambda`, as in R.

`Y` is traits x observations; `species[o]` is the label of observation `o`.
Species are matched to the phylogeny by label (R's factor-level rule), never
by position. `species_levels` optionally declares the full level set (R's
factor levels, which may include labels with no observations); it defaults
to the sorted distinct labels of `species`. Tips of a tree that carry no
observation are legal and are integrated out.

Exactly one phylogeny source is required:

* `tree`: a Newick string or an [`AugmentedPhy`](@ref). The sparse precision
  over every non-root node is built as R builds it (internal nodes first, tips
  last, entries `height / edge_length`), so polytomies are admitted with
  `n_aug = n_tip + Nnode - 1`. The tree must be ultrametric.
* `vcv` (or its alias `A`): a dense tip covariance with row labels
  `tip_labels`. R's `1e-8` diagonal ridge is added before inversion, exactly as
  in R, so the dense route and the tree route agree only to roughly `1e-5` in
  log-density. A condition number above `1e8` warns; it never refuses.

Refusals port R's sentences: `d` above the number of traits, no source, two
sources, a non-phylogeny `tree`, a non-ultrametric tree, `vcv` without labels,
and species labels the source does not cover (naming `droplevels()` when the
uncovered level has no observations). Two Julia scope fences are labelled as
such: `rho != 1` (`GJL-GATE-PHYLO-LATENT-RHO`; R accepts `rho`, the twin does
not yet) and `Ainv` (`GJL-GATE-PHYLO-LATENT-AINV`; R's in-keyword `Ainv` takes
the dense ridged route while a global sparse `phylo_vcv` takes the sparse
route, and which one this keyword twins is a pending maintainer decision).

`unique = true` adds a per-trait phylogenetic unique variance (positive log
link); it is a documented extra, not part of the twin.

Optimisation uses finite-difference gradients because the sparse Cholesky
does not accept automatic-differentiation numbers. Returns a
[`PrecisionMultivariateFit`](@ref) with `residual_mode = :shared`; use
`extract_Sigma(fit; level = :phy)` and [`extract_phylo_signal`](@ref).
"""
function fit_phylo_latent_gllvm(Y::AbstractMatrix{<:Real}, species::AbstractVector;
        d::Integer = 1, tree = nothing, vcv = nothing, A = nothing, Ainv = nothing,
        tip_labels = nothing, species_levels = nothing, unique::Bool = false,
        rho::Real = 1, X = nothing, start = nothing, g_tol::Real = 1e-5,
        iterations::Integer = 400)
    rho == 1 || _phylo_latent_refuse(
        "rho is outside the A14/A15 scope of the Julia twin; fit rho = 1 or use R.",
        "GJL-GATE-PHYLO-LATENT-RHO")
    source_tag = "GJL-GATE-PHYLO-LATENT-SOURCE"
    (A !== nothing && vcv !== nothing) && _phylo_latent_refuse(
        "phylo_latent() got both A and vcv.", source_tag, "These are aliases -- supply only one.")
    (Ainv !== nothing && vcv !== nothing) && _phylo_latent_refuse(
        "phylo_latent() got both Ainv and vcv.", source_tag, "These are aliases -- supply only one.")
    dense = A !== nothing ? A : vcv
    n_sources = count(!isnothing, (tree, dense, Ainv))
    n_sources == 0 && _phylo_latent_refuse(
        "phylo_latent() / phylo_slope() found in formula but phylo_vcv (or phylo_tree) is NULL.",
        source_tag, "Pass tree = ... or vcv = ... to fit_phylo_latent_gllvm.")
    n_sources > 1 && _phylo_latent_refuse(
        "Supply one of tree, vcv, or A / Ainv.", source_tag)
    Ainv === nothing || _phylo_latent_refuse(
        "Ainv is not yet admitted by the Julia twin.", "GJL-GATE-PHYLO-LATENT-AINV",
        "R's in-keyword Ainv takes the dense ridged route and a global sparse phylo_vcv " *
        "takes the sparse route; which one this keyword twins awaits a maintainer decision. " *
        "Use tree or vcv.")
    n_traits, m = size(Y)
    d > n_traits && _phylo_latent_refuse(
        "phylo_latent(d = $(d)) exceeds the number of traits ($(n_traits)); the latent rank must satisfy d <= n_traits.",
        "GJL-GATE-PHYLO-LATENT-RANK", "Pass d at most $(n_traits).")
    d >= 1 || throw(ArgumentError("d must be a positive integer"))
    length(species) == m || throw(DimensionMismatch(
        "species has length $(length(species)); Y has $(m) observations (columns)"))
    obs_labels = String[string(s) for s in species]
    observed = sort(Base.unique(obs_labels))
    levels = species_levels === nothing ? observed : String[string(l) for l in species_levels]
    length(Base.unique(levels)) == length(levels) ||
        throw(ArgumentError("species_levels must not repeat a label"))
    issubset(observed, levels) || throw(ArgumentError(
        "species_levels must include every label in species; missing: " *
        join(setdiff(observed, levels), ", ")))

    phy, tip_names = if tree !== nothing
        phy_t, labels_t = _phylo_latent_tree_precision(tree)
        _phylo_latent_coverage(levels, labels_t, observed, "phylo_tree tip labels")
        (phy_t, labels_t)
    else
        dense isa AbstractMatrix || _phylo_latent_refuse(
            "phylo_vcv must be a numeric matrix.", "GJL-GATE-PHYLO-LATENT-VCV")
        tip_labels === nothing && _phylo_latent_refuse(
            "phylo_vcv must have rownames matching levels of species.",
            "GJL-GATE-PHYLO-LATENT-LABELS", "Pass tip_labels = the row labels of vcv.")
        labels_d = String[string(l) for l in tip_labels]
        size(dense) == (length(labels_d), length(labels_d)) || _phylo_latent_refuse(
            "phylo_vcv must be square with one row per tip label.", "GJL-GATE-PHYLO-LATENT-LABELS")
        length(Base.unique(labels_d)) == length(labels_d) || _phylo_latent_refuse(
            "phylo_vcv rownames must be unique.", "GJL-GATE-PHYLO-LATENT-LABELS")
        _phylo_latent_coverage(levels, labels_d, observed, "phylo_vcv rownames")
        (_phylo_latent_dense_precision(dense, labels_d, levels), levels)
    end
    tip_position = Dict(l => i for (i, l) in enumerate(tip_names))
    species_id = [tip_position[l] for l in obs_labels]
    fit = fit_precision_multivariate(Y, phy; rank = d,
        mode = unique ? :explicitunique : :barelowrank, residual_mode = :shared,
        species_id = species_id, X = X, start = start, g_tol = g_tol,
        iterations = iterations)
    return _pmv_with_labels(fit, obs_labels, tip_names)
end
