# Augmented-state sparse phylogenetic precision.
#
# Standard phylogenetic comparative methods supply a dense (p × p) Brownian-
# motion covariance Σ_phy over species at the tips of a tree. For p = 10_000
# the dense Cholesky is already 16+ seconds per evaluation. The Felsenstein
# (1981) / Hadfield (2010) / Bates (2015) workaround: augment the state with
# internal ancestral nodes, then represent the tree by a SPARSE precision
# matrix Q over all 2p − 1 nodes. Internal nodes get marginalised inside the
# sparse linear solves.
#
# Each tree edge (parent → child) with branch length b contributes
#     Q[parent, parent] += 1 / b
#     Q[child,  child ] += 1 / b
#     Q[parent, child ] -= 1 / b
#     Q[child,  parent] -= 1 / b
# i.e. a 2 × 2 block (1 / b) · [[1, -1], [-1, 1]] on rows/cols (parent, child).
# A binary tree with p leaves has p − 1 internal nodes and 2p − 2 edges so
# Q has 4 · (2p − 2) ≈ 8p non-zeros. Q is symmetric and rank-deficient by
# one: the constant-shift direction `z ≡ 1` lies in its null space (Brownian
# motion is identified only up to a common offset = root value).
#
# This file provides:
#   * `AugmentedPhy`   — container for the sparse topology precision plus
#                        identifying which augmented rows are leaves.
#   * `augmented_phy`  — Newick string parser → `AugmentedPhy`.
#   * `make_phy`       — same, but from a (parent, child, length) triple list.
#
# We deliberately do NOT depend on Phylo.jl or any other ecology package —
# the parser is ~80 lines of recursive descent and matches the minimal
# Newick grammar at
#     https://evolution.genetics.washington.edu/phylip/newicktree.html.
# Only `name:length` leaves and balanced parentheses are supported (no
# quoted names, no bootstrap labels, no internal-node names). The
# delimiter is `;`. Whitespace is ignored.

using SparseArrays
using LinearAlgebra

"""
    AugmentedPhy{T}

Augmented-state sparse phylogenetic precision for a binary tree.

Fields
------
* `n_leaves::Int`               – number of tip species (p).
* `n_total::Int`                – 2p − 1, leaves + internal ancestor nodes.
* `Q_topology::SparseMatrixCSC` – (n_total × n_total) topology contribution
  to the sparse precision. The actual phylogenetic precision is
  `Q_topology / σ²_phy`. About 8p non-zeros.
* `leaf_indices::Vector{Int}`   – maps a leaf k ∈ 1:p to its row/col in the
  augmented state. Ordering matches the order leaves were encountered in
  the Newick string (left-to-right).
* `leaf_names::Vector{String}`  – species names parsed from the Newick.
* `branch_lengths::Vector{T}`   – the 2p − 2 branch lengths in the order
  the parser walked the tree, always the RAW (unscaled) lengths — matches
  `phylo-tree-precision.R` keeping `edge_length` untouched even under
  `correlation = TRUE` (only the precision *values* are scaled, not the
  lengths).
* `root_index::Int`             – which augmented row is the root.
* `scale::Float64`              – the root-to-tip height actually baked
  into `Q_topology` (`correlation = true`, `augmented_phy`/`make_phy`);
  `1.0` when unscaled (the default, `correlation = false`).

`Q_topology` is positive **semi**-definite (rank 2p − 2). The all-ones
vector is its sole zero eigenvector — fixing the root removes the
degeneracy. The sparse log-likelihood path adds a positive contribution
to the leaf diagonals (proportional to `λ_phy² / d_total`) which renders
the active solve matrix positive definite without any explicit ridge.
"""
struct AugmentedPhy{T}
    n_leaves::Int
    n_total::Int
    Q_topology::SparseMatrixCSC{T,Int}
    leaf_indices::Vector{Int}
    leaf_names::Vector{String}
    branch_lengths::Vector{T}
    root_index::Int
    scale::Float64
end

# Positional-compat constructor: the pre-`scale`-field 7-argument call
# (every caller before phylo transport S2) still works, defaulting to the
# unscaled convention (`scale = 1.0`).
AugmentedPhy{T}(n_leaves::Int, n_total::Int, Q_topology::SparseMatrixCSC{T,Int},
                leaf_indices::Vector{Int}, leaf_names::Vector{String},
                branch_lengths::Vector{T}, root_index::Int) where {T} =
    AugmentedPhy{T}(n_leaves, n_total, Q_topology, leaf_indices, leaf_names,
                    branch_lengths, root_index, 1.0)

#     _apply_phy_correlation(raw::AugmentedPhy; correlation::Bool) :: AugmentedPhy
#
# Shared `correlation = true` gate + unit-height rescale used by both
# `augmented_phy` and `make_phy` (phylo transport S2). `correlation = false`
# returns `raw` unchanged (`scale = 1.0`, the existing behaviour — bit-
# identical to every pre-S2 caller). `correlation = true` checks
# ultrametricity via `_phylo_check_ultrametric_height`
# (`src/phylo_precision.jl`, mirrors `phylo-tree-precision.R:140-146`; raises
# `GJL-GATE-PHYLO-NONULTRAMETRIC` otherwise) and multiplies `Q_topology` by
# the root-to-tip height, matching R's `values <- scale * values`
# (`phylo-tree-precision.R:216`) — `branch_lengths` itself is left raw.
function _apply_phy_correlation(raw::AugmentedPhy{T}; correlation::Bool) where {T}
    correlation || return raw
    h = _phylo_check_ultrametric_height(raw)
    return AugmentedPhy{T}(raw.n_leaves, raw.n_total, h .* raw.Q_topology,
                           raw.leaf_indices, raw.leaf_names, raw.branch_lengths,
                           raw.root_index, h)
end

# ---------------------------------------------------------------------------
# Newick parsing
# ---------------------------------------------------------------------------
# Grammar (subset of the Felsenstein Newick standard):
#
#     tree    := node ";"
#     node    := leaf | internal
#     leaf    := name [":" length]
#     internal:= "(" node ("," node)* ")" [name] [":" length]
#     name    := [A-Za-z0-9_.\-]+
#     length  := [0-9]+ ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )?
#
# We strip whitespace before parsing. Length defaults to 0.0 if omitted.

mutable struct _NewickCursor
    s::String
    i::Int
end

@inline _peek(c::_NewickCursor) = c.i > lastindex(c.s) ? '\0' : c.s[c.i]
@inline function _advance(c::_NewickCursor)
    ch = _peek(c)
    c.i = nextind(c.s, c.i)
    return ch
end

function _parse_number!(c::_NewickCursor)
    j = c.i
    while j <= lastindex(c.s)
        ch = c.s[j]
        if ch in '0':'9' || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-'
            j = nextind(c.s, j)
        else
            break
        end
    end
    j == c.i && error("expected number at position $(c.i)")
    val = parse(Float64, c.s[c.i:prevind(c.s, j)])
    c.i = j
    return val
end

function _parse_name!(c::_NewickCursor)
    j = c.i
    while j <= lastindex(c.s)
        ch = c.s[j]
        if ch == ',' || ch == ')' || ch == ':' || ch == ';' || ch == '('
            break
        end
        j = nextind(c.s, j)
    end
    name = c.s[c.i:prevind(c.s, j)]
    c.i = j
    return name
end

# Internal builder: walks the Newick string and writes nodes + edges into
# the supplied vectors. Returns the index of the node it just consumed.
function _parse_node!(c::_NewickCursor,
                      node_parent::Vector{Int},
                      node_is_leaf::Vector{Bool},
                      node_name::Vector{String},
                      node_length::Vector{Float64},
                      leaf_indices::Vector{Int},
                      leaf_names::Vector{String})
    children_local = Int[]
    if _peek(c) == '('
        _advance(c)                       # consume "("
        # parse comma-separated children
        push!(children_local, _parse_node!(c, node_parent, node_is_leaf,
                                           node_name, node_length,
                                           leaf_indices, leaf_names))
        while _peek(c) == ','
            _advance(c)
            push!(children_local, _parse_node!(c, node_parent, node_is_leaf,
                                               node_name, node_length,
                                               leaf_indices, leaf_names))
        end
        _peek(c) == ')' ||
            error("expected ')' at position $(c.i) in Newick string")
        _advance(c)
        # internal node label (optional, discarded — minimal grammar)
        name = ""
        if _peek(c) != ':' && _peek(c) != ',' && _peek(c) != ')' && _peek(c) != ';'
            name = _parse_name!(c)
        end
        # branch length (to PARENT, optional)
        blen = 0.0
        if _peek(c) == ':'
            _advance(c)
            blen = _parse_number!(c)
        end
        # allocate this internal node and patch children's parents
        push!(node_parent, 0)             # parent set by caller
        push!(node_is_leaf, false)
        push!(node_name, name)
        push!(node_length, blen)
        my_idx = length(node_parent)
        for c_idx in children_local
            node_parent[c_idx] = my_idx
        end
        return my_idx
    else
        # leaf
        name = _parse_name!(c)
        blen = 0.0
        if _peek(c) == ':'
            _advance(c)
            blen = _parse_number!(c)
        end
        push!(node_parent, 0)             # parent set by caller
        push!(node_is_leaf, true)
        push!(node_name, name)
        push!(node_length, blen)
        my_idx = length(node_parent)
        push!(leaf_indices, my_idx)
        push!(leaf_names, name)
        return my_idx
    end
end

"""
    augmented_phy(newick::AbstractString; correlation::Bool = false) :: AugmentedPhy{Float64}

Parse a minimal Newick string and return the augmented-state sparse
precision representation.

Restrictions
------------
* Binary (bifurcating) trees only — every internal node has exactly two
  children. Multifurcations and unary nodes are rejected.
* Leaf names follow `[A-Za-z0-9_.\\-]+`. Internal node labels are tolerated
  but discarded.
* Branch lengths must all be > 0 (otherwise 1/b blows up).
* The root has no parent branch; the optional root length in
  `(…):0.0;` is read but does not enter Q.

Keyword arguments
-----------------
* `correlation::Bool = false` (opt-in) — when
  `true`, requires the tree to be ultrametric (within `sqrt(eps())` of a
  common root-to-tip height, mirroring `phylo-tree-precision.R:140-146`;
  raises `GJL-GATE-PHYLO-NONULTRAMETRIC` otherwise) and rescales
  `Q_topology` to the unit-height correlation form R's fit path always
  uses. σ²_phy fitted under `correlation = true` is `height` times the
  σ²_phy fitted under `correlation = false` for the same actual model —
  the estimand differs, not the log-likelihood. Default `false` keeps
  every existing native caller's behaviour bit-identical.

Example
-------
```julia
phy = augmented_phy("((A:0.1,B:0.2):0.3,C:0.5);")
phy.n_leaves      # 3
phy.n_total       # 5  (3 leaves + 2 internal)
length(phy.branch_lengths)   # 4
nnz(phy.Q_topology)          # 16  (4 per edge × 4 edges)
```
"""
function augmented_phy(newick::AbstractString; correlation::Bool = false)
    s = filter(!isspace, String(newick))
    !endswith(s, ";") &&
        error("Newick string must end with ';'")
    s = s[1:prevind(s, lastindex(s))]    # strip trailing ;
    c = _NewickCursor(s, firstindex(s))

    node_parent = Int[]
    node_is_leaf = Bool[]
    node_name = String[]
    node_length = Float64[]
    leaf_indices = Int[]
    leaf_names = String[]

    root_idx = _parse_node!(c, node_parent, node_is_leaf, node_name,
                            node_length, leaf_indices, leaf_names)

    c.i <= lastindex(c.s) &&
        error("extra characters after end of tree at position $(c.i)")

    # Reindex: place leaves first (1:p) in the order they were encountered,
    # then internal nodes in the order they were added (post-order, so the
    # root is last). This is the convention the likelihood code uses.
    n_total = length(node_parent)
    p = length(leaf_indices)
    n_total == 2 * p - 1 ||
        error("tree is not binary (got $n_total nodes for $p leaves; expected $(2p - 1))")
    # Binary trees only, checked per node: the global 2p-1 count alone can be
    # satisfied by a polytomy plus a unary node (#153). The check lives here, not
    # in the shared `_parse_node!`, because the phylo-latent route admits polytomies.
    n_children = zeros(Int, n_total)
    for par in node_parent
        par > 0 && (n_children[par] += 1)
    end
    for i in 1:n_total
        node_is_leaf[i] && continue
        n_children[i] == 2 ||
            error("tree is not binary (an internal node has $(n_children[i]) " *
                  "children; every internal node must have exactly two)")
    end
    perm = Vector{Int}(undef, n_total)   # perm[new_idx] = old_idx
    new_idx_of = Vector{Int}(undef, n_total)
    for (new_i, old_i) in enumerate(leaf_indices)
        perm[new_i] = old_i
        new_idx_of[old_i] = new_i
    end
    next_new = p + 1
    for old_i in 1:n_total
        node_is_leaf[old_i] && continue
        perm[next_new] = old_i
        new_idx_of[old_i] = next_new
        next_new += 1
    end
    next_new == n_total + 1 ||
        error("internal indexing error: expected $next_new == $(n_total + 1)")

    # Build sparse Q: walk every edge (non-root nodes have a parent edge of
    # length node_length[child]). 4 entries per edge: 2 diagonal + 2 off.
    I = Int[]
    J = Int[]
    V = Float64[]
    branch_lengths = Float64[]
    new_root_idx = new_idx_of[root_idx]
    for old_child in 1:n_total
        parent_old = node_parent[old_child]
        parent_old == 0 && continue       # root has no parent edge
        b = node_length[old_child]
        b > 0 ||
            error("branch length must be > 0; node $old_child has length $b")
        push!(branch_lengths, b)
        new_child  = new_idx_of[old_child]
        new_parent = new_idx_of[parent_old]
        inv_b = 1.0 / b
        # 2 × 2 block (1/b) · [[1, -1], [-1, 1]]
        push!(I, new_parent); push!(J, new_parent); push!(V,  inv_b)
        push!(I, new_child);  push!(J, new_child);  push!(V,  inv_b)
        push!(I, new_parent); push!(J, new_child);  push!(V, -inv_b)
        push!(I, new_child);  push!(J, new_parent); push!(V, -inv_b)
    end
    Q = sparse(I, J, V, n_total, n_total)

    # Verify bifurcating: every internal node should have exactly two
    # children among `node_parent`, and the topology should now be a tree
    # (n_total - 1 edges).
    length(branch_lengths) == n_total - 1 ||
        error("expected $(n_total - 1) edges, got $(length(branch_lengths))")

    leaf_idx_new = collect(1:p)           # by construction
    leaf_names_new = leaf_names           # already in encounter order
    raw = AugmentedPhy{Float64}(p, n_total, Q, leaf_idx_new, leaf_names_new,
                                branch_lengths, new_root_idx)
    return _apply_phy_correlation(raw; correlation = correlation)
end

"""
    make_phy(edges::AbstractVector{<:Tuple}, n_leaves::Integer;
             root_index::Integer = -1, correlation::Bool = false) :: AugmentedPhy{Float64}

Convenience constructor: build an `AugmentedPhy` from a list of edges
`(parent_id, child_id, branch_length)` with integer node ids 1..n_total
(leaves first, internals last is the recommended convention but not
required).

If `root_index < 0` it is auto-detected as the unique node that is not a
child in any edge.

`correlation::Bool = false` — same opt-in unit-height rescale + ultrametric
gate as `augmented_phy` (see its docstring).

This bypasses the Newick parser — useful for tests and for trees that
arrive from another tool already as edge lists.
"""
function make_phy(edges::AbstractVector, n_leaves::Integer;
                  root_index::Integer = -1,
                  leaf_names::Union{Nothing,AbstractVector{<:AbstractString}} = nothing,
                  correlation::Bool = false)
    n_total = 0
    for (p_i, c_i, _) in edges
        n_total = max(n_total, p_i, c_i)
    end
    n_total == 2 * n_leaves - 1 ||
        error("edge list spans $n_total nodes but n_leaves = $n_leaves " *
              "implies $(2 * n_leaves - 1) total nodes")

    if root_index < 0
        is_child = falses(n_total)
        for (_, c_i, _) in edges
            is_child[c_i] = true
        end
        roots = findall(.!is_child)
        length(roots) == 1 ||
            error("could not auto-detect root: $(length(roots)) candidates")
        root_index = roots[1]
    end

    I = Int[]; J = Int[]; V = Float64[]
    branch_lengths = Float64[]
    for (p_i, c_i, b) in edges
        b > 0 || error("branch length must be > 0; got $b")
        inv_b = 1.0 / b
        push!(branch_lengths, b)
        push!(I, p_i); push!(J, p_i); push!(V,  inv_b)
        push!(I, c_i); push!(J, c_i); push!(V,  inv_b)
        push!(I, p_i); push!(J, c_i); push!(V, -inv_b)
        push!(I, c_i); push!(J, p_i); push!(V, -inv_b)
    end
    Q = sparse(I, J, V, n_total, n_total)
    leaf_idx = collect(1:n_leaves)        # convention: leaves are 1:p
    names = leaf_names === nothing ?
        ["L$(t)" for t in 1:n_leaves] : collect(String.(leaf_names))
    raw = AugmentedPhy{Float64}(n_leaves, n_total, Q, leaf_idx, names,
                                branch_lengths, Int(root_index))
    return _apply_phy_correlation(raw; correlation = correlation)
end

"""
    sigma_phy_dense(phy::AugmentedPhy; σ²_phy::Real = 1.0) :: Matrix

Build the dense (p × p) leaf covariance `Σ_phy = σ²_phy · (S Q_cond⁻¹ S')`
where `Q_cond` is `phy.Q_topology` with the root row/col removed and `S`
selects leaves. This is what the existing dense path expects; used by
verification tests to compare sparse vs. dense.

This is **O(p³)** in storage and time — only intended for small trees in
tests. Do NOT call it on the real workload; the entire point of
`AugmentedPhy` is to avoid materialising Σ_phy.
"""
function sigma_phy_dense(phy::AugmentedPhy; σ²_phy::Real = 1.0)
    p = phy.n_leaves
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = Matrix(phy.Q_topology[keep, keep])
    Σ_full = inv(Symmetric(Q_cond))
    leaf_pos = [findfirst(==(phy.leaf_indices[t]), keep) for t in 1:p]
    return σ²_phy .* Σ_full[leaf_pos, leaf_pos]
end

sigma_phy_dense(phy::AugmentedPhy, σ²_phy::Real) = sigma_phy_dense(phy; σ²_phy = σ²_phy)

"""
    random_balanced_tree(p::Integer; branch_length::Real = 0.1) :: AugmentedPhy

Build a near-balanced binary tree with `p` leaves. All branch lengths
equal `branch_length`. Used in benchmarks and scaling tests.

When `p` is a power of 2 this is perfectly balanced. Otherwise the
left-over leaf at each level is carried up one extra step (so the tree
remains binary, just with slightly uneven depths). Branch lengths stay
uniform — the goal is a representative sparse-tree topology, not an
ultrametric one.
"""
function random_balanced_tree(p::Integer; branch_length::Real = 0.1)
    p > 0 || error("p must be > 0; got $p")
    edges = Tuple{Int,Int,Float64}[]
    current_level = collect(1:p)
    next_id = p + 1
    while length(current_level) > 1
        new_level = Int[]
        i = 1
        while i + 1 <= length(current_level)
            parent = next_id; next_id += 1
            push!(edges, (parent, current_level[i],     Float64(branch_length)))
            push!(edges, (parent, current_level[i + 1], Float64(branch_length)))
            push!(new_level, parent)
            i += 2
        end
        # leftover odd node (if length(current_level) is odd) carries up
        if i == length(current_level)
            push!(new_level, current_level[i])
        end
        current_level = new_level
    end
    root_idx = current_level[1]
    return make_phy(edges, p; root_index = root_idx)
end
