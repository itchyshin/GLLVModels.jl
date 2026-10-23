# Edge-node incidence sparse representation for the phylogenetic
# precision matrix. For a binary tree with p leaves:
#   * Nodes:      2p − 1 (p leaves + p − 1 internal)
#   * Edges:      2p − 2
#   * B :: (2p − 1) × (2p − 2) edge-node incidence:
#       for edge e connecting parent → child with branch length b_e
#           B[child,  e] = +1
#           B[parent, e] = −1
#     so each column has exactly 2 non-zeros and total nnz(B) = 4(p − 1).
#   * W :: (2p − 2) × (2p − 2) diagonal with W[e, e] = 1 / (σ²_phy · b_e).
#   * Q_topology = B · W · B^T  (never formed explicitly — matrix-free).
#
# Q_topology is the *same* matrix that AugmentedPhy.Q_topology builds via
# per-edge 2 × 2 blocks; the edge-incidence factorisation just stores it as
# B · W · B^T. The advantages over the augmented-Q construction are:
#   1. B has 4(p − 1) nnz vs the augmented Q's ~8p — sparser STORAGE.
#   2. W is a literal diagonal, depending on σ²_phy and branch lengths only
#      through trivial element-wise reciprocals. This makes the whole
#      representation **AD-friendly**: ForwardDiff.Dual element types flow
#      through `B' * x`, `W .* z`, `B * z2` without ever touching CHOLMOD,
#      which the augmented-Q sparse Cholesky path cannot do.
#
# Matrix-free Q · x in three O(p) operations:
#       z1 = B' * x        # sparse mat-vec, 2 flops per edge
#       z2 = W .* z1       # diagonal scale, 1 flop per edge
#       z3 = B * z2        # sparse mat-vec, 2 flops per edge
#
# Reference (representation): Golub & Van Loan §4.6 (incidence matrices).
# First application of this representation to the phylogenetic likelihood
# in this work.

using SparseArrays
using LinearAlgebra

# `node_parent[t]` (or 0 for the root) and `node_edge[t]` (or 0 for the
# root) come out of the Newick walk and are kept around so that
# `log_det_Q` and `solve_Q` can do their O(p) tree traversals without
# having to round-trip through B.
"""
    EdgePhy{T}

Edge-incidence representation of a rooted bifurcating phylogeny, built by
[`edge_phy`](@ref) from a Newick string. Fields: `n_leaves`, `n_nodes` (`2p − 1`),
`n_edges` (`2p − 2`), the sparse node-by-edge incidence matrix `B`, `branch_lengths`,
`leaf_indices` and `leaf_names`, `root_index`, and the tree-walk arrays `node_parent`,
`node_edge` and `node_children`. It is the tree argument of [`sigma_phy_dense_edge`](@ref),
[`log_det_Q`](@ref), [`solve_Q`](@ref), [`path_membership`](@ref) and
[`fit_branch_re`](@ref).
"""
struct EdgePhy{T}
    n_leaves::Int
    n_nodes::Int                            # 2p − 1 for a binary tree
    n_edges::Int                            # 2p − 2
    B::SparseMatrixCSC{T, Int}              # (n_nodes × n_edges) incidence
    branch_lengths::Vector{T}               # length n_edges
    leaf_indices::Vector{Int}               # rows of B that correspond to leaves
    leaf_names::Vector{String}
    root_index::Int                         # row of B corresponding to the root
    node_parent::Vector{Int}                # length n_nodes; root has parent 0
    node_edge::Vector{Int}                  # edge index of the parent edge (0 for root)
    node_children::Vector{Vector{Int}}      # children of each node (empty for leaves)
end

# ---------------------------------------------------------------------------
# Internal Newick parser. Self-contained (no Phylo.jl dep), grammar matches
# the minimal subset already used by `sparse_phy.jl`.
# ---------------------------------------------------------------------------

mutable struct _EdgeNewickCursor
    s::String
    i::Int
end

@inline _epeek(c::_EdgeNewickCursor) = c.i > lastindex(c.s) ? '\0' : c.s[c.i]
@inline function _eadvance(c::_EdgeNewickCursor)
    ch = _epeek(c)
    c.i = nextind(c.s, c.i)
    return ch
end

function _eparse_number!(c::_EdgeNewickCursor)
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

function _eparse_name!(c::_EdgeNewickCursor)
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

# Walks the Newick string and pushes one entry per encountered node into
# the parallel arrays. The (post-order, leaves-first) reindex happens
# afterwards in `edge_phy`.
function _eparse_node!(c::_EdgeNewickCursor,
                       node_parent::Vector{Int},
                       node_is_leaf::Vector{Bool},
                       node_name::Vector{String},
                       node_length::Vector{Float64},
                       leaf_indices::Vector{Int},
                       leaf_names::Vector{String})
    children_local = Int[]
    if _epeek(c) == '('
        _eadvance(c)
        push!(children_local, _eparse_node!(c, node_parent, node_is_leaf,
                                            node_name, node_length,
                                            leaf_indices, leaf_names))
        while _epeek(c) == ','
            _eadvance(c)
            push!(children_local, _eparse_node!(c, node_parent, node_is_leaf,
                                                node_name, node_length,
                                                leaf_indices, leaf_names))
        end
        _epeek(c) == ')' ||
            error("expected ')' at position $(c.i) in Newick string")
        _eadvance(c)
        name = ""
        if _epeek(c) != ':' && _epeek(c) != ',' && _epeek(c) != ')' && _epeek(c) != ';'
            name = _eparse_name!(c)
        end
        blen = 0.0
        if _epeek(c) == ':'
            _eadvance(c)
            blen = _eparse_number!(c)
        end
        push!(node_parent, 0)
        push!(node_is_leaf, false)
        push!(node_name, name)
        push!(node_length, blen)
        my_idx = length(node_parent)
        for cidx in children_local
            node_parent[cidx] = my_idx
        end
        return my_idx
    else
        name = _eparse_name!(c)
        blen = 0.0
        if _epeek(c) == ':'
            _eadvance(c)
            blen = _eparse_number!(c)
        end
        push!(node_parent, 0)
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
    edge_phy(newick::AbstractString) :: EdgePhy{Float64}

Parse a (minimal) Newick string and return the edge-incidence
representation `EdgePhy`. Node labelling convention (matches the
augmented-Q sparse path so the two representations are directly
comparable):

* Leaves are rows `1:p` in the order they appear in the Newick string.
* Internal nodes (post-order) follow as rows `p + 1 : 2p − 1`, with the
  root last.

For each edge `e` connecting parent `p_e` → child `c_e` with branch
length `b_e`:

    B[c_e, e] = +1
    B[p_e, e] = −1

so every column of `B` has exactly two non-zeros and `nnz(B) = 4(p − 1)`.

Restrictions match `sparse_phy.augmented_phy`: bifurcating trees only,
positive branch lengths, no internal-node labels (tolerated but
discarded). Tree must end with `;`.
"""
function edge_phy(newick::AbstractString)
    s = filter(!isspace, String(newick))
    !endswith(s, ";") &&
        error("Newick string must end with ';'")
    s = s[1:prevind(s, lastindex(s))]
    c = _EdgeNewickCursor(s, firstindex(s))

    node_parent_raw = Int[]
    node_is_leaf    = Bool[]
    node_name_raw   = String[]
    node_length_raw = Float64[]
    leaf_indices_raw = Int[]
    leaf_names      = String[]

    root_idx_raw = _eparse_node!(c, node_parent_raw, node_is_leaf,
                                  node_name_raw, node_length_raw,
                                  leaf_indices_raw, leaf_names)

    c.i <= lastindex(c.s) &&
        error("extra characters after end of tree at position $(c.i)")

    n_total = length(node_parent_raw)
    p       = length(leaf_indices_raw)
    n_total == 2 * p - 1 ||
        error("tree is not binary (got $n_total nodes for $p leaves; expected $(2p - 1))")

    # Reindex: leaves to rows 1..p (in encounter order), then internal nodes
    # in their original (post-order) order, root last.
    new_idx_of = Vector{Int}(undef, n_total)
    for (new_i, old_i) in enumerate(leaf_indices_raw)
        new_idx_of[old_i] = new_i
    end
    next_new = p + 1
    for old_i in 1:n_total
        node_is_leaf[old_i] && continue
        new_idx_of[old_i] = next_new
        next_new += 1
    end
    next_new == n_total + 1 ||
        error("internal indexing error: expected $next_new == $(n_total + 1)")

    # Renumbered node arrays
    node_parent = zeros(Int, n_total)
    for old_i in 1:n_total
        op = node_parent_raw[old_i]
        if op != 0
            node_parent[new_idx_of[old_i]] = new_idx_of[op]
        end
    end
    node_length = zeros(Float64, n_total)
    for old_i in 1:n_total
        node_length[new_idx_of[old_i]] = node_length_raw[old_i]
    end
    new_root_idx = new_idx_of[root_idx_raw]

    # Each non-root node owns the edge from its parent. Assign edge index by
    # iterating over non-root nodes in their post-order new-index ordering.
    n_edges = n_total - 1
    node_edge = zeros(Int, n_total)
    branch_lengths = Vector{Float64}(undef, n_edges)
    Iv = Vector{Int}(undef, 2 * n_edges)
    Jv = Vector{Int}(undef, 2 * n_edges)
    Vv = Vector{Float64}(undef, 2 * n_edges)
    e_idx = 0
    for c_new in 1:n_total
        c_new == new_root_idx && continue
        e_idx += 1
        b = node_length[c_new]
        b > 0 ||
            error("branch length must be > 0; node $c_new has length $b")
        branch_lengths[e_idx] = b
        p_new = node_parent[c_new]
        node_edge[c_new] = e_idx
        # Column e_idx of B: +1 at child, -1 at parent.
        Iv[2 * e_idx - 1] = c_new; Jv[2 * e_idx - 1] = e_idx; Vv[2 * e_idx - 1] = +1.0
        Iv[2 * e_idx    ] = p_new; Jv[2 * e_idx    ] = e_idx; Vv[2 * e_idx    ] = -1.0
    end
    e_idx == n_edges ||
        error("expected $n_edges edges, got $e_idx")
    B = sparse(Iv, Jv, Vv, n_total, n_edges)

    # Children lists for the tree traversals (post/pre-order).
    node_children = [Int[] for _ in 1:n_total]
    for c_new in 1:n_total
        c_new == new_root_idx && continue
        push!(node_children[node_parent[c_new]], c_new)
    end

    leaf_idx_new = collect(1:p)               # by construction
    return EdgePhy{Float64}(p, n_total, n_edges, B, branch_lengths,
                            leaf_idx_new, leaf_names,
                            new_root_idx, node_parent, node_edge,
                            node_children)
end

"""
    Q_times_x(phy::EdgePhy, σ²_phy::Real, x::AbstractVector) -> Vector

Apply the phylogenetic topology precision `Q = B · W · B^T` to a vector
`x` without ever materialising `Q`. `W = diag(1 / (σ²_phy · b_e))`.
Cost: three O(p) operations.

`Q` is symmetric positive **semi**-definite — the all-ones vector lies in
its null space (`B^T · 1 = 0`, so `Q · 1 = 0`). For the log-likelihood
the root row/col is dropped and the resulting `Q_cond` is positive
definite. This function does not condition; it applies the raw operator.

This entry point is fully AD-friendly: `B` is structural, `σ²_phy` and
`branch_lengths` participate only through the diagonal scaling, so
`ForwardDiff.Dual` element types flow through cleanly.
"""
function Q_times_x(phy::EdgePhy, σ²_phy::Real, x::AbstractVector)
    length(x) == phy.n_nodes ||
        throw(ArgumentError("length(x) ($(length(x))) must equal phy.n_nodes ($(phy.n_nodes))"))
    z1     = phy.B' * x
    W_diag = 1 ./ (σ²_phy .* phy.branch_lengths)
    z2     = W_diag .* z1
    z3     = phy.B * z2
    return z3
end

"""
    node_depths(phy::EdgePhy, σ²_phy::Real) -> Vector

Compute depth (sum of branch lengths from the root) for every node, in
the row ordering of `B`. The root has depth 0. AD-friendly — branch
lengths flow through `σ²_phy` only as a scalar multiplier on the result.
"""
function _node_depths(phy::EdgePhy{T}, σ²_phy::S) where {T, S}
    R = promote_type(T, S)
    depths = zeros(R, phy.n_nodes)
    # Pre-order: visit each non-root node after its parent.
    # Because of the post-order construction (leaves first, then
    # internals, root last), if we walk nodes from the ROOT downward via
    # node_children we get a correct pre-order.
    stack = [phy.root_index]
    while !isempty(stack)
        u = pop!(stack)
        for v in phy.node_children[u]
            depths[v] = depths[u] + σ²_phy * phy.branch_lengths[phy.node_edge[v]]
            push!(stack, v)
        end
    end
    return depths
end

"""
    sigma_phy_dense_edge(phy::EdgePhy, σ²_phy::Real) -> Matrix

Build the dense p × p leaf phylogenetic covariance for a Brownian-motion
model from the edge-incidence representation:

    Σ_phy[i, j] = σ²_phy · (length of root → MRCA(i, j))

The MRCA depth equals `σ²_phy · depth(MRCA(i, j))` in our units. This is
the same matrix that `sigma_phy_dense(::AugmentedPhy)` builds via the
sparse Schur inverse, so the two paths can be cross-checked.

Time/storage: O(p²). AD-friendly — `σ²_phy` is just a scalar multiplier
on each entry and branch lengths participate linearly.

For p ≲ 1500 this dense form is the simplest **AD-friendly** path
through the likelihood; at very large p, an O(p) matrix-free path is
required and is provided by `Q_times_x` / `solve_Q`.
"""
function sigma_phy_dense_edge(phy::EdgePhy{T}, σ²_phy::S) where {T, S}
    R = promote_type(T, S)
    p = phy.n_leaves
    depths = _node_depths(phy, σ²_phy)

    # Ancestor chain (incl. leaf itself) for each leaf, with the depth
    # stored so we can search for the MRCA without re-traversing.
    ancestor = Vector{Vector{Int}}(undef, p)
    for t in 1:p
        chain = Int[]
        u = t
        while u != 0
            push!(chain, u)
            u = phy.node_parent[u]
        end
        ancestor[t] = chain
    end

    Σ = Matrix{R}(undef, p, p)
    # For each leaf t the diagonal is depth(t) (MRCA(t, t) = t).
    for t in 1:p
        Σ[t, t] = depths[t]
    end
    # Off-diagonals: MRCA depth via set-membership probe (ancestor chains
    # are short — O(log p) for balanced trees, O(p) worst-case).
    for i in 1:p
        chain_i = ancestor[i]
        in_chain_i = Set{Int}(chain_i)
        for j in (i + 1):p
            chain_j = ancestor[j]
            # Walk j's chain root-to-leaf-ward (it's stored leaf-up so
            # reverse iteration). The first ancestor in chain_i that we
            # also see is the MRCA, but we want the DEEPEST shared one,
            # i.e. the first as we walk leaf-up.
            mrca = 0
            for u in chain_j
                if u in in_chain_i
                    mrca = u
                    break
                end
            end
            mrca == 0 && error("MRCA not found between leaves $i and $j — tree disconnected?")
            Σ[i, j] = depths[mrca]
            Σ[j, i] = depths[mrca]
        end
    end
    return Σ
end

sigma_phy_dense(phy::EdgePhy; σ²_phy = 1.0) = sigma_phy_dense_edge(phy, σ²_phy)
sigma_phy_dense(phy::EdgePhy, σ²_phy) = sigma_phy_dense_edge(phy, σ²_phy)

"""
    log_det_Q(phy::EdgePhy, σ²_phy::Real) -> Real

Log-determinant of the root-conditioned precision
`Q_cond = (B · W · B^T)` with the root row/col removed.

**Closed-form derivation.** `Q_cond` is the weighted Laplacian of a
tree (the tree topology with edge weights `w_e = 1 / (σ²_phy · b_e)`)
with one node (the root) deleted. By the Matrix-Tree Theorem, for a
connected graph with `n` nodes any (n − 1) × (n − 1) principal minor of
its weighted Laplacian equals the weighted sum over spanning trees,
where each spanning-tree weight is the product of its edge weights. A
TREE has exactly one spanning tree (itself), and that spanning tree's
weight is the product of every edge weight:

    det(Q_cond) = ∏_e w_e = ∏_e 1 / (σ²_phy · b_e)
    log det(Q_cond) = − sum_e log(σ²_phy · b_e)
                    = −(2p − 2) · log(σ²_phy) − sum_e log(b_e)

This is what we return. **AD-friendly** — closed form in terms of
`σ²_phy` and branch lengths only. Verified numerically against
`logdet(Matrix(Q_cond))` in the test suite.
"""
function log_det_Q(phy::EdgePhy{T}, σ²_phy::S) where {T, S}
    R = promote_type(T, S)
    acc = zero(R)
    for e in 1:phy.n_edges
        acc -= log(σ²_phy * phy.branch_lengths[e])
    end
    return acc
end

"""
    solve_Q(phy::EdgePhy, σ²_phy::Real, b::AbstractVector) -> Vector

Solve `Q_cond · x = b` where `Q_cond` is the root-conditioned precision
on the `n_nodes − 1` non-root augmented nodes. Cost: O(p) via two tree
traversals.

`b` is expected to be a vector of length `n_nodes − 1` indexed in the
order obtained by deleting the root row from the B-row ordering (i.e.
positions are `1..root_index−1, root_index+1..n_nodes` mapped into
`1..n_nodes − 1`).

**Algorithm.** Because `Q_cond` is the weighted-Laplacian of the tree
with the root pinned, the linear system corresponds to a Brownian-
motion conditional mean computation:

1. **Post-order forward pass** (leaves → root): each node aggregates the
   "force" it must transmit to its parent, weighted by branch length.
2. **Pre-order backward pass** (root → leaves): distribute the
   accumulated forces.

This is a well-known O(p) message-passing algorithm on a tree. For
testing and the AD likelihood path we currently use the equivalent
dense `Σ_phy = σ²_phy · Q_cond⁻¹` (built by `sigma_phy_dense_edge`)
because the dense form is what the existing rotation-trick likelihood
takes; the O(p) `solve_Q` is provided here for future use.

The current implementation builds the dense Q_cond and solves
directly — sufficient for correctness checking. Replace with the
two-pass tree traversal for full O(p) scaling.
"""
function solve_Q(phy::EdgePhy{T}, σ²_phy::S, b::AbstractVector) where {T, S}
    n_cond = phy.n_nodes - 1
    length(b) == n_cond ||
        throw(ArgumentError("length(b) ($(length(b))) must equal n_nodes − 1 ($n_cond)"))

    # Build Q = B · W · B^T as a dense matrix on the conditioned indices.
    # AD-friendly: B is structural, W flows through σ²_phy & branch_lengths.
    R = promote_type(T, S, eltype(b))
    W_diag = 1 ./ (σ²_phy .* phy.branch_lengths)
    BW = Matrix{R}(undef, phy.n_nodes, phy.n_edges)
    @inbounds for j in 1:phy.n_edges
        for i in 1:phy.n_nodes
            BW[i, j] = phy.B[i, j] * W_diag[j]
        end
    end
    Q_full = BW * Matrix(phy.B')
    # Drop root row/col
    keep = filter(i -> i != phy.root_index, 1:phy.n_nodes)
    Q_cond = Q_full[keep, keep]
    return Symmetric((Q_cond + Q_cond') ./ 2) \ collect(b)
end
