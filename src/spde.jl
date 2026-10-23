# SPDE / Matérn-GMRF spatial-field module (Lindgren, Rue & Lindström 2011).
#
# Lindgren, Rue & Lindström (2011, JRSSB) show that a Gaussian random field with
# Matérn covariance is the stationary solution of the stochastic PDE
#
#     (κ² − Δ)^{α/2} (τ u(s)) = W(s),
#
# where W is Gaussian white noise, κ > 0 is the inverse-range scale, τ > 0 a
# marginal-precision scale, and α = ν + d/2 the SPDE order (d = 2 here, so the
# Matérn smoothness is ν = α − 1). Discretising this SPDE with piecewise-linear
# (P1) finite elements on a triangular mesh turns the dense Matérn field into a
# *sparse* Gaussian Markov random field (GMRF): the field at the N mesh nodes is
# u ~ N(0, Q⁻¹) with a sparse precision Q, and the field at arbitrary observation
# sites is recovered by barycentric interpolation A·u with a sparse projector A.
#
# This module is deliberately self-contained linear algebra — it depends only on
# LinearAlgebra, SparseArrays, and `besselk`/`gamma` (SpecialFunctions, already in
# scope in the GLLVModels module). It has no dependence on the family/fit code and is
# designed to be shared with DRM.jl.
#
# Pipeline:
#   (Cdiag, G) = spde_fem(nodes, tris)            # P1 mass + stiffness
#   Q          = spde_precision(Cdiag, G, κ, τ)   # sparse Matérn precision
#   A          = spde_projector(nodes, tris, locs) # node → site interpolation
#   u ~ N(0, Q⁻¹) on the mesh nodes; observed as A·u at the M sites.
#
# References:
#   - Lindgren, Rue & Lindström 2011 (SPDE ↔ Matérn link, P1 FEM, JRSSB)
#   - Rue & Held 2005 (GMRFs; sparse precision representations)
#   - Stein 1999; Matérn 1960 (the Matérn covariance class)

using LinearAlgebra
using SparseArrays
using SpecialFunctions: besselk, gamma

"""
    spde_fem(nodes, tris) -> (Cdiag::Vector, G::SparseMatrixCSC)

Assemble the piecewise-linear (P1) finite-element matrices on a 2-D triangular
mesh, as used in the SPDE representation of a Matérn field (Lindgren, Rue &
Lindström 2011).

Arguments:

- `nodes::AbstractMatrix` — N × 2 vertex coordinates (one row per node).
- `tris::AbstractMatrix{<:Integer}` — T × 3, 1-based vertex indices, one row per
  triangle.

For each triangle with vertices `p1, p2, p3 = nodes[tris[k, :], :]`:

- `area = 0.5 |（p2−p1) × (p3−p1)|`.
- *Lumped mass*: add `area/3` to each of the three vertices' `Cdiag` entry. This
  is the standard mass-lumping that makes the FEM mass matrix diagonal, which is
  what gives the SPDE precision its sparse-GMRF structure.
- *Stiffness*: with `x = (p1ₓ, p2ₓ, p3ₓ)`, `y = (p1_y, p2_y, p3_y)`,
  `b = (y₂−y₃, y₃−y₁, y₁−y₂)`, `c = (x₃−x₂, x₁−x₃, x₂−x₁)`, the local matrix is
  `Klocal[i,j] = (b[i] b[j] + c[i] c[j]) / (4 area)` (the ∇φᵢ·∇φⱼ integrals of the
  linear basis functions), scattered into the global sparse `G`.

Returns:

- `Cdiag` — length-N vector, the lumped (diagonal) mass; `sum(Cdiag)` equals the
  total mesh area.
- `G` — N × N sparse symmetric stiffness matrix (each row sums to ≈ 0).
"""
function spde_fem(nodes::AbstractMatrix, tris::AbstractMatrix{<:Integer})
    N = size(nodes, 1)
    T = size(tris, 1)
    size(nodes, 2) == 2 || throw(ArgumentError("nodes must be N × 2; got $(size(nodes))"))
    size(tris, 2) == 3 || throw(ArgumentError("tris must be T × 3; got $(size(tris))"))

    Cdiag = zeros(Float64, N)

    # Each triangle contributes 9 entries to G.
    I = Vector{Int}(undef, 9T)
    J = Vector{Int}(undef, 9T)
    V = Vector{Float64}(undef, 9T)
    idx = 0

    @inbounds for k in 1:T
        g1 = tris[k, 1]; g2 = tris[k, 2]; g3 = tris[k, 3]
        gidx = (g1, g2, g3)

        x1 = nodes[g1, 1]; y1 = nodes[g1, 2]
        x2 = nodes[g2, 1]; y2 = nodes[g2, 2]
        x3 = nodes[g3, 1]; y3 = nodes[g3, 2]

        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        area > 0 || throw(ArgumentError("degenerate (zero-area) triangle at row $k"))

        # Lumped mass: area/3 to each vertex.
        third = area / 3
        Cdiag[g1] += third
        Cdiag[g2] += third
        Cdiag[g3] += third

        # Stiffness: gradients of the linear nodal basis functions.
        b = (y2 - y3, y3 - y1, y1 - y2)
        c = (x3 - x2, x1 - x3, x2 - x1)
        inv4A = 1.0 / (4 * area)

        for i in 1:3, j in 1:3
            idx += 1
            I[idx] = gidx[i]
            J[idx] = gidx[j]
            V[idx] = (b[i] * b[j] + c[i] * c[j]) * inv4A
        end
    end

    G = sparse(I, J, V, N, N)
    return Cdiag, G
end

"""
    spde_precision(Cdiag, G, κ, τ; α::Integer = 2) -> SparseMatrixCSC

Build the sparse Matérn-field precision `Q` from the P1 FEM matrices produced by
[`spde_fem`](@ref).

With the lumped mass `C = spdiagm(Cdiag)` and stiffness `G`, set
`K = κ² C + G`. The SPDE order `α` controls the Matérn smoothness ν = α − d/2 (in
2-D, d = 2, so ν = α − 1):

- `α = 1` (ν = 0, rough): `Q = τ² K`.
- `α = 2` (ν = 1, the common case): `Q = τ² (K C⁻¹ K)`.

Higher α follow the same recursion but are not implemented here.

The field `u ~ N(0, Q⁻¹)` on the mesh nodes is then (approximately) a Matérn
field with inverse-range κ and marginal-precision scale τ. Returns a symmetric
sparse `Q`.
"""
function spde_precision(Cdiag::AbstractVector, G::SparseMatrixCSC,
                        κ::Real, τ::Real; α::Integer = 2)
    κ > 0 || throw(ArgumentError("κ must be positive; got $κ"))
    τ > 0 || throw(ArgumentError("τ must be positive; got $τ"))

    K = spdiagm(0 => (κ^2) .* Cdiag) + G

    if α == 1
        return (τ^2) .* K
    elseif α == 2
        Cinv = spdiagm(0 => 1.0 ./ Cdiag)
        return (τ^2) .* (K * Cinv * K)
    else
        throw(ArgumentError("α must be 1 or 2; got $α"))
    end
end

# Maximum edge length on a triangular mesh (for outside snap tolerance).
# Comment only; a docstring would trip missing_docs.
function _spde_max_edge_length(nodes::AbstractMatrix, tris::AbstractMatrix{<:Integer})
    maxlen = 0.0
    @inbounds for k in 1:size(tris, 1)
        g1 = tris[k, 1]; g2 = tris[k, 2]; g3 = tris[k, 3]
        for (ga, gb) in ((g1, g2), (g2, g3), (g3, g1))
            dx = nodes[ga, 1] - nodes[gb, 1]
            dy = nodes[ga, 2] - nodes[gb, 2]
            maxlen = max(maxlen, hypot(dx, dy))
        end
    end
    return maxlen
end

"""
    spde_projector(nodes, tris, locs::AbstractMatrix; outside_tol, nsnapped) -> SparseMatrixCSC

Build the sparse M × N node → site projector `A` that interpolates the mesh-node
field to arbitrary observation locations: if `u` is the field at the N mesh nodes,
then `A·u` is the field at the M rows of `locs`.

For each location, the containing triangle is found (linear search over `tris`)
and the row of `A` is filled with that triangle's three barycentric coordinates
`(λ1, λ2, λ3)` at the corresponding node columns — exact linear interpolation
consistent with the P1 finite-element basis. A point lying outside every triangle
but within `outside_tol` of the nearest mesh vertex is snapped to that vertex
(weight 1). Locations farther away throw `ArgumentError`. Each row of `A`
therefore sums to 1.

Non-finite coordinates throw `ArgumentError` naming the row. Optional keyword
`nsnapped` (`Ref{Int}`) receives the number of rows snapped to the nearest
vertex.

`locs` is M × 2. Returns an M × N sparse matrix.
"""
function spde_projector(nodes::AbstractMatrix, tris::AbstractMatrix{<:Integer},
                        locs::AbstractMatrix;
                        outside_tol::Real = 2 * _spde_max_edge_length(nodes, tris),
                        nsnapped::Union{Nothing, Ref{Int}} = nothing)
    N = size(nodes, 1)
    T = size(tris, 1)
    M = size(locs, 1)
    size(locs, 2) == 2 || throw(ArgumentError("locs must be M × 2; got $(size(locs))"))
    outside_tol ≥ 0 || throw(ArgumentError("outside_tol must be non-negative; got $outside_tol"))

    tol = 1e-9
    snap_count = 0

    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, 3M); sizehint!(J, 3M); sizehint!(V, 3M)

    @inbounds for m in 1:M
        px = locs[m, 1]; py = locs[m, 2]
        if !isfinite(px) || !isfinite(py)
            throw(ArgumentError(
                "location row $m has non-finite coordinates ($px, $py)"))
        end
        found = false

        for k in 1:T
            g1 = tris[k, 1]; g2 = tris[k, 2]; g3 = tris[k, 3]
            x1 = nodes[g1, 1]; y1 = nodes[g1, 2]
            x2 = nodes[g2, 1]; y2 = nodes[g2, 2]
            x3 = nodes[g3, 1]; y3 = nodes[g3, 2]

            # Standard 2-D barycentric coordinates of (px, py) in triangle 123.
            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            det == 0 && continue
            λ1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
            λ2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
            λ3 = 1.0 - λ1 - λ2

            if λ1 ≥ -tol && λ2 ≥ -tol && λ3 ≥ -tol
                push!(I, m); push!(J, g1); push!(V, λ1)
                push!(I, m); push!(J, g2); push!(V, λ2)
                push!(I, m); push!(J, g3); push!(V, λ3)
                found = true
                break
            end
        end

        if !found
            # Outside every triangle: snap only if within outside_tol of a vertex.
            best = 1
            bestd = Inf
            for n in 1:N
                dx = nodes[n, 1] - px; dy = nodes[n, 2] - py
                d2 = dx * dx + dy * dy
                if d2 < bestd
                    bestd = d2
                    best = n
                end
            end
            snap_dist = sqrt(bestd)
            if snap_dist > outside_tol
                throw(ArgumentError(
                    "location row $m at ($px, $py) lies outside the mesh " *
                    "(distance to nearest vertex is $snap_dist, exceeds outside_tol = $outside_tol)"))
            end
            snap_count += 1
            push!(I, m); push!(J, best); push!(V, 1.0)
        end
    end

    if nsnapped !== nothing
        nsnapped[] = snap_count
    end
    return sparse(I, J, V, M, N)
end

"""
    matern_correlation(r, κ; ν = 1.0) -> Float64

Matérn correlation function at separation `r` with inverse-range `κ` and
smoothness `ν`:

    ρ(r) = (κ r)^ν · K_ν(κ r) / (2^{ν−1} Γ(ν)),   r > 0,    ρ(0) = 1,

where `K_ν` is the modified Bessel function of the second kind. For the common
case ν = 1 (SPDE order α = 2 in 2-D) this is `ρ(r) = κr · K₁(κr)`, with limit 1
as r → 0. Used to gate the FEM-discretised SPDE precision against the analytic
Matérn correlation it approximates.
"""
function matern_correlation(r::Real, κ::Real; ν::Real = 1.0)
    κ > 0 || throw(ArgumentError("κ must be positive; got $κ"))
    ν > 0 || throw(ArgumentError("ν must be positive; got $ν"))
    r == 0 && return 1.0
    x = κ * r
    return (x^ν) * besselk(ν, x) / (2.0^(ν - 1) * gamma(ν))
end
