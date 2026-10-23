# Gaussian SPDE spatial-field model + ML fit — the end-to-end "mesh → Q → fit"
# path. This sits on top of the SPDE / Matérn-GMRF FEM machinery in src/spde.jl
# (Lindgren, Rue & Lindström 2011) and the Optim LBFGS fit-driver style used by
# the response families (e.g. src/families/gamma.jl).
#
# Model (M observation sites, N mesh nodes):
#
#     y = μ·1 + A·u + ε,    u ~ N(0, Q⁻¹),   ε ~ N(0, σ²·I_M),
#
# with A the sparse M×N barycentric projector (spde_projector) and Q the sparse
# SPD Matérn precision (spde_precision). Marginalising the latent field gives a
# Gaussian observation law,
#
#     y ~ N(μ·1, Σ_y),    Σ_y = A Q⁻¹ A' + σ²·I_M.
#
# Σ_y is dense and M×M, but we never form it (nor Q⁻¹) in the production path.
# Instead we use the standard INLA-style sparse identities so that every solve
# and log-determinant stays in the sparse N×N world:
#
#   * log-determinant — matrix-determinant lemma:
#         logdet Σ_y = M·log σ² + logdet(Q + σ⁻²·A'A) − logdet Q.
#   * quadratic form — Woodbury, with r = y − μ:
#         Σ_y⁻¹ r = σ⁻²·r − σ⁻⁴·A·( (Q + σ⁻²·A'A) \ (A'·r) ),
#         r' Σ_y⁻¹ r = dot(r, Σ_y⁻¹ r).
#
# Both `Q` and the "capacitance" `Q + σ⁻²·A'A` are sparse SPD, so a single sparse
# Cholesky of each delivers the solve and its logdet. SPD failure (e.g. a θ that
# leaves the feasible region under finite-difference probing) is guarded and maps
# to −Inf loglik / +Inf nll, which the optimiser treats as an infeasible step.
#
# References:
#   - Lindgren, Rue & Lindström 2011 (SPDE ↔ Matérn, JRSSB)
#   - Rue & Held 2005 (GMRFs; sparse-precision Gaussian inference)
#   - the matrix-determinant lemma + Woodbury identity (standard low-rank-update
#     algebra; here applied with the *sparse* precision in the "small" position)

using LinearAlgebra
using SparseArrays

# Exceptions that mean "this parameter value is infeasible" (a non-SPD precision or
# Hessian, a value outside a function's domain). Anything else — a dimension mismatch,
# a bad mesh — is a caller error and must propagate rather than become a sentinel.
_spde_infeasible(err) = err isa PosDefException || err isa SingularException ||
                        err isa DomainError

"""
    spde_gaussian_marginal_loglik(y, A, Q, σ²; μ = 0.0) -> Float64

Gaussian marginal log-likelihood of the SPDE spatial-field model

    y ~ N(μ·1, Σ_y),   Σ_y = A Q⁻¹ A' + σ²·I_M,

evaluated *without* forming `Σ_y` or `Q⁻¹` densely. `y` is the length-M response,
`A` the M×N sparse projector, `Q` the N×N sparse SPD Matérn precision, `σ²` the
i.i.d. observation variance, and `μ` the (scalar) mean.

The marginal is

    ℓ = −½ ( M·log 2π + logdet Σ_y + r' Σ_y⁻¹ r ),    r = y − μ·1,

with `logdet Σ_y` from the matrix-determinant lemma and `r' Σ_y⁻¹ r` from the
Woodbury identity (see the file header), both reduced to sparse Cholesky factor-
isations of `Q` and the capacitance `Q + σ⁻²·A'A`. If either Cholesky fails to be
SPD the model is infeasible and `-Inf` is returned.
"""
function spde_gaussian_marginal_loglik(y::AbstractVector, A::AbstractMatrix,
        Q::AbstractMatrix, σ²::Real; μ::Real = 0.0)
    M = length(y)
    σ2 = float(σ²)
    σ2 > 0 || return -Inf

    r = y .- μ
    invσ2 = 1.0 / σ2

    # Sparse SPD Cholesky of Q and of the capacitance C = Q + σ⁻²·A'A.
    AtA = A' * A
    C = Symmetric(Q + invσ2 .* AtA)

    local cholQ, cholC
    try
        cholQ = cholesky(Symmetric(Q))
        cholC = cholesky(C)
    catch err
        _spde_infeasible(err) || rethrow()
        return -Inf
    end

    # logdet Σ_y = M·log σ² + logdet C − logdet Q   (matrix-determinant lemma).
    logdetΣy = M * log(σ2) + logdet(cholC) - logdet(cholQ)

    # Woodbury: Σ_y⁻¹ r = σ⁻²·r − σ⁻⁴·A·( C \ (A'·r) ).
    Atr = A' * r
    w = cholC \ Vector(Atr)
    Σyinv_r = invσ2 .* r .- (invσ2^2) .* (A * w)
    quad = dot(r, Σyinv_r)

    return -0.5 * (M * log(2π) + logdetΣy + quad)
end

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

"""
    SPDEGaussianFit

Result of [`fit_spde_gaussian`](@ref): the estimated Matérn inverse-range `κ` and
precision-scale `τ`, the observation variance `σ2`, the mean `μ`, the maximised
marginal `loglik`, the optimiser `converged` flag and `iterations`, and the mesh
(`nodes`, `tris`) the model was fit on.
"""
struct SPDEGaussianFit
    κ::Float64
    τ::Float64
    σ2::Float64
    μ::Float64
    loglik::Float64
    converged::Bool
    iterations::Int
    nodes::Any
    tris::Any
end

function Base.show(io::IO, f::SPDEGaussianFit)
    print(io, "SPDEGaussianFit(κ=", round(f.κ; sigdigits = 4),
          ", τ=", round(f.τ; sigdigits = 4),
          ", σ²=", round(f.σ2; sigdigits = 4),
          ", μ=", round(f.μ; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_spde_gaussian(y, nodes, tris, locs; α = 2, κ_init = 1.0, τ_init = 1.0,
                      g_tol = 1e-5, iterations = 500) -> SPDEGaussianFit

Maximum-likelihood fit of the Gaussian SPDE spatial-field model

    y = μ·1 + A·u + ε,   u ~ N(0, Q⁻¹),   ε ~ N(0, σ²·I),

to the length-M response `y` observed at the M rows of `locs`, over the triangular
mesh (`nodes`, `tris`). The FEM matrices `(Cdiag, G) = spde_fem(nodes, tris)` and
the sparse projector `A = spde_projector(nodes, tris, locs)` are computed once;
each optimiser evaluation rebuilds only the sparse precision
`Q = spde_precision(Cdiag, G, exp(logκ), exp(logτ); α)` and calls
[`spde_gaussian_marginal_loglik`](@ref).

The parameter vector is `θ = [log κ, log τ, log σ², μ]` (the three positive scales
on the log line, `μ` free). Optimisation is L-BFGS with a BackTracking line search
and a finite-difference gradient (`autodiff = :finite`). Warm start: `κ`, `τ` from
`κ_init`, `τ_init`; `σ²₀ = 0.5·var(y)`; `μ₀ = mean(y)`.
"""
function fit_spde_gaussian(y::AbstractVector, nodes::AbstractMatrix,
        tris::AbstractMatrix{<:Integer}, locs::AbstractMatrix;
        α::Integer = 2, κ_init::Real = 1.0, τ_init::Real = 1.0,
        g_tol::Real = 1e-5, iterations::Integer = 500)

    yv = collect(float.(y))

    # Precompute the mesh-dependent pieces once.
    Cdiag, G = spde_fem(nodes, tris)
    A = spde_projector(nodes, tris, locs)

    # Warm start on θ = [log κ, log τ, log σ², μ].
    σ20 = 0.5 * var(yv)
    σ20 = σ20 > 0 ? σ20 : 1.0
    μ0 = mean(yv)
    θ0 = [log(float(κ_init)), log(float(τ_init)), log(σ20), μ0]

    function negll(θ)
        κ = exp(θ[1])
        τ = exp(θ[2])
        σ2 = exp(θ[3])
        μ = θ[4]
        (isfinite(κ) && isfinite(τ) && κ > 0 && τ > 0) || return Inf
        v = try
            Q = spde_precision(Cdiag, G, κ, τ; α = α)
            -spde_gaussian_marginal_loglik(yv, A, Q, σ2; μ = μ)
        catch err
            _spde_infeasible(err) || rethrow()
            return Inf
        end
        return isfinite(v) ? v : Inf
    end

    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)

    θ̂ = Optim.minimizer(res)
    κ̂ = exp(θ̂[1])
    τ̂ = exp(θ̂[2])
    σ2̂ = exp(θ̂[3])
    μ̂ = θ̂[4]

    return SPDEGaussianFit(κ̂, τ̂, σ2̂, μ̂, _fit_verdict(res)...,
                           nodes, tris)
end
