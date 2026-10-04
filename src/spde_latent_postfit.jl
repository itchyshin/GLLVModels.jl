# Post-fit API for the SPDE-latent GLLVM (SPDELatentFit).
#
# The spatial GLLVM fits K Matérn-GMRF fields u_1,…,u_K (N×1 each) on a
# triangular mesh. The site scores at the M training locations are Z = A·U
# (M×K), where A = spde_projector(nodes, tris, locs) and U is the N×K
# matrix of field values at the mesh nodes. The linear predictor is
#
#     η = β .+ Λ * (A * U)'           (p × M)
#
# and the conditional mean is linkinv(link, η).
#
# Three entry points are provided:
#
#   getLV(fit, Y, locs)                → site latent scores Z = A·Û (M×K)
#   predict(fit, Y, locs; type)        → η or mean at the training sites (p×M)
#   predict_spatial(fit, Y, locs,      → η or mean at NEW locations (p×M′)
#                   new_locs; type)      via kriging interpolation of Û
#
# The field mode Û is found by `_spde_latent_mode` (extracted from
# spde_latent_marginal_loglik in the conservative refactor of spde_latent.jl).
# Rebuilding Q from fit.κ, fit.τ and the stored mesh (nodes, tris) via
# spde_fem + spde_precision ensures Û is computed at the fitted parameters.
#
# References:
#   - spde_latent.jl (the model; _spde_latent_mode)
#   - spde.jl        (spde_fem, spde_precision, spde_projector)
#   - postfit.jl     (signature and style conventions)

# ---------------------------------------------------------------------------
# Internal helper: rebuild Q and run the mode-finder at the fitted parameters.
# Returns (U::Matrix{Float64}, A::SparseMatrixCSC, Qs::SparseMatrixCSC)
# or throws on numerical failure.
# ---------------------------------------------------------------------------
function _spde_latent_rebuild(fit::SPDELatentFit,
                               Y::AbstractMatrix,
                               locs::AbstractMatrix;
                               α::Integer = 2,
                               maxiter::Integer = 50,
                               tol::Real = 1e-9)
    p, M = size(Y)
    Ntr = if size(fit.N) == (p, M)
        fit.N
    elseif isempty(fit.N)
        ones(Float64, p, M)
    else
        throw(ArgumentError("stored trial counts size $(size(fit.N)) does not match Y $(size(Y))"))
    end
    Cdiag, G = spde_fem(fit.nodes, fit.tris)
    Q        = spde_precision(Cdiag, G, fit.κ, fit.τ; α = α)
    Qs       = sparse(Q)
    A        = spde_projector(fit.nodes, fit.tris, locs)

    # Reconstruct the family marker (re-attach dispersion where relevant).
    fam = _spde_make_family(fit.family,
                            isnan(fit.dispersion) ? Float64[] :
                            [log(fit.dispersion)])

    U = _spde_latent_mode(fam, Y, Ntr, fit.Λ, fit.β, fit.link, A, Qs;
                          maxiter = maxiter, tol = tol)
    if U === nothing
        error("SPDE-latent mode-finder failed to converge (non-SPD Hessian). " *
              "The fit may be numerically ill-conditioned.")
    end
    return U, A, Qs
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    getLV(fit::SPDELatentFit, Y, locs; α=2, maxiter=50, tol=1e-9,
          return_nodes=false) -> Z::Matrix{Float64}

Site-level latent scores `Z = A·Û` (M×K), where `Û` (N×K) is the joint-Laplace
field mode at the fitted parameters and `A` is the projector from mesh nodes to
the M rows of `locs`.

`Y` (p×M) and `locs` (M×2) must match what was passed to
[`fit_spde_latent_gllvm`](@ref). Set `return_nodes=true` to return `(Z, Û)`
instead of just `Z`, which exposes the mesh-node field BLUPs directly.
"""
function getLV(fit::SPDELatentFit, Y::AbstractMatrix, locs::AbstractMatrix;
               α::Integer = 2, maxiter::Integer = 50, tol::Real = 1e-9,
               return_nodes::Bool = false)
    U, A, _ = _spde_latent_rebuild(fit, Y, locs; α = α, maxiter = maxiter, tol = tol)
    Z = A * U                              # M×K
    return_nodes && return Z, U
    return Z
end

"""
    predict(fit::SPDELatentFit, Y, locs; type=:response, α=2,
            maxiter=50, tol=1e-9) -> Matrix{Float64}

In-sample predictions at the training locations. `type=:link` returns the linear
predictor `η = β .+ Λ*(A·Û)'` (p×M); `type=:response` applies the inverse link to
give the fitted means (p×M).

`Y` (p×M) and `locs` (M×2) must match the fit.
"""
function predict(fit::SPDELatentFit, Y::AbstractMatrix, locs::AbstractMatrix;
                 type::Symbol = :response, α::Integer = 2,
                 maxiter::Integer = 50, tol::Real = 1e-9)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    U, A, _ = _spde_latent_rebuild(fit, Y, locs; α = α, maxiter = maxiter, tol = tol)
    Z  = A * U                             # M×K
    η  = fit.β .+ fit.Λ * Z'              # p×M
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

"""
    predict_spatial(fit::SPDELatentFit, Y, train_locs, new_locs;
                    type=:response, α=2, maxiter=50, tol=1e-9) -> Matrix{Float64}

Spatial prediction (kriging) at new, unobserved locations. The field mode `Û`
(N×K) is found using `train_locs` (M×2) and `Y` (p×M), then interpolated to
`new_locs` (M′×2) via `A_new = spde_projector(fit.nodes, fit.tris, new_locs)`:

    η_new = β .+ Λ * (A_new · Û)'          (p × M′)

`type=:link` returns `η_new`; `type=:response` applies the inverse link.

When `new_locs == train_locs` the result equals `predict(fit, Y, train_locs)` to
machine precision, because the same projector rows are produced.

This is the headline "go beyond gllvm" capability: predictions at arbitrary
spatial locations within (or near) the mesh by barycentric interpolation of the
fitted Matérn field.
"""
function predict_spatial(fit::SPDELatentFit,
                         Y::AbstractMatrix,
                         train_locs::AbstractMatrix,
                         new_locs::AbstractMatrix;
                         type::Symbol = :response, α::Integer = 2,
                         maxiter::Integer = 50, tol::Real = 1e-9)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))

    # Find the field mode at the fitted params using the training projector.
    U, _, _ = _spde_latent_rebuild(fit, Y, train_locs;
                                   α = α, maxiter = maxiter, tol = tol)

    # Build a fresh projector for the new locations and interpolate.
    A_new = spde_projector(fit.nodes, fit.tris, new_locs)
    Z_new = A_new * U                     # M′×K
    η_new = fit.β .+ fit.Λ * Z_new'      # p×M′
    type === :link && return η_new
    return linkinv.(Ref(fit.link), _clamp_eta.(η_new))
end

# ---------------------------------------------------------------------------
# Information criteria: parameter count and log-likelihood accessors so the
# generic `aic` / `bic` (src/postfit.jl) work for the SPDE-latent model.
# Free parameters: β (p) + reduced lower-triangular loadings Λ + (κ, τ) +
# a dispersion parameter for the dispersion families (Gaussian σ², NB r).
# ---------------------------------------------------------------------------
function _nparams(fit::SPDELatentFit)
    p, K = size(fit.Λ)
    ndisp = isnan(fit.dispersion) ? 0 : 1
    return p + rr_theta_len(p, K) + 2 + ndisp
end
_loglik(fit::SPDELatentFit) = fit.loglik

# Loadings accessor so the generic `getLoadings` / `ordination` (rotation via the
# SVD of Λ) work for the SPDE-latent model.
_loadings(fit::SPDELatentFit) = fit.Λ
