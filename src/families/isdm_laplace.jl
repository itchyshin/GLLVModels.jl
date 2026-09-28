# iSDM per-cell Laplace kernel over long rows (docs/design/isdm-port-spec.md
# sections 1.2, 1.3, 3.2).
#
# For long row o with trait t(o), unit (cell) s(o):
#     eta(o) = [X b](o) + offset(o) + Lambda[t(o), :] . z_s,   z_s ~ N(0, I_K)
# count rows (R fid 2):     y ~ Poisson(exp(eta))
# detection rows (fid 1):   y ~ Bernoulli(1 - exp(-exp(eta))), evaluated with a
#                           copy of R's gll_dbinom_cloglog (src/gllvmTMB_cloglog.h)
# With latent(..., unique = TRUE) (R's default) eta(o) also carries
# s_B(t(o), s) ~ N(0, exp(theta_diag_B[t])^2); the kernel takes it through the
# augmented loadings [Lambda diag(exp(theta_diag_B))] (see `_isdm_augment`).
# Rows are conditionally independent given z_s, and z is independent across
# cells, so TMB's Laplace approximation over z_B equals a per-cell Laplace with
# the observed curvature of the joint negative log density:
#     L_s = sum_o l_o(zhat) - 1/2 zhat'zhat - 1/2 logdet(A_obs),
#     A_obs = sum_o W_o lambda_t(o) lambda_t(o)' + I_K,  W_o = -d2 l_o / d eta2.
#
# This kernel is a long-table sibling of src/families/mixed.jl: one cell has
# several rows per trait, so the p x n matrix substrate of laplace.jl / mixed.jl
# does not fit. The mode search COPIES (does not share or call) the damped
# Fisher-scoring rule of `_mixed_laplace_mode` (step halving
# on a decrease of the cell log-posterior, the Newton-decrement convergence
# test, the floating-point-floor acceptance and the stalled-step acceptance);
# the shared `_laplace_mode` is neither edited nor called. With K = 0 (no
# latent() term) there is no random effect and a cell's value is the plain
# GLM log-likelihood of its rows.

# ---------------------------------------------------------------------------
# The detection-row density: a Dual-safe copy of gll_dbinom_cloglog.
# ---------------------------------------------------------------------------

# log(1 - exp(-exp(eta))), branch for branch as gll_log_cloglog_p
# (src/gllvmTMB_cloglog.h:14-41 at P1). CppAD::CondExp evaluates both branches
# and selects one; `ifelse` does the same and selects the whole Dual, so the
# unselected branch's partials never leak. Every branch evaluates at an argument
# clamped to where its arithmetic stays finite.
function _isdm_log_cloglog_p(eta)
    T = typeof(eta)
    left_cut = T(-20.0)
    right_cut = T(700.0)
    eta_left = ifelse(eta > left_cut, left_cut, eta)
    lambda_left = exp(eta_left)
    left_series = one(T) - lambda_left / T(2.0) +
        lambda_left * lambda_left / T(6.0) -
        lambda_left * lambda_left * lambda_left / T(24.0)
    left = eta_left + log(left_series)
    eta_mid = ifelse(eta < left_cut, left_cut, eta)
    eta_mid = ifelse(eta_mid > right_cut, right_cut, eta_mid)
    lambda_mid = exp(eta_mid)
    middle = log(one(T) - exp(-lambda_mid))
    not_left = ifelse(eta <= left_cut, left, middle)
    return ifelse(eta >= right_cut, zero(T), not_left)
end

"""
    _isdm_dbinom_cloglog(y, eta, n = 1)

Binomial(n)-cloglog log-density evaluated on the log scale in both tails, a copy
of R's `gll_dbinom_cloglog` (`src/gllvmTMB_cloglog.h:45-57` at gllvmTMB P1):
`lchoose(n, y) + y * log(1 - exp(-exp(eta))) - (n - y) * exp(min(eta, 700))`,
with a series expansion below `eta = -20`. Written with `ifelse` so ForwardDiff
differentiates it (any order).
"""
function _isdm_dbinom_cloglog(y, eta, n = 1)
    T = typeof(eta)
    right_cut = T(700.0)
    eta_q = ifelse(eta > right_cut, right_cut, eta)
    log_choose = loggamma(n + 1.0) - loggamma(y + 1.0) - loggamma(n - y + 1.0)
    return log_choose + y * _isdm_log_cloglog_p(eta) - (n - y) * exp(eta_q)
end

"""
    _isdm_cloglog_score(y, eta)

First derivative in `eta` of [`_isdm_dbinom_cloglog`](@ref) (ForwardDiff).
"""
_isdm_cloglog_score(y, eta) = ForwardDiff.derivative(e -> _isdm_dbinom_cloglog(y, e), eta)

"""
    _isdm_cloglog_obs_weight(y, eta)

Observed weight `-d2/d eta2` of [`_isdm_dbinom_cloglog`](@ref) itself (nested
ForwardDiff), so the curvature in the Laplace log-determinant is the curvature
of the density actually summed.
"""
_isdm_cloglog_obs_weight(y, eta) =
    -ForwardDiff.derivative(e -> ForwardDiff.derivative(x -> _isdm_dbinom_cloglog(y, x), e), eta)

# Fisher weight of a Bernoulli-cloglog row, used only to build the step matrix
# of the mode search (never the log-determinant): me^2 / (mu (1 - mu)).
function _isdm_cloglog_fisher(eta)
    link = CLogLogLink()
    mu = _clamp_mu(Binomial(), linkinv(link, eta))
    return _glm_weight(Binomial(), mu, 1, mu_eta(link, eta))
end

# Per-row log-density, score and observed weight, dispatched on R's family id.
_isdm_row_logpdf(fid::Integer, y, eta) =
    fid == 2 ? _pois_logpmf(exp(eta), y) : _isdm_dbinom_cloglog(y, eta)
_isdm_row_score(fid::Integer, y, eta) =
    fid == 2 ? y - exp(eta) : _isdm_cloglog_score(y, eta)
_isdm_row_obs_weight(fid::Integer, y, eta) =
    fid == 2 ? exp(eta) : _isdm_cloglog_obs_weight(y, eta)
_isdm_row_fisher(fid::Integer, eta) =
    fid == 2 ? exp(eta) : _isdm_cloglog_fisher(eta)

# Linear predictor of row o at score z: eta0[o] + Lambda[t(o), :] . z.
@inline function _isdm_eta(eta0o, Λ::AbstractMatrix, t::Integer, z::AbstractVector)
    acc = eta0o
    @inbounds for k in eachindex(z)
        acc += Λ[t, k] * z[k]
    end
    return acc
end

# Cell log-posterior at z (the value the step-halving line search compares).
function _isdm_cell_logpost(y, fid, tr, eta0, Λ, z)
    q = -0.5 * dot(z, z)
    @inbounds for o in eachindex(y)
        q += _isdm_row_logpdf(fid[o], y[o], _isdm_eta(eta0[o], Λ, tr[o], z))
    end
    return q
end

"""
    _isdm_cell_mode(y, fid, tr, eta0, Λ; maxiter = 100, tol = 1e-9, grad_tol = 1e-6,
                    nd_tol = 1e-12, z_init = nothing) -> (z, converged)

Damped Fisher-scoring search for the conditional mode of one cell's latent
score. Each row contributes its score `s_o` and Fisher weight `W_o` to its
trait's loading row: `g = sum_o s_o lambda_t(o) - z`, `A = sum_o W_o lambda
lambda' + I_K`, step `A \\ g`. The convergence rule is a copy of
`_mixed_laplace_mode` in src/families/mixed.jl: converged
when both the step and the Newton decrement `g'Δ` are small, accepted at the
floating-point floor or when a stalled line search leaves a decrement below
`nd_tol`; a step that lowers the cell log-posterior is halved.
"""
function _isdm_cell_mode(y::AbstractVector, fid::AbstractVector, tr::AbstractVector,
        eta0::AbstractVector, Λ::AbstractMatrix;
        maxiter::Integer = 100, tol::Real = 1e-9, grad_tol::Real = 1e-6,
        nd_tol::Real = 1e-12, z_init = nothing)
    K = size(Λ, 2)
    T = promote_type(eltype(Λ), eltype(eta0))
    z = z_init === nothing ? zeros(T, K) : collect(T, z_init)
    g = Vector{T}(undef, K)
    A = Matrix{T}(undef, K, K)
    linesearch_only = false   # set once a full step has been rejected
    prev_dmax = Inf           # previous max|Δ|: has the step stopped shrinking?
    for _ in 1:maxiter
        @. g = -z
        fill!(A, zero(T))
        @inbounds for k in 1:K
            A[k, k] = one(T)
        end
        @inbounds for o in eachindex(y)
            t = tr[o]
            η = _isdm_eta(eta0[o], Λ, t, z)
            s = _isdm_row_score(fid[o], y[o], η)
            W = _isdm_row_fisher(fid[o], η)
            for k in 1:K
                g[k] += s * Λ[t, k]
                for j in 1:K
                    A[j, k] += W * Λ[t, j] * Λ[t, k]
                end
            end
        end
        Δ = _safe_solve(Symmetric(A), g)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        decrement = abs(dot(g, Δ))
        maximum(abs, Δ) < tol && decrement < grad_tol && return z .+ Δ, true
        dmax = maximum(abs, Δ)
        at_floor = dmax <= sqrt(eps(Float64)) * (1 + norm(z))
        at_floor && dmax >= prev_dmax && decrement < nd_tol && return z .+ Δ, true
        prev_dmax = dmax
        if (!linesearch_only || at_floor) && norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _isdm_cell_logpost(y, fid, tr, eta0, Λ, z)
            if isfinite(q0)
                zprev = z
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _isdm_cell_logpost(y, fid, tr, eta0, Λ, ztrial)
                    if isfinite(q1) && q1 >= q0
                        z = ztrial
                        accepted = true
                        break
                    end
                    step *= 0.5
                    linesearch_only = true   # a full-size step was just rejected
                end
                (!accepted || z == zprev) && decrement < nd_tol && return z, true
                accepted || return z, false
            else
                z = z .+ Δ
            end
        end
    end
    return z, false
end

# Mode with the larger-budget retry of `_mixed_loglik_site` (20x `maxiter`,
# from the same start) for a cell whose default budget ran out.
function _isdm_cell_mode_retry(y, fid, tr, eta0, Λ; maxiter::Integer = 100,
        tol::Real = 1e-9, z_init = nothing)
    z, ok = _isdm_cell_mode(y, fid, tr, eta0, Λ; maxiter = maxiter, tol = tol, z_init = z_init)
    ok || ((z, ok) = _isdm_cell_mode(y, fid, tr, eta0, Λ; maxiter = 20 * maxiter,
                                     tol = tol, z_init = z_init))
    return z, ok
end

# Laplace value of one cell at a given z (no search): sum_o l_o(z) - z'z/2 -
# logdet(A_obs(z))/2. `-Inf` when A_obs is not positive definite.
function _isdm_cell_value_at(y, fid, tr, eta0, Λ, z)
    K = size(Λ, 2)
    T = promote_type(eltype(Λ), eltype(eta0), eltype(z))
    ℓ = zero(T)
    A = Matrix{T}(undef, K, K)
    fill!(A, zero(T))
    @inbounds for k in 1:K
        A[k, k] = one(T)
    end
    @inbounds for o in eachindex(y)
        t = tr[o]
        η = _isdm_eta(eta0[o], Λ, t, z)
        ℓ += _isdm_row_logpdf(fid[o], y[o], η)
        K == 0 && continue
        W = _isdm_row_obs_weight(fid[o], y[o], η)
        for k in 1:K, j in 1:K
            A[j, k] += W * Λ[t, j] * Λ[t, k]
        end
    end
    K == 0 && return ℓ
    C = cholesky(Symmetric(A); check = false)
    issuccess(C) || return T(-Inf)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(C)
end

"""
    _isdm_cell_loglik(y, fid, tr, eta0, Λ; maxiter = 100, tol = 1e-9) -> (value, z, converged)

Laplace log-marginal of one cell. A cell whose mode search does not certify a
stationary point returns `-Inf`, so the fitter's failure barrier fires instead
of a silently wrong value.
"""
function _isdm_cell_loglik(y, fid, tr, eta0, Λ; maxiter::Integer = 100, tol::Real = 1e-9,
        z_init = nothing)
    K = size(Λ, 2)
    T = promote_type(eltype(Λ), eltype(eta0))
    if K == 0
        return _isdm_cell_value_at(y, fid, tr, eta0, Λ, zeros(T, 0)), zeros(T, 0), true
    end
    z, ok = _isdm_cell_mode_retry(y, fid, tr, eta0, Λ; maxiter = maxiter, tol = tol, z_init = z_init)
    ok || return T(-Inf), z, false
    return _isdm_cell_value_at(y, fid, tr, eta0, Λ, z), z, true
end

# The augmented loading matrix of a `unique = TRUE` fit (R's theta_diag_B):
# s_B(t, s) ~ N(0, exp(theta_d[t])^2) enters eta additively with an identity
# loading, so with s_B = diag(exp(theta_d)) u, u ~ N(0, I_p), the model is the
# loadings-only kernel with Lambda_aug = [Lambda diag(exp(theta_d))] and
# z_aug = [z; u] ~ N(0, I_{K+p}). A linear change of variables leaves the
# Laplace approximation unchanged (the Jacobian cancels against the Hessian
# determinant), so this is exactly R's joint Laplace over (z_B, s_B).
function _isdm_augment(Λ::AbstractMatrix, θd::AbstractVector)
    p, K = size(Λ)
    length(θd) == p || throw(DimensionMismatch(
        "theta_diag_B has length $(length(θd)); the table has $p traits."))
    T = promote_type(eltype(Λ), eltype(θd))
    A = zeros(T, p, K + p)
    A[:, 1:K] .= Λ
    @inbounds for t in 1:p
        A[t, K + t] = exp(θd[t])
    end
    return A
end

"""
    isdm_marginal_loglik_laplace(table::IsdmTable, Λ, b; theta_diag_B = nothing,
                                 maxiter = 100, tol = 1e-9) -> Real

Laplace log-marginal of an integrated species-distribution model: the sum over
units (cells) of the per-cell Laplace value, with the observed curvature of the
summed density in the log-determinant (the curvature TMB obtains by automatic
differentiation). `Λ` is the `p x K` loading matrix (`K = 0` for a GLM fit),
`b` the fixed coefficients in the order of `table.X_names`. For a table built
from `latent(..., unique = TRUE)` (R's default), `theta_diag_B` is required: the
length-`p` vector of log standard deviations of the per-trait unit-level unique
effects (gllvmTMB's `theta_diag_B`), integrated jointly with the latent scores.
Returns `-Inf` if any cell's mode search fails.
"""
function isdm_marginal_loglik_laplace(table::IsdmTable, Λ::AbstractMatrix, b::AbstractVector;
        theta_diag_B = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    size(Λ, 1) == length(table.trait_levels) || throw(DimensionMismatch(
        "Λ has $(size(Λ, 1)) rows; the table has $(length(table.trait_levels)) traits."))
    length(b) == size(table.X, 2) || throw(DimensionMismatch(
        "b has length $(length(b)); the design has $(size(table.X, 2)) columns."))
    if table.unique
        theta_diag_B === nothing && throw(ArgumentError(
            "This table was built from latent(..., unique = TRUE): pass `theta_diag_B`, the " *
            "length-$(length(table.trait_levels)) vector of log standard deviations of the " *
            "per-trait unit-level unique effects."))
        Λ = _isdm_augment(Λ, theta_diag_B)
    else
        theta_diag_B === nothing || throw(ArgumentError(
            "`theta_diag_B` was given, but this table was built without a unique variance " *
            "(latent(..., unique = FALSE) or no latent() term)."))
    end
    eta0 = table.X * b .+ table.offset
    acc = zero(promote_type(eltype(Λ), eltype(b)))
    for rows in table.rows_by_unit
        v, _, _ = _isdm_cell_loglik(view(table.y, rows), view(table.fid, rows),
                                    view(table.trait_id, rows), view(eta0, rows), Λ;
                                    maxiter = maxiter, tol = tol)
        isfinite(v) || return oftype(acc, -Inf)
        acc += v
    end
    return acc
end
