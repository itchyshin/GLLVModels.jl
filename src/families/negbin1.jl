# Negative-binomial type-1 (NB1, linear-variance) family pieces for the generic
# Laplace core (src/families/laplace.jl). NB1 has Var = μ(1 + φ) — overdispersion
# that scales LINEARLY with the mean (quasi-Poisson-like), in contrast to NB2's
# Var = μ + μ²/r (quadratic). NB1 is exactly a negative binomial with a
# MEAN-DEPENDENT size r = μ/φ and a constant success probability p = 1/(1+φ):
#
#     y ~ NegativeBinomial(r = μ/φ, p = 1/(1+φ)),   E[y] = μ,  Var = μ(1+φ).
#
# Because the size depends on μ, the score carries digamma terms and the Fisher
# information has NO closed form — it needs E_y[ψ'(y+r)], a convergent sum over the
# NB pmf. We compute the EXACT expected information by that sum (stable pmf
# recursion, truncated at negligible tail mass), so NB1 stays consistent with the
# package's Fisher-scoring-Laplace convention (W = me²·I_μ, expected info ⇒ SPD).
# As φ → 0 (r → ∞) the score → (y−μ)/μ and I_μ → 1/μ, recovering Poisson.
#
# NB1 matches R gllvm's `negative.binomial1` family (its `negative.binomial` is the
# NB2 with quadratic variance); it gives users the linear-variance overdispersion
# model when NB2's quadratic tail is wrong. gllvm parameterises NB1 as Var=μ+μ·φ
# with the SAME φ as here, so the dispersion maps 1:1 across the R↔Julia bridge.

"""
    NB1(φ = 1.0)

Family marker for the negative-binomial type-1 (linear-variance) GLLVM,
`Var = μ(1 + φ)` — R gllvm's `negative.binomial1`. Pass it to
[`fit_gllvm`](@ref) or [`gllvm`](@ref):

```julia
fit_gllvm(Y; family = NB1(), K = 2)     # per-trait φ → NB1GroupedFit
```

The `φ` field is a **tag payload only**: no public route reads it. `φ` is always
estimated — per trait by default, matching gllvmTMB's length-`p`
`log_phi_nbinom1` — and is never used as a starting value. To seed it, pass
`φ_init` to a named fitter ([`fit_nb1_gllvm`](@ref),
[`fit_nb1_gllvm_grouped`](@ref)). Internally the Laplace kernels construct their
own per-iteration `NB1(φ)` markers, which is what the field is for.
"""
struct NB1
    φ::Float64
end

NB1() = NB1(1.0)

# Expected Fisher information of NB1 w.r.t. the mean μ:
#   I_μ = (1/φ²)[ψ'(r) − E_y ψ'(y+r)],  r = μ/φ,  y ~ NB(r, p = 1/(1+φ)).
# No closed form ⇒ summed over the NB pmf via the stable recursion
#   P₀ = p^r,  P_y = P_{y−1}·(1−p)·(y−1+r)/y,   until the tail mass is negligible.
# Returns max(I_μ, 1e-12) so the working weight is strictly positive (SPD).
function _nb1_fisher_mu(μ::Real, φ::Real)
    # For tiny φ, NB1 is in the Poisson-limit regime. The exact expression
    # below subtracts nearly equal trigamma terms and divides by φ^2, which is
    # numerically unstable even though the limit is well behaved.
    if φ <= 1e-6
        return max(inv(μ * (1 + φ)), 1e-12)
    end
    r = μ / φ
    q = φ / (1 + φ)                      # 1 − p, the NB "failure" probability
    tr_r = trigamma(r)
    P = (1 - q)^r                        # P(y = 0) = p^r
    cum = P
    Eψ = P * tr_r                        # y = 0 term: ψ'(0 + r)
    y = 0
    @inbounds while cum < 1 - 1e-12 && y < 10_000
        y += 1
        P *= q * (y - 1 + r) / y
        Eψ += P * trigamma(y + r)
        cum += P
    end
    return max((tr_r - Eψ) / φ^2, 1e-12)
end

_clamp_mu(::NB1, μ) = max(μ, 1e-12)
# Score wrt η: me·∂logf/∂μ,  ∂logf/∂μ = (1/φ)[ψ(y+r) − ψ(r) − log(1+φ)],  r = μ/φ.
_glm_score(f::NB1, μ, n, me, y) = me * (digamma(y + μ / f.φ) - digamma(μ / f.φ) - log1p(f.φ)) / f.φ
# Expected-information working weight wrt η:  me²·I_μ.
_glm_weight(f::NB1, μ, n, me)   = me^2 * _nb1_fisher_mu(μ, f.φ)
_glm_logpdf(f::NB1, μ, n, y)    = logpdf(NegativeBinomial(μ / f.φ, 1 / (1 + f.φ)), Int(y))

"""
    nb1_marginal_loglik_laplace(Y, Λ, β, φ; link=LogLink(), kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a negative-binomial
type-1 (NB1) GLLVM with dispersion `φ` (linear variance `Var = μ(1+φ)`) — a thin
wrapper over the family-generic `marginal_loglik_laplace` with the `NB1(φ)` marker.
`Y` is the p×n integer count matrix; `Λ` p×K; `β` length-p. With `Λ = 0` this
reduces exactly to the independent NB1 log-likelihood; as `φ → 0` it tends to the
Poisson marginal.
"""
nb1_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector,
        φ::Real; link::Link = LogLink(), kwargs...) =
    marginal_loglik_laplace(NB1(float(φ)), Y, ones(Int, size(Y)), Λ, β, link; kwargs...)

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

"""
    NB1Fit

Result of [`fit_nb1_gllvm`](@ref): intercepts `β` (length p), loadings `Λ` (p×K),
the estimated dispersion `φ` (linear variance `Var = μ(1+φ)`), the `link`, the
maximised Laplace `loglik`, the optimiser `converged` flag, and `iterations`.
"""
struct NB1Fit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
end

# Positional compatibility constructor (2026-08-28): every pre-existing
# construction site builds a default-curvature fit; the `hessian` field
# records the objective identity so `confint`/bootstrap can rebuild THE
# SAME objective instead of guessing (the audit's confint-consistency class).
NB1Fit(β, Λ, φ, link, loglik, converged, iterations) =
    NB1Fit(β, Λ, φ, link, loglik, converged, iterations, _default_hessian(NB1(1.0), link))

function Base.show(io::IO, f::NB1Fit)
    p, K = size(f.Λ)
    print(io, "NB1Fit(p=", p, ", K=", K, ", φ=", round(f.φ; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# NB1/log Laplace log-det defaults to the OBSERVED conditional curvature
# (decision A, 2026-08-27 — campaign: estimator preference 90–100% med/strong;
# reported-loglik cost accepted). Delegates to the hand-derived grouped
# formula (the 2026-08-24 NB1 parity fix's weight). No analytic-gradient
# coupling: this fitter is finite-difference only.
_glm_obs_weight(f::NB1, μ, n, me, y, link::LogLink, η) =
    _nb1_grouped_laplace_weight(:observed, f, μ, me, y, link)
_default_hessian(::NB1, ::LogLink) = :observed

# Opt into the damped mode-search backtracking (2026-08-27 audit rider): the
# grouped solver's gate `any(_laplace_mode_should_backtrack, fams)` silently
# no-opped for NB1, leaving its grouped mode search undamped.
_laplace_mode_should_backtrack(::NB1) = true

"""
    fit_nb1_gllvm(Y; K, link=LogLink(), mask=nothing, offset=nothing, φ_init=nothing, …) -> NB1Fit

Fit a negative-binomial type-1 (NB1) GLLVM by L-BFGS over `[β; vec(Λ); log φ]` on
the Laplace marginal ([`nb1_marginal_loglik_laplace`](@ref)), jointly estimating
the linear-variance dispersion `φ` (`Var = μ(1+φ)`). `Y` is a p×n integer count
matrix (may contain `missing`); `K` the latent dimension. Finite-difference
gradient; warm start = empirical log-mean intercepts + an SVD loadings init + a
moderate `φ₀`.

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or `missing` entries in
`Y`; masked cells are dropped from the marginal and the warm start.

`hessian` selects the Laplace log-det curvature only (`:fisher` expected /
`:observed` joint — TMB's choice); the inner mode search is always
Fisher-scored. Default: NB1/log default `:fisher`. Omitting it is exactly the pre-kwarg
behaviour.
"""
function fit_nb1_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), mask = nothing, offset = nothing,
        hessian::Symbol = _default_hessian(NB1(1.0), link),
        β_init = nothing, Λ_init = nothing, φ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    rr = rr_theta_len(p, K)
    hessian in (:fisher, :observed) || throw(ArgumentError(
        "fit_nb1_gllvm: hessian must be :fisher or :observed; got :$hessian"))

    msk = mask === nothing ? (any(ismissing, Y) ? observed_mask(Y) : nothing) : mask
    Yc = Integer.(_sanitize_missing(Y, 0))

    Zemp = [linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)
    if msk !== nothing
        @inbounds for t in 1:p
            cnt = count(view(msk, t, :))
            rowmean = cnt > 0 ? sum(Zemp[t, i] for i in 1:n if msk[t, i]) / cnt : 0.0
            for i in 1:n
                msk[t, i] || (Zemp[t, i] = rowmean)
            end
        end
    end
    β0 = β_init === nothing ? vec(sum(Zemp; dims = 2)) ./ n : collect(float.(β_init))
    Λ0 = if Λ_init === nothing
        Zc = Zemp .- β0
        F = svd(Zc)
        kk = min(K, length(F.S))
        L = zeros(p, K)
        @inbounds for j in 1:kk
            L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        L
    else
        collect(float.(Λ_init))
    end
    logφ0 = φ_init === nothing ? log(1.0) : log(float(φ_init))

    θ0 = vcat(β0, pack_lambda(Λ0), logφ0)
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φ = exp(θ[p + rr + 1])
        v = try
            -nb1_marginal_loglik_laplace(Yc, Λ, β, φ; link = link, mask = msk, offset = offset,
                                         hessian = hessian,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    φ̂ = exp(θ̂[p + rr + 1])
    return NB1Fit(β̂, Λ̂, φ̂, link, _fit_verdict(res)..., hessian)
end
