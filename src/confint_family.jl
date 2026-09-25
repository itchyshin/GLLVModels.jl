# Confidence intervals for the non-Gaussian (Laplace) family fitters.
#
# The Gaussian path (src/confint.jl / confint_profile.jl / confint_bootstrap.jl)
# is tightly coupled to GllvmFit and the legacy θ_packed layout. The family
# fitters (Poisson / Binomial / NB / Beta / Gamma) each carry their own struct
# and a `[β; pack_lambda(Λ); (log-dispersion)]` working vector, so this file
# provides a SINGLE generic CI layer over a small per-family adapter, exposing
# all three R-style methods through one entry point:
#
#     confint(fit, Y; method = :wald | :profile | :bootstrap, ...)
#
# - :wald      — observed-information (finite-difference Hessian) → inv → SEs →
#                θ̂ ± z·SE, with an exp() back-transform for log-scale dispersion.
# - :profile   — invert the LRT: D(c) = 2(ℓ̂ − ℓ_p(c)) ~ χ²₁; bracket-then-bisect
#                each side (reuses `_profile_bisect_side` from confint_profile.jl).
# - :bootstrap — parametric: simulate Yᵇ ~ fitted model, refit, take percentiles.
#                Embarrassingly parallel — `parallel = true` runs the replicates
#                over `Threads.@threads`; each replicate seeds its own RNG
#                (`seed + b`) so results are independent of thread scheduling
#                (identical single- vs multi-core).
#
# The family fitters use finite-difference gradients (the Laplace inner
# mode-finder is not forward-AD-friendly), so the Hessian here is a
# central-difference one rather than ForwardDiff — consistent with how the
# fitters themselves are optimised.

using Distributions: Normal, Chisq, TDist, quantile
using Random: AbstractRNG, MersenneTwister, randn

# Families handled by this layer (single latent block, optional scalar dispersion).
const _FamilyFit = Union{PoissonFit, BinomialFit, NBFit, NB1Fit, GP1Fit, BetaFit, GammaFit, ExponentialFit,
                         TweedieFit, BetaBinomialFit, RowRandomFit, LognormalFit, TruncatedPoissonFit,
                         TruncatedNegBin2Fit, StudentTFit}

# Two-part families ([βz; βc; pack_lambda(Λc); (log-dispersion)] layout).
const _TwoPartFit = Union{DeltaLogNormalFit, DeltaGammaFit, HurdlePoissonFit,
                          HurdleNBFit, ZIPFit, ZINBFit, ZIBFit, BetaHurdleFit}

# Everything the unified confint(fit, Y; method=…) entry accepts.
const _GroupedDispersionFit = Union{NBGroupedFit, NB1GroupedFit, BetaGroupedFit, GammaGroupedFit,
                                    BetaBinomialGroupedFit, TweedieGroupedFit, TweediePerTraitPowerFit}
const _GroupedDispersionCovFit = Union{NBGroupedCovFit, NB1GroupedCovFit, BetaGroupedCovFit, GammaGroupedCovFit,
                                       BetaBinomialGroupedCovFit}

const _CIFit = Union{_FamilyFit, _TwoPartFit, _GroupedDispersionFit, _GroupedDispersionCovFit,
                     OrdinalFit, OrdinalPerTraitFit, OrdinalPerTraitCovFit, MultinomialFit,
                     GllvmCovFit, ZIPCovFit, ZINBCovFit, OrderedBetaFit, QuadraticFit, RowEffectFit}

# ---------------------------------------------------------------------------
# Per-family adapter. Bundles everything the generic routines need:
#   θ        — MLE working vector (matches the fitter's negll layout)
#   nll      — negative Laplace log-likelihood as a function of that vector
#   names    — term names in θ order
#   kinds    — :linear (β, Λ) or :log (a log-scale dispersion: r / φ / α)
#   simulate — rng -> Yᵇ (a fresh parametric draw from the fitted model)
#   refit    — Yᵇ -> working vector θ̂ᵇ (or `nothing` on refit failure)
#   boundary — per-parameter KNOWN-boundary flag (T14 F1, 2026-09-02): true at
#              positions carried over from a fit's own `dispersion_boundary`
#              (grouped NB2/NB1/Beta/Gamma). Defaults to all-`false` via the
#              6-arg constructor below, so every pre-existing `_FamilyCI(...)`
#              call site is unaffected.
# ---------------------------------------------------------------------------
struct _FamilyCI
    θ::Vector{Float64}
    nll::Function
    names::Vector{String}
    kinds::Vector{Symbol}
    simulate::Function
    refit::Function
    boundary::Vector{Bool}
end

_FamilyCI(θ, nll, names, kinds, simulate, refit) =
    _FamilyCI(θ, nll, names, kinds, simulate, refit, falses(length(θ)))

# Shared GLM term names: β[t] for t in 1:p, then Λ[i,k] in pack_lambda order.
_glm_lin_names(p::Integer, K::Integer) =
    vcat(["beta[$t]" for t in 1:p], _confint_lambda_term_names("Lambda", p, K))

function _ci_mask(mask, Y::AbstractMatrix)
    mask === nothing && return nothing
    M = Matrix{Bool}(mask)
    size(M) == size(Y) || throw(ArgumentError(
        "confint mask must have size $(size(Y)); got $(size(M))"))
    all(M) && return nothing
    any(M) || throw(ArgumentError("confint mask has no observed cells"))
    return M
end

# --- Poisson ---------------------------------------------------------------
function _family_ci(fit::PoissonFit, Y::AbstractMatrix;
                    mask = nothing,
                    objective::Symbol = :fit,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    if _is_poisson_aghq(fit)
        objective in (:fit,:aghq) || throw(ArgumentError("AGHQ inference must use objective=:fit; Laplace/VA would change the estimator"))
        q,_=_poisson_aghq_problem(fit,Y;mask=mask,require_identity=true)
        i=fit.integration;p,K=size(fit.Λ);n=size(Y,2)
        theta=copy(fit.theta_packed);nll=t->q.objective(t,i.caches)
        draw=rng->_poisson_aghq_simulate(fit,n;rng=rng)
        refit=function(Yb)
            fb=try
                fit_poisson_gllvm(Yb;_poisson_aghq_refit_kwargs(fit)...)
            catch e
                e isa InterruptException && rethrow()
                return nothing
            end
            return _is_poisson_aghq(fb) && fb.converged ? copy(fb.theta_packed) : nothing
        end
        return _FamilyCI(theta,nll,_glm_lin_names(p,K),fill(:linear,length(theta)),draw,refit)
    end
    objective===:fit && (objective=:laplace)
    fit.alpha_lv === nothing || throw(ArgumentError(
        "confint for fit_poisson_gllvm(...; X_lv=...) is not carried by confint(fit, Y); " *
        "use confint_lv_effects(fit, Y, X_lv) for Wald intervals on B_lv, " *
        "or extract_lv_effects(fit) for point estimates"))
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        v = try
            objective === :va ?
                -poisson_marginal_loglik_va(Y, Λ, β; maxiter = newton_maxiter, tol = newton_tol) :
                -poisson_marginal_loglik_laplace(Y, Λ, β, link; mask = M, hessian = fit.hessian,
                                                 maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = rng -> _glm_simulate_counts(rng, fit.β, fit.Λ, link, n,
                                           (r, μ) -> Poisson(max(μ, 1e-12)))
    refit = function (Yb)
        fb = try fit_poisson_gllvm(Yb; K = K, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ))
    end
    return _FamilyCI(θ, nll, _glm_lin_names(p, K), fill(:linear, length(θ)), simulate, refit)
end

# --- Binomial --------------------------------------------------------------
function _family_ci(fit::BinomialFit, Y::AbstractMatrix;
                    N::Union{Nothing, AbstractMatrix} = nothing,
                    mask = nothing,
                    objective::Symbol = :fit,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    if _is_binomial_aghq(fit)
        objective in (:fit,:aghq) || throw(ArgumentError("AGHQ inference must use objective=:fit; Laplace/VA would change the estimator"))
        q,_=_binomial_aghq_problem(fit,Y;N=N,mask=mask,require_identity=true)
        i=fit.integration;p,K=size(fit.Λ);n=size(Y,2)
        theta=copy(fit.theta_packed);nll=t->q.objective(t,i.caches)
        draw=rng->_binomial_aghq_simulate(fit,n;rng=rng)
        refit=function(Yb)
            fb=try
                fit_binomial_gllvm(Yb;_binomial_aghq_refit_kwargs(fit)...)
            catch e
                e isa InterruptException && rethrow()
                return nothing
            end
            return _is_binomial_aghq(fb) && fb.converged ? copy(fb.theta_packed) : nothing
        end
        return _FamilyCI(theta,nll,_glm_lin_names(p,K),fill(:linear,length(theta)),draw,refit)
    end
    objective===:fit && (objective=:laplace)
    fit.alpha_lv === nothing || throw(ArgumentError(
        "confint for fit_binomial_gllvm(...; X_lv=...) is not carried by confint(fit, Y); " *
        "use confint_lv_effects(fit, Y, X_lv) for Wald intervals on B_lv, " *
        "or extract_lv_effects(fit) for point estimates"))
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    Nm = N === nothing ? fill(1, p, n) : Matrix{Int}(N)
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        v = try
            objective === :va ?
                -binomial_marginal_loglik_va(Y, Nm, Λ, β; maxiter = newton_maxiter, tol = newton_tol) :
                -binomial_marginal_loglik_laplace(Y, Nm, Λ, β, link; mask = M, hessian = fit.hessian,
                                                  maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = clamp(linkinv(link, _clamp_eta(η[t])), 1e-12, 1 - 1e-12)
                Yb[t, s] = rand(rng, Binomial(Nm[t, s], μ))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_binomial_gllvm(Yb; K = K, link = link, N = Nm, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ))
    end
    return _FamilyCI(θ, nll, _glm_lin_names(p, K), fill(:linear, length(θ)), simulate, refit)
end

# --- Negative binomial -----------------------------------------------------
function _family_ci(fit::NBFit, Y::AbstractMatrix;
                    mask = nothing,
                    objective::Symbol = :laplace,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    fit.alpha_lv === nothing || throw(ArgumentError(
        "confint for fit_nb_gllvm(...; X_lv=...) is not carried by confint(fit, Y); " *
        "use confint_lv_effects(fit, Y, X_lv) for Wald intervals on B_lv, " *
        "or extract_lv_effects(fit) for point estimates"))
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log(fit.r))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K); r = exp(θv[p + rr + 1])
        v = try
            objective === :va ?
                -nb_marginal_loglik_va(Y, Λ, β, r; maxiter = newton_maxiter, tol = newton_tol) :
                -nb_marginal_loglik_laplace(Y, Λ, β, r; link = link, mask = M, hessian = fit.hessian,
                                            maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = rng -> _glm_simulate_counts(rng, fit.β, fit.Λ, link, n,
                                           (rg, μ) -> (m = max(μ, 1e-12); NegativeBinomial(fit.r, fit.r / (fit.r + m))))
    refit = function (Yb)
        fb = try fit_nb_gllvm(Yb; K = K, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log(fb.r))
    end
    names = vcat(_glm_lin_names(p, K), "r")
    kinds = vcat(fill(:linear, length(θ) - 1), :log)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# --- Negative binomial type-1 (NB1, linear variance Var = μ(1+φ)) -----------
function _family_ci(fit::NB1Fit, Y::AbstractMatrix;
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log(fit.φ))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K); φ = exp(θv[p + rr + 1])
        v = try
            -nb1_marginal_loglik_laplace(Y, Λ, β, φ; link = link, mask = M, hessian = fit.hessian,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = rng -> _glm_simulate_counts(rng, fit.β, fit.Λ, link, n,
                                           (rg, μ) -> (m = max(μ, 1e-12); NegativeBinomial(m / fit.φ, 1 / (1 + fit.φ))))
    refit = function (Yb)
        fb = try fit_nb1_gllvm(Yb; K = K, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log(fb.φ))
    end
    names = vcat(_glm_lin_names(p, K), "phi")
    kinds = vcat(fill(:linear, length(θ) - 1), :log)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# --- Generalized Poisson type-1 (GP-1, Var = μ(1+α μ)², signed dispersion α) ----
function _family_ci(fit::GP1Fit, Y::AbstractMatrix;
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), fit.α)            # α packed RAW (signed, not log)
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K); α = θv[p + rr + 1]
        v = try
            -gp1_marginal_loglik_laplace(Y, Λ, β, α; link = link, mask = M, hessian = fit.hessian,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        fam = GeneralizedPoisson1(fit.α)
        Yb = Matrix{Int}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                Yb[t, s] = _rand_gp1(rng, fam, linkinv(link, _clamp_eta(η[t])))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_gp1_gllvm(Yb; K = K, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), fb.α)
    end
    names = vcat(_glm_lin_names(p, K), "alpha")
    kinds = fill(:linear, length(θ))                      # α is raw/linear, not log
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# --- Beta ------------------------------------------------------------------
function _family_ci(fit::BetaFit, Y::AbstractMatrix;
                    mask = nothing,
                    objective::Symbol = :laplace,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    fit.alpha_lv === nothing || throw(ArgumentError(
        "confint for fit_beta_gllvm(...; X_lv=...) is not carried by confint(fit, Y); " *
        "use confint_lv_effects(fit, Y, X_lv) for Wald intervals on B_lv, " *
        "or extract_lv_effects(fit) for point estimates"))
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log(fit.φ))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K); φ = exp(θv[p + rr + 1])
        v = try
            objective === :va ?
                -beta_marginal_loglik_va(Y, Λ, β, φ; maxiter = newton_maxiter, tol = newton_tol) :
                -beta_marginal_loglik_laplace(Y, Λ, β, φ; link = link, mask = M, hessian = fit.hessian,
                                              maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n); φ = fit.φ
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = clamp(linkinv(link, _clamp_eta(η[t])), 1e-6, 1 - 1e-6)
                Yb[t, s] = clamp(rand(rng, Beta(μ * φ, (1 - μ) * φ)), 1e-6, 1 - 1e-6)
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_beta_gllvm(Yb; K = K, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log(fb.φ))
    end
    names = vcat(_glm_lin_names(p, K), "phi")
    kinds = vcat(fill(:linear, length(θ) - 1), :log)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# --- Gamma -----------------------------------------------------------------
function _family_ci(fit::GammaFit, Y::AbstractMatrix;
                    mask = nothing,
                    objective::Symbol = :laplace,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    fit.alpha_lv === nothing || throw(ArgumentError(
        "confint for fit_gamma_gllvm(...; X_lv=...) is not carried by confint(fit, Y); " *
        "use confint_lv_effects(fit, Y, X_lv) for Wald intervals on B_lv, " *
        "or extract_lv_effects(fit) for point estimates"))
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log(fit.α))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K); α = exp(θv[p + rr + 1])
        v = try
            objective === :va ?
                -gamma_marginal_loglik_va(Y, Λ, β, α; maxiter = newton_maxiter, tol = newton_tol) :
                -gamma_marginal_loglik_laplace(Y, Λ, β, α; link = link, mask = M, hessian = fit.hessian,
                                               maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n); α = fit.α
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                Yb[t, s] = rand(rng, Gamma(α, μ / α))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_gamma_gllvm(Yb; K = K, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log(fb.α))
    end
    names = vcat(_glm_lin_names(p, K), "alpha")
    kinds = vcat(fill(:linear, length(θ) - 1), :log)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# One-part lognormal: packing [β; pack(Λ); log σ]; closed-form y-scale NLL
# (`lognormal_marginal_loglik`). No public docstring — keeps Documenter
# `:missing_docs` clean (adapters are internal).
function _family_ci(fit::LognormalFit, Y::AbstractMatrix; kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log(fit.σ))
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        σ = exp(θv[p + rr + 1])
        v = try
            -lognormal_marginal_loglik(Y, Λ, β, σ)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        σ = fit.σ
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                Yb[t, s] = exp(η[t] + σ * randn(rng))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_lognormal_gllvm(Yb; K = K) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log(fb.σ))
    end
    names = vcat(_glm_lin_names(p, K), "sigma")
    kinds = vcat(fill(:linear, length(θ) - 1), :log)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# Student-t fixed-ν only (contract §6 free-ν holdout). Packing matches
# fit_studentt_gllvm with `nu` pinned: [β; pack(Λ); log σ…] — ν is a plug-in.
# Free / estimated ν stays OUT (Wald SE pathology at the ν→∞ boundary).
function _family_ci(fit::StudentTFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    fit.estimated_nu && throw(ArgumentError(
        "StudentTFit Wald CI: estimated ν (free degrees of freedom) is held out of " *
        "_CIFit until the ν-boundary second-order ruling clears; refit with a finite " *
        "`nu = …` pin (e.g. fit_studentt_gllvm(Y; nu = 4.0))"))
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    link = fit.link; hess = fit.hessian; ν = fit.ν
    shared = fit.disp_group === :shared
    ndisp = shared ? 1 : p
    logσ = shared ? [log(fit.σ)] : log.(fit.σ)
    θ = vcat(fit.β, pack_lambda(fit.Λ), logσ)
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        σ = shared ? exp(θv[p + rr + 1]) : exp.(θv[(p + rr + 1):(p + rr + ndisp)])
        v = try
            -studentt_marginal_loglik_laplace(Y, Λ, β, σ; ν = ν, link = link,
                                              hessian = hess,
                                              maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = linkinv(link, _clamp_eta(η[t]))
                σ_t = shared ? fit.σ : fit.σ[t]
                ν_t = ν isa Real ? ν : ν[t]
                Yb[t, s] = μ + σ_t * rand(rng, TDist(ν_t))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_studentt_gllvm(Yb; K = K, nu = ν, link = link, hessian = hess,
                               disp_group = fit.disp_group)
        catch
            return nothing
        end
        logσb = shared ? [log(fb.σ)] : log.(fb.σ)
        return vcat(fb.β, pack_lambda(fb.Λ), logσb)
    end
    σ_names = shared ? ["sigma"] : ["sigma[$t]" for t in 1:p]
    names = vcat(_glm_lin_names(p, K), σ_names)
    kinds = vcat(fill(:linear, p + rr), fill(:log, ndisp))
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# Zero-truncated Poisson: packing [β; pack(Λ)]; Laplace NLL via
# truncated_poisson_marginal_loglik_laplace. Bridge ci_method guard stays
# until after foreign #357.
function _family_ci(fit::TruncatedPoissonFit, Y::AbstractMatrix;
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    M = _ci_mask(mask, Y)
    Yi = round.(Int, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        v = try
            -truncated_poisson_marginal_loglik_laplace(Yi, Λ, β, link;
                mask = M, maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                # Reject-sample zeros (support y ≥ 1).
                y = 0
                while y < 1
                    y = rand(rng, Poisson(μ))
                end
                Yb[t, s] = y
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_truncated_poisson_gllvm(Yb; K = K, link = link, mask = M) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ))
    end
    return _FamilyCI(θ, nll, _glm_lin_names(p, K), fill(:linear, length(θ)), simulate, refit)
end

# Zero-truncated NB2 (shared r): packing [β; pack(Λ); log r]. Fit object does
# not store hessian — NLL uses :observed (family default / twin TMB).
function _family_ci(fit::TruncatedNegBin2Fit, Y::AbstractMatrix;
                    mask = nothing,
                    hessian::Symbol = :observed,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    M = _ci_mask(mask, Y)
    Yi = round.(Int, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log(fit.r))
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        r = exp(θv[p + rr + 1])
        v = try
            -truncated_nbinom2_marginal_loglik_laplace(Yi, Λ, β, r;
                link = link, mask = M, hessian = hessian,
                maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                Yb[t, s] = _rand_ztnb(rng, fit.r, μ)
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_truncated_nbinom2_gllvm(Yb; K = K, link = link, mask = M, hessian = hessian)
        catch
            return nothing
        end
        return vcat(fb.β, pack_lambda(fb.Λ), log(fb.r))
    end
    names = vcat(_glm_lin_names(p, K), "r")
    kinds = vcat(fill(:linear, length(θ) - 1), :log)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# --- Grouped / per-trait dispersion bridge families -----------------------
_grouped_dispersion_names(p::Integer, K::Integer, parameter::AbstractString, G::Integer) =
    vcat(_glm_lin_names(p, K), ["$(parameter)[$g]" for g in 1:G])

function _family_ci(fit::NBGroupedFit, Y::AbstractMatrix;
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.r_group)
    Yi = round.(Int, Y)
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log.(fit.r_group))
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        rg = exp.(θv[(p + rr + 1):(p + rr + G)])
        rvec = [rg[group[t]] for t in 1:p]
        v = try
            -nb_grouped_marginal_loglik_laplace(Yi, Λ, β, rvec; link = link,
                                                mask = M, hessian = fit.hessian,
                                                maxiter = newton_maxiter,
                                                tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                r = fit.r_group[group[t]]
                Yb[t, s] = rand(rng, NegativeBinomial(r, r / (r + μ)))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_nb_gllvm_grouped(Yb; K = K, group = group, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log.(fb.r_group))
    end
    names = _grouped_dispersion_names(p, K, "r", G)
    kinds = vcat(fill(:linear, p + rr), fill(:log, G))
    boundary = vcat(falses(p + rr), fit.dispersion_boundary)   # T14 F1: trailing G entries are r[1..G]
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

function _family_ci(fit::NB1GroupedFit, Y::AbstractMatrix;
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.φ)
    Yi = round.(Int, Y)
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log.(fit.φ))
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        φg = exp.(θv[(p + rr + 1):(p + rr + G)])
        φvec = [φg[group[t]] for t in 1:p]
        v = try
            -nb1_grouped_marginal_loglik_laplace(Yi, Λ, β, φvec; link = link,
                                                 mask = M, hessian = fit.hessian,
                                                 maxiter = newton_maxiter,
                                                 tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                φ = fit.φ[group[t]]
                Yb[t, s] = rand(rng, NegativeBinomial(μ / φ, 1 / (1 + φ)))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_nb1_gllvm_grouped(Yb; K = K, group = group, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log.(fb.φ))
    end
    names = _grouped_dispersion_names(p, K, "phi", G)
    kinds = vcat(fill(:linear, p + rr), fill(:log, G))
    boundary = vcat(falses(p + rr), fit.dispersion_boundary)   # T14 F1: trailing G entries are phi[1..G]
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

function _family_ci(fit::BetaGroupedFit, Y::AbstractMatrix;
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.φ)
    Yf = clamp.(Float64.(Y), 1e-6, 1 - 1e-6)
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log.(fit.φ))
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        φg = exp.(θv[(p + rr + 1):(p + rr + G)])
        φvec = [φg[group[t]] for t in 1:p]
        v = try
            -beta_grouped_marginal_loglik_laplace(Yf, Λ, β, φvec; link = link,
                                                  mask = M, hessian = fit.hessian,
                                                  maxiter = newton_maxiter,
                                                  tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = clamp(linkinv(link, _clamp_eta(η[t])), 1e-6, 1 - 1e-6)
                φ = fit.φ[group[t]]
                Yb[t, s] = clamp(rand(rng, Beta(μ * φ, (1 - μ) * φ)), 1e-6, 1 - 1e-6)
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_beta_gllvm_grouped(Yb; K = K, group = group, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log.(fb.φ))
    end
    names = _grouped_dispersion_names(p, K, "phi", G)
    kinds = vcat(fill(:linear, p + rr), fill(:log, G))
    boundary = vcat(falses(p + rr), fit.dispersion_boundary)   # T14 F1: trailing G entries are phi[1..G]
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

# --- Grouped dispersion + shared site-X (NB2 / Beta API B under X) ---------
function _family_ci(fit::NBGroupedCovFit, Y::AbstractMatrix;
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    X === nothing && throw(ArgumentError(
        "confint on an NBGroupedCovFit needs the design `X`: confint(fit, Y; method=…, X=X)"))
    p, K = size(fit.Λ); n = size(Y, 2); q_full = length(fit.γ); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.r_group)
    Xfit, γ_free_idx = _slice_fixed_X(X, fit.γ_fixed)
    q = length(γ_free_idx)
    Yi = round.(Int, Y)
    M = _ci_mask(mask, Y)
    γ_free = fit.γ[γ_free_idx]
    θ = vcat(fit.β, γ_free, pack_lambda(fit.Λ), log.(fit.r_group))
    nll = function (θv)
        β = θv[1:p]; γ = θv[(p + 1):(p + q)]
        Λ = unpack_lambda(θv[(p + q + 1):(p + q + rr)], p, K)
        rg = exp.(θv[(p + q + rr + 1):(p + q + rr + G)])
        rvec = [rg[group[t]] for t in 1:p]
        O = _build_offset(Xfit, γ)
        v = try
            -nb_grouped_marginal_loglik_laplace(Yi, Λ, β, rvec; link = link,
                                                mask = M, offset = O, hessian = fit.hessian,
                                                maxiter = newton_maxiter,
                                                tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        O = _build_offset(X, fit.γ)
        @inbounds for s in 1:n
            η = fit.β .+ view(O, :, s) .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                r = fit.r_group[group[t]]
                Yb[t, s] = rand(rng, NegativeBinomial(r, r / (r + μ)))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_nb_gllvm_grouped_cov(Yb; X = X, K = K, group = group, link = link,
                                     mask = M, γ_fixed = fit.γ_fixed, hessian = fit.hessian)
        catch
            return nothing
        end
        return vcat(fb.β, fb.γ[γ_free_idx], pack_lambda(fb.Λ), log.(fb.r_group))
    end
    names = vcat(["beta[$t]" for t in 1:p], ["gamma[$k]" for k in γ_free_idx],
                 _confint_lambda_term_names("Lambda", p, K),
                 ["r[$g]" for g in 1:G])
    kinds = vcat(fill(:linear, p + q + rr), fill(:log, G))
    boundary = vcat(falses(p + q + rr), fit.dispersion_boundary)   # T14 F1
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

function _family_ci(fit::NB1GroupedCovFit, Y::AbstractMatrix;
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    X === nothing && throw(ArgumentError(
        "confint on an NB1GroupedCovFit needs the design `X`: confint(fit, Y; method=…, X=X)"))
    p, K = size(fit.Λ); n = size(Y, 2); q_full = length(fit.γ); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.φ)
    Xfit, γ_free_idx = _slice_fixed_X(X, fit.γ_fixed)
    q = length(γ_free_idx)
    Yi = round.(Int, Y)
    M = _ci_mask(mask, Y)
    γ_free = fit.γ[γ_free_idx]
    θ = vcat(fit.β, γ_free, pack_lambda(fit.Λ), log.(fit.φ))
    nll = function (θv)
        β = θv[1:p]; γ = θv[(p + 1):(p + q)]
        Λ = unpack_lambda(θv[(p + q + 1):(p + q + rr)], p, K)
        φg = exp.(θv[(p + q + rr + 1):(p + q + rr + G)])
        φvec = [φg[group[t]] for t in 1:p]
        O = _build_offset(Xfit, γ)
        v = try
            -nb1_grouped_marginal_loglik_laplace(Yi, Λ, β, φvec; link = link,
                                                 mask = M, offset = O, hessian = fit.hessian,
                                                 maxiter = newton_maxiter,
                                                 tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        O = _build_offset(X, fit.γ)
        @inbounds for s in 1:n
            η = fit.β .+ view(O, :, s) .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                φ = fit.φ[group[t]]
                Yb[t, s] = rand(rng, NegativeBinomial(μ / φ, 1 / (1 + φ)))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_nb1_gllvm_grouped_cov(Yb; X = X, K = K, group = group, link = link,
                                      mask = M, γ_fixed = fit.γ_fixed, hessian = fit.hessian)
        catch
            return nothing
        end
        return vcat(fb.β, fb.γ[γ_free_idx], pack_lambda(fb.Λ), log.(fb.φ))
    end
    names = vcat(["beta[$t]" for t in 1:p], ["gamma[$k]" for k in γ_free_idx],
                 _confint_lambda_term_names("Lambda", p, K),
                 ["phi[$g]" for g in 1:G])
    kinds = vcat(fill(:linear, p + q + rr), fill(:log, G))
    boundary = vcat(falses(p + q + rr), fit.dispersion_boundary)   # T14 F1
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

function _family_ci(fit::BetaGroupedCovFit, Y::AbstractMatrix;
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    X === nothing && throw(ArgumentError(
        "confint on a BetaGroupedCovFit needs the design `X`: confint(fit, Y; method=…, X=X)"))
    p, K = size(fit.Λ); n = size(Y, 2); q_full = length(fit.γ); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.φ)
    Xfit, γ_free_idx = _slice_fixed_X(X, fit.γ_fixed)
    q = length(γ_free_idx)
    Yf = clamp.(Float64.(Y), 1e-6, 1 - 1e-6)
    M = _ci_mask(mask, Y)
    γ_free = fit.γ[γ_free_idx]
    θ = vcat(fit.β, γ_free, pack_lambda(fit.Λ), log.(fit.φ))
    nll = function (θv)
        β = θv[1:p]; γ = θv[(p + 1):(p + q)]
        Λ = unpack_lambda(θv[(p + q + 1):(p + q + rr)], p, K)
        φg = exp.(θv[(p + q + rr + 1):(p + q + rr + G)])
        φvec = [φg[group[t]] for t in 1:p]
        O = _build_offset(Xfit, γ)
        v = try
            -beta_grouped_marginal_loglik_laplace(Yf, Λ, β, φvec; link = link,
                                                  mask = M, offset = O, hessian = fit.hessian,
                                                  maxiter = newton_maxiter,
                                                  tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        O = _build_offset(X, fit.γ)
        @inbounds for s in 1:n
            η = fit.β .+ view(O, :, s) .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = clamp(linkinv(link, _clamp_eta(η[t])), 1e-6, 1 - 1e-6)
                φ = fit.φ[group[t]]
                Yb[t, s] = clamp(rand(rng, Beta(μ * φ, (1 - μ) * φ)), 1e-6, 1 - 1e-6)
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_beta_gllvm_grouped_cov(Yb; X = X, K = K, group = group, link = link,
                                       mask = M, γ_fixed = fit.γ_fixed, hessian = fit.hessian)
        catch
            return nothing
        end
        return vcat(fb.β, fb.γ[γ_free_idx], pack_lambda(fb.Λ), log.(fb.φ))
    end
    names = vcat(["beta[$t]" for t in 1:p], ["gamma[$k]" for k in γ_free_idx],
                 _confint_lambda_term_names("Lambda", p, K),
                 ["phi[$g]" for g in 1:G])
    kinds = vcat(fill(:linear, p + q + rr), fill(:log, G))
    boundary = vcat(falses(p + q + rr), fit.dispersion_boundary)   # T14 F1
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

function _family_ci(fit::BetaBinomialGroupedFit, Y::AbstractMatrix;
                    N::Union{Nothing, AbstractMatrix} = nothing,
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.φ)
    Yi = round.(Int, Y)
    Nm = N === nothing ? fill(1, p, n) : Matrix{Int}(N)
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log.(fit.φ))
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        φg = exp.(θv[(p + rr + 1):(p + rr + G)])
        φvec = [φg[group[t]] for t in 1:p]
        v = try
            -betabinomial_grouped_marginal_loglik_laplace(Yi, Nm, Λ, β, φvec; link = link,
                                                          mask = M,
                                                          maxiter = newton_maxiter,
                                                          tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = clamp(linkinv(link, _clamp_eta(η[t])), 1e-12, 1 - 1e-12)
                φ = fit.φ[group[t]]
                pdraw = clamp(rand(rng, Beta(μ * φ, (1 - μ) * φ)), 1e-12, 1 - 1e-12)
                Yb[t, s] = rand(rng, Binomial(Nm[t, s], pdraw))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_beta_binomial_gllvm_grouped(Yb; K = K, N = Nm, group = group,
                                            link = link, mask = M)
        catch
            return nothing
        end
        return vcat(fb.β, pack_lambda(fb.Λ), log.(fb.φ))
    end
    names = _grouped_dispersion_names(p, K, "phi", G)
    kinds = vcat(fill(:linear, p + rr), fill(:log, G))
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

function _family_ci(fit::BetaBinomialGroupedCovFit, Y::AbstractMatrix;
                    N::Union{Nothing, AbstractMatrix} = nothing,
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    X === nothing && throw(ArgumentError(
        "confint on a BetaBinomialGroupedCovFit needs the design `X`: confint(fit, Y; method=…, X=X)"))
    p, K = size(fit.Λ); n = size(Y, 2); q_full = length(fit.γ); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.φ)
    Xfit, γ_free_idx = _slice_fixed_X(X, fit.γ_fixed)
    q = length(γ_free_idx)
    Yi = round.(Int, Y)
    Nm = N === nothing ? fill(1, p, n) : Matrix{Int}(N)
    M = _ci_mask(mask, Y)
    γ_free = fit.γ[γ_free_idx]
    θ = vcat(fit.β, γ_free, pack_lambda(fit.Λ), log.(fit.φ))
    nll = function (θv)
        β = θv[1:p]; γ = θv[(p + 1):(p + q)]
        Λ = unpack_lambda(θv[(p + q + 1):(p + q + rr)], p, K)
        φg = exp.(θv[(p + q + rr + 1):(p + q + rr + G)])
        φvec = [φg[group[t]] for t in 1:p]
        O = _build_offset(Xfit, γ)
        v = try
            -betabinomial_grouped_marginal_loglik_laplace(Yi, Nm, Λ, β, φvec; link = link,
                                                          mask = M, offset = O,
                                                          maxiter = newton_maxiter,
                                                          tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        O = _build_offset(X, fit.γ)
        @inbounds for s in 1:n
            η = fit.β .+ view(O, :, s) .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = clamp(linkinv(link, _clamp_eta(η[t])), 1e-12, 1 - 1e-12)
                φ = fit.φ[group[t]]
                pdraw = clamp(rand(rng, Beta(μ * φ, (1 - μ) * φ)), 1e-12, 1 - 1e-12)
                Yb[t, s] = rand(rng, Binomial(Nm[t, s], pdraw))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_beta_binomial_gllvm_grouped_cov(Yb; X = X, K = K, N = Nm, group = group,
                                                link = link, mask = M, γ_fixed = fit.γ_fixed)
        catch
            return nothing
        end
        return vcat(fb.β, fb.γ[γ_free_idx], pack_lambda(fb.Λ), log.(fb.φ))
    end
    names = vcat(["beta[$t]" for t in 1:p], ["gamma[$k]" for k in γ_free_idx],
                 _confint_lambda_term_names("Lambda", p, K),
                 ["phi[$g]" for g in 1:G])
    kinds = vcat(fill(:linear, p + q + rr), fill(:log, G))
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

function _family_ci(fit::GammaGroupedFit, Y::AbstractMatrix;
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.α)
    Yf = max.(Float64.(Y), 1e-9)
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log.(fit.α))
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        αg = exp.(θv[(p + rr + 1):(p + rr + G)])
        αvec = [αg[group[t]] for t in 1:p]
        v = try
            -gamma_grouped_marginal_loglik_laplace(Yf, Λ, β, αvec; link = link,
                                                   mask = M, hessian = fit.hessian,
                                                   maxiter = newton_maxiter,
                                                   tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                α = fit.α[group[t]]
                Yb[t, s] = rand(rng, Gamma(α, μ / α))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_gamma_gllvm_grouped(Yb; K = K, group = group, link = link, mask = M, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log.(fb.α))
    end
    names = _grouped_dispersion_names(p, K, "alpha", G)
    kinds = vcat(fill(:linear, p + rr), fill(:log, G))
    boundary = vcat(falses(p + rr), fit.dispersion_boundary)   # T14 F1
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

function _family_ci(fit::GammaGroupedCovFit, Y::AbstractMatrix;
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    X === nothing && throw(ArgumentError(
        "confint on a GammaGroupedCovFit needs the design `X`: confint(fit, Y; method=…, X=X)"))
    p, K = size(fit.Λ); n = size(Y, 2); q_full = length(fit.γ); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.α)
    Xfit, γ_free_idx = _slice_fixed_X(X, fit.γ_fixed)
    q = length(γ_free_idx)
    Yf = max.(Float64.(Y), 1e-9)
    M = _ci_mask(mask, Y)
    γ_free = fit.γ[γ_free_idx]
    θ = vcat(fit.β, γ_free, pack_lambda(fit.Λ), log.(fit.α))
    nll = function (θv)
        β = θv[1:p]; γ = θv[(p + 1):(p + q)]
        Λ = unpack_lambda(θv[(p + q + 1):(p + q + rr)], p, K)
        αg = exp.(θv[(p + q + rr + 1):(p + q + rr + G)])
        αvec = [αg[group[t]] for t in 1:p]
        O = _build_offset(Xfit, γ)
        v = try
            -gamma_grouped_marginal_loglik_laplace(Yf, Λ, β, αvec; link = link,
                                                   mask = M, offset = O, hessian = fit.hessian,
                                                   maxiter = newton_maxiter,
                                                   tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        O = _build_offset(X, fit.γ)
        @inbounds for s in 1:n
            η = fit.β .+ view(O, :, s) .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                α = fit.α[group[t]]
                Yb[t, s] = rand(rng, Gamma(α, μ / α))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_gamma_gllvm_grouped_cov(Yb; X = X, K = K, group = group, link = link,
                                        mask = M, γ_fixed = fit.γ_fixed, hessian = fit.hessian)
        catch
            return nothing
        end
        return vcat(fb.β, fb.γ[γ_free_idx], pack_lambda(fb.Λ), log.(fb.α))
    end
    names = vcat(["beta[$t]" for t in 1:p], ["gamma[$k]" for k in γ_free_idx],
                 _confint_lambda_term_names("Lambda", p, K),
                 ["alpha[$g]" for g in 1:G])
    kinds = vcat(fill(:linear, p + q + rr), fill(:log, G))
    boundary = vcat(falses(p + q + rr), fit.dispersion_boundary)   # T14 F1
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

# Compound Poisson–Gamma draw from a Tweedie (1 < p < 2) with mean μ, dispersion
# φ: N ~ Poisson(λ), λ = μ^{2-p}/(φ(2-p)); Y = Σᵢ Gammaᵢ(shape=(2-p)/(p-1),
# scale=φ(p-1)μ^{p-1}); N = 0 ⇒ exact zero (Dunn & Smyth 2005). No dedicated
# Tweedie sampler ships with the family pieces, so the bootstrap draws here.
function _rand_tweedie(rng::AbstractRNG, μ, φ, p)
    μ = max(float(μ), 1e-12)
    λ = μ^(2.0 - p) / (φ * (2.0 - p))
    N = rand(rng, Poisson(λ))
    N == 0 && return 0.0
    shape = (2.0 - p) / (p - 1.0)
    scale = φ * (p - 1.0) * μ^(p - 1.0)
    s = 0.0
    @inbounds for _ in 1:N
        s += rand(rng, Gamma(shape, scale))
    end
    return s
end

# --- Tweedie (compound Poisson–Gamma, power p ∈ (1,2) fixed at the fit) ------
# The fitter jointly estimates φ and the power p, but the CI layer profiles only
# the dispersion φ here (one log-scale term, like NB's r / Gamma's α); the power
# is held fixed at its fitted value `fit.p` throughout the Hessian / profile /
# bootstrap, so the working vector is [β; pack_lambda(Λ); log φ].
function _family_ci(fit::TweedieFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    pw = fit.p
    θ = vcat(fit.β, pack_lambda(fit.Λ), log(fit.φ))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K); φ = exp(θv[p + rr + 1])
        v = try
            -tweedie_marginal_loglik_laplace(Y, Λ, β, φ, pw; link = link, hessian = fit.hessian,
                                             maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n); φ = fit.φ
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                Yb[t, s] = _rand_tweedie(rng, μ, φ, pw)
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_tweedie_gllvm(Yb; K = K, link = link, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log(fb.φ))
    end
    names = vcat(_glm_lin_names(p, K), "phi")
    kinds = vcat(fill(:linear, length(θ) - 1), :log)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# --- Tweedie grouped dispersion (shared power held at fit; φ per group) -----
# Mirrors `TweedieFit`: the CI layer profiles only the group φ's on the log
# scale; `fit.power` is fixed for Hessian / profile / bootstrap. Estimated
# shared power on the fitter is therefore treated as a plug-in (same contract
# as shared-φ Tweedie). `TweediePerTraitPowerFit` stays out of `_CIFit`.
function _family_ci(fit::TweedieGroupedFit, Y::AbstractMatrix;
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.φ)
    pw = fit.power
    Yf = max.(Float64.(Y), 0.0)   # Tweedie allows exact zeros
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log.(fit.φ))
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        φg = exp.(θv[(p + rr + 1):(p + rr + G)])
        φvec = [φg[group[t]] for t in 1:p]
        v = try
            -tweedie_grouped_marginal_loglik_laplace(Yf, Λ, β, φvec, pw; link = link,
                                                     mask = M, hessian = fit.hessian,
                                                     maxiter = newton_maxiter,
                                                     tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                Yb[t, s] = _rand_tweedie(rng, μ, fit.φ[group[t]], pw)
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            if fit.power_fixed
                fit_tweedie_gllvm_grouped(Yb; K = K, group = group, power = pw,
                                          link = link, mask = M, hessian = fit.hessian)
            else
                fit_tweedie_gllvm_grouped(Yb; K = K, group = group, power_group = :shared,
                                          link = link, mask = M, hessian = fit.hessian)
            end
        catch
            return nothing
        end
        fb isa TweedieGroupedFit || return nothing
        return vcat(fb.β, pack_lambda(fb.Λ), log.(fb.φ))
    end
    names = _grouped_dispersion_names(p, K, "phi", G)
    kinds = vcat(fill(:linear, p + rr), fill(:log, G))
    # No T14 `dispersion_boundary` field on TweedieGroupedFit yet — treat all φ free.
    boundary = falses(p + rr + G)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

# --- Tweedie per-trait estimated power (φ per group; power plug-in) --------
# Same contract as `TweedieGroupedFit`: profile only group φ on the log scale;
# each `fit.power[t]` is held fixed for Wald / profile / bootstrap.
function _family_ci(fit::TweediePerTraitPowerFit, Y::AbstractMatrix;
                    mask = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    link = fit.link; group = collect(Int, fit.group); G = length(fit.φ)
    pw = collect(fit.power)
    Yf = max.(Float64.(Y), 0.0)
    M = _ci_mask(mask, Y)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log.(fit.φ))
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        φg = exp.(θv[(p + rr + 1):(p + rr + G)])
        φvec = [φg[group[t]] for t in 1:p]
        v = try
            -tweedie_grouped_marginal_loglik_laplace(Yf, Λ, β, φvec, pw; link = link,
                                                     mask = M, hessian = fit.hessian,
                                                     maxiter = newton_maxiter,
                                                     tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = max(linkinv(link, _clamp_eta(η[t])), 1e-12)
                Yb[t, s] = _rand_tweedie(rng, μ, fit.φ[group[t]], pw[t])
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_tweedie_gllvm_grouped(Yb; K = K, group = group, power_group = :species,
                                      link = link, mask = M, hessian = fit.hessian)
        catch
            return nothing
        end
        fb isa TweediePerTraitPowerFit || return nothing
        return vcat(fb.β, pack_lambda(fb.Λ), log.(fb.φ))
    end
    names = _grouped_dispersion_names(p, K, "phi", G)
    kinds = vcat(fill(:linear, p + rr), fill(:log, G))
    boundary = falses(p + rr + G)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit, boundary)
end

# --- Exponential (positive continuous, no dispersion) ----------------------
function _family_ci(fit::ExponentialFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    θ = vcat(fit.β, pack_lambda(fit.Λ))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        v = try
            -exponential_marginal_loglik_laplace(Y, Λ, β; link = link, hessian = fit.hessian, maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                Yb[t, s] = rand(rng, Exponential(max(linkinv(link, _clamp_eta(η[t])), 1e-12)))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_exponential_gllvm(Yb; K = K, link = link, hessian = fit.hessian) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ))
    end
    return _FamilyCI(θ, nll, _glm_lin_names(p, K), fill(:linear, length(θ)), simulate, refit)
end

# --- Beta-binomial (overdispersed binomial; one Beta-precision φ) -----------
# Working vector [β; pack_lambda(Λ); log φ] — the NB/Beta scalar-dispersion
# template. Like the Binomial CIs, the trial counts N are NOT stored in the fit,
# so they are taken as data through the `N` kwarg (default all-ones / Bernoulli-
# overdispersed) and threaded through the marginal, the bootstrap draw, and the
# refit exactly as families/binomial.jl threads them.
function _family_ci(fit::BetaBinomialFit, Y::AbstractMatrix;
                    N::Union{Nothing, AbstractMatrix} = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); link = fit.link
    Nm = N === nothing ? fill(1, p, n) : Matrix{Int}(N)
    θ = vcat(fit.β, pack_lambda(fit.Λ), log(fit.φ))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K); φ = exp(θv[p + rr + 1])
        v = try
            -betabinomial_marginal_loglik_laplace(Y, Nm, Λ, β, φ; link = link,
                                                  maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Int}(undef, p, n); φ = fit.φ
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = clamp(linkinv(link, _clamp_eta(η[t])), 1e-12, 1 - 1e-12)
                # beta-binomial draw: success prob ~ Beta(μφ, (1−μ)φ), then Binomial(N, ·).
                pdraw = clamp(rand(rng, Beta(μ * φ, (1 - μ) * φ)), 1e-12, 1 - 1e-12)
                Yb[t, s] = rand(rng, Binomial(Nm[t, s], pdraw))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_beta_binomial_gllvm(Yb; K = K, link = link, N = Nm) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), log(fb.φ))
    end
    names = vcat(_glm_lin_names(p, K), "phi")
    kinds = vcat(fill(:linear, length(θ) - 1), :log)
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# --- Random row effect (per-site random intercept σ_row, + family dispersion) ---
# Working vector [β; pack_lambda(Λ); log σ_row; (log-dispersion)] — `σ_row` is the
# row-effect SD on the log scale (it stays ≥ 0), and the underlying family's
# dispersion (NB `r` / Beta `φ` / Gamma `α`) is the optional trailing log-scale
# term, exactly as the fitter lays it out. The marginal is the augmented
# (K+1)-latent reparameterisation `Λ̃ = [Λ | σ_row·𝟙_p]`; the bootstrap draws
# z_s ~ N(0,I_K) and ρ_s ~ N(0, σ_row²) then the per-family response.
function _family_ci(fit::RowRandomFit, Y::AbstractMatrix;
                    N::Union{Nothing, AbstractMatrix} = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    lk = fit.link; fam0 = fit.family; hasd = _cov_has_disp(fam0)
    Nm = N === nothing ? fill(1, p, n) : N
    θ = hasd ? vcat(fit.β, pack_lambda(fit.Λ), log(fit.σ_row), log(fit.dispersion)) :
               vcat(fit.β, pack_lambda(fit.Λ), log(fit.σ_row))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        σ_row = exp(θv[p + rr + 1])
        disp = hasd ? exp(θv[p + rr + 2]) : NaN
        fam = _cov_family(fam0, disp)
        v = try
            -row_random_marginal_loglik_laplace(fam, Y, Nm, Λ, β, σ_row; link = lk,
                                                maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    simulate = function (rng)
        Yb = Matrix{Float64}(undef, p, n); fam = _cov_family(fam0, fit.dispersion)
        @inbounds for s in 1:n
            ρ = fit.σ_row * randn(rng)                       # per-site random intercept
            η = fit.β .+ ρ .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = linkinv(lk, _clamp_eta(η[t]))
                Yb[t, s] = _cov_sample(fam, μ, Nm[t, s], rng)
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_row_random_gllvm(Yb; family = fam0, K = K, link = lk,
                                 N = fam0 isa Binomial ? Nm : nothing)
        catch
            return nothing
        end
        return hasd ? vcat(fb.β, pack_lambda(fb.Λ), log(fb.σ_row), log(fb.dispersion)) :
                      vcat(fb.β, pack_lambda(fb.Λ), log(fb.σ_row))
    end
    names = vcat(_glm_lin_names(p, K), "sigma_row")
    kinds = vcat(fill(:linear, p + rr), :log)
    if hasd
        push!(names, _cov_dispname(fam0)); push!(kinds, :log)
    end
    return _FamilyCI(θ, nll, names, kinds, simulate, refit)
end

# Count-family simulation (Poisson / NB share the loop; `make` builds the
# per-cell Distributions sampler from (rng, μ)).
function _glm_simulate_counts(rng::AbstractRNG, β::AbstractVector, Λ::AbstractMatrix,
                              link::Link, n::Integer, make)
    p, K = size(Λ)
    Yb = Matrix{Int}(undef, p, n)
    @inbounds for s in 1:n
        η = β .+ Λ * randn(rng, K)
        for t in 1:p
            μ = linkinv(link, _clamp_eta(η[t]))
            Yb[t, s] = rand(rng, make(rng, μ))
        end
    end
    return Yb
end

# ---------------------------------------------------------------------------
# Two-part family adapters. Shared layout [βz; βc; pack_lambda(Λc); (log-disp)];
# βz are occurrence/zero-inflation logits, βc the positive/count log-(or log-mean)
# intercepts. `make_nll` closes over the family's marginal; `sim` draws Yᵇ; `refit`
# returns the working vector or `nothing`.
# ---------------------------------------------------------------------------
_twopart_lin_names(p::Integer, K::Integer) =
    vcat(["betaz[$t]" for t in 1:p], ["betac[$t]" for t in 1:p],
         _confint_lambda_term_names("Lambda", p, K))

# gllvmTMB twin-identity mode (`predictor = :shared`): one η = β + Λz for both parts.
_twopart_shared_lin_names(p::Integer, K::Integer) =
    vcat(["beta[$t]" for t in 1:p], _confint_lambda_term_names("Lambda", p, K))

# Zero-truncated count samplers (rejection; rare fall-through returns 1).
function _rand_ztpois(rng::AbstractRNG, μ)
    d = Poisson(max(μ, 1e-12))
    for _ in 1:10_000
        y = rand(rng, d); y > 0 && return y
    end
    return 1
end
function _rand_ztnb(rng::AbstractRNG, r, μ)
    d = NegativeBinomial(r, r / (r + max(μ, 1e-12)))
    for _ in 1:10_000
        y = rand(rng, d); y > 0 && return y
    end
    return 1
end

# --- Delta-lognormal -------------------------------------------------------
# Shared-σ packing (#347) plus per-trait σ packing for Option A twin alignment
# (`disp_group=:species`). Public fitter default is `:species` since
# `accept delta dispersion A` (maintainer paste, 2026-09-24); `:shared` remains
# an explicit opt-in and both branches below stay exercised.
function _family_ci(fit::DeltaLogNormalFit, Y::AbstractMatrix;
                    hessian::Symbol = :observed,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K)
    shared = fit.disp_group === :shared
    fit.disp_group in (:shared, :species) || throw(ArgumentError(
        "DeltaLogNormalFit Wald CI: disp_group must be :shared or :species; got :$(fit.disp_group)"))
    ndisp = shared ? 1 : p
    if shared
        fit.σ isa Real || throw(ArgumentError(
            "DeltaLogNormalFit Wald CI: disp_group=:shared requires scalar σ"))
        logσ = [log(fit.σ)]
    else
        fit.σ isa AbstractVector || throw(ArgumentError(
            "DeltaLogNormalFit Wald CI: disp_group=:species requires length-p σ"))
        length(fit.σ) == p || throw(ArgumentError(
            "DeltaLogNormalFit Wald CI: length(σ)=$(length(fit.σ)) must equal p=$p"))
        logσ = log.(fit.σ)
    end
    σ_names = shared ? ["sigma"] : ["sigma[$t]" for t in 1:p]
    if fit.predictor === :shared
        β = fit.βc
        θ = vcat(β, pack_lambda(fit.Λc), logσ)
        nll = function (θv)
            βv = θv[1:p]
            Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
            σv = shared ? exp(θv[p + rr + 1]) : exp.(θv[(p + rr + 1):(p + rr + ndisp)])
            v = try
                -delta_lognormal_marginal_loglik_laplace(Y, Λ, βv, βv, σv; Λz = Λ,
                    hessian = hessian, maxiter = newton_maxiter, tol = newton_tol)
            catch
                return 1e12
            end
            return isfinite(v) ? v : 1e12
        end
        sim = function (rng)
            Yb = zeros(Float64, p, n)
            @inbounds for s in 1:n
                η = β .+ fit.Λc * randn(rng, K)
                for t in 1:p
                    π = inv(1 + exp(-η[t]))
                    σt = shared ? fit.σ : fit.σ[t]
                    rand(rng) < π && (Yb[t, s] = exp(η[t] + σt * randn(rng)))
                end
            end
            return Yb
        end
        refit = function (Yb)
            fb = try
                fit_delta_lognormal_gllvm(Yb; K = K, predictor = :shared,
                    disp_group = fit.disp_group)
            catch
                return nothing
            end
            logσb = shared ? (fb.σ isa Real ? [log(fb.σ)] : nothing) : (fb.σ isa AbstractVector ? log.(fb.σ) : nothing)
            logσb === nothing && return nothing
            return vcat(fb.βc, pack_lambda(fb.Λc), logσb)
        end
        names = vcat(_twopart_shared_lin_names(p, K), σ_names)
        return _FamilyCI(θ, nll, names, vcat(fill(:linear, p + rr), fill(:log, ndisp)), sim, refit)
    end
    θ = vcat(fit.βz, fit.βc, pack_lambda(fit.Λc), logσ)
    nll = function (θv)
        βz = θv[1:p]; βc = θv[(p + 1):(2p)]
        Λc = unpack_lambda(θv[(2p + 1):(2p + rr)], p, K)
        σv = shared ? exp(θv[2p + rr + 1]) : exp.(θv[(2p + rr + 1):(2p + rr + ndisp)])
        v = try
            -delta_lognormal_marginal_loglik_laplace(Y, Λc, βz, βc, σv; hessian = hessian,
                maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Float64, p, n)
        @inbounds for s in 1:n
            ηc = fit.βc .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-fit.βz[t]))
                σt = shared ? fit.σ : fit.σ[t]
                rand(rng) < π && (Yb[t, s] = exp(ηc[t] + σt * randn(rng)))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_delta_lognormal_gllvm(Yb; K = K, disp_group = fit.disp_group)
        catch
            return nothing
        end
        logσb = shared ? (fb.σ isa Real ? [log(fb.σ)] : nothing) : (fb.σ isa AbstractVector ? log.(fb.σ) : nothing)
        logσb === nothing && return nothing
        return vcat(fb.βz, fb.βc, pack_lambda(fb.Λc), logσb)
    end
    names = vcat(_twopart_lin_names(p, K), σ_names)
    return _FamilyCI(θ, nll, names, vcat(fill(:linear, 2p + rr), fill(:log, ndisp)), sim, refit)
end

# --- Delta-Gamma -----------------------------------------------------------
function _family_ci(fit::DeltaGammaFit, Y::AbstractMatrix;
                    objective::Symbol = :laplace,
                    hessian::Symbol = :observed,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K)
    shared = fit.disp_group === :shared
    fit.disp_group in (:shared, :species) || throw(ArgumentError(
        "DeltaGammaFit Wald CI: disp_group must be :shared or :species; got :$(fit.disp_group)"))
    ndisp = shared ? 1 : p
    if shared
        fit.α isa Real || throw(ArgumentError(
            "DeltaGammaFit Wald CI: disp_group=:shared requires scalar α"))
        logα = [log(fit.α)]
    else
        fit.α isa AbstractVector || throw(ArgumentError(
            "DeltaGammaFit Wald CI: disp_group=:species requires length-p α"))
        length(fit.α) == p || throw(ArgumentError(
            "DeltaGammaFit Wald CI: length(α)=$(length(fit.α)) must equal p=$p"))
        logα = log.(fit.α)
    end
    α_names = shared ? ["alpha"] : ["alpha[$t]" for t in 1:p]
    if fit.predictor === :shared
        β = fit.βc
        θ = vcat(β, pack_lambda(fit.Λc), logα)
        nll = function (θv)
            βv = θv[1:p]
            Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
            αv = shared ? exp(θv[p + rr + 1]) : exp.(θv[(p + rr + 1):(p + rr + ndisp)])
            v = try
                objective === :va ?
                    -delta_gamma_marginal_loglik_va(Y, Λ, βv, βv, αv; Λz = Λ,
                        maxiter = newton_maxiter, tol = newton_tol) :
                    -delta_gamma_marginal_loglik_laplace(Y, Λ, βv, βv, αv; Λz = Λ,
                        hessian = hessian, maxiter = newton_maxiter, tol = newton_tol)
            catch
                return 1e12
            end
            return isfinite(v) ? v : 1e12
        end
        sim = function (rng)
            Yb = zeros(Float64, p, n)
            @inbounds for s in 1:n
                η = β .+ fit.Λc * randn(rng, K)
                for t in 1:p
                    π = inv(1 + exp(-η[t]))
                    if rand(rng) < π
                        αt = shared ? fit.α : fit.α[t]
                        μ = exp(η[t]); Yb[t, s] = rand(rng, Gamma(αt, μ / αt))
                    end
                end
            end
            return Yb
        end
        refit = function (Yb)
            fb = try
                fit_delta_gamma_gllvm(Yb; K = K, predictor = :shared,
                    disp_group = fit.disp_group)
            catch
                return nothing
            end
            logαb = shared ? (fb.α isa Real ? [log(fb.α)] : nothing) : (fb.α isa AbstractVector ? log.(fb.α) : nothing)
            logαb === nothing && return nothing
            return vcat(fb.βc, pack_lambda(fb.Λc), logαb)
        end
        names = vcat(_twopart_shared_lin_names(p, K), α_names)
        return _FamilyCI(θ, nll, names, vcat(fill(:linear, p + rr), fill(:log, ndisp)), sim, refit)
    end
    θ = vcat(fit.βz, fit.βc, pack_lambda(fit.Λc), logα)
    nll = function (θv)
        βz = θv[1:p]; βc = θv[(p + 1):(2p)]
        Λc = unpack_lambda(θv[(2p + 1):(2p + rr)], p, K)
        αv = shared ? exp(θv[2p + rr + 1]) : exp.(θv[(2p + rr + 1):(2p + rr + ndisp)])
        v = try
            objective === :va ?
                -delta_gamma_marginal_loglik_va(Y, Λc, βz, βc, αv; maxiter = newton_maxiter, tol = newton_tol) :
                -delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc, αv; hessian = hessian,
                    maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Float64, p, n)
        @inbounds for s in 1:n
            ηc = fit.βc .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-fit.βz[t]))
                if rand(rng) < π
                    αt = shared ? fit.α : fit.α[t]
                    μ = exp(ηc[t]); Yb[t, s] = rand(rng, Gamma(αt, μ / αt))
                end
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_delta_gamma_gllvm(Yb; K = K, disp_group = fit.disp_group)
        catch
            return nothing
        end
        logαb = shared ? (fb.α isa Real ? [log(fb.α)] : nothing) : (fb.α isa AbstractVector ? log.(fb.α) : nothing)
        logαb === nothing && return nothing
        return vcat(fb.βz, fb.βc, pack_lambda(fb.Λc), logαb)
    end
    names = vcat(_twopart_lin_names(p, K), α_names)
    return _FamilyCI(θ, nll, names, vcat(fill(:linear, 2p + rr), fill(:log, ndisp)), sim, refit)
end

# --- Beta-hurdle (Bernoulli occurrence × positive Beta) --------------------
function _family_ci(fit::BetaHurdleFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K)
    θ = vcat(fit.βz, fit.βc, pack_lambda(fit.Λc), log(fit.φ))
    nll = function (θv)
        βz = θv[1:p]; βc = θv[(p + 1):(2p)]
        Λc = unpack_lambda(θv[(2p + 1):(2p + rr)], p, K); φ = exp(θv[2p + rr + 1])
        v = try
            -beta_hurdle_marginal_loglik_laplace(Y, Λc, βz, βc, φ;
                                                 maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Float64, p, n)
        @inbounds for s in 1:n
            ηc = fit.βc .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-fit.βz[t]))
                if rand(rng) < π
                    μ = clamp(inv(1 + exp(-ηc[t])), 1e-6, 1 - 1e-6)
                    Yb[t, s] = rand(rng, Beta(μ * fit.φ, (1 - μ) * fit.φ))
                end
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_beta_hurdle_gllvm(Yb; K = K) catch; return nothing end
        return vcat(fb.βz, fb.βc, pack_lambda(fb.Λc), log(fb.φ))
    end
    names = vcat(_twopart_lin_names(p, K), "phi")
    return _FamilyCI(θ, nll, names, vcat(fill(:linear, length(θ) - 1), :log), sim, refit)
end

# --- Ordered-beta (proportions with point masses at 0 and 1) ---------------
function _family_ci(fit::OrderedBetaFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); rr = rr_theta_len(p, K)
    θ = vcat(fit.β, pack_lambda(fit.Λ), fit.c0, fit.c1, log(fit.φ))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        c0 = θv[p + rr + 1]; c1 = θv[p + rr + 2]; φ = exp(θv[p + rr + 3])
        v = try
            -ordered_beta_marginal_loglik_laplace(Y, Λ, β, c0, c1, φ;
                                                  maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim   = _ -> error("bootstrap is not supported for ordered-beta CIs")
    refit = function (Yb)
        fb = try fit_ordered_beta_gllvm(Yb; K = K) catch; return nothing end
        return vcat(fb.β, pack_lambda(fb.Λ), fb.c0, fb.c1, log(fb.φ))
    end
    names = vcat(_glm_lin_names(p, K), "cut0", "cut1", "phi")
    kinds = vcat(fill(:linear, p + rr + 2), :log)
    return _FamilyCI(θ, nll, names, kinds, sim, refit)
end

# ---------------------------------------------------------------------------
# Structural models (quadratic, RRR, row-effects, species-cov, fourth-corner,
# concurrent/constrained). Each adapter mirrors that fitter's own negll layout
# and reconstructs the dispersion via _cov_family (covariates.jl). Bootstrap is
# unsupported (sim stub errors); Wald + profile work. Dispersion (when present)
# is the last θ entry on the log scale.
# ---------------------------------------------------------------------------
function _family_ci(fit::QuadraticFit, Y::AbstractMatrix;
                    N::Union{Nothing, AbstractMatrix} = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); pK = p * K
    lk = fit.link; fam0 = fit.family; hasd = _cov_has_disp(fam0)
    Nm = N === nothing ? fill(1, p, n) : N
    θ = hasd ? vcat(fit.β, pack_lambda(fit.Λ), vec(fit.D), log(fit.dispersion)) :
               vcat(fit.β, pack_lambda(fit.Λ), vec(fit.D))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        D = reshape(θv[(p + rr + 1):(p + rr + pK)], p, K)
        disp = hasd ? exp(θv[p + rr + pK + 1]) : NaN
        v = try
            -quadratic_marginal_loglik_laplace(_cov_family(fam0, disp), Y, Nm, Λ, D, β, lk;
                                               maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    names = vcat(_glm_lin_names(p, K), ["D[$t,$k]" for k in 1:K for t in 1:p])
    kinds = fill(:linear, p + rr + pK)
    hasd && (push!(names, "dispersion"); push!(kinds, :log))
    return _FamilyCI(θ, nll, names, kinds,
                     _ -> error("bootstrap not supported for quadratic CIs"), _ -> nothing)
end

function _family_ci(fit::RowEffectFit, Y::AbstractMatrix;
                    N::Union{Nothing, AbstractMatrix} = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); nfree = n - 1
    lk = fit.link; fam0 = fit.family; hasd = _cov_has_disp(fam0)
    Nm = N === nothing ? fill(1, p, n) : N
    ρfree0 = fit.ρ[2:end]                       # ρ_1 ≡ 0 reference
    θ = hasd ? vcat(fit.β, ρfree0, pack_lambda(fit.Λ), log(fit.dispersion)) :
               vcat(fit.β, ρfree0, pack_lambda(fit.Λ))
    nll = function (θv)
        β = θv[1:p]; ρfree = θv[(p + 1):(p + nfree)]
        Λ = unpack_lambda(θv[(p + nfree + 1):(p + nfree + rr)], p, K)
        disp = hasd ? exp(θv[p + nfree + rr + 1]) : NaN
        ρ = vcat(zero(eltype(ρfree)), ρfree); O = _build_offset_row(ρ, p)
        v = try
            -_marginal_loglik_offset(_cov_family(fam0, disp), Y, Nm, Λ, β, O, lk;
                                     maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    names = vcat(["beta[$t]" for t in 1:p], ["rho[$(s + 1)]" for s in 1:nfree],
                 _confint_lambda_term_names("Lambda", p, K))
    kinds = fill(:linear, p + nfree + rr)
    hasd && (push!(names, "dispersion"); push!(kinds, :log))
    return _FamilyCI(θ, nll, names, kinds,
                     _ -> error("bootstrap not supported for row-effect CIs"), _ -> nothing)
end

# The covariate structural models need their design matrix (not stored in the fit),
# so they get dedicated Wald entries (like confint_spde_latent), reusing _family_wald.

"""
    confint_speciescov(fit::GllvmSpeciesCovFit, Y, X; level=0.95, parm=nothing, N=nothing) -> NamedTuple

Wald CIs for the species-specific-covariate GLLVM. `X` is the p×n×q design used in the fit.
"""
function confint_speciescov(fit::GllvmSpeciesCovFit, Y::AbstractMatrix, X::AbstractArray{<:Real,3};
        level::Real = 0.95, parm = nothing, N::Union{Nothing,AbstractMatrix} = nothing,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); q = size(X, 3); pq = p * q
    lk = fit.link; fam0 = fit.family; hasd = _cov_has_disp(fam0)
    Nm = N === nothing ? fill(1, p, n) : N
    θ = hasd ? vcat(fit.β, vec(fit.B), pack_lambda(fit.Λ), log(fit.dispersion)) :
               vcat(fit.β, vec(fit.B), pack_lambda(fit.Λ))
    nll = function (θv)
        β = θv[1:p]; B = reshape(θv[(p + 1):(p + pq)], p, q)
        Λ = unpack_lambda(θv[(p + pq + 1):(p + pq + rr)], p, K)
        disp = hasd ? exp(θv[p + pq + rr + 1]) : NaN
        v = try
            -_marginal_loglik_offset(_cov_family(fam0, disp), Y, Nm, Λ, β,
                                     _build_offset_species(X, B), lk;
                                     maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    names = vcat(["beta[$t]" for t in 1:p], ["B[$t,$j]" for j in 1:q for t in 1:p],
                 _confint_lambda_term_names("Lambda", p, K))
    kinds = fill(:linear, p + pq + rr)
    hasd && (push!(names, "dispersion"); push!(kinds, :log))
    ad = _FamilyCI(θ, nll, names, kinds, _ -> error("bootstrap unsupported"), _ -> nothing)
    sel = _family_select(parm, ad.names)
    isempty(sel) && throw(ArgumentError("parm selector matched no parameters"))
    return _family_wald(ad, sel, level)
end

"""
    confint_fourthcorner(fit::FourthCornerFit, Y, Xenv, TR; level=0.95, parm=nothing, N=nothing) -> NamedTuple

Wald CIs for the fourth-corner trait–environment GLLVM (`Xenv` n×q, `TR` p×r).
"""
function confint_fourthcorner(fit::FourthCornerFit, Y::AbstractMatrix,
        Xenv::AbstractMatrix, TR::AbstractMatrix;
        level::Real = 0.95, parm = nothing, N::Union{Nothing,AbstractMatrix} = nothing,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K)
    q = size(Xenv, 2); r = size(TR, 2); qr = q * r
    lk = fit.link; fam0 = fit.family; hasd = _cov_has_disp(fam0)
    Nm = N === nothing ? fill(1, p, n) : N
    θ = hasd ? vcat(fit.β, vec(fit.C), pack_lambda(fit.Λ), log(fit.dispersion)) :
               vcat(fit.β, vec(fit.C), pack_lambda(fit.Λ))
    nll = function (θv)
        β = θv[1:p]; C = reshape(θv[(p + 1):(p + qr)], q, r)
        Λ = unpack_lambda(θv[(p + qr + 1):(p + qr + rr)], p, K)
        disp = hasd ? exp(θv[p + qr + rr + 1]) : NaN
        v = try
            -_marginal_loglik_offset(_cov_family(fam0, disp), Y, Nm, Λ, β,
                                     _build_offset_fourthcorner(Xenv, TR, C), lk;
                                     maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    names = vcat(["beta[$t]" for t in 1:p], ["C[$i,$j]" for j in 1:r for i in 1:q],
                 _confint_lambda_term_names("Lambda", p, K))
    kinds = fill(:linear, p + qr + rr)
    hasd && (push!(names, "dispersion"); push!(kinds, :log))
    ad = _FamilyCI(θ, nll, names, kinds, _ -> error("bootstrap unsupported"), _ -> nothing)
    sel = _family_select(parm, ad.names)
    isempty(sel) && throw(ArgumentError("parm selector matched no parameters"))
    return _family_wald(ad, sel, level)
end

"""
    confint_rrr(fit::RRRFit, Y, X; level=0.95, parm=nothing, N=nothing) -> NamedTuple

Wald CIs for the reduced-rank-regression (constrained) ordination (`X` n×q).
"""
function confint_rrr(fit::RRRFit, Y::AbstractMatrix, X::AbstractMatrix;
        level::Real = 0.95, parm = nothing, N::Union{Nothing,AbstractMatrix} = nothing)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); q = size(X, 2); qK = q * K
    lk = fit.link; fam0 = fit.family; hasd = _cov_has_disp(fam0)
    Nm = N === nothing ? fill(1, p, n) : N
    θ = hasd ? vcat(fit.β, pack_lambda(fit.Λ), vec(fit.B), log(fit.dispersion)) :
               vcat(fit.β, pack_lambda(fit.Λ), vec(fit.B))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        B = reshape(θv[(p + rr + 1):(p + rr + qK)], q, K)
        disp = hasd ? exp(θv[p + rr + qK + 1]) : NaN
        v = try
            -rrr_marginal_loglik(_cov_family(fam0, disp), Y, Nm, Λ, B, β, X, lk)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    names = vcat(_glm_lin_names(p, K), ["B[$i,$k]" for k in 1:K for i in 1:q])
    kinds = fill(:linear, p + rr + qK)
    hasd && (push!(names, "dispersion"); push!(kinds, :log))
    ad = _FamilyCI(θ, nll, names, kinds, _ -> error("bootstrap unsupported"), _ -> nothing)
    sel = _family_select(parm, ad.names)
    isempty(sel) && throw(ArgumentError("parm selector matched no parameters"))
    return _family_wald(ad, sel, level)
end

"""
    confint_constrained(fit::ConstrainedOrdinationFit, Y, X; level=0.95, parm=nothing, N=nothing) -> NamedTuple

Wald CIs for the concurrent / constrained ordination (`X` n×q).
"""
function confint_constrained(fit::ConstrainedOrdinationFit, Y::AbstractMatrix, X::AbstractMatrix;
        level::Real = 0.95, parm = nothing, N::Union{Nothing,AbstractMatrix} = nothing,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); q = size(X, 2); qK = q * K
    lk = fit.link; fam0 = fit.family; hasd = _cov_has_disp(fam0)
    Nm = N === nothing ? fill(1, p, n) : N
    θ = hasd ? vcat(fit.β, pack_lambda(fit.Λ), vec(fit.B), log(fit.dispersion)) :
               vcat(fit.β, pack_lambda(fit.Λ), vec(fit.B))
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        B = reshape(θv[(p + rr + 1):(p + rr + qK)], q, K)
        disp = hasd ? exp(θv[p + rr + qK + 1]) : NaN
        v = try
            -_marginal_loglik_offset(_cov_family(fam0, disp), Y, Nm, Λ, β,
                                     _build_offset_constrained(Λ, B, X), lk;
                                     maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    names = vcat(_glm_lin_names(p, K), ["B[$i,$k]" for k in 1:K for i in 1:q])
    kinds = fill(:linear, p + rr + qK)
    hasd && (push!(names, "dispersion"); push!(kinds, :log))
    ad = _FamilyCI(θ, nll, names, kinds, _ -> error("bootstrap unsupported"), _ -> nothing)
    sel = _family_select(parm, ad.names)
    isempty(sel) && throw(ArgumentError("parm selector matched no parameters"))
    return _family_wald(ad, sel, level)
end

# --- Hurdle-Poisson --------------------------------------------------------
function _family_ci(fit::HurdlePoissonFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K)
    θ = vcat(fit.βz, fit.βc, pack_lambda(fit.Λc))
    nll = function (θv)
        βz = θv[1:p]; βc = θv[(p + 1):(2p)]
        Λc = unpack_lambda(θv[(2p + 1):(2p + rr)], p, K)
        v = try
            -hurdle_poisson_marginal_loglik_laplace(Y, Λc, βz, βc; maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Int, p, n)
        @inbounds for s in 1:n
            ηc = fit.βc .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-fit.βz[t]))
                rand(rng) < π && (Yb[t, s] = _rand_ztpois(rng, exp(ηc[t])))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_hurdle_poisson_gllvm(Yb; K = K) catch; return nothing end
        return vcat(fb.βz, fb.βc, pack_lambda(fb.Λc))
    end
    return _FamilyCI(θ, nll, _twopart_lin_names(p, K), fill(:linear, length(θ)), sim, refit)
end

# --- Hurdle-NB -------------------------------------------------------------
function _family_ci(fit::HurdleNBFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K)
    θ = vcat(fit.βz, fit.βc, pack_lambda(fit.Λc), log(fit.r))
    nll = function (θv)
        βz = θv[1:p]; βc = θv[(p + 1):(2p)]
        Λc = unpack_lambda(θv[(2p + 1):(2p + rr)], p, K); r = exp(θv[2p + rr + 1])
        v = try
            -hurdle_nb_marginal_loglik_laplace(Y, Λc, βz, βc, r; maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Int, p, n)
        @inbounds for s in 1:n
            ηc = fit.βc .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-fit.βz[t]))
                rand(rng) < π && (Yb[t, s] = _rand_ztnb(rng, fit.r, exp(ηc[t])))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_hurdle_nb_gllvm(Yb; K = K) catch; return nothing end
        return vcat(fb.βz, fb.βc, pack_lambda(fb.Λc), log(fb.r))
    end
    names = vcat(_twopart_lin_names(p, K), "r")
    return _FamilyCI(θ, nll, names, vcat(fill(:linear, length(θ) - 1), :log), sim, refit)
end

# --- Zero-inflated Poisson -------------------------------------------------
function _family_ci(fit::ZIPFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K)
    θ = vcat(fit.βz, fit.βc, pack_lambda(fit.Λc))
    nll = function (θv)
        βz = θv[1:p]; βc = θv[(p + 1):(2p)]
        Λc = unpack_lambda(θv[(2p + 1):(2p + rr)], p, K)
        v = try
            -zip_marginal_loglik_laplace(Y, Λc, βz, βc; maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Int, p, n)
        @inbounds for s in 1:n
            ηc = fit.βc .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-fit.βz[t]))
                Yb[t, s] = rand(rng) < π ? 0 : rand(rng, Poisson(exp(ηc[t])))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_zip_gllvm(Yb; K = K) catch; return nothing end
        return vcat(fb.βz, fb.βc, pack_lambda(fb.Λc))
    end
    return _FamilyCI(θ, nll, _twopart_lin_names(p, K), fill(:linear, length(θ)), sim, refit)
end

# --- Zero-inflated Poisson + shared site-X (dual γz/γc; Λz=0) ---------------
function _family_ci(fit::ZIPCovFit, Y::AbstractMatrix;
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    X === nothing && throw(ArgumentError(
        "confint on a ZIPCovFit needs the design `X`: confint(fit, Y; method=…, X=X)"))
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K)
    Xfit, γ_free_idx = _slice_fixed_X(X, fit.γ_fixed)
    q = length(γ_free_idx)
    γz_free = fit.γz[γ_free_idx]
    γc_free = fit.γc[γ_free_idx]
    θ = vcat(fit.βz, γz_free, fit.βc, γc_free, pack_lambda(fit.Λc))
    nll = function (θv)
        βz = θv[1:p]
        γz = θv[(p + 1):(p + q)]
        βc = θv[(p + q + 1):(2p + q)]
        γc = θv[(2p + q + 1):(2p + 2q)]
        Λc = unpack_lambda(θv[(2p + 2q + 1):(2p + 2q + rr)], p, K)
        Oz = _build_offset(Xfit, γz)
        Oc = _build_offset(Xfit, γc)
        v = try
            -zip_marginal_loglik_laplace(Y, Λc, βz, βc;
                                         offsetz = Oz, offsetc = Oc,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Int, p, n)
        Oz = _build_offset(X, fit.γz)
        Oc = _build_offset(X, fit.γc)
        @inbounds for s in 1:n
            ηc = fit.βc .+ view(Oc, :, s) .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-(fit.βz[t] + Oz[t, s])))
                Yb[t, s] = rand(rng) < π ? 0 : rand(rng, Poisson(exp(ηc[t])))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_zip_gllvm_cov(Yb; X = X, K = K, γ_fixed = fit.γ_fixed)
        catch
            return nothing
        end
        return vcat(fb.βz, fb.γz[γ_free_idx], fb.βc, fb.γc[γ_free_idx], pack_lambda(fb.Λc))
    end
    names = vcat(["betaz[$t]" for t in 1:p],
                 ["gammaz[$k]" for k in γ_free_idx],
                 ["betac[$t]" for t in 1:p],
                 ["gammac[$k]" for k in γ_free_idx],
                 _confint_lambda_term_names("Lambda", p, K))
    return _FamilyCI(θ, nll, names, fill(:linear, length(θ)), sim, refit)
end

# --- Zero-inflated NB ------------------------------------------------------
function _family_ci(fit::ZINBFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K)
    θ = vcat(fit.βz, fit.βc, pack_lambda(fit.Λc), log(fit.r))
    nll = function (θv)
        βz = θv[1:p]; βc = θv[(p + 1):(2p)]
        Λc = unpack_lambda(θv[(2p + 1):(2p + rr)], p, K); r = exp(θv[2p + rr + 1])
        v = try
            -zinb_marginal_loglik_laplace(Y, Λc, βz, βc, r; maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Int, p, n)
        @inbounds for s in 1:n
            ηc = fit.βc .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-fit.βz[t])); μ = exp(ηc[t])
                Yb[t, s] = rand(rng) < π ? 0 : rand(rng, NegativeBinomial(fit.r, fit.r / (fit.r + μ)))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_zinb_gllvm(Yb; K = K) catch; return nothing end
        return vcat(fb.βz, fb.βc, pack_lambda(fb.Λc), log(fb.r))
    end
    names = vcat(_twopart_lin_names(p, K), "r")
    return _FamilyCI(θ, nll, names, vcat(fill(:linear, length(θ) - 1), :log), sim, refit)
end

# --- Zero-inflated NB + shared site-X (dual γz/γc; Λz=0; shared scalar r) ---
function _family_ci(fit::ZINBCovFit, Y::AbstractMatrix;
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    X === nothing && throw(ArgumentError(
        "confint on a ZINBCovFit needs the design `X`: confint(fit, Y; method=…, X=X)"))
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K)
    Xfit, γ_free_idx = _slice_fixed_X(X, fit.γ_fixed)
    q = length(γ_free_idx)
    γz_free = fit.γz[γ_free_idx]
    γc_free = fit.γc[γ_free_idx]
    θ = vcat(fit.βz, γz_free, fit.βc, γc_free, pack_lambda(fit.Λc), log(fit.r))
    nll = function (θv)
        βz = θv[1:p]
        γz = θv[(p + 1):(p + q)]
        βc = θv[(p + q + 1):(2p + q)]
        γc = θv[(2p + q + 1):(2p + 2q)]
        Λc = unpack_lambda(θv[(2p + 2q + 1):(2p + 2q + rr)], p, K)
        r = exp(θv[2p + 2q + rr + 1])
        Oz = _build_offset(Xfit, γz)
        Oc = _build_offset(Xfit, γc)
        v = try
            -zinb_marginal_loglik_laplace(Y, Λc, βz, βc, r;
                                         offsetz = Oz, offsetc = Oc,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Int, p, n)
        Oz = _build_offset(X, fit.γz)
        Oc = _build_offset(X, fit.γc)
        @inbounds for s in 1:n
            ηc = fit.βc .+ view(Oc, :, s) .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-(fit.βz[t] + Oz[t, s])))
                μ = exp(ηc[t])
                Yb[t, s] = rand(rng) < π ? 0 : rand(rng, NegativeBinomial(fit.r, fit.r / (fit.r + μ)))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_zinb_gllvm_cov(Yb; X = X, K = K, γ_fixed = fit.γ_fixed)
        catch
            return nothing
        end
        return vcat(fb.βz, fb.γz[γ_free_idx], fb.βc, fb.γc[γ_free_idx],
                    pack_lambda(fb.Λc), log(fb.r))
    end
    names = vcat(["betaz[$t]" for t in 1:p],
                 ["gammaz[$k]" for k in γ_free_idx],
                 ["betac[$t]" for t in 1:p],
                 ["gammac[$k]" for k in γ_free_idx],
                 _confint_lambda_term_names("Lambda", p, K),
                 "r")
    return _FamilyCI(θ, nll, names, vcat(fill(:linear, length(θ) - 1), :log), sim, refit)
end

# --- Zero-inflated binomial ------------------------------------------------
function _family_ci(fit::ZIBFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λc); n = size(Y, 2); rr = rr_theta_len(p, K); Ntr = fit.N
    θ = vcat(fit.βz, fit.βc, pack_lambda(fit.Λc))
    nll = function (θv)
        βz = θv[1:p]; βc = θv[(p + 1):(2p)]
        Λc = unpack_lambda(θv[(2p + 1):(2p + rr)], p, K)
        v = try
            -zib_marginal_loglik_laplace(Y, Λc, βz, βc, Ntr; maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = zeros(Int, p, n)
        @inbounds for s in 1:n
            ηc = fit.βc .+ fit.Λc * randn(rng, K)
            for t in 1:p
                π = inv(1 + exp(-fit.βz[t])); μ = inv(1 + exp(-ηc[t]))
                Yb[t, s] = rand(rng) < π ? 0 : rand(rng, Binomial(Ntr, μ))
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_zib_gllvm(Yb; K = K, N = Ntr) catch; return nothing end
        return vcat(fb.βz, fb.βc, pack_lambda(fb.Λc))
    end
    return _FamilyCI(θ, nll, _twopart_lin_names(p, K), fill(:linear, length(θ)), sim, refit)
end
# Working vector [pack_lambda(Λ); τ] in the NATURAL cutpoint parameterisation
# (the fitter optimises ψ-increments, but the MLE τ̂ is strictly ordered, so the
# Wald Hessian / profile / bootstrap all run directly in τ-space — the
# interpretable scale). The ordinal marginal clamps category probabilities at
# 1e-12, so the small perturbations used by the Hessian / refits stay finite even
# if a τ momentarily loses ordering.
function _family_ci(fit::OrdinalFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); C = fit.C
    θ = vcat(pack_lambda(fit.Λ), fit.τ)
    nll = function (θv)
        Λ = unpack_lambda(θv[1:rr], p, K)
        τ = θv[(rr + 1):(rr + C - 1)]
        v = try
            -ordinal_marginal_loglik_laplace(Y, Λ, τ; link = fit.link,
                                             maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.Λ * randn(rng, K)
            for t in 1:p
                u = rand(rng); cum = 0.0; cat = C
                for c in 1:C
                    cum += _ord_prob(c, η[t], fit.τ, fit.link)
                    if u <= cum
                        cat = c; break
                    end
                end
                Yb[t, s] = cat
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_ordinal_gllvm(Yb; K = K, link = fit.link) catch; return nothing end
        fb.C == C || return nothing                 # category-count mismatch ⇒ drop replicate
        return vcat(pack_lambda(fb.Λ), fb.τ)
    end
    names = vcat(_confint_lambda_term_names("Lambda", p, K), ["tau[$c]" for c in 1:(C - 1)])
    return _FamilyCI(θ, nll, names, fill(:linear, length(θ)), sim, refit)
end

# Per-trait ordinal cutpoints (twin τ₁=0): free natural-scale entries are
# τ[t,c] for c = 2:(C[t]−1). Wald / profile / bootstrap run in that τ-space
# (same natural-scale choice as shared-cutpoint OrdinalFit). Bridge ci_method
# guards stay until a separate bridge lift (do not edit bridge.jl here).
function _pack_free_tau_pertrait(τ::AbstractMatrix, C::AbstractVector{<:Integer})
    pieces = Float64[]
    @inbounds for t in eachindex(C)
        for c in 2:(C[t] - 1)
            push!(pieces, Float64(τ[t, c]))
        end
    end
    return pieces
end

function _unpack_free_tau_pertrait(θτ::AbstractVector, C::AbstractVector{<:Integer})
    p = length(C)
    Cmax = maximum(C)
    τ = fill(NaN, p, max(Cmax - 1, 0))
    pos = 1
    @inbounds for t in 1:p
        m = C[t] - 1
        m >= 1 || continue
        τ[t, 1] = 0.0
        for c in 2:m
            τ[t, c] = θτ[pos]
            pos += 1
        end
    end
    return τ
end

function _free_tau_pertrait_names(C::AbstractVector{<:Integer})
    names = String[]
    @inbounds for t in eachindex(C)
        for c in 2:(C[t] - 1)
            push!(names, "tau[$t,$c]")
        end
    end
    return names
end

# Working vector [β; pack_lambda(Λ); free τ] — τ₁ fixed at 0 per trait.
function _family_ci(fit::OrdinalPerTraitFit, Y::AbstractMatrix;
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    p, K = size(fit.Λ); n = size(Y, 2); rr = rr_theta_len(p, K); C = fit.C
    ncut = sum(C .- 2)
    τ_free = _pack_free_tau_pertrait(fit.τ, C)
    length(τ_free) == ncut || throw(ArgumentError(
        "OrdinalPerTraitFit free cutpoint length $(length(τ_free)) ≠ sum(C−2)=$ncut"))
    θ = vcat(fit.β, pack_lambda(fit.Λ), τ_free)
    nll = function (θv)
        β = θv[1:p]
        Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        τ = _unpack_free_tau_pertrait(θv[(p + rr + 1):(p + rr + ncut)], C)
        v = try
            -ordinal_marginal_loglik_laplace_pertrait(Y, Λ, β, τ, C;
                link = fit.link, maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        @inbounds for s in 1:n
            η = fit.β .+ fit.Λ * randn(rng, K)
            for t in 1:p
                τt = _trait_cutpoints(fit.τ, C, t)
                u = rand(rng); cum = 0.0; cat = C[t]
                for c in 1:C[t]
                    cum += _ord_prob(c, η[t], τt, fit.link)
                    if u <= cum
                        cat = c; break
                    end
                end
                Yb[t, s] = cat
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try fit_ordinal_gllvm_pertrait(Yb; K = K, link = fit.link) catch; return nothing end
        fb.C == C || return nothing
        return vcat(fb.β, pack_lambda(fb.Λ), _pack_free_tau_pertrait(fb.τ, C))
    end
    names = vcat(["beta[$t]" for t in 1:p],
                 _confint_lambda_term_names("Lambda", p, K),
                 _free_tau_pertrait_names(C))
    return _FamilyCI(θ, nll, names, fill(:linear, length(θ)), sim, refit)
end

# Working vector [β; γ_free; pack_lambda(Λ); free τ]. Needs design `X`.
function _family_ci(fit::OrdinalPerTraitCovFit, Y::AbstractMatrix;
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    X === nothing && throw(ArgumentError(
        "confint on an OrdinalPerTraitCovFit needs the design `X` " *
        "(the same array passed to fit_ordinal_gllvm_pertrait_cov): " *
        "confint(fit, Y; method=…, X=X)"))
    p, K = size(fit.Λ); n = size(Y, 2); q_full = length(fit.γ); rr = rr_theta_len(p, K)
    C = fit.C; ncut = sum(C .- 2)
    length(fit.γ_fixed) == q_full || throw(ArgumentError(
        "fit.γ_fixed length ($(length(fit.γ_fixed))) must equal length(fit.γ) = $q_full"))
    Xfit, γ_free_idx = _slice_fixed_X(X, fit.γ_fixed)
    q = length(γ_free_idx)
    γ_free = fit.γ[γ_free_idx]
    τ_free = _pack_free_tau_pertrait(fit.τ, C)
    length(τ_free) == ncut || throw(ArgumentError(
        "OrdinalPerTraitCovFit free cutpoint length $(length(τ_free)) ≠ sum(C−2)=$ncut"))
    θ = vcat(fit.β, γ_free, pack_lambda(fit.Λ), τ_free)
    nll = function (θv)
        β = θv[1:p]
        γ = θv[(p + 1):(p + q)]
        Λ = unpack_lambda(θv[(p + q + 1):(p + q + rr)], p, K)
        τ = _unpack_free_tau_pertrait(θv[(p + q + rr + 1):(p + q + rr + ncut)], C)
        O = _build_offset(Xfit, γ)
        v = try
            -ordinal_marginal_loglik_laplace_pertrait(Y, Λ, β, τ, C;
                link = fit.link, offset = O,
                maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = Matrix{Int}(undef, p, n)
        O = _build_offset(X, fit.γ)
        @inbounds for s in 1:n
            η = fit.β .+ view(O, :, s) .+ fit.Λ * randn(rng, K)
            for t in 1:p
                τt = _trait_cutpoints(fit.τ, C, t)
                u = rand(rng); cum = 0.0; cat = C[t]
                for c in 1:C[t]
                    cum += _ord_prob(c, η[t], τt, fit.link)
                    if u <= cum
                        cat = c; break
                    end
                end
                Yb[t, s] = cat
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_ordinal_gllvm_pertrait_cov(Yb; X = X, K = K, link = fit.link,
                                           γ_fixed = fit.γ_fixed)
        catch
            return nothing
        end
        fb.C == C || return nothing
        return vcat(fb.β, fb.γ[γ_free_idx], pack_lambda(fb.Λ),
                    _pack_free_tau_pertrait(fb.τ, C))
    end
    names = vcat(["beta[$t]" for t in 1:p],
                 ["gamma[$k]" for k in γ_free_idx],
                 _confint_lambda_term_names("Lambda", p, K),
                 _free_tau_pertrait_names(C))
    return _FamilyCI(θ, nll, names, fill(:linear, length(θ)), sim, refit)
end

# Unordered categorical FE softmax (v1; twin fid 16): packing
# [β₂…β_K; γ₂ (p); …; γ_K (p)] with η₁≡0. No LV / no φ. Design `X` is n×p
# (site covariates), not the 3-array GllvmCovFit layout — pass via kwargs as
# AbstractMatrix. No-X fits omit X.
function _multinomial_ci_names(n_categories::Integer, n_covariates::Integer)
    names = String["beta[$k]" for k in 2:n_categories]
    @inbounds for k in 2:n_categories
        for j in 1:n_covariates
            push!(names, "gamma[$k,$j]")
        end
    end
    return names
end

function _family_ci(fit::MultinomialFit, Y::AbstractMatrix;
                    X::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                    kwargs...)
    K = fit.n_categories
    p = size(fit.γ, 2)
    n = size(Y, 2)
    if p == 0
        X === nothing || throw(ArgumentError(
            "MultinomialFit was fit without covariates; omit X in confint"))
    else
        X === nothing && throw(ArgumentError(
            "confint on a covariate MultinomialFit needs the design `X` " *
            "(the same n×p matrix passed to fit_multinomial_gllvm): " *
            "confint(fit, Y; method=…, X=X)"))
        size(X) == (n, p) || throw(DimensionMismatch(
            "multinomial confint X must be n×p = ($n, $p); got $(size(X))"))
    end
    θ = copy(fit.theta_packed)
    length(θ) == multinomial_pack_len(K, p) || throw(DimensionMismatch(
        "MultinomialFit.theta_packed length $(length(θ)) ≠ (K-1)(1+p) = " *
        "$(multinomial_pack_len(K, p))"))
    nll = function (θv)
        v = try
            -multinomial_loglik(Y, θv; X = X, n_categories = K)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = Matrix{Int}(undef, 1, n)
        xbuf = p == 0 ? Float64[] : Vector{Float64}(undef, p)
        @inbounds for i in 1:n
            if p > 0
                for j in 1:p
                    xbuf[j] = X[i, j]
                end
            end
            η = multinomial_eta(fit.β, fit.γ, xbuf)
            # Softmax draw (η₁≡0 already in multinomial_eta).
            m = η[1]
            for k in 2:K
                η[k] > m && (m = η[k])
            end
            s = 0.0
            for k in 1:K
                s += exp(η[k] - m)
            end
            u = rand(rng) * s
            cum = 0.0
            cat = K
            for k in 1:K
                cum += exp(η[k] - m)
                if u <= cum
                    cat = k
                    break
                end
            end
            Yb[1, i] = cat
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_multinomial_gllvm(Yb; X = X, n_categories = K, link = fit.link)
        catch
            return nothing
        end
        fb.n_categories == K || return nothing
        size(fb.γ, 2) == p || return nothing
        return copy(fb.theta_packed)
    end
    return _FamilyCI(θ, nll, _multinomial_ci_names(K, p), fill(:linear, length(θ)),
                     sim, refit)
end

# --- Covariate fit (GllvmCovFit: β + Xγ + Λz) ------------------------------
# Working vector [β; γ_free; pack_lambda(Λ); (log-dispersion)]. Fixed-zero γ
# entries are structural constraints, not free Hessian/profile/bootstrap terms.
# Requires the full (p,n,q) design `X` (and Binomial trial counts `N`) via
# confint(...; X=…, N=…); the free-column slice is reconstructed from fit.γ_fixed.
function _family_ci(fit::GllvmCovFit, Y::AbstractMatrix;
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    N::Union{Nothing, AbstractMatrix} = nothing,
                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, kwargs...)
    X === nothing && throw(ArgumentError("confint on a GllvmCovFit needs the design `X` (the same array passed to fit_gllvm_cov): confint(fit, Y; method=…, X=X)"))
    p, K = size(fit.Λ); n = size(Y, 2); q_full = length(fit.γ); rr = rr_theta_len(p, K)
    length(fit.γ_fixed) == q_full || throw(ArgumentError(
        "fit.γ_fixed length ($(length(fit.γ_fixed))) must equal length(fit.γ) = $q_full"))
    Xfit, γ_free_idx = _slice_fixed_X(X, fit.γ_fixed)
    q = length(γ_free_idx)
    lk = fit.link; has_disp = !isnan(fit.dispersion)
    Nm = N === nothing ? fill(1, p, n) : N
    γ_free = fit.γ[γ_free_idx]
    θ = has_disp ? vcat(fit.β, γ_free, pack_lambda(fit.Λ), log(fit.dispersion)) :
                   vcat(fit.β, γ_free, pack_lambda(fit.Λ))
    nll = function (θv)
        β = θv[1:p]; γ = θv[(p + 1):(p + q)]
        Λ = unpack_lambda(θv[(p + q + 1):(p + q + rr)], p, K)
        disp = has_disp ? exp(θv[p + q + rr + 1]) : NaN
        O = _build_offset(Xfit, γ)
        v = try
            # Inside the `try`: exp(log-dispersion) can underflow to 0.0 (see fit_gllvm_cov).
            fam = _cov_family(fit.family, disp)
            -_marginal_loglik_offset(fam, Y, Nm, Λ, β, O, lk; maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    sim = function (rng)
        Yb = Matrix{Float64}(undef, p, n)
        O = _build_offset(X, fit.γ); fam = _cov_family(fit.family, fit.dispersion)
        @inbounds for s in 1:n
            η = fit.β .+ view(O, :, s) .+ fit.Λ * randn(rng, K)
            for t in 1:p
                μ = linkinv(lk, _clamp_eta(η[t]))
                Yb[t, s] = _cov_sample(fam, μ, Nm[t, s], rng)
            end
        end
        return Yb
    end
    refit = function (Yb)
        fb = try
            fit_gllvm_cov(Yb; family = fit.family, X = X, K = K, link = lk,
                          N = fit.family isa Binomial ? Nm : nothing,
                          γ_fixed = fit.γ_fixed)
        catch
            return nothing
        end
        fb_γ_free = fb.γ[γ_free_idx]
        return has_disp ? vcat(fb.β, fb_γ_free, pack_lambda(fb.Λ), log(fb.dispersion)) :
                          vcat(fb.β, fb_γ_free, pack_lambda(fb.Λ))
    end
    names = vcat(["beta[$t]" for t in 1:p], ["gamma[$k]" for k in γ_free_idx],
                 _confint_lambda_term_names("Lambda", p, K))
    kinds = fill(:linear, length(names))
    if has_disp
        names = vcat(names, _cov_dispname(fit.family)); kinds = vcat(kinds, :log)
    end
    return _FamilyCI(θ, nll, names, kinds, sim, refit)
end

# ---------------------------------------------------------------------------
# Generic numerics
# ---------------------------------------------------------------------------

# Objectives in this package return a large FINITE penalty (`_PHYLO_PENALTY`,
# `_TWEEDIE_FAIL_PENALTY`, the bare 1e12 in constrained_ordination.jl) when they cannot be
# evaluated — near a domain edge, for instance. Differencing such an arm is not a curvature
# measurement: it produces enormous apparent curvature and therefore a VANISHING standard
# error. The failure direction is toward false certainty, which is the dangerous one.
#
# Measured before this guard existed: `fit_gp1_gllvm(fill(600, 40, 5); K = 1)` put α̂ about
# 0.05·h from the GP-1 domain edge, 17 stencil arms returned the sentinel, and the reported
# 95% interval for α was 4.8e-10 wide on an estimate of 1.7e-3 — with `pd_hessian` true.
const _FD_FAIL_THRESHOLD = 1e11

_fd_failed(v) = !isfinite(v) || abs(v) >= _FD_FAIL_THRESHOLD

# Central-difference Hessian of `f` at `x`. Step ∝ eps^(1/4) (the optimum for
# the 3-point second derivative, balancing O(h²) truncation against O(eps/h²)
# rounding). O(m²) function evaluations.
#
# Any entry whose stencil touches a failed arm is `NaN`. Callers already treat a non-finite
# Hessian as "not positive definite" and report `NaN` intervals, so a failure surfaces as
# "no interval" rather than as a confident wrong one.
function _fd_hessian(f, x::AbstractVector)
    m = length(x)
    h = [eps()^(1 / 4) * max(abs(x[i]), 1.0) for i in 1:m]
    H = Matrix{Float64}(undef, m, m)
    f0 = f(x)
    f0_bad = _fd_failed(f0)
    @inbounds for i in 1:m
        xp = copy(x); xp[i] += h[i]
        xm = copy(x); xm[i] -= h[i]
        fp = f(xp); fm = f(xm)
        # NB `2 * f0`, not `2f0`: the latter lexes as the Float32 literal 2.0f0,
        # dropping the central term and corrupting every observed-information SE.
        H[i, i] = (f0_bad || _fd_failed(fp) || _fd_failed(fm)) ? NaN :
                  (fp - 2 * f0 + fm) / h[i]^2
    end
    @inbounds for i in 1:m, j in (i + 1):m
        xpp = copy(x); xpp[i] += h[i]; xpp[j] += h[j]
        xpm = copy(x); xpm[i] += h[i]; xpm[j] -= h[j]
        xmp = copy(x); xmp[i] -= h[i]; xmp[j] += h[j]
        xmm = copy(x); xmm[i] -= h[i]; xmm[j] -= h[j]
        a = f(xpp); b = f(xpm); c = f(xmp); d = f(xmm)
        H[i, j] = H[j, i] = any(_fd_failed, (a, b, c, d)) ? NaN :
                            (a - b - c + d) / (4 * h[i] * h[j])
    end
    return H
end

# Resolve a `parm` selector against the term names. `nothing` → all; an exact
# name, a group prefix ("beta", "Lambda"), a dispersion name ("r"/"phi"/"alpha"),
# or a vector of any of these.
function _family_select(parm, names::Vector{String})
    parm === nothing && return collect(eachindex(names))
    sels = parm isa AbstractVector ? parm : [parm]
    idx = Int[]
    for s in sels
        ss = String(s)
        exact = findall(==(ss), names)
        if !isempty(exact)
            append!(idx, exact)
        elseif ss in ("beta", "Lambda")
            pref = ss == "beta" ? "beta[" : "Lambda["
            append!(idx, findall(n -> startswith(n, pref), names))
        else
            grp = findall(n -> startswith(n, ss), names)
            isempty(grp) && throw(ArgumentError("parm selector \"$ss\" matched no terms"))
            append!(idx, grp)
        end
    end
    return unique(idx)
end

# Natural-scale point estimate for entry i (exp() for log-scale dispersion).
_family_estimate(ad::_FamilyCI, i::Integer) = ad.kinds[i] === :log ? exp(ad.θ[i]) : ad.θ[i]

# T14 F1 (docs/dev-log/core070/t14-nb2-wald-nan-diagnosis.md), per-parameter Wald
# degradation. When the FULL joint Hessian is not usable (Cholesky fails, or any
# resulting variance is non-finite/non-positive), the previous behaviour NaN'd
# EVERY parameter — including well-identified β/γ/Λ entries sharing a joint
# Hessian with one degenerate direction (e.g. a grouped-dispersion trait at the
# Poisson/near-Bernoulli/near-deterministic boundary, `dispersion_boundary`).
# R's `sdreport` degrades per-block instead. This conditions the KNOWN-boundary
# parameters (`known::Vector{Int}`, from a fit's own `dispersion_boundary` via
# `ad.boundary`) out of the joint Hessian and, if the remaining sub-Hessian is
# still not PD, greedily removes the remaining direction whose eigenvector has
# the largest weight on the smallest eigenvalue — the same numerical PD test
# (`isposdef`, a Cholesky attempt under the hood, i.e. the same machine-epsilon-
# scale tolerance already used above) applied to a shrinking index set, rather
# than a hand-rolled eigenvalue cutoff. Bounded to at most `m` iterations (can
# drop at most every parameter, at which point the old all-NaN behaviour is
# recovered exactly).
function _wald_boundary_indices(Hsym::Symmetric, known::Vector{Int})
    m = size(Hsym, 1)
    idx = Set(known)
    for _ in 1:m
        rem = sort(setdiff(1:m, idx))
        isempty(rem) && break
        Hsub = Symmetric(Hsym[rem, rem])
        isposdef(Hsub) && return sort(collect(idx))
        vals, vecs = eigen(Hsub)
        j = argmin(vals)
        worst_local = argmax(abs.(view(vecs, :, j)))
        push!(idx, rem[worst_local])
    end
    return sort(collect(idx))
end

# ---------------------------------------------------------------------------
# Wald
# ---------------------------------------------------------------------------
function _family_wald(ad::_FamilyCI, sel::Vector{Int}, level::Real; hessian=nothing)
    m = length(ad.θ)
    H = hessian===nothing ? _fd_hessian(ad.nll, ad.θ) : hessian
    size(H)==(m,m) || throw(DimensionMismatch("Wald Hessian dimension mismatch"))
    se=fill(NaN,m);pd=false
    boundary_terms = String[]
    if all(isfinite,H)
        Hsym = Symmetric((H .+ H') ./ 2)
        factor=try
            cholesky(Hsym;check=true)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        if factor!==nothing
            covariance=factor \ Matrix{Float64}(I,m,m)
            variances=diag(covariance)
            pd=all(v->isfinite(v) && v>0,variances)
            pd && (se .= sqrt.(variances))
        end
        # T14 F1 (2026-09-03): a parameter the FIT flags as at a boundary
        # (`dispersion_boundary`) is conditioned out even when the joint
        # Hessian happens to be barely positive definite — otherwise a
        # Poisson-limit dispersion gets a meaningless huge "finite" SE and an
        # `Inf` bound, and the outcome flips with the optimizer's stopping
        # point (the seed-523 regime A/B knife edge). `pd_hessian` is `false`
        # whenever any term is conditioned out; R's sdreport NaNs that block.
        if pd && any(ad.boundary)
            pd = false
            se .= NaN
        end
        if !pd
            # `pd_hessian` keeps its existing meaning below (true only when the
            # FULL joint Hessian's Cholesky succeeded with all-positive
            # variances) — this branch only fills in what it can on top of the
            # NaN default, it never flips `pd` back to true.
            bidx = _wald_boundary_indices(Hsym, findall(ad.boundary))
            rem = setdiff(1:m, bidx)
            if !isempty(rem)
                Hsub = Symmetric((H[rem, rem] .+ H[rem, rem]') ./ 2)
                if isposdef(Hsub)
                    Σsub = inv(Hsub)
                    dΣ = diag(Σsub)
                    for (k, i) in enumerate(rem)
                        v = dΣ[k]
                        (isfinite(v) && v > 0) && (se[i] = sqrt(v))
                    end
                end
            end
            boundary_terms = ad.names[bidx]
        end
    end
    z = quantile(Normal(), 0.5 + level / 2)
    term = String[]; est = Float64[]; lo = Float64[]; hi = Float64[]; ses = Float64[]
    for i in sel
        push!(term, ad.names[i]); push!(ses, se[i])
        θi = ad.θ[i]; sei = se[i]
        if ad.kinds[i] === :log
            push!(est, exp(θi))
            push!(lo, isfinite(sei) ? exp(θi - z * sei) : NaN)
            push!(hi, isfinite(sei) ? exp(θi + z * sei) : NaN)
        else
            push!(est, θi)
            push!(lo, isfinite(sei) ? θi - z * sei : NaN)
            push!(hi, isfinite(sei) ? θi + z * sei : NaN)
        end
    end
    return (term = term, estimate = est, lower = lo, upper = hi, se = ses,
            method = :wald, pd_hessian = pd, boundary_terms = boundary_terms)
end

# ---------------------------------------------------------------------------
# Profile likelihood
# ---------------------------------------------------------------------------

# Constrained refit: minimise nll over θ_{-i} with θ_i fixed at c. Returns
# (ℓ_profile, ok, θ_red_solution). Finite-difference gradient (nll is not
# AD-friendly through the Laplace mode-finder).
function _family_profile_refit(ad::_FamilyCI, i::Integer, c::Real, θ_red_warm::AbstractVector;
                               g_tol::Real = 1e-4,
                               iterations::Integer = 200)
    m = length(ad.θ)
    cf = float(c)
    full = function (θr)
        θf = Vector{Float64}(undef, m)
        @inbounds for j in 1:(i - 1); θf[j] = θr[j]; end
        θf[i] = cf
        @inbounds for j in (i + 1):m; θf[j] = θr[j - 1]; end
        return θf
    end
    nll_red = θr -> ad.nll(full(θr))
    res = try
        Optim.optimize(nll_red, collect(Float64, θ_red_warm),
                       Optim.LBFGS(), Optim.Options(g_tol = g_tol, iterations = iterations);
                       autodiff = :finite)
    catch
        return (NaN, false, collect(Float64, θ_red_warm))
    end
    nmin = Optim.minimum(res)
    isfinite(nmin) || return (NaN, false, collect(Float64, θ_red_warm))
    return (-nmin, true, Optim.minimizer(res))
end

function _family_profile(ad::_FamilyCI, sel::Vector{Int}, level::Real;
                         profile_iterations::Integer = 200,
                         profile_g_tol::Real = 1e-4,
                         profile_max_expand::Integer = 20,
                         profile_max_bisect::Integer = 30)
    profile_iterations > 0 ||
        throw(ArgumentError("profile_iterations must be positive; got $profile_iterations"))
    isfinite(profile_g_tol) && profile_g_tol > 0 ||
        throw(ArgumentError("profile_g_tol must be positive and finite; got $profile_g_tol"))
    profile_max_expand > 0 ||
        throw(ArgumentError("profile_max_expand must be positive; got $profile_max_expand"))
    profile_max_bisect > 0 ||
        throw(ArgumentError("profile_max_bisect must be positive; got $profile_max_bisect"))
    m = length(ad.θ)
    cutoff = quantile(Chisq(1), level)
    ll_full = -ad.nll(ad.θ)
    # Wald SEs to seed the bracket steps (cheap relative to the refits).
    H = _fd_hessian(ad.nll, ad.θ)
    se_all = fill(NaN, m)
    if all(isfinite, H)
        Σ = try inv(Symmetric((H .+ H') ./ 2)) catch; nothing end
        if Σ !== nothing
            for i in 1:m
                v = Σ[i, i]; (isfinite(v) && v > 0) && (se_all[i] = sqrt(v))
            end
        end
    end

    term = String[]; est = Float64[]; lo = Float64[]; hi = Float64[]; meth = Symbol[]
    for i in sel
        θi = float(ad.θ[i])
        sei = (isnan(se_all[i]) || se_all[i] ≤ 0) ? max(abs(θi) / 2, 0.1) : se_all[i]
        warm_lo = vcat(ad.θ[1:(i - 1)], ad.θ[(i + 1):m])
        warm_hi = copy(warm_lo)
        function dev_lo(c)
            ll, ok, sol = _family_profile_refit(ad, i, c, warm_lo;
                                                g_tol = profile_g_tol,
                                                iterations = profile_iterations)
            ok ? (warm_lo = sol; 2.0 * (ll_full - ll)) : NaN
        end
        function dev_hi(c)
            ll, ok, sol = _family_profile_refit(ad, i, c, warm_hi;
                                                g_tol = profile_g_tol,
                                                iterations = profile_iterations)
            ok ? (warm_hi = sol; 2.0 * (ll_full - ll)) : NaN
        end
        # Seed the first candidate near the Wald bound (θ̂ ± √cutoff·SE) so the
        # bracket is found in ~1 refit; false-position root-finding does the rest.
        step = max(sqrt(cutoff) * sei, 1e-3)
        lower = _profile_bisect_side(dev_lo, θi, -step, cutoff;
                                     max_expand = profile_max_expand,
                                     max_bisect = profile_max_bisect)
        upper = _profile_bisect_side(dev_hi, θi,  step, cutoff;
                                     max_expand = profile_max_expand,
                                     max_bisect = profile_max_bisect)
        if ad.kinds[i] === :log
            lower = isnan(lower) ? NaN : exp(lower)
            upper = isnan(upper) ? NaN : exp(upper)
        end
        push!(term, ad.names[i]); push!(est, _family_estimate(ad, i))
        push!(lo, lower); push!(hi, upper)
        push!(meth, (isnan(lower) && isnan(upper)) ? :failed :
                    (isnan(lower) || isnan(upper)) ? :partial : :profile)
    end
    return (term = term, estimate = est, lower = lo, upper = hi, status = meth, method = :profile)
end

# ---------------------------------------------------------------------------
# Parametric bootstrap (optionally threaded)
# ---------------------------------------------------------------------------
function _family_bootstrap(ad::_FamilyCI, sel::Vector{Int}, level::Real,
                           n_boot::Integer, seed::Integer, parallel::Bool; retain_replicates::Bool=false)
    m = length(ad.θ)
    reps = fill(NaN, n_boot, m)
    ok = fill(false, n_boot)   # Vector{Bool} (one byte/elt) — safe for concurrent distinct-index writes (a BitVector is not)
    work = function (b)
        rng = MersenneTwister(seed + b)
        θb = try
            ad.refit(ad.simulate(rng))     # guard both sim + refit so one bad replicate can't crash the run
        catch
            nothing
        end
        if θb !== nothing && length(θb) == m && all(isfinite, θb)
            @inbounds reps[b, :] .= θb
            ok[b] = true
        end
        return nothing
    end
    if parallel
        Threads.@threads for b in 1:n_boot
            work(b)
        end
    else
        for b in 1:n_boot
            work(b)
        end
    end

    a = (1 - level) / 2
    term = String[]; est = Float64[]; lo = Float64[]; hi = Float64[]
    for i in sel
        col = Float64[]
        for b in 1:n_boot
            isnan(reps[b, i]) || push!(col, ad.kinds[i] === :log ? exp(reps[b, i]) : reps[b, i])
        end
        push!(term, ad.names[i]); push!(est, _family_estimate(ad, i))
        if length(col) ≥ 10
            push!(lo, quantile(col, a)); push!(hi, quantile(col, 1 - a))
        else
            push!(lo, NaN); push!(hi, NaN)
        end
    end
    result=(term = term, estimate = est, lower = lo, upper = hi,
            n_converged = count(ok), method = :bootstrap)
    return retain_replicates ? merge(result,(replicates=reps,converged=ok,seed=seed,)) : result
end

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
"""
    confint(fit, Y; method = :wald, level = 0.95, parm = nothing, N = nothing,
            mask = nothing,
            n_boot = 200, seed = 0, parallel = false, objective = :fit,
            newton_maxiter = 100, newton_tol = 1e-9,
            profile_iterations = 200, profile_g_tol = 1e-4,
            profile_max_expand = 20, profile_max_bisect = 30) -> NamedTuple

Confidence intervals for a non-Gaussian family GLLVM fit — the scalar-μ GLM
families (`PoissonFit`, `BinomialFit`, `NBFit`, `BetaFit`, `GammaFit`,
`TweedieFit`, `BetaBinomialFit`, `LognormalFit`, `TruncatedPoissonFit`,
`TruncatedNegBin2Fit`, fixed-ν `StudentTFit`), shared-cutpoint `OrdinalFit`, per-trait
cutpoint `OrdinalPerTraitFit` / `OrdinalPerTraitCovFit` (free `tau[t,c]` for
`c≥2`; `τ₁=0` fixed; CovFit needs `X`), FE softmax `MultinomialFit`
(contrast `beta[k]` / `gamma[k,j]` for categories `k≥2`; `η₁≡0`; n×p `X` when
covariates were fit), the random-row-effect fit (`RowRandomFit`,
which adds a `sigma_row` term plus the underlying family's dispersion), and the
two-part families (`DeltaLogNormalFit`, `DeltaGammaFit`, `HurdlePoissonFit`,
`HurdleNBFit`, `ZIPFit`, `ZIPCovFit`, `ZINBFit`, `ZINBCovFit`, `ZIBFit`). `Y` is the same
response matrix passed to the fitter; it is needed to reconstruct the marginal
likelihood. For `BinomialFit` / `BetaBinomialFit` (and a `Binomial`-family
`RowRandomFit`) supply the trial counts via `N` (default all-ones). For
`ZIPCovFit` / `ZINBCovFit` (and other covariate fits) supply the design via `X`. For
`MultinomialFit` with covariates, `X` is the same n×p matrix passed to the fitter
(not the 3-array `GllvmCovFit` layout). For
response-mask fits, pass the same Boolean `mask` matrix used by the fitter;
masked cells are ignored by the likelihood and by bootstrap refits.
`StudentTFit` admits Wald only when `ν` was pinned at fit time
(`estimated_nu == false`); free / estimated ν remains a contract §6 holdout.

Term names are `beta[t]` / `Lambda[i,k]` (+ a dispersion `r`/`phi`/`alpha`/`sigma`) for
the GLM families, `beta[k]` / `gamma[k,j]` (category indices `k≥2`) for
`MultinomialFit`, `... + phi` for `BetaBinomialFit` (the Beta precision) and
`... + sigma_row (+ r/phi/alpha)` for `RowRandomFit`, and `betaz[t]` (occurrence / zero-inflation logits) / `betac[t]`
(positive / count intercepts) / `Lambda[i,k]` (+ `sigma`/`alpha`/`r`) for the
two-part families. `ZIPCovFit` / `ZINBCovFit` add free dual slopes `gammaz[k]` / `gammac[k]`
between the intercept blocks (`ZINBCovFit` also reports shared scalar `r` on the
`:log` scale). For `TweedieFit` the dispersion term is `phi`; the
power `p ∈ (1,2)` is held fixed at its fitted value, so only `phi` is profiled.

`method` selects the inference:

  - `:wald`      — observed-information Wald intervals. The Hessian of the
                   negative Laplace log-likelihood is formed by central finite
                   differences at the MLE, inverted for the asymptotic
                   covariance. Returns an extra `pd_hessian::Bool`.
  - `:profile`   — profile-likelihood intervals: invert `D(c)=2(ℓ̂−ℓ_p(c)) ~ χ²₁`
                   by bracket-then-bisection on each side (a constrained refit
                   per candidate). Returns an extra per-term `status` vector
                   (`:profile` / `:partial` / `:failed`). `profile_iterations`,
                   `profile_g_tol`, `profile_max_expand`, and
                   `profile_max_bisect` tune the constrained refits and
                   bracketing budget without changing the default route.
  - `:bootstrap` — parametric bootstrap: simulate `n_boot` datasets from the
                   fitted model, refit each, take percentile bounds. Set
                   `parallel = true` to run replicates over `Threads.@threads`
                   (each replicate seeds its own RNG `seed + b`, so multi-core
                   and single-core give identical results). Returns an extra
                   `n_converged::Int`.

All methods return `term`, `estimate` (dispersion on its natural scale),
`lower`, `upper`, and `method`. Dispersion parameters (`r`, `phi`, `alpha`) are
parameterised on the log scale internally; their bounds are reported on the
natural (positive) scale.

`parm` subsets the terms by name: an exact name (`"beta[1]"`, `"Lambda[2,1]"`,
`"r"`), a group (`"beta"`, `"Lambda"`), or a vector of these. `N` supplies the
Binomial trial counts (default all-ones / Bernoulli).

`objective` selects which marginal the Hessian is taken from. The default
`:fit` uses the fitted AGHQ frozen-node objective for public Poisson AGHQ fits,
and the existing Laplace objective otherwise. AGHQ Wald uses automatic
second derivatives; profiles hold the final adaptation fixed, and bootstrap
refits use the stored AGHQ controls. AGHQ results identify their objective;
bootstrap additionally returns every packed replicate (NaN on failed refits)
and convergence flag. These are functional inference routes, not a coverage
validation. AGHQ rejects requests for Laplace or VA intervals to avoid changing
the estimator silently. `:laplace` explicitly selects Laplace for other fits. `:va` instead uses the negative variational (ELBO) marginal and is
available only for the scalar-μ GLM families (`PoissonFit`, `NBFit`,
`BinomialFit`, `BetaFit`, `GammaFit`) with `method = :wald`; combine it with a VA
fit (e.g. `fit_poisson_gllvm_va`) for VA-consistent standard errors. Masked
confidence intervals currently require `objective = :laplace`.

```julia
fit = fit_poisson_gllvm(Y; K = 2)
confint(fit, Y; method = :wald)
confint(fit, Y; method = :profile, parm = "beta[1]")
confint(fit, Y; method = :bootstrap, n_boot = 500, parallel = true)
```
"""
function confint(fit::_CIFit, Y::AbstractMatrix;
                 method::Symbol = :wald,
                 level::Real = 0.95,
                 parm = nothing,
                 N::Union{Nothing, AbstractMatrix} = nothing,
                 X::Union{Nothing, AbstractMatrix{<:Real}, AbstractArray{<:Real, 3}} = nothing,
                 mask = nothing,
                 n_boot::Integer = 200,
                 seed::Integer = 0,
                 parallel::Bool = false,
                 objective::Symbol = :fit,
                 newton_maxiter::Integer = 100,
                 newton_tol::Real = 1e-9,
                 profile_iterations::Integer = 200,
                 profile_g_tol::Real = 1e-4,
                 profile_max_expand::Integer = 20,
                 profile_max_bisect::Integer = 30)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    is_aghq=_is_aghq_fit(fit)
    objective===:fit && (objective=is_aghq ? :aghq : :laplace)
    is_aghq && objective!==:aghq && throw(ArgumentError("AGHQ intervals require the fitted frozen objective; use objective=:fit"))
    objective in (:laplace, :va, :aghq) && (objective!==:aghq || is_aghq) ||
        throw(ArgumentError("objective must select :fit or an available estimator; got :$objective"))
    if objective === :va && !(fit isa Union{PoissonFit, NBFit, BinomialFit, BetaFit, GammaFit, DeltaGammaFit})
        throw(ArgumentError("objective=:va is only available for Poisson/NB/Binomial/Beta/Gamma/Delta-Gamma fits"))
    end
    if objective === :va && method !== :wald
        throw(ArgumentError("objective=:va currently supports method=:wald only"))
    end
    if objective === :va && mask !== nothing
        throw(ArgumentError("objective=:va is not routed for masked confidence intervals; use objective=:laplace"))
    end
    ad = _family_ci(fit, Y; N = N, X = X, objective = objective,
                    mask = mask,
                    newton_maxiter = newton_maxiter, newton_tol = newton_tol)
    sel = _family_select(parm, ad.names)
    isempty(sel) && throw(ArgumentError("parm selector matched no parameters"))
    if method === :wald
        result=_family_wald(ad,sel,level;hessian=is_aghq ? ForwardDiff.hessian(ad.nll,ad.θ) : nothing)
        return is_aghq ? merge(result,(objective=:aghq,gradient_kind=:frozen_surrogate,)) : result
    elseif method === :profile
        result = _family_profile(ad, sel, level;
                               profile_iterations = profile_iterations,
                               profile_g_tol = profile_g_tol,
                               profile_max_expand = profile_max_expand,
                               profile_max_bisect = profile_max_bisect)
        return is_aghq ? merge(result,(objective=:aghq,gradient_kind=:frozen_surrogate,)) : result
    elseif method === :bootstrap
        result=_family_bootstrap(ad,sel,level,n_boot,seed,parallel;retain_replicates=is_aghq)
        return is_aghq ? merge(result,(objective=:aghq,gradient_kind=:frozen_surrogate,)) : result
    else
        throw(ArgumentError("method must be :wald, :profile, or :bootstrap; got :$method"))
    end
end

# ---------------------------------------------------------------------------
# Wald CIs for the SPDE-latent model. It needs the observation locations (not
# stored in the fit) to rebuild the projector, so it gets a dedicated entry
# rather than the generic confint(fit, Y) dispatch — but reuses the same Wald
# machinery (_FamilyCI + _family_wald).
# ---------------------------------------------------------------------------
"""
    confint_spde_latent(fit::SPDELatentFit, Y, locs; level=0.95, parm=nothing,
                        α=2, newton_maxiter=50, newton_tol=1e-9) -> NamedTuple

Wald confidence intervals for the SPDE-latent GLLVM. `Y` (p×M) and `locs` (M×2) must
match the fit. β and Λ are reported on their natural scale; κ, τ, and any dispersion
on the positive scale. SEs come from the observed information (finite-difference
Hessian of the θ-marginal, which rebuilds the Matérn precision each evaluation).
"""
function confint_spde_latent(fit::SPDELatentFit, Y::AbstractMatrix, locs::AbstractMatrix;
                             level::Real = 0.95, parm = nothing, α::Integer = 2,
                             newton_maxiter::Integer = 50, newton_tol::Real = 1e-9)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    p, K = size(fit.Λ); rr = rr_theta_len(p, K)
    Cdiag, G = spde_fem(fit.nodes, fit.tris)
    A = spde_projector(fit.nodes, fit.tris, locs)
    Ntr = ones(eltype(Y), size(Y))
    nd = _spde_disp_len(fit.family)
    dbase = p + rr + 2
    θ = vcat(fit.β, pack_lambda(fit.Λ), log(fit.κ), log(fit.τ),
             nd == 0 ? Float64[] : [log(fit.dispersion)])
    nll = function (θv)
        β = θv[1:p]; Λ = unpack_lambda(θv[(p + 1):(p + rr)], p, K)
        κ = exp(θv[p + rr + 1]); τ = exp(θv[p + rr + 2])
        fam = _spde_make_family(fit.family, view(θv, (dbase + 1):(dbase + nd)))
        v = try
            Q = spde_precision(Cdiag, G, κ, τ; α = α)
            -spde_latent_marginal_loglik(fam, Y, Ntr, Λ, β, fit.link, A, Q;
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    names = vcat(["beta[$t]" for t in 1:p],
                 _confint_lambda_term_names("Lambda", p, K), "kappa", "tau")
    kinds = vcat(fill(:linear, p + rr), :log, :log)
    if nd > 0
        push!(names, "dispersion"); push!(kinds, :log)
    end
    sim   = _ -> error("bootstrap is not supported for SPDE-latent CIs")
    refit = _ -> nothing
    ad = _FamilyCI(θ, nll, names, kinds, sim, refit)
    sel = _family_select(parm, ad.names)
    isempty(sel) && throw(ArgumentError("parm selector matched no parameters"))
    return _family_wald(ad, sel, level)
end

# ---------------------------------------------------------------------------
# Wald CIs for the predictor-informed latent-score trait effects B_lv = Λ·α'.
#
# The X_lv fitters (fit_*_gllvm(...; X_lv=...)) carry the packed working vector
# θ = [β; vec(α_lv); pack_lambda(Λ); (log-dispersion)] in `fit.theta_packed`,
# with objective `*_lv_nll_packed`. The user-facing estimand is the rotation-/
# sign-stable B_lv = Λ·α' — a DERIVED quantity, not a raw θ entry — so this entry
# takes the observed-information covariance Σ = inv(H) of θ̂ (finite-difference
# Hessian, as everywhere else in this file) and pushes it through the delta
# method onto B_lv: J = ∂vec(B_lv)/∂θ (finite difference of a cheap algebraic
# map), Cov(B_lv) = J Σ Jᵀ. B_lv is rotation-invariant for ANY K (Λ→ΛQ, α→αQ
# leaves Λα' fixed), so the interval is well-posed at K ≥ 1; at K = 1 it is also
# sign-identified. Profile and bootstrap paths are below.
# ---------------------------------------------------------------------------

# vec(B_lv) = vec(Λ(θ)·α_lv(θ)ᵀ) from the packed X_lv working vector.
function _lv_effects_from_packed(θ::AbstractVector, p::Integer, K::Integer, q_lv::Integer)
    rr = rr_theta_len(p, K)
    a  = reshape(θ[(p + 1):(p + q_lv * K)], q_lv, K)
    Λ  = unpack_lambda(θ[(p + q_lv * K + 1):(p + q_lv * K + rr)], p, K)
    return vec(Λ * a')
end

# Forward/central finite-difference Jacobian of a cheap vector map g: ℝᵐ → ℝⁿ.
function _fd_jacobian(g::Function, x::AbstractVector)
    f0 = g(x); m = length(x); nout = length(f0)
    J = Matrix{Float64}(undef, nout, m)
    @inbounds for j in 1:m
        h = eps()^(1 / 3) * max(abs(x[j]), 1.0)
        xp = copy(x); xp[j] += h
        xm = copy(x); xm[j] -= h
        J[:, j] = (g(xp) .- g(xm)) ./ (2h)
    end
    return J
end

# Shared post-Hessian delta-method core over the p·q_lv entries of vec(B_lv)
# (column-major: entry (t, c) at index t + (c−1)·p). `extractor(θ, p, K, q_lv)`
# maps the packed vector to vec(B_lv) (the layout differs Gaussian vs GLM).
function _lv_wald_t_unit_df(n_units::Integer, K::Integer)
    return max(Int(n_units) - Int(K) - 1, 1)
end

function _lv_wald_critical(level::Real, method::Symbol, critical_df::Union{Nothing, Integer})
    if method === :wald
        return quantile(Normal(), 0.5 + level / 2)
    elseif method === :wald_t_unit
        critical_df === nothing &&
            throw(ArgumentError("method=:wald_t_unit requires a unit-level degrees-of-freedom value"))
        return quantile(TDist(max(Int(critical_df), 1)), 0.5 + level / 2)
    end
    throw(ArgumentError("unknown LV Wald method :$method"))
end

function _lv_wald_from_hessian(H::AbstractMatrix, x::AbstractVector, p::Integer,
                               K::Integer, q_lv::Integer, level::Real, extractor;
                               method::Symbol = :wald,
                               critical_df::Union{Nothing, Integer} = nothing)
    Σ = all(isfinite, H) ? (try inv(Symmetric((H .+ H') ./ 2)) catch; nothing end) : nothing
    b̂ = extractor(x, p, K, q_lv)
    nb = length(b̂); se = fill(NaN, nb); pd = Σ !== nothing
    if pd
        J = _fd_jacobian(t -> extractor(t, p, K, q_lv), x)
        C = J * Σ * J'
        @inbounds for i in 1:nb
            v = C[i, i]
            (isfinite(v) && v > 0) && (se[i] = sqrt(v))
        end
    end
    crit = _lv_wald_critical(level, method, critical_df)
    term = ["B_lv[$t,$c]" for c in 1:q_lv for t in 1:p]
    lo = [isfinite(se[i]) ? b̂[i] - crit * se[i] : NaN for i in 1:nb]
    hi = [isfinite(se[i]) ? b̂[i] + crit * se[i] : NaN for i in 1:nb]
    return (term = term, estimate = b̂, lower = lo, upper = hi, se = se,
            level = level, method = method, pd_hessian = pd)
end

# GLM families: finite-difference observed-information Hessian of the packed
# objective (the Laplace marginal is not AD-friendly through its inner solve).
function _lv_effect_wald(nll::Function, θ::AbstractVector, p::Integer, K::Integer,
                         q_lv::Integer, level::Real)
    x = collect(Float64, θ)
    safenll = function (v)
        val = try nll(v) catch; return 1e12 end
        return isfinite(val) ? val : 1e12
    end
    return _lv_wald_from_hessian(_fd_hessian(safenll, x), x, p, K, q_lv, level,
                                 _lv_effects_from_packed)
end

# Gaussian packed layout is [β(q); vec(α_lv); log σ; pack_lambda(Λ)] — α_lv FIRST,
# then a scalar log σ, then Λ; no per-trait β for the centred unit-tier X_lv fit.
function _lv_effects_from_packed_gaussian(θ::AbstractVector, p::Integer, K::Integer, q_lv::Integer)
    rr = rr_theta_len(p, K)
    a  = reshape(θ[1:(q_lv * K)], q_lv, K)
    Λ  = unpack_lambda(θ[(q_lv * K + 2):(q_lv * K + 1 + rr)], p, K)   # skip log σ
    return vec(Λ * a')
end

# Per-family packed-objective closure for the observed-information Hessian.
_lv_packed_nll(fit::PoissonFit, Y, X_lv, q_lv, N) =
    θ -> poisson_lv_nll_packed(θ, Y, size(fit.Λ, 1), size(fit.Λ, 2), fit.link; X_lv = X_lv, q_lv = q_lv)
function _lv_packed_nll(fit::BinomialFit, Y, X_lv, q_lv, N)
    Nm = N === nothing ? fill(1, size(Y, 1), size(Y, 2)) : Matrix{Int}(N)
    return θ -> binomial_lv_nll_packed(θ, Y, Nm, size(fit.Λ, 1), size(fit.Λ, 2), fit.link;
                                       X_lv = X_lv, q_lv = q_lv)
end
_lv_packed_nll(fit::NBFit, Y, X_lv, q_lv, N) =
    θ -> nb_lv_nll_packed(θ, Y, size(fit.Λ, 1), size(fit.Λ, 2), fit.link; X_lv = X_lv, q_lv = q_lv)
_lv_packed_nll(fit::GammaFit, Y, X_lv, q_lv, N) =
    θ -> gamma_lv_nll_packed(θ, Y, size(fit.Λ, 1), size(fit.Λ, 2), fit.link; X_lv = X_lv, q_lv = q_lv)
_lv_packed_nll(fit::BetaFit, Y, X_lv, q_lv, N) =
    θ -> beta_lv_nll_packed(θ, Y, size(fit.Λ, 1), size(fit.Λ, 2), fit.link; X_lv = X_lv, q_lv = q_lv)

function _lv_effects_from_packed_ordinal(θ::AbstractVector, p::Integer, K::Integer,
                                         q_lv::Integer)
    rr = rr_theta_len(p, K)
    a = reshape(θ[1:(q_lv * K)], q_lv, K)
    Λ = unpack_lambda(θ[(q_lv * K + 1):(q_lv * K + rr)], p, K)
    return vec(Λ * a')
end

_lv_packed_nll(fit::OrdinalFit, Y, X_lv, q_lv, N) =
    θ -> ordinal_lv_nll_packed(θ, Y, size(fit.Λ, 1), size(fit.Λ, 2), fit.link, fit.C;
                               X_lv = X_lv, q_lv = q_lv)

# Profile-likelihood CIs for each entry of vec(B_lv). For entry `idx` the profile
# deviance D(c) = 2[ℓ_constrained(B_lv[idx]=c) − ℓ̂] is inverted against the χ²₁
# cutoff: CI = {c : D(c) ≤ qchisq(level, 1)}. The constraint B_lv[idx]=c is imposed
# by an escalating quadratic penalty and ALL OTHER parameters are RE-MAXIMISED at
# each c — a genuine profile, NOT the nuisance-fixed "estimated likelihood" (ELR)
# shortcut, which under-covers when the target correlates with the nuisances.
# `wald_se` (from a prior Wald pass) sets the per-entry stepping scale. `ad=true`
# (Gaussian, AD-friendly objective) uses LBFGS; GLM Laplace objectives are not
# AD-friendly through the inner solve, so `ad=false` uses derivative-free
# NelderMead. NOTE: χ²₁ is the interior asymptotic reference; the boundary
# chi-bar-square correction (variance→0, |ρ|→1, loading→0) is a separate,
# not-yet-implemented refinement.
function _lv_profile_entry_indices(indices, nb::Integer)
    if indices === nothing
        return collect(1:nb)
    end
    idx = collect(Int, indices)
    isempty(idx) && throw(ArgumentError("profile indices must not be empty"))
    for i in idx
        1 <= i <= nb || throw(ArgumentError("profile index $i outside 1:$nb"))
    end
    length(unique(idx)) == length(idx) ||
        throw(ArgumentError("profile indices must be unique"))
    return idx
end

function _profile_positive_integer(name::AbstractString, value::Integer)
    value > 0 || throw(ArgumentError("$name must be positive; got $value"))
    return Int(value)
end

function _profile_positive_real(name::AbstractString, value::Real)
    isfinite(value) && value > 0 ||
        throw(ArgumentError("$name must be positive and finite; got $value"))
    return float(value)
end

function _lv_effect_profile(nll::Function, x̂::AbstractVector, p::Integer, K::Integer,
                            q_lv::Integer, level::Real, extractor, wald_se::AbstractVector;
                            ad::Bool = false, maxstep::Integer = 40,
                            indices = nothing,
                            profile_indices = nothing,
                            profile_iterations::Union{Nothing, Integer} = nothing,
                            profile_g_tol::Real = 1e-8,
                            profile_max_expand::Union{Nothing, Integer} = nothing,
                            profile_max_bisect::Integer = 40)
    x  = collect(Float64, x̂)
    b̂  = extractor(x, p, K, q_lv)
    nb = length(b̂)
    ℓ0 = nll(x)
    cutoff = quantile(Chisq(1), level)
    selected = profile_indices === nothing ? indices : profile_indices
    profile_idx = _lv_profile_entry_indices(selected, nb)
    iterations_eff = profile_iterations === nothing ? (ad ? 1000 : 3000) :
                     _profile_positive_integer("profile_iterations", profile_iterations)
    profile_g_tol_eff = _profile_positive_real("profile_g_tol", profile_g_tol)
    profile_max_expand_eff = _profile_positive_integer(
        "profile_max_expand", profile_max_expand === nothing ? maxstep : profile_max_expand)
    profile_max_bisect_eff = _profile_positive_integer("profile_max_bisect", profile_max_bisect)
    term_all = ["B_lv[$t,$c]" for c in 1:q_lv for t in 1:p]
    lo = fill(NaN, length(profile_idx)); hi = fill(NaN, length(profile_idx))

    # Constrained re-optimisation at B_lv[idx] = c → unpenalised deviance 2(ℓc − ℓ̂).
    constrained_dev = function (idx, c, start)
        g = θ -> extractor(θ, p, K, q_lv)[idx]
        θc = copy(start)
        for w in (1e2, 1e3, 1e4, 1e5, 1e6)
            obj = function (θ)
                val = try nll(θ) catch; return 1e12 end
                isfinite(val) || return 1e12
                return val + 0.5 * w * (g(θ) - c)^2
            end
            res = try
                if ad
                    Optim.optimize(obj, θc, Optim.LBFGS(),
                                   Optim.Options(g_tol = profile_g_tol_eff,
                                                 iterations = iterations_eff);
                                   autodiff = :forward)
                else
                    # GLM Laplace objective is not AD-friendly through the inner
                    # solve, so use derivative-free NelderMead. This re-optimises
                    # the marginal at every grid point and is EXPENSIVE — GLM
                    # profile is best reserved for small problems or a few entries;
                    # Wald and bootstrap are the practical GLM defaults.
                    Optim.optimize(obj, θc, Optim.NelderMead(),
                                   Optim.Options(iterations = iterations_eff))
                end
            catch
                nothing
            end
            res === nothing && break
            θc = Optim.minimizer(res)
        end
        return 2 * (nll(θc) - ℓ0), θc
    end

    # Step out in SE units to bracket the D = cutoff crossing on side `dir` (±1),
    # then bisect. Returns NaN if the profile does not close within `maxstep`.
    crossing = function (idx, dir, s)
        c0 = b̂[idx]; clo = c0; chi = NaN
        θlo = copy(x)
        θhi = copy(x)
        for k in 1:profile_max_expand_eff
            c = c0 + dir * s * k
            D, θc = constrained_dev(idx, c, θlo)
            if isfinite(D) && D >= cutoff
                chi = c; θhi = θc; break
            end
            clo = c
            θlo = θc
        end
        isnan(chi) && return NaN
        for _ in 1:profile_max_bisect_eff
            cm = (clo + chi) / 2
            start = abs(cm - clo) <= abs(chi - cm) ? θlo : θhi
            Dm, θm = constrained_dev(idx, cm, start)
            if isfinite(Dm) && Dm >= cutoff
                chi = cm
                θhi = θm
            else
                clo = cm
                θlo = θm
            end
            abs(chi - clo) < 1e-6 * max(1.0, abs(c0)) && break
        end
        return (clo + chi) / 2
    end

    for (j, idx) in pairs(profile_idx)
        s = (isfinite(wald_se[idx]) && wald_se[idx] > 0) ? wald_se[idx] :
            max(0.1, 0.1 * abs(b̂[idx]))
        lo[j] = crossing(idx, -1, s)
        hi[j] = crossing(idx, +1, s)
    end
    return (term = term_all[profile_idx], estimate = b̂[profile_idx],
            lower = lo, upper = hi, se = fill(NaN, length(profile_idx)),
            level = level, method = :profile, pd_hessian = true)
end

"""
    confint_lv_effects(fit, Y, X_lv; N=nothing, level=0.95) -> NamedTuple

Wald confidence intervals for the predictor-informed latent-score trait-effect
matrix `B_lv = Λ·α'` of an `X_lv` fit (`fit_*_gllvm(...; X_lv=...)` for Poisson,
Binomial, NB2, Gamma, or Beta). `Y` and `X_lv` must match the fit; `N` is the
binomial trial-count matrix. SEs come from the observed-information covariance of
the packed MLE pushed through the delta method onto `B_lv`, returning
`(term, estimate, lower, upper, se, level, method, pd_hessian)` over the `p·q_lv`
entries of `vec(B_lv)`. `method = :wald` (delta method), `:profile` (invert the
likelihood-ratio statistic by constrained refit — asymmetry- and
boundary-respecting; `se` is `NaN` since the interval need not be symmetric), or
`:bootstrap` (percentiles of `B_lv`). For `method = :profile`,
`profile_indices` selects entries of `vec(B_lv)` in column-major order; `nothing`
profiles every entry. Bootstrap refits use each family's default optimiser
iteration cap unless `bootstrap_iterations` is supplied.
Admitted for `K ≥ 1` (`B_lv` is rotation-invariant), complete responses, single
ordinary latent block. Shared-cutpoint Julia-side `OrdinalFit` is admitted
through its own method; per-trait ordinal bridge CIs stay gated. Every other
structure (masks, `X` + `X_lv`, mixed-family, W-tier,
phylo/animal/spatial/kernel sources) stays gated.
"""
function confint_lv_effects(fit::Union{PoissonFit, BinomialFit, NBFit, GammaFit, BetaFit},
                            Y::AbstractMatrix, X_lv::AbstractMatrix;
                            N::Union{Nothing, AbstractMatrix} = nothing, level::Real = 0.95,
                            method::Symbol = :wald, n_boot::Integer = 200,
                            seed::Integer = 0,
                            bootstrap_iterations::Union{Nothing, Integer} = nothing,
                            profile_indices = nothing,
                            profile_iterations::Union{Nothing, Integer} = nothing,
                            profile_g_tol::Real = 1e-8,
                            profile_max_expand::Union{Nothing, Integer} = nothing,
                            profile_max_bisect::Integer = 40)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    fit.alpha_lv === nothing && throw(ArgumentError(
        "confint_lv_effects requires an X_lv fit (fit_*_gllvm(...; X_lv=...)); this fit has none"))
    p, K = size(fit.Λ)
    # B_lv = Λ·α' is invariant under the K×K orthogonal rotation Λ→ΛQ, α→αQ, so
    # the interval is well-posed for any K (not just K = 1).
    K >= 1 || throw(ArgumentError("confint_lv_effects requires K >= 1; got K = $K"))
    q_lv = size(X_lv, 2)
    size(X_lv, 1) == size(Y, 2) || throw(ArgumentError(
        "X_lv must have one row per site: got $(size(X_lv, 1)), need $(size(Y, 2))"))
    size(fit.alpha_lv) == (q_lv, K) || throw(ArgumentError(
        "X_lv has $(q_lv) column(s) but the fit carries α_lv of size $(size(fit.alpha_lv))"))
    method in (:wald, :bootstrap, :profile) ||
        throw(ArgumentError("method must be :wald, :bootstrap, or :profile; got :$method"))
    profile_indices === nothing || method === :profile ||
        throw(ArgumentError("profile_indices is only supported with method=:profile"))
    method === :bootstrap &&
        return _lv_bootstrap(fit, Y, X_lv, N, q_lv, level, n_boot, seed;
                             bootstrap_iterations = bootstrap_iterations)
    nll = _lv_packed_nll(fit, Y, X_lv, q_lv, N)
    if method === :profile
        wse = _lv_effect_wald(nll, fit.theta_packed, p, K, q_lv, level).se
        return _lv_effect_profile(nll, fit.theta_packed, p, K, q_lv, level,
                                  _lv_effects_from_packed, wse; ad = false,
                                  profile_indices = profile_indices,
                                  profile_iterations = profile_iterations,
                                  profile_g_tol = profile_g_tol,
                                  profile_max_expand = profile_max_expand,
                                  profile_max_bisect = profile_max_bisect)
    end
    return _lv_effect_wald(nll, fit.theta_packed, p, K, q_lv, level)
end

function confint_lv_effects(fit::OrdinalFit, Y::AbstractMatrix, X_lv::AbstractMatrix;
                            N::Union{Nothing, AbstractMatrix} = nothing, level::Real = 0.95,
                            method::Symbol = :wald, n_boot::Integer = 200,
                            seed::Integer = 0,
                            bootstrap_iterations::Union{Nothing, Integer} = nothing,
                            profile_indices = nothing,
                            profile_iterations::Union{Nothing, Integer} = nothing,
                            profile_g_tol::Real = 1e-8,
                            profile_max_expand::Union{Nothing, Integer} = nothing,
                            profile_max_bisect::Integer = 40)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    fit.alpha_lv === nothing && throw(ArgumentError(
        "confint_lv_effects requires an X_lv fit (fit_ordinal_gllvm(...; X_lv=...)); this fit has none"))
    p, K = size(fit.Λ)
    K >= 1 || throw(ArgumentError("confint_lv_effects requires K >= 1; got K = $K"))
    q_lv = size(X_lv, 2)
    size(X_lv, 1) == size(Y, 2) || throw(ArgumentError(
        "X_lv must have one row per site: got $(size(X_lv, 1)), need $(size(Y, 2))"))
    size(fit.alpha_lv) == (q_lv, K) || throw(ArgumentError(
        "X_lv has $(q_lv) column(s) but the fit carries α_lv of size $(size(fit.alpha_lv))"))
    method in (:wald, :bootstrap, :profile) ||
        throw(ArgumentError("method must be :wald, :bootstrap, or :profile; got :$method"))
    profile_indices === nothing || method === :profile ||
        throw(ArgumentError("profile_indices is only supported with method=:profile"))
    method === :bootstrap &&
        return _lv_bootstrap(fit, Y, X_lv, N, q_lv, level, n_boot, seed;
                             bootstrap_iterations = bootstrap_iterations)
    nll = _lv_packed_nll(fit, Y, X_lv, q_lv, N)
    x = collect(Float64, fit.theta_packed)
    safenll = function (v)
        val = try nll(v) catch; return 1e12 end
        return isfinite(val) ? val : 1e12
    end
    H = _fd_hessian(safenll, x)
    if method === :profile
        wse = _lv_wald_from_hessian(H, x, p, K, q_lv, level,
                                    _lv_effects_from_packed_ordinal).se
        return _lv_effect_profile(nll, x, p, K, q_lv, level,
                                  _lv_effects_from_packed_ordinal, wse; ad = false,
                                  profile_indices = profile_indices,
                                  profile_iterations = profile_iterations,
                                  profile_g_tol = profile_g_tol,
                                  profile_max_expand = profile_max_expand,
                                  profile_max_bisect = profile_max_bisect)
    end
    return _lv_wald_from_hessian(H, x, p, K, q_lv, level,
                                 _lv_effects_from_packed_ordinal)
end

"""
    confint_lv_effects(fit::GllvmFit, Y, X_lv; level=0.95) -> NamedTuple

Wald intervals for `B_lv = Λ·α'` of a Gaussian `X_lv` fit
(`fit_gaussian_gllvm(...; X_lv=...)`). The Gaussian marginal is closed-form, so
the observed information is the **exact ForwardDiff Hessian** of
`gaussian_lv_nll_packed` at the packed MLE (no finite differencing). `method =
:wald`, `:wald_t_unit` (same delta-method SE with a unit-df t critical,
`df = max(n_sites - K - 1, 1)`), `:profile` (LR inversion via constrained refit;
the Gaussian objective is AD-friendly so the constrained refits use LBFGS), or
`:bootstrap`. For `method = :profile`, `profile_indices` selects entries of
`vec(B_lv)` in column-major order; `nothing` profiles every entry. Bootstrap
refits use the Gaussian fitter's default optimiser iteration cap unless
`bootstrap_iterations` is supplied. `K ≥ 1` (`B_lv` rotation-invariant),
complete responses, single unit-tier latent block, no fixed-effect `X`.
"""
function confint_lv_effects(fit::GllvmFit, Y::AbstractMatrix, X_lv::AbstractMatrix;
                            N::Union{Nothing, AbstractMatrix} = nothing, level::Real = 0.95,
                            method::Symbol = :wald, n_boot::Integer = 200,
                            seed::Integer = 0,
                            bootstrap_iterations::Union{Nothing, Integer} = nothing,
                            profile_indices = nothing,
                            profile_iterations::Union{Nothing, Integer} = nothing,
                            profile_g_tol::Real = 1e-8,
                            profile_max_expand::Union{Nothing, Integer} = nothing,
                            profile_max_bisect::Integer = 40)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    fit.pars.alpha_lv === nothing && throw(ArgumentError(
        "confint_lv_effects requires a Gaussian X_lv fit (fit_gaussian_gllvm(...; X_lv=...)); this fit has none"))
    isempty(fit.pars.β) || throw(ArgumentError(
        "confint_lv_effects supports X_lv-only Gaussian fits (no fixed-effect X); got q = $(length(fit.pars.β))"))
    p, K = size(fit.pars.Λ)
    # B_lv = Λ·α' is invariant under the K×K orthogonal rotation Λ→ΛQ, α→αQ, so
    # the interval is well-posed for any K (not just K = 1).
    K >= 1 || throw(ArgumentError("confint_lv_effects requires K >= 1; got K = $K"))
    q_lv = size(X_lv, 2)
    size(X_lv, 1) == size(Y, 2) || throw(ArgumentError(
        "X_lv must have one row per site: got $(size(X_lv, 1)), need $(size(Y, 2))"))
    size(fit.pars.alpha_lv) == (q_lv, K) || throw(ArgumentError(
        "X_lv has $(q_lv) column(s) but the fit carries α_lv of size $(size(fit.pars.alpha_lv))"))
    method in (:wald, :wald_t_unit, :bootstrap, :profile) ||
        throw(ArgumentError("method must be :wald, :wald_t_unit, :bootstrap, or :profile; got :$method"))
    profile_indices === nothing || method === :profile ||
        throw(ArgumentError("profile_indices is only supported with method=:profile"))
    method === :bootstrap &&
        return _lv_bootstrap(fit, Y, X_lv, N, q_lv, level, n_boot, seed;
                             bootstrap_iterations = bootstrap_iterations)
    x = collect(Float64, fit.pars.θ_packed)
    # Model A: carry any phylo block so the observed-information Hessian is built
    # on the SAME augmented objective the fit used. The B_lv extractor is unchanged
    # (the phylo tail is appended after Λ_B), and the delta method correctly uses
    # only the α/Λ block of Σ = inv(H), inflated by the phylo params being estimated.
    K_phy = fit.model.K_phy
    has_phy_unique = fit.model.has_phy_unique
    Σ_phy = hasproperty(fit.pars, :Σ_phy) ? fit.pars.Σ_phy : nothing
    nll = θv -> gaussian_lv_nll_packed(θv, Y, p, K; X_lv = X_lv, q_lv = q_lv,
                                       K_phy = K_phy, has_phy_unique = has_phy_unique,
                                       Σ_phy = Σ_phy)
    H = try
        ForwardDiff.hessian(nll, x)
    catch
        safenll = function (v)
            val = try nll(v) catch; return 1e12 end
            return isfinite(val) ? val : 1e12
        end
        _fd_hessian(safenll, x)
    end
    if method === :profile
        wse = _lv_wald_from_hessian(H, x, p, K, q_lv, level, _lv_effects_from_packed_gaussian).se
        return _lv_effect_profile(nll, x, p, K, q_lv, level,
                                  _lv_effects_from_packed_gaussian, wse; ad = true,
                                  profile_indices = profile_indices,
                                  profile_iterations = profile_iterations,
                                  profile_g_tol = profile_g_tol,
                                  profile_max_expand = profile_max_expand,
                                  profile_max_bisect = profile_max_bisect)
    end
    df = method === :wald_t_unit ? _lv_wald_t_unit_df(size(X_lv, 1), K) : nothing
    return _lv_wald_from_hessian(H, x, p, K, q_lv, level, _lv_effects_from_packed_gaussian;
                                 method = method, critical_df = df)
end

# ---------------------------------------------------------------------------
# Parametric bootstrap for B_lv. Percentiles of the DERIVED B_lv = Λ·α' across
# refits of simulate(fit; X_lv) draws — a useful complement to Wald because B_lv
# is a product of parameters whose finite-sample distribution can be skewed. Per
# family: (simfn(rng) -> Y^b, refitfn(Y^b) -> fit^b or nothing).
# ---------------------------------------------------------------------------
function _lv_boot_kwargs(bootstrap_iterations::Union{Nothing, Integer})
    bootstrap_iterations === nothing && return NamedTuple()
    n = Int(bootstrap_iterations)
    n > 0 || throw(ArgumentError("bootstrap_iterations must be positive; got $bootstrap_iterations"))
    return (; iterations = n)
end

function _lv_boot_fns(fit::PoissonFit, Y, X_lv, N,
                      bootstrap_iterations::Union{Nothing, Integer})
    K = size(fit.Λ, 2); n = size(Y, 2); link = fit.link
    boot_kwargs = _lv_boot_kwargs(bootstrap_iterations)
    return (rng -> simulate(fit, n; X_lv = X_lv, rng = rng),
            Yb -> (try fit_poisson_gllvm(Yb; K = K, link = link, X_lv = X_lv,
                                         boot_kwargs...) catch; nothing end))
end
function _lv_boot_fns(fit::BinomialFit, Y, X_lv, N,
                      bootstrap_iterations::Union{Nothing, Integer})
    K = size(fit.Λ, 2); n = size(Y, 2); link = fit.link
    Nm = N === nothing ? fill(1, size(Y, 1), n) : Matrix{Int}(N)
    boot_kwargs = _lv_boot_kwargs(bootstrap_iterations)
    return (rng -> simulate(fit, n; N = Nm, X_lv = X_lv, rng = rng),
            Yb -> (try fit_binomial_gllvm(Yb; K = K, N = Nm, link = link, X_lv = X_lv,
                                          boot_kwargs...) catch; nothing end))
end
function _lv_boot_fns(fit::NBFit, Y, X_lv, N,
                      bootstrap_iterations::Union{Nothing, Integer})
    K = size(fit.Λ, 2); n = size(Y, 2); link = fit.link
    boot_kwargs = _lv_boot_kwargs(bootstrap_iterations)
    return (rng -> simulate(fit, n; X_lv = X_lv, rng = rng),
            Yb -> (try fit_nb_gllvm(Yb; K = K, link = link, X_lv = X_lv,
                                    boot_kwargs...) catch; nothing end))
end
function _lv_boot_fns(fit::GammaFit, Y, X_lv, N,
                      bootstrap_iterations::Union{Nothing, Integer})
    K = size(fit.Λ, 2); n = size(Y, 2); link = fit.link
    boot_kwargs = _lv_boot_kwargs(bootstrap_iterations)
    return (rng -> simulate(fit, n; X_lv = X_lv, rng = rng),
            Yb -> (try fit_gamma_gllvm(Yb; K = K, link = link, X_lv = X_lv,
                                       boot_kwargs...) catch; nothing end))
end
function _lv_boot_fns(fit::BetaFit, Y, X_lv, N,
                      bootstrap_iterations::Union{Nothing, Integer})
    K = size(fit.Λ, 2); n = size(Y, 2); link = fit.link
    boot_kwargs = _lv_boot_kwargs(bootstrap_iterations)
    return (rng -> simulate(fit, n; X_lv = X_lv, rng = rng),
            Yb -> (try fit_beta_gllvm(Yb; K = K, link = link, X_lv = X_lv,
                                      boot_kwargs...) catch; nothing end))
end
function _lv_boot_fns(fit::OrdinalFit, Y, X_lv, N,
                      bootstrap_iterations::Union{Nothing, Integer})
    K = size(fit.Λ, 2); n = size(Y, 2); link = fit.link
    boot_kwargs = _lv_boot_kwargs(bootstrap_iterations)
    return (rng -> simulate(fit, n; X_lv = X_lv, rng = rng),
            Yb -> (try fit_ordinal_gllvm(Yb; K = K, link = link, X_lv = X_lv,
                                         boot_kwargs...) catch; nothing end))
end
function _lv_boot_fns(fit::GllvmFit, Y, X_lv, N,
                      bootstrap_iterations::Union{Nothing, Integer})
    p, K = size(fit.pars.Λ); n = size(Y, 2)
    boot_kwargs = _lv_boot_kwargs(bootstrap_iterations)
    Λ = fit.pars.Λ; Zmean = X_lv * fit.pars.alpha_lv; σ = fit.pars.σ_eps  # n×K score mean (α_lv is q_lv×K)
    # Model A: simulate + refit must carry the phylo block. φ ~ N(0, B) is the
    # shared species random effect (B = (Λ_phy_aug Λ_phy_aug') .* Σ_phy); the refit
    # must re-estimate the same phylo structure. No-phylo fits → all nothing/0 → unchanged.
    Λ_phy = fit.pars.Λ_phy; σ_phy = fit.pars.σ_phy
    Σ_phy = hasproperty(fit.pars, :Σ_phy) ? fit.pars.Σ_phy : nothing
    K_phy = Λ_phy === nothing ? 0 : size(Λ_phy, 2)
    has_phy_unique = σ_phy !== nothing
    L_phy = if Σ_phy !== nothing && (Λ_phy !== nothing || σ_phy !== nothing)
        aug = (Λ_phy !== nothing && σ_phy !== nothing) ? hcat(Λ_phy, σ_phy) :
              (Λ_phy !== nothing ? Λ_phy : reshape(σ_phy, p, 1))
        cholesky(Symmetric((aug * aug') .* Σ_phy + 1e-10 * I)).L
    else
        nothing
    end
    simfn = function (rng)
        φ = L_phy === nothing ? zeros(p) : L_phy * randn(rng, p)
        Λ * (Zmean .+ randn(rng, n, K))' .+ φ .+ σ .* randn(rng, p, n)
    end
    refitfn = Yb -> (try fit_gaussian_gllvm(Yb; K = K, X_lv = X_lv, K_phy = K_phy,
                                            has_phy_unique = has_phy_unique, Σ_phy = Σ_phy,
                                            boot_kwargs...)
                     catch; nothing end)
    return (simfn, refitfn)
end

function _lv_bootstrap(fit, Y, X_lv, N, q_lv::Integer, level::Real,
                       n_boot::Integer, seed::Integer;
                       bootstrap_iterations::Union{Nothing, Integer} = nothing)
    simfn, refitfn = _lv_boot_fns(fit, Y, X_lv, N, bootstrap_iterations)
    b̂ = vec(extract_lv_effects(fit)); nb = length(b̂); p = nb ÷ q_lv
    reps = Vector{Vector{Float64}}()
    for b in 1:n_boot
        rng = MersenneTwister(seed + b)
        Bb = try
            fb = refitfn(simfn(rng))
            fb === nothing ? nothing : vec(extract_lv_effects(fb))
        catch
            nothing
        end
        (Bb === nothing || length(Bb) != nb || any(!isfinite, Bb)) && continue
        # B_lv = Lambda * alpha' is already rotation/sign stable; flipping
        # bootstrap replicates would hide failed refits instead of diagnosing them.
        push!(reps, Bb)
    end
    nconv = length(reps); a = (1 - level) / 2
    lo = fill(NaN, nb); hi = fill(NaN, nb)
    if nconv >= 10
        M = reduce(hcat, reps)    # nb × nconv
        @inbounds for i in 1:nb
            lo[i] = quantile(view(M, i, :), a); hi[i] = quantile(view(M, i, :), 1 - a)
        end
    end
    term = ["B_lv[$t,$c]" for c in 1:q_lv for t in 1:p]
    return (term = term, estimate = b̂, lower = lo, upper = hi,
            level = level, method = :bootstrap, n_converged = nconv)
end
