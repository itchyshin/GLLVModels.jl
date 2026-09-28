# Parametric bootstrap confidence intervals for the Gaussian GLLVM.
#
# Sample y_b ~ N(μ̂, Σ̂_y) for b = 1..n_boot, refit each, then take
# percentiles of the resulting parameter (or derived-quantity) distributions.
#
# Why bootstrap (vs Wald in src/confint.jl and profile in src/confint_profile.jl):
# For derived quantities (Σ_y entries, communality, ICC, H²) neither Wald
# (assumes Gaussian on the working scale) nor profile (per-parameter only)
# is ideal. Parametric bootstrap is the gold standard for arbitrary
# derived quantities. This file is a NEW addition alongside src/confint.jl
# and src/confint_profile.jl; sister agents PERF+F (Wald) and PERF+H
# (profile) own those files and the integration agent will dispatch the
# three methods through a unified `confint(fit; method = ...)` later.
#
# Required for the ADEMP simulation derived-quantity coverage gates
# (Σ_y entries, communality) and cross-check vs R-side
# `extract_Sigma_bootstrap`. Cost: n_boot fits at per-fit time — practical
# at n_boot = 100 for moderate fixture sizes.
#
# Active plan: ~/.claude/plans/please-have-a-robust-elephant.md

using Random
using Statistics
using LinearAlgebra

# ---------------------------------------------------------------------------
# Term-name builder (legacy θ_packed layout).
#
# Kept local to this file so PERF+I does not have to touch src/confint.jl
# (owned by PERF+F). The integration agent will deduplicate this with the
# version in src/confint.jl after all three CI methods land.
# ---------------------------------------------------------------------------

# Λ packing order (see src/packing.jl, _lower_index): diagonals k = 1..K
# first, then strict-lower column-by-column.
function _bootstrap_lambda_term_names(prefix::String, p::Integer, K::Integer)
    out = String[]
    for k in 1:K
        push!(out, "$(prefix)[$k,$k]")
    end
    for k in 1:K
        for i in (k + 1):p
            push!(out, "$(prefix)[$i,$k]")
        end
    end
    return out
end

# Names in θ_packed order; mirrors _all_term_names in src/confint.jl.
function _bootstrap_term_names(fit::GllvmFit)
    model = fit.model
    p   = model.p
    K_B = model.K
    K_W = model.K_W
    has_diag = model.has_diag
    K_phy    = model.K_phy
    has_phy_unique = model.has_phy_unique
    q = fit.pars.β === nothing ? 0 : length(fit.pars.β)
    β_fixed = _pars_fixed_mask(fit.pars, :β_fixed, q)
    β_free = _free_coeff_indices(β_fixed)

    terms = String[]

    for j in β_free
        push!(terms, "beta[$j]")
    end

    push!(terms, "sigma_eps")

    if has_diag
        for t in 1:p
            push!(terms, "sigma_B[$t]")
        end
        for t in 1:p
            push!(terms, "sigma_W[$t]")
        end
    end

    append!(terms, _bootstrap_lambda_term_names("Lambda_B", p, K_B))

    if K_W > 0
        append!(terms, _bootstrap_lambda_term_names("Lambda_W", p, K_W))
    end

    if has_phy_unique
        for t in 1:p
            push!(terms, "sigma_phy[$t]")
        end
    end

    if K_phy > 0
        append!(terms, _bootstrap_lambda_term_names("Lambda_phy", p, K_phy))
    end

    return terms
end

# ---------------------------------------------------------------------------
# Per-(t, s) mean from fixed effects.
#
# μ̂[t, s] = sum_k X[t, s, k] * β̂[k]. With no X the mean is zero. *Random*
# contributions (phy block, W tier, diag RE) are NOT part of μ̂ — they are
# realised per replicate via parametric simulation.
# ---------------------------------------------------------------------------
function _bootstrap_mean_from_X(fit::GllvmFit, X::AbstractArray{<:Real, 3})
    p = fit.model.p
    n = size(X, 2)
    β̂ = fit.pars.β
    if β̂ === nothing || length(β̂) == 0
        return zeros(Float64, p, n)
    end
    μ = zeros(Float64, p, n)
    q = size(X, 3)
    @inbounds for s in 1:n, t in 1:p
        v = 0.0
        for k in 1:q
            v += X[t, s, k] * β̂[k]
        end
        μ[t, s] = v
    end
    return μ
end

# ---------------------------------------------------------------------------
# Per-site covariance reconstruction.
#
# A = Λ_B Λ_B' + diag(d_total)
# d_total[t] = (Λ_W Λ_W')[t,t] + σ²_B[t] + σ²_W[t] + σ²_eps
# ---------------------------------------------------------------------------
function _bootstrap_site_cov(fit::GllvmFit)
    p   = fit.model.p
    K_W = fit.model.K_W
    has_diag = fit.model.has_diag
    σ² = fit.pars.σ_eps^2
    Λ_B = fit.pars.Λ
    A = Λ_B * Λ_B'
    @inbounds for t in 1:p
        v = σ²
        if K_W > 0 && fit.pars.Λ_W !== nothing
            for k in 1:size(fit.pars.Λ_W, 2)
                v += fit.pars.Λ_W[t, k]^2
            end
        end
        if has_diag && fit.pars.σ²_B !== nothing
            v += fit.pars.σ²_B[t]
        end
        if has_diag && fit.pars.σ²_W !== nothing
            v += fit.pars.σ²_W[t]
        end
        A[t, t] += v
    end
    return A
end

# ---------------------------------------------------------------------------
# Single replicate simulation.
#
# Non-phy: y_b[:, s] iid ~ N(μ̂[:, s], A); cholesky of A is reused.
# Phy: y_b[:, s] = μ̂[:, s] + L_site * z_s + phy_contrib (species shared).
# ---------------------------------------------------------------------------
function _bootstrap_simulate!(rng::AbstractRNG, y_out::AbstractMatrix,
                              μ̂::AbstractMatrix,
                              L_site::LowerTriangular,
                              L_phy::Union{Nothing, LowerTriangular},
                              Λ_phy_aug::Union{Nothing, AbstractMatrix})
    p, n = size(y_out)

    # Per-site Gaussian noise: A^{1/2} z, z ~ N(0, I_p)
    Z = randn(rng, p, n)
    mul!(y_out, L_site, Z)

    # Phylogenetic block (species-level, shared across all sites):
    #   Each axis k: φ_k ~ MVN(0, Σ_phy); contribution Λ_phy_aug[:, k] .* φ_k
    if L_phy !== nothing && Λ_phy_aug !== nothing
        K_aug = size(Λ_phy_aug, 2)
        Φ = L_phy * randn(rng, p, K_aug)   # p × K_aug — each column ~ MVN(0, Σ_phy)
        phy_contrib = vec(sum(Λ_phy_aug .* Φ, dims = 2))   # length p, shared across sites
        @inbounds for s in 1:n, t in 1:p
            y_out[t, s] += phy_contrib[t]
        end
    end

    # Add the fixed-effect mean
    @inbounds for s in 1:n, t in 1:p
        y_out[t, s] += μ̂[t, s]
    end
    return y_out
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    bootstrap_ci(fit::GllvmFit;
                 n_boot = 100,
                 level = 0.95,
                 seed = 0,
                 y = nothing,
                 n_sites = nothing,
                 X = nothing,
                 Σ_phy = nothing,
                 parms = nothing,
                 verbose = false)
        -> NamedTuple

Parametric bootstrap CIs for the fitted parameters in `fit`. Returns a
NamedTuple with fields:

  - `term::Vector{String}`         — parameter names (θ_packed order)
  - `estimate::Vector{Float64}`    — original MLE (`fit.pars.θ_packed`)
  - `lower::Vector{Float64}`       — percentile `100·(1-level)/2`
  - `upper::Vector{Float64}`       — percentile `100·(1+level)/2`
  - `n_converged::Int`             — number of bootstrap fits that converged
  - `replicates::Matrix{Float64}`  — `n_boot × n_params` matrix of bootstrap θ̂_b

`n_sites` is required because `GllvmFit` does not record it. Supply
either `n_sites` directly, or pass the original `y` (or `X`) — the
function infers `n_sites` from `size(y, 2)` or `size(X, 2)`.

`X` and `Σ_phy` must be supplied when the original fit had fixed effects
(`q > 0`) or a phylogenetic block (`K_phy > 0` or `has_phy_unique`).
Otherwise the bootstrap model spec would not match the fitted spec.

`parms` selects a subset of returned terms (default `nothing` = all).
Accepts a `String` (single term name) or `Vector{String}`.

`n_boot` defaults to 100; publication-grade is 500–2000.

# Algorithm

1. Reconstruct Σ̂_y at the fitted parameters (`Λ̂ Λ̂' + diag(d_total)`,
   plus the phy block when present).
2. For b = 1..n_boot:
   - Simulate y_b ~ N(μ̂, Σ̂_y) using a Cholesky factor of Σ̂_y_site
     (independent across sites for J1 / J2; J3 adds species-shared
     phylogenetic contributions).
   - Refit via `fit_gaussian_gllvm(y_b; K, K_W, has_diag, K_phy,
     has_phy_unique, Σ_phy, X)` so the bootstrap model matches the
     original spec.
   - Record the resulting θ_packed and convergence flag.
3. Compute percentile CIs over the non-NaN replicates per parameter.

Replicates whose refit errors out are recorded as `NaN` (and excluded
from the percentile calculation); a parameter with fewer than 10
converged replicates returns `NaN` bounds.

# Example

```julia
fit = fit_gaussian_gllvm(y; K = 1)
ci  = bootstrap_ci(fit; y = y, n_boot = 200, seed = 42)
ci.term      # parameter names
ci.lower     # 2.5% percentile bounds
ci.upper     # 97.5% percentile bounds
```
"""
function bootstrap_ci(fit::GllvmFit;
                      n_boot::Integer = 100,
                      level::Real = 0.95,
                      seed::Integer = 0,
                      y::Union{Nothing, AbstractMatrix} = nothing,
                      n_sites::Union{Nothing, Integer} = nothing,
                      X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                      Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                      parms::Union{Nothing, AbstractString, AbstractVector} = nothing,
                      verbose::Bool = false)

    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    n_boot ≥ 1   || throw(ArgumentError("n_boot must be ≥ 1; got $n_boot"))
    _has_lv_predictor(fit) && throw(ArgumentError(
        "bootstrap_ci for fit_gaussian_gllvm(...; X_lv=...) is not admitted in the C1 predictor-informed latent-score path; use extract_lv_effects for point estimates"))

    if _has_gaussian_record(fit)
        data=y===nothing ? fit.integration.data.responses : y
        n_sites===nothing || n_sites==size(data,2) || throw(ArgumentError("Gaussian integration bootstrap preserves the fitted site count"))
        ad=_gaussian_record_ci(fit,data;X=X,Σ_phy=Σ_phy)
        # This legacy alias returns working-scale sigma, unlike confint.
        working=_FamilyCI(ad.θ,ad.nll,ad.names,fill(:linear,length(ad.θ)),ad.simulate,ad.refit)
        sel=_confint_select_indices(parms,ad.names)
        r=_family_bootstrap(working,sel,level,n_boot,seed,false;retain_replicates=true)
        return (term=r.term,estimate=r.estimate,lower=r.lower,upper=r.upper,n_converged=r.n_converged,
            replicates=r.replicates[:,sel],converged=r.converged,objective=fit.integration.actual)
    end

    model = fit.model
    p     = model.p
    K_B   = model.K
    K_W   = model.K_W
    has_diag = model.has_diag
    K_phy    = model.K_phy
    has_phy_unique = model.has_phy_unique
    has_phy_block = (K_phy > 0) || has_phy_unique

    q = fit.pars.β === nothing ? 0 : length(fit.pars.β)
    β_fixed = _pars_fixed_mask(fit.pars, :β_fixed, q)

    # ----- Determine n_sites
    n = if n_sites !== nothing
        Int(n_sites)
    elseif X !== nothing
        size(X, 2)
    elseif y !== nothing
        size(y, 2)
    else
        throw(ArgumentError(
            "bootstrap_ci needs n_sites. Pass one of: " *
            "`y = ...` (original data), `X = ...` (covariates), or `n_sites = ...` explicitly."))
    end

    if y !== nothing
        size(y, 1) == p || throw(ArgumentError(
            "y first dim ($(size(y, 1))) must equal fit.model.p ($p)"))
        size(y, 2) == n || throw(ArgumentError(
            "y has $(size(y, 2)) sites but n_sites was determined to be $n"))
    end

    X = _mean_X(fit, X, n)
    if q > 0 && X === nothing
        throw(ArgumentError(
            "The fitted model has q = $q fixed effects but X was not supplied. " *
            "Pass X = ... (same 3-D array used in fit_gaussian_gllvm) to bootstrap_ci."))
    end
    if X !== nothing
        size(X, 1) == p || throw(ArgumentError(
            "X first dim ($(size(X, 1))) must equal p ($p)"))
        size(X, 2) == n || throw(ArgumentError(
            "X second dim ($(size(X, 2))) must equal n_sites ($n)"))
        size(X, 3) == q || throw(ArgumentError(
            "X third dim ($(size(X, 3))) must equal fitted q ($q)"))
    end
    if has_phy_block && Σ_phy === nothing
        throw(ArgumentError(
            "The fitted model has a phylogenetic block but Σ_phy was not supplied. " *
            "Pass Σ_phy = ... (same p × p matrix used in fit_gaussian_gllvm) to bootstrap_ci."))
    end
    if Σ_phy !== nothing
        size(Σ_phy, 1) == p && size(Σ_phy, 2) == p || throw(ArgumentError(
            "Σ_phy must be p × p; got $(size(Σ_phy)) for p = $p"))
    end

    θ̂ = fit.pars.θ_packed
    n_params = length(θ̂)

    # ----- Reconstruct μ̂[t, s]
    μ̂ = X === nothing ? zeros(Float64, p, n) : _bootstrap_mean_from_X(fit, X)

    # ----- Reconstruct Σ̂_y_site and cholesky factor
    A = _bootstrap_site_cov(fit)
    # Symmetrise to defeat round-off-induced asymmetry before cholesky.
    A_sym = Symmetric((A + A') ./ 2)
    L_site = cholesky(A_sym).L

    # ----- Phylogenetic Cholesky (if applicable)
    L_phy = nothing
    Λ_phy_aug = nothing
    if has_phy_block
        Σ_phy_sym = Symmetric((Σ_phy + Σ_phy') ./ 2)
        L_phy = cholesky(Σ_phy_sym).L
        # Λ_phy_aug = hcat(Λ_phy, σ_phy) (each present only if its flag is on)
        pieces = AbstractMatrix{Float64}[]
        if K_phy > 0 && fit.pars.Λ_phy !== nothing
            push!(pieces, fit.pars.Λ_phy)
        end
        if has_phy_unique && fit.pars.σ_phy !== nothing
            push!(pieces, reshape(collect(Float64, fit.pars.σ_phy), p, 1))
        end
        Λ_phy_aug = isempty(pieces) ? nothing : reduce(hcat, pieces)
    end

    rng = MersenneTwister(seed)

    replicates = fill(NaN, n_boot, n_params)
    n_converged = 0

    y_b = Matrix{Float64}(undef, p, n)

    for b in 1:n_boot
        # Sample y_b ~ N(μ̂, Σ̂_y_site)
        _bootstrap_simulate!(rng, y_b, μ̂, L_site, L_phy, Λ_phy_aug)

        # Refit on y_b under the original spec.
        try
            fit_b = fit_gaussian_gllvm(y_b;
                                       K = K_B,
                                       K_W = K_W,
                                       has_diag = has_diag,
                                       K_phy = K_phy,
                                       has_phy_unique = has_phy_unique,
                                       Σ_phy = Σ_phy,
                                       X = X,
                                       β_fixed = β_fixed)
            θ_b = fit_b.pars.θ_packed
            if length(θ_b) == n_params
                replicates[b, :] = θ_b
                if fit_b.converged
                    n_converged += 1
                end
            else
                verbose && @info "Bootstrap rep $b: θ_packed length mismatch ($(length(θ_b)) vs $n_params)"
            end
        catch e
            verbose && @info "Bootstrap rep $b failed: $e"
            # replicates[b, :] stays NaN
        end
    end

    # ----- Percentile CIs
    α = (1 - level) / 2
    lower = fill(NaN, n_params)
    upper = fill(NaN, n_params)
    for j in 1:n_params
        col = filter(!isnan, replicates[:, j])
        if length(col) ≥ 10
            lower[j] = quantile(col, α)
            upper[j] = quantile(col, 1 - α)
        end
    end

    # ----- Build canonical term names
    terms = _bootstrap_term_names(fit)
    length(terms) == n_params || error(
        "Internal: term-name vector length ($(length(terms))) does not match " *
        "θ_packed length ($n_params). This is a packing layout bug.")

    # ----- Optionally subset by `parms`
    sel = if parms === nothing
        1:n_params
    elseif parms isa AbstractString
        idx = findall(==(String(parms)), terms)
        isempty(idx) ? throw(ArgumentError("parms selector \"$parms\" matched no terms")) : idx
    else
        idxs = Int[]
        for p_ in parms
            append!(idxs, findall(==(String(p_)), terms))
        end
        isempty(idxs) && throw(ArgumentError("parms selector $(parms) matched no terms"))
        idxs
    end

    return (term       = terms[sel],
            estimate   = θ̂[sel],
            lower      = lower[sel],
            upper      = upper[sel],
            n_converged = n_converged,
            replicates = replicates[:, sel])
end
