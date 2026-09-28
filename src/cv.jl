# Cross-Validation (CV) Engine for GLLVModels.jl
#
# Implements K-fold cross-validation supporting:
#   - Random entry-wise (cell-level) split
#   - Site-level block split (leave out entire site columns)
#   - Species-level block split (leave out entire species rows)
#
# Evaluates out-of-sample:
#   - Log-likelihood (conditional and marginal cell/site log-densities)
#   - Mean Squared Error (MSE)
#   - Dunn–Smyth randomized quantile residuals (approx N(0, 1) under well-specified models)
#
# References:
#   - Warton et al. 2015 (So many variables: Joint species distribution modelling)
#   - Niku et al. 2019 (gllvm: Fast analysis of multivariate abundance data)
#   - Dunn & Smyth 1996 (Randomized quantile residuals)

using Random
using LinearAlgebra
using Statistics
using Distributions
using Printf

@doc raw"""
    CVResult

Summary container holding out-of-sample cross-validation evaluation results.

Fields:
- `k_folds::Int` — number of folds ($K$)
- `split::Symbol` — split strategy (`:random`, `:site`, or `:species`)
- `loglik::Float64` — total out-of-sample log-likelihood summed across all test cells
- `mse::Float64` — overall out-of-sample Mean Squared Error
- `residual_mean::Float64` — sample mean of out-of-sample Dunn–Smyth randomized quantile residuals (target ≈ 0)
- `residual_std::Float64` — sample standard deviation of out-of-sample randomized quantile residuals (target ≈ 1)
- `fold_logliks::Vector{Float64}` — out-of-sample log-likelihood for each fold
- `fold_mses::Vector{Float64}` — out-of-sample MSE for each fold
- `fold_residual_means::Vector{Float64}` — mean of quantile residuals per fold
- `fold_residual_stds::Vector{Float64}` — standard deviation of quantile residuals per fold
- `predictions::Matrix{Float64}` — $p \times n$ matrix of out-of-sample predicted response values $\hat{\mu}$
- `residuals::Matrix{Float64}` — $p \times n$ matrix of out-of-sample Dunn–Smyth randomized quantile residuals
- `fits::Vector{Any}` — fitted model objects from each training fold
"""
struct CVResult
    k_folds::Int
    split::Symbol
    loglik::Float64
    mse::Float64
    residual_mean::Float64
    residual_std::Float64
    fold_logliks::Vector{Float64}
    fold_mses::Vector{Float64}
    fold_residual_means::Vector{Float64}
    fold_residual_stds::Vector{Float64}
    predictions::Matrix{Float64}
    residuals::Matrix{Float64}
    fits::Vector{Any}
end

# StatsAPI extractors
StatsAPI.loglikelihood(res::CVResult) = res.loglik
StatsAPI.residuals(res::CVResult; type::Symbol = :dunnsmyth) = res.residuals
StatsAPI.predict(res::CVResult) = res.predictions
StatsAPI.fitted(res::CVResult) = res.predictions

function Base.show(io::IO, ::MIME"text/plain", res::CVResult)
    println(io, "GLLVModels $(res.k_folds)-Fold Cross-Validation ($(res.split) split)")
    println(io, "  Out-of-sample logLik: ", @sprintf("%.4f", res.loglik))
    println(io, "  Out-of-sample MSE:    ", @sprintf("%.6f", res.mse))
    println(io, "  Residual Mean:        ", @sprintf("%.4f", res.residual_mean), " (target ≈ 0)")
    println(io, "  Residual Std:         ", @sprintf("%.4f", res.residual_std), " (target ≈ 1)")
    println(io, "\nFold breakdown:")
    println(io, "  Fold     logLik        MSE   Resid.Mean   Resid.Std")
    for k in 1:res.k_folds
        @printf(io, "   %3d  %10.4f  %10.6f  %10.4f  %10.4f\n",
                k, res.fold_logliks[k], res.fold_mses[k],
                res.fold_residual_means[k], res.fold_residual_stds[k])
    end
end

function Base.show(io::IO, res::CVResult)
    print(io, "CVResult($(res.k_folds)-fold $(res.split), logLik=", round(res.loglik, digits=2), ", MSE=", round(res.mse, digits=4), ")")
end

# -----------------------------------------------------------------------------
# Parameter extraction & helper dispatch
# -----------------------------------------------------------------------------

function _cv_extract_params(fit)
    if fit isa GllvmFit
        p_fit = size(fit.pars.Λ, 1)
        β = (fit.pars.β === nothing || isempty(fit.pars.β)) ? zeros(Float64, p_fit) : fit.pars.β
        return (fit.pars.Λ, β, IdentityLink())
    elseif hasfield(typeof(fit), :Λ) && hasfield(typeof(fit), :β) && hasfield(typeof(fit), :link)
        return (fit.Λ, fit.β, fit.link)
    elseif hasfield(typeof(fit), :Λ) && hasfield(typeof(fit), :β)
        return (fit.Λ, fit.β, IdentityLink())
    elseif hasfield(typeof(fit), :pars)
        Λ = haskey(fit.pars, :Λ) ? fit.pars.Λ : fit.pars.Λ_B
        p_fit = size(Λ, 1)
        β = (haskey(fit.pars, :β) && fit.pars.β !== nothing && !isempty(fit.pars.β)) ? fit.pars.β : zeros(Float64, p_fit)
        return (Λ, β, IdentityLink())
    else
        throw(ArgumentError("Cannot extract parameters from fitted model of type $(typeof(fit))"))
    end
end

# Fixed-effect mean `X[t, s, :] ⋅ β` of a Gaussian fold fit; 0 when it has no β.
function _cv_gaussian_mean(fit::GllvmFit, X, t::Int, s::Int)
    β = fit.pars.β
    (β === nothing || isempty(β)) && return 0.0
    X === nothing && _has_trait_intercepts(fit) && return β[t]   # per-trait intercepts (#519)
    return dot(view(X, t, s, :), β)
end

function _cv_site_mode(fit::GllvmFit, family::Normal, Y::AbstractMatrix, s::Int, mask_s, N_mat, train_species::Vector{Int};
                       X = nothing)
    p_fit = size(fit.pars.Λ, 1)
    K = size(fit.pars.Λ, 2)

    obs_in_train = [i for (i, t) in enumerate(train_species) if mask_s === nothing || mask_s[t]]
    isempty(obs_in_train) && return zeros(Float64, K)

    Λ_obs = fit.pars.Λ[obs_in_train, :]
    y_obs = Y[train_species[obs_in_train], s]
    Xs = X === nothing ? nothing : X[train_species, s:s, :]
    μ_obs = _fitted_mean(fit, view(Y, train_species, s:s), Xs)[obs_in_train, 1]

    σ_eps = fit.pars.σ_eps
    Ψ_obs = (σ_eps^2) * I(length(obs_in_train))
    ΨiΛ = Ψ_obs \ Λ_obs
    M = Symmetric(I(K) + Λ_obs' * ΨiΛ)
    z = M \ (ΨiΛ' * (y_obs .- μ_obs))
    return z
end

function _cv_site_mode(fit, family, Y::AbstractMatrix, s::Int, mask_s, N_mat, train_species::Vector{Int};
                       X = nothing)
    Λ, β, link = _cv_extract_params(fit)
    K = size(Λ, 2)
    p_fit = size(Λ, 1)

    obs_in_train = [i for (i, t) in enumerate(train_species) if mask_s === nothing || mask_s[t]]
    isempty(obs_in_train) && return zeros(Float64, K)

    y_sub = Y[train_species, s]
    n_sub = N_mat === nothing ? fill(1, p_fit) : N_mat[train_species, s]
    mask_sub = [mask_s === nothing ? true : mask_s[t] for t in train_species]

    return _laplace_mode(family, y_sub, n_sub, Λ, β, link; mask = mask_sub)
end

# -----------------------------------------------------------------------------
# Out-of-sample cell-level evaluation (log-likelihood + Dunn–Smyth residual)
# -----------------------------------------------------------------------------

function _cv_cell_eval(fit, family::Normal, y::Real, μ::Real, t::Int, s::Int, N_mat, rng::AbstractRNG)
    σ = if fit isa GllvmFit
        fit.pars.σ_eps
    elseif hasfield(typeof(fit), :σ_eps)
        fit.σ_eps
    elseif hasfield(typeof(fit), :pars) && haskey(fit.pars, :σ_eps)
        fit.pars.σ_eps
    else
        1.0
    end
    d = Normal(μ, max(σ, 1e-12))
    ll = logpdf(d, y)
    r = (y - μ) / max(σ, 1e-12)
    return (ll, r)
end

function _cv_cell_eval(fit, family::Poisson, y::Real, μ::Real, t::Int, s::Int, N_mat, rng::AbstractRNG)
    μ_c = max(Float64(μ), 1e-12)
    d = Poisson(μ_c)
    yi = round(Int, y)
    ll = logpdf(d, yi)
    Flo = cdf(d, yi - 1)
    Fhi = cdf(d, yi)
    u = Flo + (Fhi - Flo) * rand(rng)
    r = quantile(Normal(), clamp(u, 1e-12, 1.0 - 1e-12))
    return (ll, r)
end

function _cv_cell_eval(fit, family::Binomial, y::Real, μ::Real, t::Int, s::Int, N_mat, rng::AbstractRNG)
    μ_c = clamp(Float64(μ), 1e-12, 1.0 - 1e-12)
    N = N_mat === nothing ? 1 : Int(N_mat[t, s])
    d = Binomial(N, μ_c)
    yi = round(Int, y)
    ll = logpdf(d, yi)
    Flo = cdf(d, yi - 1)
    Fhi = cdf(d, yi)
    u = Flo + (Fhi - Flo) * rand(rng)
    r = quantile(Normal(), clamp(u, 1e-12, 1.0 - 1e-12))
    return (ll, r)
end

function _cv_cell_eval(fit, family::NegativeBinomial, y::Real, μ::Real, t::Int, s::Int, N_mat, rng::AbstractRNG)
    μ_c = max(Float64(μ), 1e-12)
    r_raw = if hasfield(typeof(fit), :r)
        fit.r
    elseif hasfield(typeof(fit), :rvec)
        fit.rvec
    else
        1.0
    end
    r_disp = (r_raw isa AbstractVector) ? Float64(r_raw[min(t, length(r_raw))]) : Float64(r_raw)
    prob = r_disp / (r_disp + μ_c)
    d = NegativeBinomial(r_disp, clamp(prob, 1e-12, 1.0 - 1e-12))
    yi = round(Int, y)
    ll = logpdf(d, yi)
    Flo = cdf(d, yi - 1)
    Fhi = cdf(d, yi)
    u = Flo + (Fhi - Flo) * rand(rng)
    r = quantile(Normal(), clamp(u, 1e-12, 1.0 - 1e-12))
    return (ll, r)
end

function _cv_cell_eval(fit, family::Beta, y::Real, μ::Real, t::Int, s::Int, N_mat, rng::AbstractRNG)
    μ_c = clamp(Float64(μ), 1e-6, 1.0 - 1e-6)
    φ_raw = if hasfield(typeof(fit), :φ)
        fit.φ
    elseif hasfield(typeof(fit), :φvec)
        fit.φvec
    else
        10.0
    end
    φ = (φ_raw isa AbstractVector) ? Float64(φ_raw[min(t, length(φ_raw))]) : Float64(φ_raw)
    d = Beta(μ_c * φ, (1.0 - μ_c) * φ)
    yc = clamp(Float64(y), 1e-6, 1.0 - 1e-6)
    ll = logpdf(d, yc)
    u = cdf(d, yc)
    r = quantile(Normal(), clamp(u, 1e-12, 1.0 - 1e-12))
    return (ll, r)
end

function _cv_cell_eval(fit, family::Gamma, y::Real, μ::Real, t::Int, s::Int, N_mat, rng::AbstractRNG)
    μ_c = max(Float64(μ), 1e-12)
    α_raw = if hasfield(typeof(fit), :α)
        fit.α
    elseif hasfield(typeof(fit), :αvec)
        fit.αvec
    else
        1.0
    end
    α = (α_raw isa AbstractVector) ? Float64(α_raw[min(t, length(α_raw))]) : Float64(α_raw)
    θ = μ_c / α
    d = Gamma(α, θ)
    yc = max(Float64(y), 1e-12)
    ll = logpdf(d, yc)
    u = cdf(d, yc)
    r = quantile(Normal(), clamp(u, 1e-12, 1.0 - 1e-12))
    return (ll, r)
end

function _cv_cell_eval(fit, family::Exponential, y::Real, μ::Real, t::Int, s::Int, N_mat, rng::AbstractRNG)
    μ_c = max(Float64(μ), 1e-12)
    d = Exponential(μ_c)
    yc = max(Float64(y), 1e-12)
    ll = logpdf(d, yc)
    u = cdf(d, yc)
    r = quantile(Normal(), clamp(u, 1e-12, 1.0 - 1e-12))
    return (ll, r)
end

# Fallback for other families
function _cv_cell_eval(fit, family, y::Real, μ::Real, t::Int, s::Int, N_mat, rng::AbstractRNG)
    d = Normal(μ, 1.0)
    ll = logpdf(d, y)
    r = y - μ
    return (ll, r)
end

# -----------------------------------------------------------------------------
# Main cv_gllvm entry point
# -----------------------------------------------------------------------------

@doc raw"""
    cv_gllvm(Y::AbstractMatrix;
             k_folds::Integer = 5,
             split::Symbol = :random,
             family = Normal(),
             K::Integer = 1,
             num_lv::Union{Nothing, Integer} = nothing,
             rng::AbstractRNG = Random.default_rng(),
             mask = nothing,
             N = nothing,
             kwargs...) -> CVResult

Run $K$-fold cross-validation on community response matrix `Y` ($p \times n$,
species/traits $\times$ sites/observations).

Supported `split` strategies:
- `:random` (or `:cell`, `:entry`) — Random entry-wise split across all observed cells.
- `:site` (or `:site_block`, `:site_level`, `:column`) — Site-level block cross-validation holding out entire site columns.
- `:species` (or `:species_block`, `:species_level`, `:row`) — Species-level block cross-validation holding out entire species rows.

For the Gaussian family, a fixed-effect design `X` (a `p×n×q` array, as for
[`fit_gaussian_gllvm`](@ref)) may be passed as `X = X`; it is subset with each
fold's training data, and held-out cells are predicted at `X[t, s, :] ⋅ β̂`.
Other families reject `X`.

Returns a [`CVResult`](@ref) containing out-of-sample log-likelihood, MSE,
Dunn–Smyth randomized quantile residuals, predictions, and per-fold metrics.

# Examples
```julia
using GLLVModels, Random

# 5-fold random cell-level CV on count data
Y = rand(0:10, 6, 40)
cv_res = cv_gllvm(Y; k_folds = 5, split = :random, family = Poisson(), K = 2)
cv_res.mse
cv_res.loglik

# 4-fold site-block CV
cv_site = cv_gllvm(Y; k_folds = 4, split = :site, family = Poisson(), K = 1)
```
"""
function cv_gllvm(Y::AbstractMatrix;
                  k_folds::Integer = 5,
                  split::Symbol = :random,
                  family = Normal(),
                  K::Integer = 1,
                  num_lv::Union{Nothing, Integer} = nothing,
                  rng::AbstractRNG = Random.default_rng(),
                  mask = nothing,
                  N = nothing,
                  kwargs...)
    p, n = size(Y)
    K_actual = num_lv === nothing ? K : num_lv
    k_folds >= 2 || throw(ArgumentError("k_folds must be ≥ 2; got $k_folds"))

    # Canonicalize split symbol
    split_mode = if split in (:random, :cell, :entry)
        :random
    elseif split in (:site, :site_block, :site_level, :column)
        :site
    elseif split in (:species, :species_block, :species_level, :row)
        :species
    else
        throw(ArgumentError("Unknown split :$split. Supported: :random, :site, :species"))
    end

    # A Gaussian fixed-effect design X (p×n×q) is subset alongside each fold's data.
    X = get(kwargs, :X, nothing)
    fit_kw = Base.structdiff(values(kwargs), NamedTuple{(:X,)})
    if X !== nothing
        family isa Normal || throw(ArgumentError(
            "cv_gllvm supports a fixed-effect design X only for the Gaussian family"))
        (X isa AbstractArray{<:Real, 3} && size(X, 1) == p && size(X, 2) == n) ||
            throw(ArgumentError("X must be a p×n×q array matching Y ($p×$n); got size $(size(X))"))
    end

    # Determine base observation mask (ignoring missing values)
    base_mask = trues(p, n)
    @inbounds for s in 1:n, t in 1:p
        if ismissing(Y[t, s])
            base_mask[t, s] = false
        end
    end
    if mask !== nothing
        base_mask .&= mask
    end

    n_obs = count(base_mask)
    n_obs > 0 || throw(ArgumentError("No observed cells in response matrix Y"))

    # Partitioning into K folds
    test_cell_groups = Vector{Vector{Tuple{Int, Int}}}(undef, k_folds)

    if split_mode === :random
        k_folds <= n_obs || throw(ArgumentError("k_folds ($k_folds) cannot exceed number of observed cells ($n_obs)"))
        valid_cells = Tuple{Int, Int}[]
        sizehint!(valid_cells, n_obs)
        @inbounds for s in 1:n, t in 1:p
            if base_mask[t, s]
                push!(valid_cells, (t, s))
            end
        end
        shuffled = Random.shuffle(rng, valid_cells)
        for k in 1:k_folds
            test_cell_groups[k] = Tuple{Int, Int}[]
        end
        for (i, cell) in enumerate(shuffled)
            fold_idx = mod1(i, k_folds)
            push!(test_cell_groups[fold_idx], cell)
        end
    elseif split_mode === :site
        k_folds <= n || throw(ArgumentError("k_folds ($k_folds) cannot exceed number of sites ($n)"))
        sites = Random.shuffle(rng, collect(1:n))
        site_groups = [Int[] for _ in 1:k_folds]
        for (i, s) in enumerate(sites)
            push!(site_groups[mod1(i, k_folds)], s)
        end
        for k in 1:k_folds
            test_cell_groups[k] = Tuple{Int, Int}[]
            for s in site_groups[k]
                for t in 1:p
                    if base_mask[t, s]
                        push!(test_cell_groups[k], (t, s))
                    end
                end
            end
        end
    elseif split_mode === :species
        k_folds <= p || throw(ArgumentError("k_folds ($k_folds) cannot exceed number of species ($p)"))
        species = Random.shuffle(rng, collect(1:p))
        species_groups = [Int[] for _ in 1:k_folds]
        for (i, t) in enumerate(species)
            push!(species_groups[mod1(i, k_folds)], t)
        end
        for k in 1:k_folds
            test_cell_groups[k] = Tuple{Int, Int}[]
            for t in species_groups[k]
                for s in 1:n
                    if base_mask[t, s]
                        push!(test_cell_groups[k], (t, s))
                    end
                end
            end
        end
    end

    predictions = fill(NaN, p, n)
    residuals_mat = fill(NaN, p, n)
    fits = Vector{Any}(undef, k_folds)

    fold_logliks = zeros(Float64, k_folds)
    fold_mses    = zeros(Float64, k_folds)
    fold_res_means = zeros(Float64, k_folds)
    fold_res_stds  = zeros(Float64, k_folds)

    all_sq_errs = Float64[]
    all_resids  = Float64[]
    total_ll    = 0.0

    for k in 1:k_folds
        test_cells = test_cell_groups[k]
        train_mask = copy(base_mask)
        for (t, s) in test_cells
            train_mask[t, s] = false
        end

        # Fit model on training fold
        fit_k = if family isa Normal
            if all(train_mask)
                X === nothing ? _fit_gaussian_trait_intercepts(Y; K = K_actual, fit_kw...) :
                                fit_gaussian_gllvm(Y; K = K_actual, X = X, fit_kw...)
            elseif split_mode === :site
                train_sites = findall(s -> any(view(train_mask, :, s)), 1:n)
                Xk = X === nothing ? nothing : X[:, train_sites, :]
                Xk === nothing ? _fit_gaussian_trait_intercepts(view(Y, :, train_sites); K = K_actual, fit_kw...) :
                                 fit_gaussian_gllvm(view(Y, :, train_sites); K = K_actual, X = Xk, fit_kw...)
            elseif split_mode === :species
                train_species = findall(t -> any(view(train_mask, t, :)), 1:p)
                Xk = X === nothing ? nothing : X[train_species, :, :]
                Xk === nothing ? _fit_gaussian_trait_intercepts(view(Y, train_species, :); K = K_actual, fit_kw...) :
                                 fit_gaussian_gllvm(view(Y, train_species, :); K = K_actual, X = Xk, fit_kw...)
            else
                # Random cell split: impute unobserved cells with species mean + PPCA refinement
                Y_imputed = Matrix{Float64}(undef, p, n)
                for t in 1:p
                    obs_t = findall(view(train_mask, t, :))
                    m_t = isempty(obs_t) ? 0.0 : mean(view(Y, t, obs_t))
                    for s in 1:n
                        Y_imputed[t, s] = train_mask[t, s] ? Float64(Y[t, s]) : m_t
                    end
                end
                fit_init = X === nothing ? _fit_gaussian_trait_intercepts(Y_imputed; K = K_actual, fit_kw...) :
                                           fit_gaussian_gllvm(Y_imputed; K = K_actual, X = X, fit_kw...)
                Z_init = getLV(fit_init, Y_imputed; X = X, rotate = false)
                for (t, s) in test_cells
                    Y_imputed[t, s] = _cv_gaussian_mean(fit_init, X, t, s) +
                                      dot(view(fit_init.pars.Λ, t, :), view(Z_init, s, :))
                end
                X === nothing ? _fit_gaussian_trait_intercepts(Y_imputed; K = K_actual, fit_kw...) :
                                fit_gaussian_gllvm(Y_imputed; K = K_actual, X = X, fit_kw...)
            end
        else
            if N !== nothing
                fit_gllvm(Y; family = family, K = K_actual, mask = train_mask, N = N, kwargs...)
            else
                fit_gllvm(Y; family = family, K = K_actual, mask = train_mask, kwargs...)
            end
        end
        fits[k] = fit_k

        Λ, β, link = _cv_extract_params(fit_k)
        p_fit = size(Λ, 1)

        train_spec_vec = split_mode === :species ? findall(t -> any(view(train_mask, t, :)), 1:p) : collect(1:p)

        # Precompute per-site conditional latent scores using training observations
        Z_hat = zeros(Float64, n, size(Λ, 2))
        for s in 1:n
            Z_hat[s, :] = _cv_site_mode(fit_k, family, Y, s, view(train_mask, :, s), N, train_spec_vec; X = X)
        end

        fold_ll = 0.0
        fold_errs = Float64[]
        fold_rvec = Float64[]

        for (t, s) in test_cells
            y_val = Y[t, s]
            z_s = @view Z_hat[s, :]

            # For species-block split, map to fitted row or use average
            t_idx = findfirst(==(t), train_spec_vec)
            β_t = if fit_k isa GllvmFit && X !== nothing
                _cv_gaussian_mean(fit_k, X, t, s)   # β is a covariate vector, not per-species
            else
                (t_idx !== nothing && t_idx <= length(β)) ? β[t_idx] : (isempty(β) ? 0.0 : mean(β))
            end
            Λ_t_dot_z = (t_idx !== nothing && t_idx <= size(Λ, 1)) ? dot(@view(Λ[t_idx, :]), z_s) : 0.0
            η = β_t + Λ_t_dot_z
            μ = linkinv(link, η)

            predictions[t, s] = μ
            sq_err = (Float64(y_val) - Float64(μ))^2
            push!(fold_errs, sq_err)
            push!(all_sq_errs, sq_err)

            cell_ll, cell_res = _cv_cell_eval(fit_k, family, y_val, μ, t_idx !== nothing ? t_idx : 1, s, N, rng)
            residuals_mat[t, s] = cell_res
            fold_ll += cell_ll
            push!(fold_rvec, cell_res)
            push!(all_resids, cell_res)
        end

        fold_logliks[k] = fold_ll
        total_ll += fold_ll
        fold_mses[k] = isempty(fold_errs) ? 0.0 : mean(fold_errs)
        fold_res_means[k] = isempty(fold_rvec) ? 0.0 : mean(fold_rvec)
        fold_res_stds[k]  = length(fold_rvec) > 1 ? std(fold_rvec) : 0.0
    end

    overall_mse = isempty(all_sq_errs) ? 0.0 : mean(all_sq_errs)
    overall_res_mean = isempty(all_resids) ? 0.0 : mean(all_resids)
    overall_res_std  = length(all_resids) > 1 ? std(all_resids) : 0.0

    return CVResult(
        Int(k_folds),
        split_mode,
        total_ll,
        overall_mse,
        overall_res_mean,
        overall_res_std,
        fold_logliks,
        fold_mses,
        fold_res_means,
        fold_res_stds,
        predictions,
        residuals_mat,
        fits
    )
end
