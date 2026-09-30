# Unordered categorical / multinomial FE softmax (twin gllvmTMB family_id 16).
#
# Identity: docs/dev-log/decisions/2026-08-18-multinomial-identity.md
# v1 = one unordered trait, fixed effects only. No LV, no TMB K−1 pseudo-rows,
# no ordinal cutpoints, no φ. Ledger stays `missing`.
#
# Model (one observation i, declared baseline = category 1):
#   η₁ ≡ 0
#   η_k = β_k + x_i' γ_k    k = 2, …, K
#   P(y = k) = exp(η_k) / Σ_j exp(η_j)
# Packing (contrast-major): [β₂…β_K; γ₂ (p); …; γ_K (p)] length (K−1)(1+p).

"""
    Multinomial()

Marker for unordered categorical (twin `multinomial()`, family_id 16).
Not `Categorical` / `categorical()` (missing-predictor imputation) and not
`Distributions.Multinomial` (count vectors). Support `y ∈ {1, …, K}` with
`K ≥ 3` (`K = 2` is binomial-logit). v1 is fixed-effects softmax only.
"""
struct Multinomial end

default_link(::Multinomial) = LogitLink()

"""
    multinomial_pack_len(n_categories, n_covariates) -> Int

Free-parameter count `(K−1)(1+p)` for baseline-category softmax.
`K ≥ 3`; `K = 2` is binomial-logit and is rejected.
"""
function multinomial_pack_len(n_categories::Integer, n_covariates::Integer)
    n_categories >= 3 || throw(ArgumentError(
        "multinomial requires K ≥ 3 categories; K = 2 is binomial-logit — " *
        "use Binomial() / LogitLink()"))
    n_covariates >= 0 || throw(ArgumentError(
        "multinomial: n_covariates must be ≥ 0 (got $n_covariates)"))
    return (Int(n_categories) - 1) * (1 + Int(n_covariates))
end

"""
    unpack_multinomial(θ, n_categories, n_covariates) -> (β, γ)

Split a packed vector into contrast intercepts `β` (length `K−1`) and
contrast-major slopes `γ` (`(K−1) × p`).
"""
function unpack_multinomial(θ::AbstractVector, n_categories::Integer,
        n_covariates::Integer)
    n_free = n_categories - 1
    length(θ) == n_free * (1 + n_covariates) || throw(DimensionMismatch(
        "multinomial pack length $(length(θ)) ≠ (K-1)(1+p) = $(n_free * (1 + n_covariates))"))
    β = θ[1:n_free]
    if n_covariates == 0
        γ = reshape(similar(θ, 0), n_free, 0)
        return β, γ
    end
    γ = permutedims(reshape(θ[(n_free + 1):end], n_covariates, n_free))
    return β, γ
end

"""
    multinomial_eta(β, γ, x) -> Vector

Linear predictor with `η₁ ≡ 0` and `η_k = β_{k-1} + xᵀ γ_{k-1,:}` for
`k = 2, …, K`. One vector per observation.
"""
function multinomial_eta(β::AbstractVector, γ::AbstractMatrix, x::AbstractVector)
    n_free = length(β)
    T = promote_type(eltype(β), eltype(γ), eltype(x), Float64)
    η = zeros(T, n_free + 1)   # η[1] stays 0
    p = size(γ, 2)
    @inbounds for k in 1:n_free
        ηk = β[k]
        for j in 1:p
            ηk += γ[k, j] * x[j]
        end
        η[k + 1] = ηk
    end
    return η
end

function _multinomial_logsumexp(η::AbstractVector)
    m = η[1]
    @inbounds for i in 2:length(η)
        η[i] > m && (m = η[i])
    end
    s = zero(m)
    @inbounds for ηj in η
        s += exp(ηj - m)
    end
    return m + log(s)
end

function _multinomial_int_y(y::AbstractVector, n_categories)
    n = length(y)
    n >= 1 || throw(ArgumentError("multinomial: empty response"))
    yi = Vector{Int}(undef, n)
    ymax = 0
    @inbounds for i in 1:n
        v = Int(y[i])
        v < 1 && throw(ArgumentError(
            "multinomial requires y ∈ {1, …, K}; found y=$v"))
        yi[i] = v
        v > ymax && (ymax = v)
    end
    K = n_categories === nothing ? ymax : Int(n_categories)
    K >= 3 || throw(ArgumentError(
        "multinomial requires K ≥ 3 categories; K = 2 is binomial-logit — " *
        "use Binomial() / LogitLink()"))
    @inbounds for v in yi
        v > K && throw(ArgumentError(
            "multinomial requires y ∈ {1, …, $K}; found y=$v"))
    end
    return yi, K
end

function _is_onehot_columns(Y::AbstractMatrix)
    r, c = size(Y)
    r >= 3 || return false
    @inbounds for s in 1:c
        ssum = 0
        for t in 1:r
            v = Y[t, s]
            (v == 0 || v == 1) || return false
            ssum += Int(v)
        end
        ssum == 1 || return false
    end
    return true
end

function _multinomial_response(Y::AbstractMatrix; n_categories = nothing)
    r, c = size(Y)
    if r == 1
        return _multinomial_int_y(vec(Y), n_categories)
    end
    if c == 1 && (n_categories === nothing || n_categories != r)
        return _multinomial_int_y(vec(Y), n_categories)
    end
    if _is_onehot_columns(Y)
        ncat = r
        ncat >= 3 || throw(ArgumentError(
            "multinomial requires K ≥ 3 categories; K = 2 is binomial-logit — " *
            "use Binomial() / LogitLink()"))
        y = Vector{Int}(undef, c)
        @inbounds for s in 1:c
            y[s] = findfirst(isone, view(Y, :, s))
        end
        return y, ncat
    end
    throw(ArgumentError(
        "multinomial v1 is one unordered trait per fit (1×n integer categories " *
        "or K×n one-hot); do not expand TMB K−1 pseudo-rows"))
end

_multinomial_response(y::AbstractVector; n_categories = nothing) =
    _multinomial_int_y(y, n_categories)

"""
    multinomial_loglik(Y, θ; X=nothing, n_categories=nothing) -> Real

Fixed-effect softmax log-likelihood at packed `θ` of length `(K−1)(1+p)`.
One softmax per observation; `η₁ ≡ 0`. AD-clean (no TMB `log(1e-12)` floor).
"""
function multinomial_loglik(Y::AbstractVecOrMat, θ::AbstractVector;
        X = nothing, n_categories = nothing)
    y, K = _multinomial_response(Y; n_categories = n_categories)
    n = length(y)
    p = X === nothing ? 0 : size(X, 2)
    X === nothing || size(X, 1) == n || throw(DimensionMismatch(
        "multinomial X must be n×p = ($n, $p); got $(size(X))"))
    length(θ) == multinomial_pack_len(K, p) || throw(DimensionMismatch(
        "multinomial θ length $(length(θ)) ≠ (K-1)(1+p) = $(multinomial_pack_len(K, p))"))
    β, γ = unpack_multinomial(θ, K, p)
    acc = zero(eltype(θ))
    xbuf = p == 0 ? eltype(θ)[] : Vector{eltype(θ)}(undef, p)
    @inbounds for i in 1:n
        if p == 0
            η = multinomial_eta(β, γ, xbuf)
        else
            for j in 1:p
                xbuf[j] = X[i, j]
            end
            η = multinomial_eta(β, γ, xbuf)
        end
        acc += η[y[i]] - _multinomial_logsumexp(η)
    end
    return acc
end

"""
    MultinomialFit

Result of [`fit_multinomial_gllvm`](@ref): contrast intercepts `β` (length
`K−1`), contrast-major slopes `γ` (`(K−1)×p`), `n_categories`, logit `link`,
maximised `loglik`, `converged`, `iterations`, and packed
`theta_packed` of length `(K−1)(1+p)`. v1 has no loadings.
"""
struct MultinomialFit
    β::Vector{Float64}
    γ::Matrix{Float64}
    n_categories::Int
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    theta_packed::Vector{Float64}
end

function Base.show(io::IO, f::MultinomialFit)
    p = size(f.γ, 2)
    print(io, "MultinomialFit(K=", f.n_categories, ", p=", p,
          ", pack=", length(f.theta_packed),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

function _multinomial_β_init(y::AbstractVector{Int}, K::Integer)
    counts = zeros(Float64, K)
    @inbounds for yi in y
        counts[yi] += 1
    end
    β = Vector{Float64}(undef, K - 1)
    @inbounds for k in 2:K
        # add-half so an empty contrast does not send β → −∞
        β[k - 1] = log(counts[k] + 0.5) - log(counts[1] + 0.5)
    end
    return β
end

# Complete-separation threshold for `_multinomial_verdict`. Under complete
# separation (some linear combination of the covariates orders every observation
# into its own category) the softmax MLE does not exist: the log-likelihood rises
# toward its saturated bound 0 only as the coefficients run to infinity, and L-BFGS
# stops wherever the gradient first falls below `g_tol`. Along that ridge the
# gradient entries are sums of the residuals 1{y_i = k} − p̂_ik, so a stop on the
# default `g_tol = 1e-5` leaves the per-observation negative log-likelihoods
# −log p̂_i(y_i) ≈ 1 − p̂_i(y_i) of roughly that size (the fixture stopped at a
# TOTAL of 1.19e-5 over 12 observations). The threshold 1e-4, applied to EVERY
# observation, means each observed category is fitted with probability above
# 0.9999, a logit gap of at least log(9999) ≈ 9.2 over every other category; that
# is ten times the default `g_tol`, so a gradient stop on separated data is caught
# with room to spare. Healthy data are nowhere near it: the six healthy fits in
# `test/fixtures/multinomial_separation.toml` have a worst observation at
# p̂ ≈ 0.007 to 0.28. Not covered: quasi-complete separation (tied observations stay
# far from p̂ = 1, so the maximum below stays large), and a caller-supplied `g_tol`
# far above 1e-4, which can stop the ridge walk before every observation clears the
# threshold.
const _MN_SEPARATION_NLL = 1e-4

"""
    _multinomial_max_obs_nll(y, θ, X, K) -> Float64

Largest per-observation negative log-likelihood `−log p̂_i(y_i)` at packed `θ`.
"""
function _multinomial_max_obs_nll(y::AbstractVector{Int}, θ::AbstractVector, X,
        K::Integer)
    p = X === nothing ? 0 : size(X, 2)
    β, γ = unpack_multinomial(θ, K, p)
    worst = 0.0
    @inbounds for i in eachindex(y)
        x = p == 0 ? Float64[] : vec(X[i, :])
        η = multinomial_eta(β, γ, x)
        worst = max(worst, Float64(_multinomial_logsumexp(η) - η[y[i]]))
    end
    return worst
end

"""
    _multinomial_verdict(optim_converged, nll, max_obs_nll) -> (converged, loglik, reason)

Convergence contract for [`fit_multinomial_gllvm`](@ref) (per the #502/#505 rule:
a per-family verdict rather than a change to the shared `_fit_verdict`). A
failed objective (`_nll_failed`) returns `(false, -Inf, :objective_impossible)`,
as `_fit_verdict` does. If every observation's negative log-likelihood is at or
below `_MN_SEPARATION_NLL` (`max_obs_nll`, see that constant's note) the data are
completely separated, no finite MLE exists, and the result is
`(false, -nll, :separation)`: the log-likelihood is reported as computed, but the
fit is not converged. Otherwise `(optim_converged, -nll, :ok)`.
"""
function _multinomial_verdict(optim_converged::Bool, nll::Real, max_obs_nll::Real)
    _nll_failed(nll) && return (false, -Inf, :objective_impossible)
    ll = -Float64(nll)
    max_obs_nll <= _MN_SEPARATION_NLL && return (false, ll, :separation)
    return (optim_converged, ll, :ok)
end

"""
    fit_multinomial_gllvm(Y; X=nothing, n_categories=nothing, …) -> MultinomialFit

Fit a one-trait unordered-categorical softmax (twin fid 16) by L-BFGS over
the packed FE vector `[β_contrast; γ_contrast]` of length `(K−1)(1+p)`.
`Y` is a `1×n` integer category matrix (or length-`n` vector, or `K×n`
one-hot). `X` is `n×p` site covariates (no intercept column). `K ≥ 3`;
`K = 2` redirects to binomial-logit. No latent variables (v1).
"""
function fit_multinomial_gllvm(Y::AbstractVecOrMat;
        X = nothing, n_categories = nothing,
        K = nothing, num_lv = nothing,
        link::Link = LogitLink(),
        β_init = nothing, γ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500)
    (K !== nothing && K != 0) && throw(ArgumentError(
        "fit_multinomial_gllvm: v1 is fixed-effects softmax only — no LV " *
        "(got K=$K). Leave K / num_lv unset."))
    (num_lv !== nothing && num_lv != 0) && throw(ArgumentError(
        "fit_multinomial_gllvm: v1 is fixed-effects softmax only — no LV " *
        "(got num_lv=$num_lv). Leave K / num_lv unset."))
    link isa LogitLink || throw(ArgumentError(
        "fit_multinomial_gllvm: only LogitLink is supported (twin multinomial)"))
    y, ncat = _multinomial_response(Y; n_categories = n_categories)
    n = length(y)
    p = X === nothing ? 0 : size(X, 2)
    X === nothing || size(X, 1) == n || throw(DimensionMismatch(
        "multinomial X must be n×p = ($n, $p); got $(size(X))"))
    β0 = β_init === nothing ? _multinomial_β_init(y, ncat) : collect(float.(β_init))
    length(β0) == ncat - 1 || throw(DimensionMismatch(
        "β_init length $(length(β0)) ≠ K-1 = $(ncat - 1)"))
    γ0 = if γ_init === nothing
        zeros(ncat - 1, p)
    else
        collect(float.(γ_init))
    end
    size(γ0) == (ncat - 1, p) || throw(DimensionMismatch(
        "γ_init size $(size(γ0)) ≠ ((K-1)×p) = ($(ncat - 1), $p)"))
    θ0 = p == 0 ? collect(β0) : vcat(β0, vec(permutedims(γ0)))
    Ymat = Y isa AbstractMatrix ? Y : reshape(y, 1, n)
    function negll(θ)
        v = try
            -multinomial_loglik(Ymat, θ; X = X, n_categories = ncat)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂, γ̂ = unpack_multinomial(θ̂, ncat, p)
    # `_multinomial_verdict` adds the complete-separation check to `_fit_verdict`'s
    # sentinel screen (reported converged = true at loglik -1.19e-5 with the slopes
    # running off, test/fixtures/multinomial_separation.toml).
    conv, ll, _reason = _multinomial_verdict(Optim.converged(res), Optim.minimum(res),
                                             _multinomial_max_obs_nll(y, θ̂, X, ncat))
    return MultinomialFit(collect(Float64, β̂), Matrix{Float64}(γ̂), ncat,
                          LogitLink(), ll, conv, Optim.iterations(res),
                          collect(Float64, θ̂))
end
