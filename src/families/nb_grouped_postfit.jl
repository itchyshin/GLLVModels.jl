# Post-fit methods for the per-species / grouped-dispersion NB2 fits (issue #555):
# `predict`, `fitted`, `residuals` (Dunn-Smyth and Pearson) and `getResidualCor` for
# `NBGroupedFit` (`fit_nb_gllvm_grouped`) and `NBGroupedCovFit`
# (`fit_nb_gllvm_grouped_cov`).
#
# These mirror the shared-dispersion `NBFit` methods (src/postfit.jl) and the covariate
# analogue `GllvmSpeciesCovFit` (src/families/species_covariates.jl). The only
# difference is the NB2 size: species `t` uses `r_group[group[t]]`, not one scalar `r`.
# Fitted values are conditional on the Laplace mode of each site's latent vector,
# exactly as in `getLV` (the posterior mode of the same objective the fit maximised).

# Per-species NB2 size, r[t] = r_group[group[t]].
_nb_grouped_rvec(fit::Union{NBGroupedFit, NBGroupedCovFit}) = fit.r_group[fit.group]

# Conditional distribution of a count with mean μ and NB2 size r (Var = μ + μ²/r).
# As r grows NB2 collapses to Poisson(μ). The package flags r > 1e6 as the Poisson
# limit and fitted groups do land there (r from 1e9 to 1e24). The Distributions.jl NB2
# CDF loses accuracy from about r = 1e12, and once r / (r + μ) rounds to 1 (r above
# about 1e16 μ) it returns 1 for every count, which would saturate every residual of
# the group at the clamp. The Poisson CDF is used once μ / r < 1e-6: there the NB2 and
# Poisson CDFs differ by about √μ / r, below 1e-6.
_nb_grouped_dist(μ::Real, r::Real) =
    r >= 1e6 * max(μ, 1.0) ? Poisson(μ) : NegativeBinomial(r, r / (r + μ))

# Shared residual kernel. `μ` is the p×n conditional mean, `rvec` the per-species size.
# Cells with `mask[t, s] == false` are NaN and consume no random draw.
function _nb_grouped_residuals(Y::AbstractMatrix, μ::AbstractMatrix, rvec::AbstractVector,
                               type::Symbol, rng::AbstractRNG, mask)
    p, n = size(Y)
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        if mask !== nothing && !mask[t, s]
            R[t, s] = NaN
        elseif type === :pearson
            m = μ[t, s]
            R[t, s] = (Y[t, s] - m) / sqrt(m + m^2 / rvec[t])
        else
            d = _nb_grouped_dist(μ[t, s], rvec[t])
            Flo = cdf(d, Y[t, s] - 1)
            Fhi = cdf(d, Y[t, s])
            u = Flo + (Fhi - Flo) * rand(rng)
            R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
        end
    end
    return R
end

function _nb_grouped_check_type(type::Symbol)
    type in (:response, :mean, :link) ||
        throw(ArgumentError("type must be :response, :mean, or :link; got :$type"))
    return nothing
end

function _nb_grouped_check_residual_type(type::Symbol)
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    return nothing
end

# ---------------------------------------------------------------------------
# NBGroupedFit
# ---------------------------------------------------------------------------

"""
    predict(fit::NBGroupedFit, Y; type=:response, N=nothing, mask=nothing, offset=nothing) -> p×n matrix

In-sample fitted values of a grouped-dispersion NB2 fit at the Laplace mode `ẑ`
(see [`getLV`](@ref)): `type=:link` returns `η = β + offset + Λẑ`; `type=:response`
(= `:mean`) returns the inverse-link fitted means `μ = linkinv(link, η)`, `exp(η)` for
the default log link. Pass the same `mask` given to [`fit_nb_gllvm_grouped`](@ref): the
latent mode, and so the prediction, depends on it.

On a fit made with an `offset`, `η` and the latent mode include it: the stored training
offset (`fit.offset`) when `Y` has the training size, otherwise the `offset` you pass (a
p×n matrix, a scalar or a length-p vector). New units from an offset fit without an
`offset` are refused, as gllvmTMB refuses `newdata` that lacks the offset variable.
"""
function predict(fit::NBGroupedFit, Y::AbstractMatrix{<:Integer};
                 type::Symbol = :response,
                 N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                 mask = nothing, offset = nothing)
    _nb_grouped_check_type(type)
    O = _grouped_prediction_offset(fit, Y, offset, mask, "predict")
    Z = getLV(fit, Y; N = N, rotate = false, mask = mask, offset = O)
    η = fit.β .+ fit.Λ * Z'
    O === nothing || (η = η .+ O)
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

"""
    residuals(fit::NBGroupedFit, Y; type=:dunnsmyth, rng=Random.default_rng(),
              N=nothing, mask=nothing, offset=nothing) -> p×n matrix

Conditional residuals for a grouped-dispersion NB2 fit, with the per-species size
`r_group[group[t]]`. `:dunnsmyth` returns Dunn-Smyth randomized quantile residuals,
`Φ⁻¹(u)` with `u` uniform on `[F(y−1), F(y)]` under `NegativeBinomial(r, r/(r+μ))`;
they are approximately N(0,1) under a correct model (pass a fixed `rng` to reproduce).
A group whose fitted `r` is at the Poisson limit uses the Poisson CDF. `:pearson`
returns `(Y − μ) / √(μ + μ²/r)`. Cells with `mask[t, s] == false` are `NaN`. The fitted
means `μ` include the fit's offset by the rule of [`predict`](@ref).
"""
function residuals(fit::NBGroupedFit, Y::AbstractMatrix{<:Integer};
                   type::Symbol = :dunnsmyth,
                   rng::AbstractRNG = Random.default_rng(),
                   N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                   mask = nothing, offset = nothing)
    _nb_grouped_check_residual_type(type)
    μ = predict(fit, Y; type = :response, N = N, mask = mask, offset = offset)
    return _nb_grouped_residuals(Y, μ, _nb_grouped_rvec(fit), type, rng, mask)
end

# ---------------------------------------------------------------------------
# NBGroupedCovFit
# ---------------------------------------------------------------------------

"""
    predict(fit::NBGroupedCovFit, Y, X; type=:response, mask=nothing) -> p×n matrix

In-sample fitted values of a grouped-dispersion NB2 fit with covariates at the Laplace
mode `ẑ` (see [`getLV`](@ref)): `type=:link` returns `η = β + Xγ + Λẑ`; `type=:response`
(= `:mean`) returns `μ = linkinv(link, η)`. `X` is the `(p, n, q)` design given to
[`fit_nb_gllvm_grouped_cov`](@ref) (a coefficient fixed at zero in `γ` ignores its
covariate).
"""
function predict(fit::NBGroupedCovFit, Y::AbstractMatrix{<:Integer},
                 X::AbstractArray{<:Real, 3};
                 type::Symbol = :response, mask = nothing)
    _nb_grouped_check_type(type)
    Z = getLV(fit, Y, X; rotate = false, mask = mask)
    O = _build_offset(X, fit.γ)
    η = fit.β .+ O .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

"""
    fitted(fit::NBGroupedCovFit, Y, X; mask=nothing) -> p×n matrix of fitted means.
"""
fitted(fit::NBGroupedCovFit, Y::AbstractMatrix{<:Integer}, X::AbstractArray{<:Real, 3};
       mask = nothing) =
    predict(fit, Y, X; type = :response, mask = mask)

"""
    residuals(fit::NBGroupedCovFit, Y, X; type=:dunnsmyth, rng=Random.default_rng(),
              mask=nothing) -> p×n matrix

Conditional residuals for a grouped-dispersion NB2 fit with covariates, with the
per-species size `r_group[group[t]]` and the mean `μ = exp(β + Xγ + Λẑ)`. `:dunnsmyth`
returns Dunn-Smyth randomized quantile residuals (approximately N(0,1) under a correct
model; pass a fixed `rng` to reproduce), `:pearson` returns `(Y − μ) / √(μ + μ²/r)`.
A group whose fitted `r` is at the Poisson limit uses the Poisson CDF. Cells with
`mask[t, s] == false` are `NaN`.
"""
function residuals(fit::NBGroupedCovFit, Y::AbstractMatrix{<:Integer},
                   X::AbstractArray{<:Real, 3};
                   type::Symbol = :dunnsmyth,
                   rng::AbstractRNG = Random.default_rng(),
                   mask = nothing)
    _nb_grouped_check_residual_type(type)
    μ = predict(fit, Y, X; type = :response, mask = mask)
    return _nb_grouped_residuals(Y, μ, _nb_grouped_rvec(fit), type, rng, mask)
end

# ---------------------------------------------------------------------------
# getResidualCor (both fits)
# ---------------------------------------------------------------------------

"""
    getResidualCor(fit::Union{NBGroupedFit, NBGroupedCovFit}; level=:unit) -> p×p matrix

Implied species correlation `cov2cor(ΛΛᵀ)` of a grouped-dispersion NB2 fit. This is the
latent-scale correlation without a link-implicit residual, the same matrix as gllvmTMB's
`getResidualCor()` (`extract_Sigma(link_residual = "none")`, with no unique variance in
these fits). It does not depend on the dispersion, the covariates or the loadings'
rotation. The diagonal is exactly 1; a species whose loadings row is zero has no
latent variance, and its row and column are `NaN`. These fits have a single latent tier,
so only `level = :unit` (or its alias `:B`) is accepted.
"""
function getResidualCor(fit::Union{NBGroupedFit, NBGroupedCovFit}; level::Symbol = :unit)
    lvl = _canonical_level(level)
    lvl === :unit || throw(ArgumentError(
        "getResidualCor(::$(nameof(typeof(fit)))) has a single latent tier, level = :unit " *
        "(or the alias :B); got :$level"))
    return _latent_correlation(_latent_sigma(fit.Λ, zeros(size(fit.Λ, 1))))
end
