# Ordination-score uncertainty — per-site latent-score intervals for biplots.
#
# A fitted GLLVM places each SITE at a latent score ẑ_s = the per-site Laplace
# mode (or Gaussian posterior mean) given that site's response y_s and the fitted
# parameters (Λ, β, link, dispersion); see `getLV`. The score is a point estimate,
# but a biplot usually wants the *spread* around each point — the CI ellipse you
# draw around a site.
#
# We quantify that spread with a CONDITIONAL (posterior-predictive) parametric
# bootstrap of the scores, holding the fitted parameters fixed:
#
#   1. fitted means       μ̂ = predict(fit, Y; type=:response)      (p×n)
#      — μ̂[:, s] encodes site s's estimated latent position ẑ_s.
#   2. for b = 1..B: resample each cell y*_{ts} ~ family(μ̂_{ts}; fitted disp),
#      recompute the score matrix S*_b = getLV(fit, Y*_b; rotate=false),
#      then PROCRUSTES-align S*_b to the reference scores S0 (centred, best
#      orthogonal R) to remove the K×K rotation/reflection ambiguity between
#      replicates.
#   3. per site s and axis k: SE = std of the aligned replicate scores,
#      interval = empirical quantiles at level α.
#
# The per-site identity is preserved (replicate site s always maps to original
# site s, because each y*_s is drawn at site s's own fitted mean), so the
# returned intervals are genuinely per-site. This is the score analogue of the
# parametric bootstrap already used for parameters (`confint_bootstrap.jl`); here
# the fitted parameters are frozen and only the data — hence the modes — vary.
#
# Scope: the single-`Y` ordination fits whose `getLV(fit, Y; rotate)` /
# `predict(fit, Y; type=:response)` take just the response matrix and which expose
# a fitted family + dispersion via `_sim_family` (Poisson, NB, Beta, Gamma,
# Exponential) or are `BinomialFit`. This is the headline ordination point cloud.

using Random: AbstractRNG, default_rng

# Procrustes alignment of A (n×K) onto reference B (n×K): centre both, find the
# best orthogonal R (R'R = I) minimising ‖Ac R − Bc‖_F (Schönemann 1966), and
# return the aligned, RE-CENTRED-TO-B coordinates Ac R .+ mean(B). Used to put
# every bootstrap score matrix in the reference fit's orientation before taking
# per-site spreads.
function _procrustes_align(A::AbstractMatrix, B::AbstractMatrix)
    μA = mean(A; dims = 1)
    μB = mean(B; dims = 1)
    Ac = A .- μA
    Bc = B .- μB
    F = svd(Bc' * Ac)            # K×K
    R = F.V * F.U'               # orthogonal: aligns Ac onto Bc
    return (Ac * R) .+ μB
end

# Conditional per-cell resampler: draw a fresh response matrix at the FIXED fitted
# means μ̂ (p×n), using the fit's family marker + dispersion. Mirrors the samplers
# in simulate_fit.jl / covariates.jl but conditions on μ̂ rather than drawing fresh
# latent z — that is what keeps the per-site identity.
function _resample_at_mean(fit::_ScalarMuFit, μ̂::AbstractMatrix, rng::AbstractRNG)
    fam, _ = _sim_family(fit)
    p, n = size(μ̂)
    Y = Matrix{_sim_eltype(fam)}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        Y[t, s] = _cov_sample(fam, μ̂[t, s], 1, rng)
    end
    return Y
end

function _resample_at_mean(fit::GllvmFit, μ̂::AbstractMatrix, rng::AbstractRNG)
    σ = fit.pars.σ_eps
    p, n = size(μ̂)
    Y = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        Y[t, s] = μ̂[t, s] + σ * randn(rng)
    end
    return Y
end

function _resample_at_mean(fit::BinomialFit, μ̂::AbstractMatrix, Nm::AbstractMatrix,
                           rng::AbstractRNG)
    p, n = size(μ̂)
    Y = Matrix{Int}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        Y[t, s] = rand(rng, Binomial(Nm[t, s], clamp(μ̂[t, s], 1e-12, 1 - 1e-12)))
    end
    return Y
end

"""
    ordination_uncertainty(fit, Y; n_boot=200, level=0.95, rotate=true,
                           rng=Random.default_rng(), N=nothing)
        -> NamedTuple

Per-site latent-score uncertainty for a fitted GLLVM, by a conditional
(posterior-predictive) parametric bootstrap of the ordination scores. The fitted
parameters `(Λ, β, link, dispersion)` are held fixed; for each of `n_boot`
replicates the response matrix is resampled at the fitted means
`μ̂ = predict(fit, Y; type=:response)`, the per-site scores are recomputed
(`getLV`), and each replicate is **procrustes-aligned** to the reference scores
to remove the `K×K` rotation/reflection ambiguity. Per site and axis we then
report the bootstrap standard error and an empirical quantile interval — the
spread you draw as a CI / ellipse on a biplot.

`Y` (the `p×n` response matrix) must match the fitting call — the fit does not
store the data. `rotate=true` (default) reports the scores and intervals in the
canonical principal orientation (the same orientation as `getLV(fit, Y)` /
`ordination(fit, Y).sites`); the procrustes alignment is applied in that frame.
`N` supplies Binomial trial counts (default all-ones). Returns a `NamedTuple`:

- `scores` — `n×K` reference site scores (`getLV(fit, Y; rotate)`).
- `se`     — `n×K` bootstrap standard error per site/axis.
- `lower`  — `n×K` lower interval bound (quantile `(1−level)/2`).
- `upper`  — `n×K` upper interval bound (quantile `1−(1−level)/2`).
- `level`  — the requested coverage level.
- `n_boot` — the number of successful bootstrap replicates used.

Supported fits: the single-`Y` ordination fits — `GllvmFit` (Gaussian),
`PoissonFit`, `NBFit`, `BetaFit`, `GammaFit`, `ExponentialFit`, and `BinomialFit`.

```julia
fit = fit_poisson_gllvm(Y; K = 2)
u   = ordination_uncertainty(fit, Y; n_boot = 200)
# u.scores[s, :]  -> site s point;  u.lower/u.upper[s, :] -> its CI box per axis
```
"""
function ordination_uncertainty(fit::Union{_ScalarMuFit, BinomialFit, GllvmFit},
        Y::AbstractMatrix; n_boot::Integer = 200, level::Real = 0.95,
        rotate::Bool = true, rng::AbstractRNG = default_rng(),
        N::Union{Nothing, AbstractMatrix} = nothing)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    n_boot ≥ 2 || throw(ArgumentError("n_boot must be ≥ 2; got $n_boot"))
    fit isa GllvmFit && N !== nothing &&
        throw(ArgumentError("ordination_uncertainty on a Gaussian GllvmFit does not take `N`"))

    isbin = fit isa BinomialFit
    Nm = isbin ? (N === nothing ? fill(1, size(Y)...) : N) : nothing

    # Reference scores (the point cloud) and fitted means (the resampling targets).
    S0 = isbin ? getLV(fit, Y; N = Nm, rotate = rotate) : getLV(fit, Y; rotate = rotate)
    μ̂  = isbin ? predict(fit, Y; N = Nm, type = :response) :
                 predict(fit, Y; type = :response)
    n, K = size(S0)

    # Accumulate aligned replicate scores: n×K×B (only successful replicates kept).
    boots = Array{Float64}(undef, n, K, n_boot)
    nok = 0
    for _ in 1:n_boot
        Yb = if isbin
            _resample_at_mean(fit, μ̂, Nm, rng)
        elseif fit isa GllvmFit
            _resample_at_mean(fit, μ̂, rng)
        else
            _resample_at_mean(fit, μ̂, rng)
        end
        Sb = try
            isbin ? getLV(fit, Yb; N = Nm, rotate = rotate) :
                    getLV(fit, Yb; rotate = rotate)
        catch
            continue
        end
        (Sb === nothing || !all(isfinite, Sb)) && continue
        nok += 1
        boots[:, :, nok] = _procrustes_align(Sb, S0)
    end
    nok ≥ 2 || error("ordination_uncertainty: fewer than 2 usable bootstrap replicates")

    se    = Matrix{Float64}(undef, n, K)
    lower = Matrix{Float64}(undef, n, K)
    upper = Matrix{Float64}(undef, n, K)
    αlo = (1 - level) / 2
    αhi = 1 - αlo
    @inbounds for k in 1:K, s in 1:n
        col = @view boots[s, k, 1:nok]
        se[s, k]    = std(col)
        lower[s, k] = quantile(col, αlo)
        upper[s, k] = quantile(col, αhi)
    end

    return (scores = S0, se = se, lower = lower, upper = upper,
            level = float(level), n_boot = nok)
end
