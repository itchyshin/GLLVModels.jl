# Non-Gaussian missing predictor (mi() axis) — augmented (z, x) Laplace.
#
# A non-Gaussian GLLVM with ONE missing site-level continuous predictor x_s and a
# broadcast slope b_x, x_s ~ N(μ_x, σ_x²), integrated by full-information ML:
#   y[t,s] ~ family(link⁻¹(η[t,s])),  η[t,s] = β_t + b_x x_s + (Λ z_s)_t,  z_s ~ N(0,I_K)
# Supported families: Poisson/log and Binomial/logit (canonical, where the Fisher and
# observed Hessian weights coincide), plus the dispersion families NegativeBinomial/log,
# Gamma/log, and Beta/logit (Track T2). For the dispersion families the implicit step
# uses the OBSERVED Hessian weight W_obs = −∂s/∂η while the log-det uses the Fisher
# weight W_F — exactly the split in src/laplace_grad.jl.
#
# Per site:
#   • x_s OBSERVED  — fold b_x x_s into the offset, Laplace over z (reuse the primal
#       mode + the implicit differentiable Newton step), plus the full x-prior
#       density logN(x_s; μ_x, σ_x²).
#   • x_s MISSING   — integrate x_s: augment the latent to (z, x), joint Laplace
#       with the rank-1 bordered Hessian
#         H = [ Λ'WΛ+I        b_x Λ'W ;
#               b_x (Λ'W)'  b_x²Σ_t W_t + 1/σ_x² ],
#       built with W_obs for the (z,x) mode + implicit step and with W_F for the
#       marginal's log-det,
#       marginal  ℓ(ẑ,x̂) − ½ẑ'ẑ − ½(x̂−μ_x)²/σ_x² − ½log σ_x² − ½logdet(H_F).
#       (The (2π) constants cancel as in the K-dim code; limits to the observed-
#       at-μ_x value as σ_x→0. Verified vs 2-D Gauss–Hermite quadrature.)
#
# AD-clean via the implicit "one differentiable Newton step at the mode" trick
# (mirrors laplace_grad.jl) on the (K+1) augmented system. The dispersion (r/α/φ) is
# carried in the family marker and packed into θ as a log-dispersion by fit_gllvm_mi.
# See docs/dev-log/2026-06-13-nongaussian-mi-design.md.

using LinearAlgebra

# AD-friendly (μ, score, observed Hessian weight, Fisher weight) for each family.
# The bordered (z, x) Newton mode and the implicit dẑ/dθ step use the OBSERVED
# weight W_obs (= −∂s/∂η), the only correct Hessian off the canonical link; the
# log-det term uses the Fisher weight W_F (matching the standalone marginal's
# logdet). For the canonical families the two coincide (W_obs == W_F), so this
# generalises the earlier `_xs_glm` returning a single weight. See laplace_grad.jl
# for the same observed-weight / Fisher-weight split (issue #65, NB/Gamma/Beta).
_xs_glm(::Poisson, ::LogLink, η, n, y) = (μ = exp.(η); W = μ; (μ, y .- μ, W, W))
function _xs_glm(::Binomial, ::LogitLink, η, n, y)
    μ = 1 ./ (1 .+ exp.(-η))
    W = n .* μ .* (1 .- μ)
    return (μ, y .- n .* μ, W, W)
end
# NB2 (log): score (y−μ)/(1+μ/r); W_obs = μr(r+y)/(r+μ)²; W_F = μ/(1+μ/r).
function _xs_glm(f::NegativeBinomial, ::LogLink, η, n, y)
    r = f.r
    μ = exp.(η)
    s = (y .- μ) ./ (1 .+ μ ./ r)
    Wobs = μ .* r .* (r .+ y) ./ (r .+ μ) .^ 2
    WF = μ ./ (1 .+ μ ./ r)
    return (μ, s, Wobs, WF)
end
# Gamma (log, shape α): score α(y−μ)/μ; W_obs = αy/μ; W_F = α.
function _xs_glm(f::Gamma, ::LogLink, η, n, y)
    α = f.α
    μ = exp.(η)
    s = α .* (y .- μ) ./ μ
    Wobs = α .* y ./ μ
    WF = fill(α, length(μ))
    return (μ, s, Wobs, WF)
end
# Beta (logit, precision φ): Ferrari & Cribari-Neto score; W_obs = −∂s/∂η via a 1-D
# AD-derivative of the scalar score (digamma/trigamma terms, exact, low-risk); the
# log-det uses the Fisher weight φ²[ψ′(μφ)+ψ′((1−μ)φ)] me². (Mirrors laplace_grad.jl.)
_beta_xs_score_scalar(η, φ, y) = begin
    μ = 1 / (1 + exp(-η)); me = μ * (1 - μ)
    ystar = log(y) - log1p(-y)
    μstar = digamma(μ * φ) - digamma((1 - μ) * φ)
    return φ * (ystar - μstar) * me
end
function _xs_glm(f::Beta, ::LogitLink, η, n, y)
    φ = f.α
    μ = 1 ./ (1 .+ exp.(-η))
    me = μ .* (1 .- μ)
    ystar = log.(y) .- log1p.(-y)
    μstar = digamma.(μ .* φ) .- digamma.((1 .- μ) .* φ)
    s = φ .* (ystar .- μstar) .* me
    ηv = ForwardDiff.value.(η); φv = ForwardDiff.value(φ)
    Wobs = [-ForwardDiff.derivative(η_ -> _beta_xs_score_scalar(η_, φv, y[t]), ηv[t]) for t in eachindex(y)]
    νF = trigamma.(μ .* φ) .+ trigamma.((1 .- μ) .* φ)
    WF = φ .^ 2 .* νF .* me .^ 2
    return (μ, s, Wobs, WF)
end

# Full AD-friendly log-pmf (the normalising constants are θ-independent, so they do
# not affect the gradient — but they ARE needed for the marginal VALUE to be a
# proper likelihood and to match the full-pmf oracles).
_binom_logpmf_full(μ, n, y) = y * log(μ) + (n - y) * log1p(-μ) +
                              loggamma(n + 1.0) - loggamma(y + 1.0) - loggamma(n - y + 1.0)
# NB/Gamma/Beta log-density kernels include the dispersion-dependent normalisers
# (functions of r/α/φ, which ARE θ parameters here), so they must be kept and
# differentiated; the dispersion-FREE constants (e.g. loggamma(y+1)) are still dropped.
# NB2 goes through the stable `_nb2_logpdf_mean` (src/families/negbin.jl): the
# expanded loggamma(r+y) − loggamma(r) + r log r − (r+y) log(r+μ) cancels at large r (#871).
_nb_logpmf_full(μ, r, y) = _nb2_logpdf_mean(μ, r, Int(y))
_gamma_logpdf_full(μ, α, y) = (α - 1) * log(y) - y * α / μ - α * log(μ) + α * log(α) - loggamma(α)
_beta_logpdf_full(μ, φ, y) = (μ * φ - 1) * log(y) + ((1 - μ) * φ - 1) * log1p(-y) -
                             (loggamma(μ * φ) + loggamma((1 - μ) * φ) - loggamma(φ))
_xs_logker_sum(::Poisson, μ, n, y) = sum(_pois_logpmf(μ[t], y[t]) for t in eachindex(y))
_xs_logker_sum(::Binomial, μ, n, y) = sum(_binom_logpmf_full(μ[t], n[t], y[t]) for t in eachindex(y))
_xs_logker_sum(f::NegativeBinomial, μ, n, y) = sum(_nb_logpmf_full(μ[t], f.r, y[t]) for t in eachindex(y))
_xs_logker_sum(f::Gamma, μ, n, y) = sum(_gamma_logpdf_full(μ[t], f.α, y[t]) for t in eachindex(y))
_xs_logker_sum(f::Beta, μ, n, y) = sum(_beta_logpdf_full(μ[t], f.α, y[t]) for t in eachindex(y))

# Strip any ForwardDiff dual from the family's dispersion field, so the primal
# Float64 mode solves never see a Dual dispersion (canonical families pass through).
_mi_primal_family(f::NegativeBinomial) = NegativeBinomial(ForwardDiff.value(f.r), 0.5; check_args = false)
_mi_primal_family(f::Gamma) = Gamma(ForwardDiff.value(f.α), 1.0; check_args = false)
_mi_primal_family(f::Beta) = Beta(ForwardDiff.value(f.α), 1.0; check_args = false)
_mi_primal_family(f) = f

# Primal augmented (z, x) Newton mode (Float64; called on primal parameter values).
function _mode_xs(family, y, n, Λ, β, link, b_x::Real, μ_x::Real, σ_x2::Real;
                  maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    z = zeros(K)
    x = float(μ_x)
    for _ in 1:maxiter
        η = _clamp_eta.(β .+ b_x * x .+ Λ * z)
        μ, s, Wobs, _ = _xs_glm(family, link, η, n, y)
        gz = Λ' * s .- z
        gx = b_x * sum(s) - (x - μ_x) / σ_x2
        A = Λ' * (Wobs .* Λ) + I
        c = b_x .* (Λ' * Wobs)
        dd = b_x^2 * sum(Wobs) + 1 / σ_x2
        H = [A c; c' dd]
        rhs = vcat(gz, gx)
        (all(isfinite, H) && all(isfinite, rhs)) || break   # non-finite weights ⇒ stop
        Δ = _safe_solve(H, rhs)                              # nothing on singular/failed
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ[1:K]
        x = x + Δ[K + 1]
        maximum(abs, Δ) < tol && break
    end
    return z, x
end

# Safe linear solve for the differentiable (possibly Dual) augmented system: returns a
# NaN-filled vector (right element type) instead of throwing when the matrix/RHS carry
# Infs/NaNs (LAPACK's chkfinite throws). A transient L-BFGS probe driving the
# dispersion to an overflow regime then yields a NaN objective, and Optim's line search
# steps back — it does not abort the fit. The interior optimum is unaffected (the
# weights are finite there, so this path is never taken).
function _safe_solve_xs(M, b)
    (all(isfinite, M) && all(isfinite, b)) || return fill(eltype(b)(NaN), length(b))
    return try
        M \ b
    catch
        fill(eltype(b)(NaN), length(b))
    end
end

# logdet that returns NaN (right element type) on a non-finite matrix instead of
# throwing inside LAPACK (same transient-probe rationale as `_safe_solve_xs`).
function _safe_logdet_xs(M)
    all(isfinite, M) || return eltype(M)(NaN)
    return try
        logdet(M)
    catch
        eltype(M)(NaN)
    end
end

# Differentiable per-site marginal with the site-predictor x_s (observed or missing).
# β, Λ, b_x, μ_x, σ_x2 may carry duals; the mode is primal (no dual leakage).
function _site_xs_diffable(family, y, n, Λ, β, link, x_obs, b_x, μ_x, σ_x2)
    K = size(Λ, 2)
    Λv = ForwardDiff.value.(Λ)
    βv = ForwardDiff.value.(β)
    bxv = ForwardDiff.value(b_x)
    mxv = ForwardDiff.value(μ_x)
    sx2v = ForwardDiff.value(σ_x2)
    famv = _mi_primal_family(family)            # primal-dispersion family for mode solves

    if x_obs !== nothing
        ẑ = _laplace_mode(famv, y, n, Λv, βv, link; offset = bxv * x_obs)
        off = b_x * x_obs
        η = _clamp_eta.(β .+ off .+ Λ * ẑ)
        μ, s, Wobs, _ = _xs_glm(family, link, η, n, y)
        A = Λ' * (Wobs .* Λ) + I             # observed weight ⇒ implicit dẑ/dθ
        z = ẑ .+ _safe_solve_xs(A, Λ' * s .- ẑ)
        ηz = _clamp_eta.(β .+ off .+ Λ * z)
        μz, _, _, WFz = _xs_glm(family, link, ηz, n, y)
        Az = Λ' * (WFz .* Λ) + I             # Fisher weight ⇒ logdet
        ℓ = _xs_logker_sum(family, μz, n, y)
        lpx = -0.5 * log(2π) - 0.5 * log(σ_x2) - 0.5 * (x_obs - μ_x)^2 / σ_x2
        return ℓ - 0.5 * dot(z, z) - 0.5 * _safe_logdet_xs(Az) + lpx
    else
        ẑ, x̂ = _mode_xs(famv, y, n, Λv, βv, link, bxv, mxv, sx2v)
        η = _clamp_eta.(β .+ b_x * x̂ .+ Λ * ẑ)
        μ, s, Wobs, _ = _xs_glm(family, link, η, n, y)
        gz = Λ' * s .- ẑ
        gx = b_x * sum(s) - (x̂ - μ_x) / σ_x2
        A = Λ' * (Wobs .* Λ) + I             # observed weight ⇒ implicit d(ẑ,x̂)/dθ
        c = b_x .* (Λ' * Wobs)
        dd = b_x^2 * sum(Wobs) + 1 / σ_x2
        H = [A c; c' dd]
        Δ = _safe_solve_xs(H, vcat(gz, gx))
        z = ẑ .+ Δ[1:K]
        x = x̂ + Δ[K + 1]
        ηz = _clamp_eta.(β .+ b_x * x .+ Λ * z)
        μz, _, _, WFz = _xs_glm(family, link, ηz, n, y)
        Az = Λ' * (WFz .* Λ) + I             # Fisher weight ⇒ logdet
        cz = b_x .* (Λ' * WFz)
        ddz = b_x^2 * sum(WFz) + 1 / σ_x2
        Hz = [Az cz; cz' ddz]
        ℓ = _xs_logker_sum(family, μz, n, y)
        return ℓ - 0.5 * dot(z, z) - 0.5 * (x - μ_x)^2 / σ_x2 -
               0.5 * log(σ_x2) - 0.5 * _safe_logdet_xs(Hz)
    end
end

_xs_supported(family, link) =
    (family isa Poisson && link isa LogLink) || (family isa Binomial && link isa LogitLink) ||
    (family isa NegativeBinomial && link isa LogLink) || (family isa Gamma && link isa LogLink) ||
    (family isa Beta && link isa LogitLink)

"""
    laplace_loglik_site_xs(family, y, n, Λ, β, link; x_obs, b_x, μ_x, σ_x2) -> Float64

Per-site Laplace marginal for a non-Gaussian GLLVM with one site-level predictor
`x_s` (observed value `x_obs`, or `nothing` if missing-and-integrated), broadcast
slope `b_x`, predictor model `x_s ~ N(μ_x, σ_x²)`. Canonical families:
`Poisson()` + `LogLink` and `Binomial()` + `LogitLink`.
"""
function laplace_loglik_site_xs(family, y::AbstractVector, n::AbstractVector,
                                Λ::AbstractMatrix, β::AbstractVector, link;
                                x_obs, b_x, μ_x, σ_x2)
    _xs_supported(family, link) || throw(ArgumentError(
        "laplace_loglik_site_xs: supports Poisson()/NegativeBinomial()/Gamma() + LogLink and " *
        "Binomial()/Beta() + LogitLink."))
    return _site_xs_diffable(family, y, n, Λ, β, link, x_obs, b_x, μ_x, σ_x2)
end

"""
    marginal_loglik_laplace_xs(family, Y, N, Λ, β, link; x, b_x, μ_x, σ_x2) -> Float64

Total Laplace log-marginal over sites for a non-Gaussian GLLVM with one missing
site-level continuous predictor integrated by FIML (mi() axis). Supported families:
Poisson/log and Binomial/logit (canonical), plus the dispersion families
NegativeBinomial/log, Gamma/log, and Beta/logit (the dispersion `r`/`α`/`φ` is
carried in the family marker). `x` is length-n (entries may be `missing`/`NaN`).
Observed sites contribute the joint `(y_s, x_s)` density; missing sites contribute
the marginal of `y_s` with `x_s` integrated out.
"""
function marginal_loglik_laplace_xs(family, Y::AbstractMatrix, N::AbstractMatrix,
                                    Λ::AbstractMatrix, β::AbstractVector, link;
                                    x::AbstractVector, b_x, μ_x, σ_x2)
    acc = zero(promote_type(eltype(Λ), eltype(β), typeof(b_x), typeof(μ_x), typeof(σ_x2)))
    @inbounds for s in axes(Y, 2)
        xs = x[s]
        x_obs = (ismissing(xs) || (xs isa Real && isnan(xs))) ? nothing : Float64(xs)
        acc += laplace_loglik_site_xs(family, view(Y, :, s), view(N, :, s), Λ, β, link;
                                      x_obs = x_obs, b_x = b_x, μ_x = μ_x, σ_x2 = σ_x2)
    end
    return acc
end

_mi_canonical_link(::Poisson) = LogLink()
_mi_canonical_link(::Binomial) = LogitLink()
_mi_canonical_link(::NegativeBinomial) = LogLink()
_mi_canonical_link(::Gamma) = LogLink()
_mi_canonical_link(::Beta) = LogitLink()

# Dispersion bookkeeping for the mi() fitter. Canonical families (Poisson/Binomial)
# carry no dispersion in θ; the three dispersion families do — packed as the LAST
# θ entry on the log scale (mirroring fit_nb_gllvm / fit_gamma_gllvm / fit_beta_gllvm).
_mi_has_dispersion(::Poisson) = false
_mi_has_dispersion(::Binomial) = false
_mi_has_dispersion(::NegativeBinomial) = true
_mi_has_dispersion(::Gamma) = true
_mi_has_dispersion(::Beta) = true

_mi_logdisp0(::NegativeBinomial) = log(10.0)   # mirrors fit_nb_gllvm r₀
_mi_logdisp0(::Gamma) = log(2.0)               # mirrors fit_gamma_gllvm α₀
_mi_logdisp0(::Beta) = log(10.0)               # mirrors fit_beta_gllvm φ₀

# Rebuild the family marker carrying a (dual-or-Float) dispersion value `d`. Only the
# dispersion field is used by the `_xs_glm` pieces (NB: r; Gamma: α; Beta: φ).
# `check_args=false`: the L-BFGS line search transiently probes θ whose `exp(·)`
# underflows the dispersion to ≈0; the marginal math is finite there, so the family
# marker must not throw on a non-positive value (Optim then steps away on its own).
_mi_with_disp(::NegativeBinomial, d) = NegativeBinomial(d, 0.5; check_args = false)
_mi_with_disp(::Gamma, d) = Gamma(d, one(d); check_args = false)
_mi_with_disp(::Beta, d) = Beta(d, one(d); check_args = false)

function _mi_init_lambda(family, Y, N, β0, K)
    p, n = size(Y)
    Z = Matrix{Float64}(undef, p, n)
    if family isa Poisson || family isa NegativeBinomial
        @. Z = log(Y + 0.5) - β0
    elseif family isa Gamma
        @. Z = log(max(Y, 1e-6)) - β0
    elseif family isa Beta
        @. Z = log(clamp(Y, 1e-6, 1 - 1e-6) / (1 - clamp(Y, 1e-6, 1 - 1e-6))) - β0
    else
        @. Z = log((Y + 0.5) / (N - Y + 0.5)) - β0
    end
    C = Symmetric((Z * Z') ./ n)
    E = eigen(C)
    idx = sortperm(E.values, rev = true)[1:K]
    return E.vectors[:, idx] .* sqrt.(max.(E.values[idx], 1e-2))'
end

"""
    fit_gllvm_mi(family, Y, x; K, N=nothing, link=canonical, ...) -> NamedTuple

Fit a non-Gaussian GLLVM with one site-level continuous predictor `x` (length
`n`, entries may be `missing`/`NaN`), where the missing `x_s` are integrated out
by full-information ML (the augmented (z,x) Laplace of
`marginal_loglik_laplace_xs`). Supported families: `Poisson()`+`LogLink` and
`Binomial()`+`LogitLink` (canonical), plus the dispersion families
`NegativeBinomial()`+`LogLink`, `Gamma()`+`LogLink`, `Beta()`+`LogitLink` (their
dispersion `r`/`α`/`φ` is estimated jointly). `Y` is `p × n`; `N` is the `p × n`
trial-count matrix for Binomial (`ones` for the others). The predictor enters with a
single broadcast slope `b_x` and predictor model `x ~ N(μ_x, σ_x²)`.

Returns a NamedTuple with `β`, `Λ` (`p × K`), `b_x`, `μ_x`, `σ_x`, `dispersion`
(`nothing` for canonical families), `logLik`, `converged`, `n_missing`. Optimised
with L-BFGS over the AD-clean marginal.
"""
function fit_gllvm_mi(family, Y::AbstractMatrix, x::AbstractVector; K::Integer,
                      N::Union{Nothing,AbstractMatrix} = nothing,
                      link = _mi_canonical_link(family),
                      g_tol::Real = 1e-8, iterations::Integer = 1000)
    p, n = size(Y)
    length(x) == n || throw(ArgumentError("length(x) = $(length(x)) must equal n = $n."))
    K ≥ 1 || throw(ArgumentError("K must be ≥ 1."))
    _xs_supported(family, link) || throw(ArgumentError(
        "fit_gllvm_mi: supports Poisson()/NegativeBinomial()/Gamma() + LogLink and " *
        "Binomial()/Beta() + LogitLink."))
    Nm = N === nothing ? ones(Int, p, n) : N

    isobs = [!(ismissing(xi) || (xi isa Real && isnan(xi))) for xi in x]
    any(isobs) || throw(ArgumentError("x has no observed values."))
    xo = Float64[Float64(x[i]) for i in findall(isobs)]
    μ_x0 = Statistics.mean(xo)
    σ_x0 = max(Statistics.std(xo), 1e-2)

    rowmean = vec(sum(Y, dims = 2)) ./ n
    β0 = if family isa Poisson || family isa NegativeBinomial
        log.(max.(rowmean, 0.5))
    elseif family isa Gamma
        log.(max.(rowmean, 1e-3))
    elseif family isa Beta
        log.(clamp.(rowmean, 1e-3, 1 - 1e-3) ./ (1 .- clamp.(rowmean, 1e-3, 1 - 1e-3)))
    else
        log.((rowmean .+ 0.5) ./ (vec(sum(Nm, dims = 2)) ./ n .- rowmean .+ 0.5))
    end
    Λ0 = _mi_init_lambda(family, Y, Nm, β0, K)

    has_disp = _mi_has_dispersion(family)
    base = p + p * K
    θ0 = has_disp ? vcat(β0, vec(Λ0), 0.0, μ_x0, log(σ_x0^2), _mi_logdisp0(family)) :
                    vcat(β0, vec(Λ0), 0.0, μ_x0, log(σ_x0^2))
    function negll(θ)
        β = θ[1:p]
        Λ = reshape(θ[(p + 1):base], p, K)
        bx = θ[base + 1]
        mx = θ[base + 2]
        sx2 = exp(θ[base + 3])
        fam = has_disp ? _mi_with_disp(family, exp(θ[base + 4])) : family
        return -marginal_loglik_laplace_xs(fam, Y, Nm, Λ, β, link;
                                           x = x, b_x = bx, μ_x = mx, σ_x2 = sx2)
    end
    # Dispersion families need the more forgiving BackTracking line search (matches
    # fit_nb/gamma/beta_gllvm): the dispersion direction can over-probe under the
    # default HagerZhang search. Canonical families keep the default for regression.
    ls = has_disp ? Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)) :
                    Optim.LBFGS()
    res = Optim.optimize(negll, θ0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :forward)
    θ = Optim.minimizer(res)
    β = θ[1:p]
    Λ = reshape(θ[(p + 1):base], p, K)
    b_x = θ[base + 1]
    μ_x = θ[base + 2]
    σ_x = exp(0.5 * θ[base + 3])
    dispersion = has_disp ? exp(θ[base + 4]) : nothing
    return (β = β, Λ = Λ, b_x = b_x, μ_x = μ_x, σ_x = σ_x, dispersion = dispersion,
            logLik = _fit_verdict(res)[1], converged = _fit_verdict(res)[2],
            n_missing = count(!, isobs))
end
