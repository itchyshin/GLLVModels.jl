using GLLVModels, Test, LinearAlgebra, StableRNGs, Statistics, Distributions, Optim, SpecialFunctions, ForwardDiff

# `_beta_grouped_loglik_site` (the per-site Laplace kernel behind `fit_beta_gllvm_grouped`,
# the default Beta route of `fit_gllvm`) ran undamped Fisher scoring and scored whatever z
# it held at maxiter (#503 class). Where responses sit near 0 or 1 the observed curvature
# exceeds about twice the Fisher curvature, the step overshoots, and the loop 2-cycles:
# on site 14 of the data below it alternated between z = -0.318 and -0.983 around a mode
# at -0.555, and the site value was off by up to 15 log-likelihood units.

const G_BETA = GLLVModels

# Closed-form Beta/logit log-posterior (shares no code with the kernel). μ is clamped at
# 1e-10 here (the package clamps Beta μ at 1e-6); the difference is immaterial on this panel.
function _bgm_logpost(y, Λ, β, φ, z)
    μ = clamp.(1 ./ (1 .+ exp.(-(β .+ Λ * z))), 1e-10, 1 - 1e-10)
    a = μ .* φ; b = (1 .- μ) .* φ
    return sum(loggamma(φ) .- loggamma.(a) .- loggamma.(b) .+ (a .- 1) .* log.(y) .+
               (b .- 1) .* log1p.(-y)) - 0.5 * dot(z, z)
end
# Independent reference mode: L-BFGS with forward-mode AD on the closed form.
function _bgm_refmode(y, Λ, β, φ)
    r = Optim.optimize(z -> -_bgm_logpost(y, Λ, β, φ, z), zeros(size(Λ, 2)), LBFGS(),
                       Optim.Options(g_tol = 1e-11, iterations = 5000); autodiff = :forward)
    return Optim.minimizer(r), Optim.g_residual(r)
end
# Laplace site value at a given z, same formula as the kernel's scoring block.
function _bgm_value(fams, y, Λ, β, z)
    link = G_BETA.LogitLink(); η = β .+ Λ * z
    μ = G_BETA._clamp_mu.(fams, G_BETA.linkinv.(Ref(link), η)); me = G_BETA.mu_eta.(Ref(link), η)
    W = G_BETA._beta_grouped_laplace_weight.(Ref(:observed), fams, μ, me, y, Ref(link), η)
    return sum(G_BETA._glm_logpdf(fams[t], μ[t], 1, y[t]) for t in eachindex(y)) -
           0.5 * dot(z, z) - 0.5 * logdet(Symmetric(Λ' * (W .* Λ) + I))
end

@testset "Beta grouped kernel: every site scores at the mode (part of #503)" begin
    rng = StableRNG(6); p, n, φ, K = 20, 300, 10.0, 1
    Λt = 0.8 .* randn(rng, p, 2); βt = 0.5 .* randn(rng, p)
    μ = 1 ./ (1 .+ exp.(-(βt .+ Λt * randn(rng, 2, n))))
    Y = [clamp(rand(rng, Beta(μ[i] * φ, (1 - μ[i]) * φ)), 1e-6, 1 - 1e-6) for i in CartesianIndices(μ)]
    L = log.(Y ./ (1 .- Y)); β = vec(mean(L; dims = 2)); F = svd(L .- β)
    Λ = F.U[:, 1:K] .* (F.S[1:K]' ./ sqrt(n))       # the fitter's warm start
    fams = [Beta(φ, 1.0) for _ in 1:p]
    worst = 0.0
    for s in 1:n
        y = Y[:, s]
        zr, gr = _bgm_refmode(y, Λ, β, φ)
        @test gr < 1e-8
        v = G_BETA._beta_grouped_loglik_site(fams, y, ones(Int, p), Λ, β, G_BETA.LogitLink())
        @test isfinite(v)
        worst = max(worst, abs(v - _bgm_value(fams, y, Λ, β, zr)))
    end
    @test worst < 1e-6            # before the fix: 15.0 (sites 14 and 78)
end

@testset "Beta grouped kernel: the 2-cycle site" begin
    # Damped Fisher scoring alone still cannot certify a mode at site 14 (the cycle's
    # steps fall under the small-step shortcut, as in #521), so it must report failure
    # rather than a point; the max(observed, Fisher) fallback then reaches the mode.
    rng = StableRNG(6); p, n, φ = 20, 300, 10.0
    Λt = 0.8 .* randn(rng, p, 2); βt = 0.5 .* randn(rng, p)
    μ = 1 ./ (1 .+ exp.(-(βt .+ Λt * randn(rng, 2, n))))
    Y = [clamp(rand(rng, Beta(μ[i] * φ, (1 - μ[i]) * φ)), 1e-6, 1 - 1e-6) for i in CartesianIndices(μ)]
    L = log.(Y ./ (1 .- Y)); β = vec(mean(L; dims = 2)); F = svd(L .- β)
    Λ = F.U[:, 1:1] .* (F.S[1:1]' ./ sqrt(n))
    fams = [Beta(φ, 1.0) for _ in 1:p]; y = Y[:, 14]; link = G_BETA.LogitLink()
    _, okf = G_BETA._beta_grouped_mode(fams, y, ones(Int, p), Λ, β, link, :fisher)
    @test !okf
    z, ok = G_BETA._beta_grouped_mode(fams, y, ones(Int, p), Λ, β, link, :dominant)
    zr, _ = _bgm_refmode(y, Λ, β, φ)
    @test ok
    @test maximum(abs, z .- zr) < 1e-7
end

@testset "Beta grouped restart: precision-plateau trigger" begin
    # θ = [β (2); packed Λ (2); log φ (G)]; the log φ block starts at index 5.
    θ(lφ) = vcat(zeros(4), lφ)
    @test G_BETA._beta_grouped_phi_plateau(θ(log.([2.5, 2.9, 1.7, 2.1, 1139.0])), 5)  # d05 after the kernel fix
    @test !G_BETA._beta_grouped_phi_plateau(θ(log.([2.6, 2.9, 1.8, 2.1, 1.7])), 5)    # d05 optimum
    @test !G_BETA._beta_grouped_phi_plateau(θ([log(5.0)]), 5)                          # one group: never
    @test !G_BETA._beta_grouped_phi_plateau(θ(log.([2.0, 2.0, 190.0])), 5)             # 95x: below the line
    @test G_BETA._beta_grouped_phi_plateau(θ(log.([2.0, 2.0, 210.0])), 5)              # 105x: above it
end

# getLV must return the mode the likelihood used. Before, `_grouped_getLV` used the generic
# `_grouped_laplace_mode`, which stops off the mode at the 2-cycle sites (sites 14 and 78 of
# the panel below, off by up to about 4e-4).
_bgm_grad(y, Λ, β, φ, z) = ForwardDiff.gradient(zz -> _bgm_logpost(y, Λ, β, φ, zz), z)

function _bgm_panel()
    rng = StableRNG(6); p, n, φ = 20, 300, 10.0
    Λt = 0.8 .* randn(rng, p, 2); βt = 0.5 .* randn(rng, p)
    μ = 1 ./ (1 .+ exp.(-(βt .+ Λt * randn(rng, 2, n))))
    Y = [clamp(rand(rng, Beta(μ[i] * φ, (1 - μ[i]) * φ)), 1e-6, 1 - 1e-6) for i in CartesianIndices(μ)]
    L = log.(Y ./ (1 .- Y)); β = vec(mean(L; dims = 2)); F = svd(L .- β)
    return Y, F.U[:, 1:1] .* (F.S[1:1]' ./ sqrt(n)), β, φ
end

@testset "getLV (Beta grouped) returns the mode at every site" begin
    Y, Λ, β, φ = _bgm_panel(); p = size(Y, 1)
    fit = BetaGroupedFit(β, Λ, fill(φ, p), collect(1:p), G_BETA.LogitLink(), NaN, true, 0, :observed)
    Z = getLV(fit, Y; rotate = false)
    @test maximum(s -> norm(_bgm_grad(Y[:, s], Λ, β, φ, Z[s, :])), axes(Y, 2)) < 1e-6
end

@testset "getLV (Beta grouped, covariates) is stationary with the covariate offset" begin
    Y, Λ, β, φ = _bgm_panel(); p, n = size(Y)
    X = randn(StableRNG(540), p, n, 1); γ = [0.2]
    fit = BetaGroupedCovFit(β, γ, [false], Λ, fill(φ, p), collect(1:p), G_BETA.LogitLink(),
                            NaN, true, 0)
    Z = getLV(fit, Y, X; rotate = false)
    O = G_BETA._build_offset(X, γ)
    @test maximum(s -> norm(_bgm_grad(Y[:, s], Λ, β .+ O[:, s], φ, Z[s, :])), axes(Y, 2)) < 1e-6
end
