# Regression: σ_phy[t] has an identity (signed) link, so `profile_ci` must
# return bounds on the same (signed) scale as the estimate and the Wald CI.
# Before the fix `_profile_all_term_names` marked it :log_sd and the profile
# bounds were exp()-ed, giving an interval that did not contain the estimate.

using Test
using Random
using LinearAlgebra
using GLLVModels

@testset "profile_ci sigma_phy[t]: identity scale (not exp'd)" begin
    rng = MersenneTwister(7)
    p, n, K = 4, 80, 1
    Λtrue = randn(rng, p, K)
    σ_eps = 0.4
    z = randn(rng, n, K)
    y = Λtrue * z' .+ σ_eps .* randn(rng, p, n)   # draw kept for RNG parity
    Σ_phy = Matrix{Float64}(I, p, p)
    σ_phy_true = fill(0.3, p)
    φ = Σ_phy * randn(rng, p)
    y2 = Λtrue * z' .+ σ_eps .* randn(rng, p, n)
    y2 .+= σ_phy_true .* φ
    fit = fit_gaussian_gllvm(y2; K = K, has_phy_unique = true, Σ_phy = Σ_phy)

    names, kinds = GLLVModels._profile_all_term_names(fit)
    i = findfirst(==("sigma_phy[1]"), names)
    @test kinds[i] === :linear
    est = fit.pars.θ_packed[i]

    prof = GLLVModels.profile_ci(fit, "sigma_phy[1]"; y = y2, Σ_phy = Σ_phy, max_expand = 10)
    @test prof.method === :profile
    @test prof.lower <= est <= prof.upper

    wald = confint(fit, y2; Σ_phy = Σ_phy, parm = "sigma_phy[1]")
    @test wald.estimate[1] ≈ est
    # Same scale as Wald: the intervals overlap. The Wald interval straddles 0
    # (sigma_phy is signed and weakly identified here), so a signed-scale
    # profile interval must reach below 0; an exp()-ed one never could.
    @test max(prof.lower, wald.lower[1]) <= min(prof.upper, wald.upper[1])
    @test wald.lower[1] < 0 < wald.upper[1]
    @test prof.lower < 0 < prof.upper
end
