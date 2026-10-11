# #753: em_fit_phylo must record the σ_eps² = eps() clamp and must not
# report converged = true on that floor. Rank-1 repro from the issue;
# on origin/main e6cfcd4ee this returns σ_eps = sqrt(eps()), converged=true
# after a handful of iterations.

using GLLVModels, Test, Random

@testset "em_fit_phylo σ_eps floor (#753)" begin
    Random.seed!(2)
    p, n = 5, 30
    Σ = [0.6^abs(i - j) for i in 1:p, j in 1:p]
    λ = [1.0, 0.8, -0.5, 0.3, 0.9]
    y = λ * randn(n)'
    fit = em_fit_phylo(y, 1, Σ; max_iter = 3000, assert_monotone = false)

    @test fit.σ_eps == sqrt(eps())
    @test fit.sigma_eps_boundary
    @test !fit.converged
    @test isfinite(fit.logLik)
end
