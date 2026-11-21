using Test, GLLVModels, Random
const G = GLLVModels

# refs #1149: loglikelihood / dof / aic / bic / nobs for OrderedBetaFit, QuadraticFit and
# MixedFamilyFit (each has a `loglik` field but had no StatsAPI methods).
@testset "loglikelihood/aic/bic for OrderedBeta, Quadratic, MixedFamily fits (#1149)" begin
    Random.seed!(3)
    p, n = 6, 60
    Y  = Float64.(rand(0:6, p, n))
    Yb = clamp.(rand(p, n), 0.05, 0.95)
    # (label, fit, data, expected dof) with p = 6, K = 1:
    #   ordered beta: beta 6 + Lambda 6 + c0, c1, phi 3 = 15
    #   quadratic Poisson (no dispersion): beta 6 + Lambda 6 + D 6 = 18
    #   mixed, all Poisson (n_disp = 0): beta 6 + Lambda 6 = 12
    cases = (
        ("OrderedBetaFit", fit_ordered_beta_gllvm(Yb; K = 1), Yb, 15),
        ("QuadraticFit",   fit_quadratic_gllvm(Y; K = 1),     Y,  18),
        ("MixedFamilyFit", fit_mixed_gllvm(Y; families = [G.Poisson() for _ in 1:p], K = 1), Y, 12),
    )
    for (name, f, y, k) in cases
        @testset "$name" begin
            ll = G.StatsAPI.loglikelihood(f)
            @test ll == f.loglik
            @test G.StatsAPI.dof(f) == k
            @test G.StatsAPI.nobs(f, y) == p * n
            @test G.StatsAPI.nobs(f, y; mask = trues(p, n)) == p * n
            @test G.StatsAPI.aic(f) ≈ 2k - 2ll
            @test G.StatsAPI.bic(f, n) ≈ k * log(n) - 2ll
            @test G.StatsAPI.bic(f, y) ≈ k * log(p * n) - 2ll
        end
    end
end
