# refs #1549 — bootstrap_ci dispatches on ZIPFit / ZINBFit like confint(..., method = :bootstrap).
using Test, GLLVModels, Random
using GLLVModels: bootstrap_ci, confint

@testset "bootstrap_ci ZIPFit (#1549)" begin
    Random.seed!(1004)
    Y = rand(0:3, 5, 60)
    fit = fit_zip_gllvm(Y; K = 1, iterations = 60)
    ci_conf = confint(fit, Y; method = :bootstrap, n_boot = 25, seed = 2, parm = "betac[1]")
    ci_boot = bootstrap_ci(fit, Y; parm = "betac[1]", n_boot = 25, seed = 2)
    @test ci_boot.method == :bootstrap
    @test ci_boot.term == ci_conf.term
    @test ci_boot.estimate ≈ ci_conf.estimate
    @test ci_boot.lower ≈ ci_conf.lower
    @test ci_boot.upper ≈ ci_conf.upper
    @test ci_boot.n_converged == ci_conf.n_converged
end