# Issue #1482: uniform weights + legacy BFGS continuation must not report
# `converged = true` when the retained stage's gradient residual fails the
# scale-aware #505 test (`_nb1_grouped_g_met`), even if Optim claims success.
using Random, Distributions, GLLVModels, Optim, Test

@testset "#1482 weighted Poisson: no converged=true on huge continuation gradient" begin
    Random.seed!(777)
    p, n = 4, 25
    Y = [rand(Poisson(2)) for _ in 1:p, _ in 1:n]
    f = fit_gllvm(Y; family = Poisson(), K = 1, weights = fill(1e8, p, n), iterations = 200)
    @test !f.converged
    info = optimizer_info(f)
    @test info !== nothing
    cont = only(stage for stage in info.stages if stage.name === :legacy_bfgs_continuation)
    @test cont.accepted
    gres = Optim.g_residual(cont.result)
    g_tol = Float64(info.controls.g_tol)
    @test !(Optim.converged(cont.result) && GLLVModels._nb1_grouped_g_met(cont.result, g_tol))
    @test gres > g_tol
end