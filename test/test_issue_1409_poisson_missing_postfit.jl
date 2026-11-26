using GLLVModels, Distributions, Random, Test

# #1409: Poisson Laplace post-fit accepts the same Matrix{Union{Missing,Int}} used at fit time.
@testset "issue 1409: Poisson missing-mask post-fit on training Ym" begin
    Random.seed!(42611)
    Y = rand(Poisson(2), 3, 8)
    Ym = Matrix{Union{Missing, Int}}(Y)
    Ym[1, 1] = missing

    fit = fit_poisson_gllvm(Ym; K = 1, iterations = 120)
    @test fit.converged
    @test isfinite(fit.loglik)

    @test size(predict(fit, Ym)) == size(Ym)
    @test size(fitted(fit, Ym)) == size(Ym)
    @test size(residuals(fit, Ym; type = :pearson)) == size(Ym)
    @test size(getLV(fit, Ym)) == (size(Ym, 2), 1)
    ci = confint(fit, Ym; method = :wald)
    @test ci.pd_hessian
    @test all(isfinite, ci.estimate)
end
