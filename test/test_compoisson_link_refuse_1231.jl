# Issue #1231: COMPoisson rate model is log-link only; non-log links must be refused.
using GLLVModels, Test, Random, Distributions

@testset "COMPoisson refuses non-log link (#1231)" begin
    rng = MersenneTwister(3)
    p, n = 4, 40
    Y = [rand(rng, Poisson(3)) for _ in 1:p, _ in 1:n]
    msg = "fit_compoisson_gllvm: only LogLink is supported on the rate (λ = exp(η))"
    fit = fit_compoisson_gllvm(Y; K = 1, iterations = 2)
    @test fit.link isa LogLink
    for l in (IdentityLink(), LogitLink(), ProbitLink())
        @test_throws ArgumentError(msg) fit_compoisson_gllvm(Y; K = 1, link = l, iterations = 2)
    end
    @test_throws ArgumentError(msg) fit_gllvm(Y; family = COMPoisson(), K = 1, link = IdentityLink(), iterations = 2)
    Λ = fit.Λ
    β = fit.β
    @test isfinite(compoisson_marginal_loglik_laplace(Y, Λ, β, fit.ν))
    @test_throws ArgumentError compoisson_marginal_loglik_laplace(Y, Λ, β, fit.ν; link = IdentityLink())
end