# Issue #1422: all-zero Poisson GLLVM is not reported converged (warning unchanged).
using Test, GLLVModels, Distributions

@testset "All-zero Poisson fit is not converged (#1422)" begin
    Y = zeros(Int, 6, 45)
    fit = fit_gllvm(Y; family = Poisson(), K = 2, iterations = 80)
    @test isfinite(fit.loglik)
    @test !fit.converged
end