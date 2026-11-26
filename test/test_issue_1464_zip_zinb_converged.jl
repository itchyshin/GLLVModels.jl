using Test, GLLVModels, Distributions, Random

@testset "issue 1464: ZIP/ZINB all-zero species column => converged false" begin
    Random.seed!(1126)
    Y = rand(Poisson(2), 4, 10)
    Y[2, :] .= 0

    fit_z = fit_zip_gllvm(copy(Y); K = 1, iterations = 150)
    fit_i = fit_zinb_gllvm(copy(Y); K = 1, iterations = 150)
    @test !fit_z.converged
    @test !fit_i.converged
    @test isfinite(fit_z.loglik) && isfinite(fit_i.loglik)
end