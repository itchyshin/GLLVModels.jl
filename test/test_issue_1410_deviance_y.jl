using GLLVModels, Random, Test

# #1410: deviance(fit, Ym) after a masked Gaussian fit matches deviance(fit).
@testset "issue 1410: deviance(fit, Y) with Missing eltype" begin
    Random.seed!(1)
    Ym = Matrix{Union{Missing, Float64}}(randn(3, 5))
    Ym[1, 1] = missing
    fit = fit_gaussian_gllvm(Ym; K = 1, iterations = 100)
    @test deviance(fit, Ym) ≈ deviance(fit) atol = 0.0
    @test deviance(fit, Ym) ≈ -2 * loglikelihood(fit) atol = 0.0
end
