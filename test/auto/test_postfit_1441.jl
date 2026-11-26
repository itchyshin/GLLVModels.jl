using GLLVModels, Test, Random
@testset "StudentTFit post-fit (#1441)" begin
    Random.seed!(99)
    Y = abs.(randn(5, 40)) .+ 0.5
    fit = fit_studentt_gllvm(Y; K = 1, nu = 5.0, g_tol = 1e-5, iterations = 500)
    @test size(predict(fit, Y)) == size(Y)
    @test size(fitted(fit, Y)) == size(Y)
    @test size(getLV(fit, Y)) == (40, 1)
    @test size(ordination(fit, Y).sites) == (40, 1)
    @test size(residuals(fit, Y)) == size(Y)
end
