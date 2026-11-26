using GLLVModels, Test, Random
@testset "LognormalFit predict (#1418)" begin
    Random.seed!(39)
    p, K, n = 3, 1, 40
    Y = abs.(randn(p, n)) .+ 0.05
    fit = fit_lognormal_gllvm(Y; K = K, iterations = 80)
    P = predict(fit, Y)
    @test size(P) == (p, n) && all(P .> 0)
end
