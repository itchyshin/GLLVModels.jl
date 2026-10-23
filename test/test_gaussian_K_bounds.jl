using GLLVModels, Test, Random

@testset "fit_gaussian_gllvm K bounds (exact path)" begin
    Random.seed!(1)
    p, n = 3, 5
    Y = randn(p, n)
    K = p + 1
    @test_throws ArgumentError("K must lie in 1:p (got K = $K, p = $p)") fit_gaussian_gllvm(Y; K = K)
end
