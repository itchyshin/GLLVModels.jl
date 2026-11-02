using GLLVModels, Test, Random, Distributions, StatsModels

@testset "default route honors no-intercept formula (#766)" begin
    Random.seed!(4)
    p, n, K = 4, 80, 1
    x = randn(n)
    Y = 5.0 .+ 0.8 .* reshape(x, 1, n) .+ randn(p, n)
    d = (x = x,)

    fit1 = gllvm(@formula(y ~ 1 + x), Y, d; family = Normal(), K = K)
    fit0 = gllvm(@formula(y ~ 0 + x), Y, d; family = Normal(), K = K)
    fitm = gllvm(@formula(y ~ -1 + x), Y, d; family = Normal(), K = K)
    @test fit0.logLik < fit1.logLik
    @test fitm.logLik ≈ fit0.logLik atol = 1e-8

    X = repeat(reshape(x, 1, n, 1), p, 1, 1)
    ref = fit_gaussian_gllvm(Y; X = X, K = K)
    @test fit0.logLik ≈ ref.logLik atol = 1e-8

    @test_throws ArgumentError gllvm(@formula(y ~ 0 + x), Y, d; family = Poisson(), K = K)
    @test_throws ArgumentError gllvm(@formula(y ~ -1 + x), Y, d; family = Poisson(), K = K)
end
