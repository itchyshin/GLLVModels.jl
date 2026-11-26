using GLLVModels, Test, Random, Distributions
@testset "simulate Delta/Lognormal (#1435)" begin
    rng = MersenneTwister(26012)
    p, n, K = 4, 35, 1
    Λ = reshape(randn(rng, p * K) .* 0.35, p, K)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n
        z = randn(rng, K)
        η = Λ * z
        for t in 1:p
            π = 1 / (1 + exp(-η[t]))
            Y[t, s] = rand(rng) > π ? 0.0 : rand(rng, LogNormal(η[t], 0.7))
        end
    end
    fit = fit_gllvm(Y; family = DeltaLogNormal(), K = K, iterations = 400)
    @test size(simulate(fit, 20; rng = MersenneTwister(1))) == (p, 20)
    Y2 = exp.(randn(rng, 4, 30))
    fit2 = fit_lognormal_gllvm(Y2; K = 1, iterations = 400)
    @test size(simulate(fit2, 20; rng = MersenneTwister(2))) == (4, 20)
    Y3 = rand(rng, Gamma(2, 1), 4, 28)
    Y3[1, 3] = 0
    fit3 = fit_gllvm(Y3; family = DeltaGamma(), K = 1, iterations = 400)
    @test size(simulate(fit3, 15; rng = MersenneTwister(3))) == (4, 15)
end
