# refs #1525: Laplace PoissonFit with a stored offset — simulate(fit, n) must use the
# same offset in η as predict(fit, Y; type = :response).
using GLLVModels, Test, Random, Distributions, Statistics

@testset "Laplace Poisson simulate uses stored offset (refs #1525)" begin
    Random.seed!(42)
    p, n = 3, 10
    Y = rand(Poisson(2), p, n)
    off = [0.3 + 0.1 * t + 0.05 * s for t in 1:p, s in 1:n]
    fp = fit_gllvm(Y; family = Poisson(), K = 1, offset = off, iterations = 80)
    @test fp.offset !== nothing

    μp = vec(mean(predict(fp, Y; type = :response); dims = 2))
    acc = zeros(p)
    for rep in 1:400
        acc .+= vec(mean(simulate(fp, n; rng = MersenneTwister(rep + 11_525)); dims = 2))
    end
    μs = acc ./ 400

    for t in 1:p
        @test μs[t] ≈ μp[t] rtol = 0.12
    end

    @test simulate(fp, n; offset = off, rng = MersenneTwister(99)) isa Matrix{Int}
    @test_throws ArgumentError simulate(fp, n + 2)
end

@testset "homogeneous offset issue repro (refs #1525)" begin
    Random.seed!(42)
    p, n = 3, 10
    Y = rand(Poisson(2), p, n)
    off = fill(0.5, p, n)
    fp = fit_gllvm(Y; family = Poisson(), K = 1, offset = off, iterations = 80)
    μp = vec(mean(predict(fp, Y; type = :response); dims = 2))
    μs = vec(mean(simulate(fp, 5000; rng = MersenneTwister(42)); dims = 2))
    for t in 1:p
        @test μs[t] ≈ μp[t] rtol = 0.1
    end
end