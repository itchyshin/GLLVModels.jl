using GLLVModels, Test, Random, Distributions

@testset "issues #1475 / #1476 post-fit guards" begin
    @testset "#1475 binomial trials omitted reuse stored N" begin
        Random.seed!(17)
        p, n, K = 4, 30, 1
        Y = [rand(Binomial(10, 0.4)) for _ in 1:p, _ in 1:n]
        Y = reshape(Y, p, n)
        N = fill(10, p, n)
        fit = fit_binomial_gllvm(Y; K = K, N = N, iterations = 50)
        @test fit.trials == N
        @test fit.n_sites == n
        S1 = getLV(fit, Y; N = N, rotate = false)
        S2 = getLV(fit, Y; rotate = false)
        @test S1 ≈ S2
        μ1 = predict(fit, Y; N = N)
        μ2 = predict(fit, Y)
        @test μ1 ≈ μ2
    end

    @testset "#1475 Bernoulli fit still defaults trials to 1" begin
        Random.seed!(3)
        p, n, K = 3, 12, 1
        Y = rand(Bernoulli(0.35), p, n)
        fit = fit_binomial_gllvm(Y; K = K)
        @test fit.trials === nothing
        @test getLV(fit, Y; rotate = false) isa Matrix{Float64}
    end

    @testset "#1476 getLV/ordination refuse wrong site count" begin
        Random.seed!(1)
        Y = [rand(Poisson(4)) for _ in 1:5, _ in 1:30]
        Y = reshape(Y, 5, 30)
        fit = fit_poisson_gllvm(Y; K = 1)
        @test fit.n_sites == 30
        @test_throws DimensionMismatch getLV(fit, Y[:, 1:20])
        Yextra = rand(Poisson(4), 5, 5)
        @test_throws DimensionMismatch getLV(fit, hcat(Y, Yextra))
        @test_throws DimensionMismatch ordination(fit, Y[:, 1:20])
    end
end