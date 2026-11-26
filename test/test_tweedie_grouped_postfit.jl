using GLLVModels, Test, Random, Statistics, LinearAlgebra

# Issue #1463: TweedieGroupedFit lacked predict, fitted, residuals and getLV.

@testset "#1463 TweedieGroupedFit post-fit predict, fitted, residuals, getLV" begin
    rng = MersenneTwister(1463)
    p, K, n = 4, 1, 70
    Random.seed!(42)
    Y = abs.(randn(p, n)) .+ 0.1

  @testset "fitted route from optimiser" begin
        fit = fit_tweedie_gllvm_grouped(Y; K = K, power = 1.5, iterations = 350)
        @test fit isa TweedieGroupedFit
        @test isfinite(fit.loglik)

        Z = getLV(fit, Y)
        @test size(Z) == (n, K)
        @test all(isfinite, Z)

        L = predict(fit, Y; type = :link)
        M = predict(fit, Y; type = :response)
        F = fitted(fit, Y)
        @test size(L) == (p, n)
        @test size(M) == (p, n)
        @test M ≈ F
        @test all(M .> 0)
        @test L ≈ fit.β .+ fit.Λ * permutedims(getLV(fit, Y; rotate = false))

        Rp = residuals(fit, Y; type = :pearson)
        @test size(Rp) == (p, n)
        @test all(isfinite, Rp)

        rng2 = MersenneTwister(77)
        Rd = residuals(fit, Y; type = :dunnsmyth, rng = rng2)
        @test size(Rd) == (p, n)
        @test all(isfinite, Rd)
        @test residuals(fit, Y; type = :dunnsmyth, rng = rng2) ≈ Rd
    end

    @testset "known parameters (no optimiser)" begin
        β = 0.2 .* randn(rng, p) .+ log.(vec(mean(Y; dims = 2)))
        Λ = 0.15 .* randn(rng, p, K)
        group = collect(1:p)
        φg = [0.8, 1.2, 0.5, 1.0]
        power = 1.5
        fit = TweedieGroupedFit(β, Λ, φg, power, group, LogLink(), 0.0, true, 0)

        Z = getLV(fit, Y; rotate = false)
        μ = predict(fit, Y; type = :response)
        @test size(μ) == (p, n)
        η = predict(fit, Y; type = :link)
        @test η ≈ β .+ Λ * Z'
        @test fitted(fit, Y) ≈ μ

        φvec = [φg[t] for t in 1:p]
        Rp = residuals(fit, Y; type = :pearson)
        @test Rp ≈ (Y .- μ) ./ [sqrt(φvec[t] * μ[t, s]^power) for t in 1:p, s in 1:n]
    end
end