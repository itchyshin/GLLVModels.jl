# Lane cursor-cloud-1125y13 — src/missing_predictor_poisson.jl (_xs_glm, safe solves)
using GLLVModels, Test, LinearAlgebra, Random, Distributions

const G = GLLVModels

@testset "cov 1125y13 missing_predictor_poisson xs" begin
    rng = MersenneTwister(1125133)
    η = randn(rng, 4)
    n = fill(5, 4)
    y = [rand(rng, Poisson(exp(clamp(η[t], -5, 5)))) for t in 1:4]

    @testset "_xs_glm canonical links" begin
        μ, s, Wobs, WF = G._xs_glm(Poisson(), LogLink(), η, n, y)
        @test μ ≈ exp.(η)
        @test s ≈ Float64.(y) .- μ
        @test Wobs ≈ WF
        yb = [rand(rng, Binomial(n[t], 1 / (1 + exp(-η[t])))) for t in 1:4]
        μb, sb, Wb, WFb = G._xs_glm(Binomial(), LogitLink(), η, n, yb)
        @test all(0 .< μb .< 1)
        @test Wb ≈ WFb
    end

    @testset "_xs_glm dispersion families" begin
        yn = [rand(rng, NegativeBinomial(10, 0.4)) for _ in 1:4]
        f = NegativeBinomial(8.0, 0.5; check_args = false)
        μn, sn, Wobsn, WFn = G._xs_glm(f, LogLink(), η, n, yn)
        @test all(μn .> 0)
        @test all(Wobsn .> 0)
        @test all(WFn .> 0)
        yg = [rand(rng, Gamma(2.0, 1.0)) for _ in 1:4]
        fg = Gamma(3.0, 1.0; check_args = false)
        μg, sg, Wobsg, WFg = G._xs_glm(fg, LogLink(), η, n, yg)
        @test all(isfinite, Wobsg)
        @test all(isfinite, WFg)
        yβ = [clamp(rand(rng, Beta(2, 3)), 1e-6, 1 - 1e-6) for _ in 1:4]
        fβ = Beta(12.0, 1.0; check_args = false)
        μβ, sβ, Wobsβ, WFβ = G._xs_glm(fβ, LogitLink(), η, n, yβ)
        @test all(isfinite, Wobsβ)
        @test all(isfinite, WFβ)
    end

    @testset "_xs_logker_sum matches summed log-pmf" begin
        μ = exp.(η)
        ref = sum(t -> logpdf(Poisson(μ[t]), y[t]), 1:length(y))
        @test G._xs_logker_sum(Poisson(), μ, n, y) ≈ ref rtol = 1e-12
    end

    @testset "_safe_solve_xs and _safe_logdet_xs" begin
        M = [1.0 0.2; 0.2 1.0]
        b = [0.3, -0.1]
        @test G._safe_solve_xs(M, b) ≈ M \ b
        @test isfinite(G._safe_logdet_xs(M))
        badM = [1.0 Inf; 0.0 1.0]
        sol = G._safe_solve_xs(badM, b)
        @test all(isnan, sol)
        @test isnan(G._safe_logdet_xs(badM))
    end

    @testset "_mode_xs Poisson fixed point" begin
        p, K = 3, 1
        Λ = reshape([0.2, -0.1, 0.15], p, K)
        β = zeros(p)
        Y = clamp.(rand(rng, Poisson(3), p), 0, 20)
        z, x = G._mode_xs(Poisson(), Y, ones(p), Λ, β, LogLink(), 0.4, 0.0, 1.0)
        @test length(z) == K
        @test isfinite(x)
        @test all(isfinite, z)
    end
end