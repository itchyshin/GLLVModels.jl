# Lane cursor-cloud-1125y12 — twolevel.jl, em_fa.jl
using GLLVModels, Test, LinearAlgebra, Random

const _TL = GLLVModels

@testset "cov 1125y12 two-level and EM-FA" begin
    rng = MersenneTwister(11251202)

    @testset "_dense_sigma" begin
        Λ = [0.5 0.0; 0.1 0.4]
        σ² = [0.2, 0.3]
        Σ = _TL._dense_sigma(Λ, σ²)
        @test Σ ≈ Λ * Λ' + Diagonal(σ²) rtol = 1e-12
        @test issymmetric(Σ)
    end

    @testset "twolevel_marginal_loglik" begin
        p, n = 3, 12
        y = randn(rng, p, n)
        indiv = repeat(1:4; inner = 3)
        Λ_B = 0.2 .* randn(rng, p, 1)
        Λ_W = 0.15 .* randn(rng, p, 1)
        σ²_B = fill(0.1, p)
        σ²_W = fill(0.2, p)
        ll = twolevel_marginal_loglik(y, indiv, Λ_B, σ²_B, Λ_W, σ²_W)
        @test isfinite(ll)
        codes, _ = _TL._code_grouping(indiv)
        idx = [findall(==(g), codes) for g in 1:maximum(codes)]
        ll2 = _TL._twolevel_loglik(y, idx, Λ_B, σ²_B, Λ_W, σ²_W)
        @test ll ≈ ll2 rtol = 1e-12
    end

    @testset "_twolevel_unpack round-trip" begin
        p, K_B, K_W = 4, 1, 1
        Λ_B = 0.3 .* randn(rng, p, K_B)
        σ²_B = fill(0.15, p)
        Λ_W = 0.2 .* randn(rng, p, K_W)
        σ²_W = fill(0.25, p)
        θ = vcat(GLLVModels.pack_lambda(Λ_B), log.(σ²_B),
                 GLLVModels.pack_lambda(Λ_W), log.(σ²_W))
        Λ_B2, σ²_B2, Λ_W2, σ²_W2 = _TL._twolevel_unpack(θ, p, K_B, K_W)
        @test Λ_B2 ≈ Λ_B rtol = 1e-12
        @test σ²_B2 ≈ σ²_B rtol = 1e-12
        @test Λ_W2 ≈ Λ_W rtol = 1e-12
        @test σ²_W2 ≈ σ²_W rtol = 1e-12
        @test length(θ) == _TL._twolevel_npar(p, K_B, K_W)
    end

    @testset "fit_twolevel_gaussian tiny" begin
        p, n = 3, 15
        y = randn(rng, p, n)
        indiv = repeat(1:5; inner = 3)
        fit = fit_twolevel_gaussian(y, indiv; K_B = 1, K_W = 1, iterations = 80, g_tol = 1e-4)
        @test fit isa TwoLevelFit
        @test fit.nindiv == 5
        @test isfinite(fit.loglik)
        shown = sprint(show, fit)
        @test occursin("TwoLevelFit", shown)
        rep = repeatability(fit)
        @test length(rep) == p
        @test all(0.0 .<= rep .<= 1.0)
    end

    @testset "em_fa convergence" begin
        p, n = 6, 40
        y = randn(rng, p, n)
        res = em_fa(y, 2; max_iter = 200, tol = 1e-8, rng = MersenneTwister(7))
        @test length(res) == 5
        Λ, ψ, ll, iters, conv = res
        @test size(Λ) == (p, 2)
        @test all(ψ .> 0)
        @test isfinite(ll)
        @test iters >= 1
        # monotone: second run from returned Λ should not decrease ll at start
        res2 = em_fa(y, 2; λ_init = Λ, ψ_init = ψ, max_iter = 2, tol = 1e-20)
        @test res2[3] >= ll - 1e-6
    end

    @testset "em_fa NA refusal" begin
        y = fill(missing, 2, 3)
        @test_throws ArgumentError em_fa(y, 1)
    end
end