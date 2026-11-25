# Lane cursor-cloud-1125y13 — src/missing_predictor_phylo.jl
using GLLVModels, Test, LinearAlgebra, Random, Statistics

const G = GLLVModels

@testset "cov 1125y13 mi_phylo" begin
    rng = MersenneTwister(1125132)
    p, n, K = 4, 8, 1

    @testset "_mi_phylo_precompute" begin
        A = Matrix{Float64}(I, p, p)
        A[1, 2] = A[2, 1] = 0.35
        A[2, 3] = A[3, 2] = 0.25
        obs = [1, 2, 4]
        mis = [3]
        pc = G._mi_phylo_precompute(A, obs, mis)
        @test size(pc.AooInv) == (3, 3)
        @test isfinite(pc.logdetAoo)
        @test size(pc.Creg, 1) == 1
        @test size(pc.Acond) == (1, 1)
        pc0 = G._mi_phylo_precompute(A, collect(1:p), Int[])
        @test size(pc0.Acond) == (0, 0)
        @test size(pc0.Creg) == (0, p)
    end

    @testset "nll finite at plausible params" begin
        y = randn(rng, p, n)
        x = [0.1, 0.2, missing, -0.1]
        A = float(I(p))
        obs = [1, 2, 4]
        mis = [3]
        xo = [x[i] for i in obs]
        pc = G._mi_phylo_precompute(A, obs, mis)
        Λ = 0.2 .* randn(rng, p, K)
        θ = vcat(0.0, 0.5, mean(xo), log(0.3), log(0.4), vec(Λ))
        nll = G._mi_phylo_nll(θ, y, xo, obs, mis, pc, p, n, K)
        @test isfinite(nll)
        @test nll > 0
    end

    @testset "tiny fit_gaussian_mi_phylo" begin
        y = 0.3 .* randn(rng, p, n) .+ repeat(0:0.3:(p - 1) * 0.3, 1, n)
        x = [0.0, 0.5, missing, -0.2]
        A = [1.0 0.4 0.2 0.1;
             0.4 1.0 0.3 0.15;
             0.2 0.3 1.0 0.25;
             0.1 0.15 0.25 1.0]
        res = fit_gaussian_mi_phylo(y, x, A; K = 1, iterations = 120, g_tol = 1e-5)
        @test isfinite(res.logLik)
        @test res.n_missing == 1
        @test length(res.eblup_x) == p
        @test all(isfinite, res.eblup_x[1:2])
        @test all(isfinite, res.eblup_x[4])
        @test isfinite(res.eblup_x[3])
    end

    @testset "refusals" begin
        y = randn(3, 5)
        x = [1.0, 2.0, 3.0]
        @test_throws ArgumentError fit_gaussian_mi_phylo(y, x, I(4); K = 1)
        @test_throws ArgumentError fit_gaussian_mi_phylo(y, x, I(3); K = 0)
        @test_throws ArgumentError fit_gaussian_mi_phylo(y, fill(missing, 3), I(3); K = 1)
    end
end