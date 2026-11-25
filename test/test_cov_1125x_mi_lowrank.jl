# Lane 1125x round 3: missing_predictor_fiml.jl Woodbury half-NLL helpers.
using Test, GLLVModels, LinearAlgebra, Random

const G = GLLVModels

@testset "1125x _mi_lowrank helpers" begin
    rng = MersenneTwister(112506)
    p, m = 5, 2
    L = 0.3 .* randn(rng, p, m)
    r = randn(rng, p)
    s2 = 0.4

    @testset "_mi_lowrank_halfnll vs dense reference" begin
        Σ = L * L' + s2 * I
        ref = 0.5 * (p * log(2π) + logdet(Σ) + dot(r, Σ \ r))
        got = G._mi_lowrank_halfnll(r, L, s2)
        @test got ≈ ref rtol = 1e-10
    end

    @testset "rank-zero (diagonal only)" begin
        L0 = zeros(p, 0)
        ref = 0.5 * (p * log(2π) + p * log(s2) + dot(r, r) / s2)
        @test G._mi_lowrank_halfnll(r, L0, s2) ≈ ref rtol = 1e-12
    end

    @testset "_mi_lowrank_solve matches Σ \\ r" begin
        Σ = L * L' + s2 * I
        sol = G._mi_lowrank_solve(r, L, s2)
        @test sol ≈ Σ \ r rtol = 1e-10
    end

end