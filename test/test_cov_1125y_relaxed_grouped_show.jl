# Coverage round 4: relaxed_clock.jl and grouped_nongaussian_postfit.jl show/helpers.
using Test
using GLLVModels
using Random
using SparseArrays
using LinearAlgebra
using Statistics

const G = GLLVModels

@testset "1125y relaxed and grouped postfit" begin
    @testset "_relaxed_sigma_eps_floor" begin
        @test G._relaxed_sigma_eps_floor(zeros(3, 4)) == 1e-8
        Y = randn(2, 10)
        f = G._relaxed_sigma_eps_floor(Y)
        @test f ≈ 1e-8 * mean(abs2, Y) rtol = 1e-6
        @test G._relaxed_sigma_eps_floor(fill(NaN, 2, 2)) == 1e-8
    end

    @testset "RelaxedClockFit show" begin
        f = G.RelaxedClockFit(
            [1.0, 2.0], [0.0, -0.1], 0.05, 0.01, 0.02, 7, true,
            [0.0, -1.0], [0.1 0.2; 0.3 0.4], true, false,
        )
        s = sprint(show, f)
        @test occursin("RelaxedClockFit", s)
        @test occursin("tau2_boundary=true", s)
        @test occursin("sigma_eps_boundary=false", s)
    end

    @testset "GroupedNonGaussianFit summary and show" begin
        p, n = 2, 3
        D = zeros(p * n, p)
        for observation in 1:n, trait in 1:p
            D[(observation - 1) * p + trait, trait] = 1.0
        end
        term = G.GroupingTerm(:unit; mode = :indep)
        fit = G.GroupedNonGaussianFit(
            [log(2.0), log(3.0)], G.Poisson(), nothing, [Matrix(Diagonal([0.25, 0.36]))],
            [term], [log(2.0), log(3.0), log(0.5), log(0.6)],
            -12.5, true, 1e-8, 0.25, true, 7, :converged, D,
            Union{String, Symbol}["trait1", "trait2"],
            ["trait1", "trait2", "unit.log_sd[1]", "unit.log_sd[2]"],
            (p, n), [1.0 2.0 1.0; 2.0 1.0 4.0], fill(5.0, 2, 3),
            [sparse(collect(1:n), collect(1:n), ones(n), n, n)], :poisson, :none, :ok,
        )
        @test occursin("Grouped poisson", G.Base.summary(fit))
        buf = sprint(show, "text/plain", fit)
        @test occursin("convergence = true", buf)
        @test occursin("observed curvature PD = true", buf)
        @test_throws ArgumentError G.extract_Sigma(fit; level = :residual)
        @test_throws ArgumentError G.extract_Sigma(fit; level = :unit, part = :nope)
    end
end