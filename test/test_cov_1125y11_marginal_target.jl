# Lane cursor-cloud-1125y11 — marginal_target_intervals.jl extra branches
using GLLVModels, Test

@testset "cov 1125y11 marginal target intervals" begin
    f_quad(x) = sum(abs2, x)
    center = [0.0, 0.0]
    targets = [(name = :a, value = θ -> θ[1], transform = :identity),
               (name = :b, value = θ -> θ[2], transform = :identity)]

    @testset "happy path" begin
        r = GLLVModels._marginal_target_intervals(f_quad, center, targets; converged = true)
        @test r.status == :available
        @test r.gradient_norm < 1e-3
        @test all(t -> t.status == :available, r.intervals)
        @test all(t -> t.lower < t.estimate < t.upper, r.intervals)
    end

    @testset "status short-circuits" begin
        @test GLLVModels._marginal_target_intervals(f_quad, center, targets;
            converged = false).status == :not_converged
        @test GLLVModels._marginal_target_intervals(f_quad, center, targets;
            converged = true, structural_redundancy = true).status == :nonidentifiable
        @test GLLVModels._marginal_target_intervals(x -> GLLVModels._NLL_SENTINEL, center,
            targets; converged = true).status == :invalid_objective
        @test GLLVModels._marginal_target_intervals(f_quad, center .+ 0.25, targets;
            converged = true).status == :not_stationary
        @test GLLVModels._marginal_target_intervals(x -> x[1]^2, [0.0, 0.0], targets;
            converged = true).status == :invalid_curvature
    end

    @testset "input validation" begin
        dup = [(name = :x, value = θ -> θ[1], transform = :identity),
               (name = :x, value = θ -> θ[2], transform = :identity)]
        @test_throws ArgumentError GLLVModels._marginal_target_intervals(
            f_quad, center, dup; converged = true)
        @test_throws ArgumentError GLLVModels._marginal_target_intervals(
            f_quad, center, targets; converged = true, level = 1.1)
        @test_throws ArgumentError GLLVModels._marginal_target_intervals(
            f_quad, [NaN, 0.0], targets; converged = true)
        bad_t = [(name = :z, value = θ -> θ[1], transform = :bogus)]
        @test_throws ArgumentError GLLVModels._marginal_target_intervals(
            f_quad, center, bad_t; converged = true)
    end

    @testset "log-scale transform" begin
        pos = [(name = :v, value = θ -> exp(θ[1]), transform = :log)]
        r = GLLVModels._marginal_target_intervals(x -> (x[1] - log(2))^2, [log(2)], pos;
            converged = true)
        @test r.status in (:available, :partial)
        @test all(isfinite, (r.intervals[1].lower, r.intervals[1].upper))
    end
end