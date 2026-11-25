# Lane cursor-cloud-1125y6: loading_profile_confirmatory_internal.jl pin plumbing.
using Test, GLLVModels, LinearAlgebra, Random

const G = GLLVModels

@testset "1125y6 loading profile pin matrix" begin
    p, K = 3, 2
    @testset "_lambda_constraint_is_pinned" begin
        pins = G._lambda_constraint_is_pinned(nothing, p, K)
        @test size(pins) == (p, K)
        @test pins[1, 2] && !pins[2, 2]
        @test !pins[1, 1]
        M = fill(NaN, p, K)
        M[2, 1] = 0.5
        pins2 = G._lambda_constraint_is_pinned(M, p, K)
        @test pins2[2, 1] && !pins2[1, 1]
        @test_throws ArgumentError G._lambda_constraint_is_pinned(zeros(2, 2), p, K)
        @test_throws ArgumentError G._lambda_constraint_is_pinned(nothing, 0, 1)
    end

    @testset "_normalize_lambda_constraint_pin_matrix" begin
        M = [NaN 1.0; 0.2 NaN; NaN NaN]
        N = G._normalize_lambda_constraint_pin_matrix(M)
        @test isnan(N[1, 2])
        @test N[2, 1] ≈ 0.2
        @test isnan(N[3, 1]) && isnan(N[3, 2])
    end

    @testset "_enumerate_free_lambda_entries" begin
        M = fill(NaN, p, K)
        M[1, 1] = 0.3
        free = G._enumerate_free_lambda_entries(M, p, K)
        @test (1, 1) ∉ free
        @test (2, 1) ∈ free
        entries = [2 1; 3 1]
        sel = G._enumerate_free_lambda_entries(M, p, K; entries = entries)
        @test sel == [(2, 1), (3, 1)]
        @test_throws ArgumentError G._enumerate_free_lambda_entries(M, p, K; entries = [4 1])
        bad = copy(M)
        bad[2, 1] = 0.1
        @test_throws ArgumentError G._enumerate_free_lambda_entries(
            bad, p, K; entries = [2 1])
        @test_throws ArgumentError G._enumerate_free_lambda_entries(
            M, p, K; entries = reshape([4, 1], 1, 2))
    end

    @testset "_profile_refit_lambda_constraint" begin
        M0 = fill(NaN, p, K)
        M1 = G._profile_refit_lambda_constraint(M0, p, K, 2, 1, 0.42)
        @test M1[2, 1] == 0.42
        @test_throws ArgumentError G._profile_refit_lambda_constraint(
            zeros(2, 2), p, K, 1, 1, 0.0)
    end

    @testset "_inv_free_block" begin
        H = [2.0 0.5; 0.5 3.0]
        Σ = G._inv_free_block(H, Int[])
        @test Σ ≈ inv(H) rtol = 1e-12
        Σp = G._inv_free_block(H, [1])
        @test Σp[1, 1] == 0.0
        @test Σp[2, 2] ≈ 1 / 3 rtol = 1e-12
    end

    @testset "_confirmatory_j1_fit_admitted" begin
        rng = MersenneTwister(112506)
        Y = randn(rng, p, 10)
        fit = fit_gaussian_gllvm(Y; K = K)
        @test G._confirmatory_j1_fit_admitted(fit)
        fit_w = fit_gaussian_gllvm(Y; K = K, K_W = 1)
        @test !G._confirmatory_j1_fit_admitted(fit_w)
    end

    @testset "_lambda_constraint_entry_pinned" begin
        rng = MersenneTwister(7)
        Y = randn(rng, 2, 8)
        M = fill(NaN, 2, 1)
        M[1, 1] = 0.2
        fit = fit_gaussian_gllvm(Y; K = 1, lambda_constraint = M)
        @test G._lambda_constraint_entry_pinned(fit, 1, 1)
        @test !G._lambda_constraint_entry_pinned(fit, 2, 1)
        @test G._lambda_constraint_pinned_theta_indices(fit) == G._lambda_constraint_pinned_theta_indices(fit)
        @test !isempty(G._lambda_constraint_pinned_theta_indices(fit))
    end

    @testset "paste authorization string" begin
        @test !G._d3_loading_profile_stage1_paste_authorized()
        @test_throws ArgumentError G._d3_loading_profile_stage1_require_paste!()
    end
end