using Test
using GLLVModels
using Random

include("parity/fixtures/loading_profile_confirmatory_substrate.jl")

const _P = LOADING_PROFILE_CONFIRMATORY_P
const _K = LOADING_PROFILE_CONFIRMATORY_K

@testset "loading_profile confirmatory internal (D3 Stage 1 plumbing)" begin
    @testset "internal substrate matches Stage 0 fixtures" begin
        pins = loading_profile_fixture_mask_b_pins()
        free = GLLVModels._enumerate_free_lambda_entries(pins, _P, _K)
        @test free == LOADING_PROFILE_ORACLE_FREE_MASK_B_PINS

        M = GLLVModels._profile_refit_lambda_constraint(pins, _P, _K, 2, 1, 0.42)
        @test M[2, 1] == 0.42
        @test M[1, 1] == -0.8
    end

    @testset "paste gate for engine refit (default require_paste)" begin
        @test !GLLVModels._d3_loading_profile_stage1_paste_authorized()
        rng = MersenneTwister(49)
        p, n, K = 4, 8, 2
        Y = randn(rng, p, n)
        fit = fit_gaussian_gllvm(Y; K = K)
        pins = loading_profile_fixture_mask_b_pins()
        @test_throws ArgumentError GLLVModels._confirmatory_profile_refit_lambda_pin(
            fit, Y, pins, 2, 1, 0.1)
    end

    if GLLVModels._d3_loading_profile_stage1_paste_authorized()
        @testset "J1 pin-and-refit smoke (paste authorized)" begin
            rng = MersenneTwister(49)
            p, n, K = 4, 12, 2
            Y = randn(rng, p, n)
            fit = fit_gaussian_gllvm(Y; K = K)
            pins = loading_profile_fixture_mask_b_pins()
            ll, ok = GLLVModels._confirmatory_profile_refit_lambda_pin(
                fit, Y, pins, 2, 1, 0.05; require_paste = true)
            @test ok
            @test isfinite(ll)
            @test ll <= fit.logLik + 1e-8
        end
    else
        @test_skip "confirmatory pin refit smoke waits for paste G0 Stage 1"
    end
end
