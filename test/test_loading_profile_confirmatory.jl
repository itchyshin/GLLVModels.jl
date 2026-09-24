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
            # p, K must match the MASK-B-PINS fixture's own dimensions (_P x _K);
            # this used to be p = 4 against 3x2 pins, which throws a BoundsError
            # once _profile_refit_lambda_constraint's size check below is added
            # (Gauss, 2026-09-24: it previously threw the same BoundsError
            # unguarded, the moment this branch ever ran under the paste gate).
            p, n, K = _P, 12, _K
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

    @testset "regression: theta_packed pins are raw-scale, not divided by sigma_eps" begin
        # Gauss verdict (2026-09-24): `_confirmatory_lambda_pin_theta_fixes` used to
        # write `pin / σ̂_eps(reference fit)` into θ_packed, but `gaussian_nll_packed`
        # (src/likelihood.jl) unpacks that Lambda_B block as RAW Λ with no σ_eps
        # rescaling anywhere on that path -- so the refit pinned the wrong value
        # (relative error `1/σ̂_eps - 1`, +35.79% on this exact seed/fit). This test
        # fails on the old `Float64(v) / σ_eps` line (since σ_eps != 1 here) and
        # passes after dropping the division.
        rng = MersenneTwister(49)
        p, n, K = _P, 20, _K
        Y = randn(rng, p, n)
        fit = fit_gaussian_gllvm(Y; K = K)
        @test fit.pars.σ_eps != 1   # a fit where the bug would be visible at all

        pins = loading_profile_fixture_mask_b_pins()   # user pins: L11 = -0.8, L32 = 0
        # Route through the (previously buggy) pin-mapping function: the extra
        # profile-grid override at (2,1) is arbitrary and not itself asserted on.
        fixes = GLLVModels._confirmatory_lambda_pin_theta_fixes(fit, pins, 2, 1, 0.3)
        ll, ok, θ_red = GLLVModels._profile_refit_with_multi_fixed(fit, fixes, Y)
        @test ok

        # Reconstruct the full refit θ and unpack Λ to check the ACTUAL fitted
        # loading at each fixed entry, not just that the refit converged.
        N = length(fit.pars.θ_packed)
        fixed_dict = Dict(fixes)
        free_idx = [j for j in 1:N if !haskey(fixed_dict, j)]
        θ_full = Vector{Float64}(undef, N)
        for (idx, v) in fixes
            θ_full[idx] = v
        end
        for (r, j) in enumerate(free_idx)
            θ_full[j] = θ_red[r]
        end
        spec = GLLVModels._profile_spec(fit)
        u = GLLVModels._derived_unpack(θ_full, spec)
        @test u.Λ_B[1, 1] == -0.8
        @test u.Λ_B[3, 2] == 0.0
        @test u.Λ_B[2, 1] == 0.3
    end
end
