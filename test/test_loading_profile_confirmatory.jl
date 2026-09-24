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

# D3 Stage 1 slice (maintainer paste `G0 Stage 1`, 2026-09-24): fit-time
# `lambda_constraint` on `fit_gaussian_gllvm` and the exported `loading_profile`.
# These exercise the actual pin-and-refit numerics for the first time — the
# tests above only reached the paste-refusal path since no paste was set in
# CI. This is a Stage 1 receipt (Stage 0 fixtures, one entry per refit), not
# full R grid parity: see the runbook's Rose fence.
@testset "loading_profile Stage 1 (fit-time lambda_constraint + export)" begin
    rng = MersenneTwister(49)
    n = 40
    Y = randn(rng, _P, n)

    @testset "fit-time pins land exactly on Stage 0 fixtures" begin
        pins = loading_profile_fixture_mask_b_pins()
        fit = fit_gaussian_gllvm(Y; K = _K, lambda_constraint = pins)
        @test fit.converged
        @test fit.pars.Λ[1, 1] == -0.8
        @test fit.pars.Λ[3, 2] == 0.0
        @test fit.pars.lambda_constraint[1, 1] == -0.8
        @test isnan(fit.pars.lambda_constraint[2, 1])

        # MASK-B-UPPER: a bogus above-diagonal entry R (and this port) ignores.
        upper = loading_profile_fixture_mask_b_upper()
        fit_upper = fit_gaussian_gllvm(Y; K = _K, lambda_constraint = upper)
        @test fit_upper.pars.Λ == fit.pars.Λ

        # MASK-B-ALLFIXED: every entry pinned, nothing left free.
        allfixed = loading_profile_fixture_mask_b_allfixed()
        fit_allfixed = fit_gaussian_gllvm(Y; K = _K, lambda_constraint = allfixed)
        @test fit_allfixed.converged
        @test fit_allfixed.pars.Λ[1, 1] == 0.8
        @test fit_allfixed.pars.Λ[2, 2] == 0.7

        # No additional user pins beyond structural zeros: numerically
        # identical to the plain unconstrained fit (only metadata differs).
        base = fit_gaussian_gllvm(Y; K = _K)
        fit_free = fit_gaussian_gllvm(Y; K = _K, lambda_constraint = fill(NaN, _P, _K))
        @test fit_free.logLik == base.logLik
        @test fit_free.pars.Λ == base.pars.Λ
    end

    @testset "lambda_constraint refusals" begin
        pins = loading_profile_fixture_mask_b_pins()
        # Structured (K_W > 0) fits are out of Stage 1 scope.
        @test_throws ArgumentError fit_gaussian_gllvm(
            Y; K = _K, K_W = 1, lambda_constraint = fill(NaN, _P, _K))
        # X-carrying fits are out of Stage 1 scope.
        X = zeros(_P, n, 1)
        @test_throws ArgumentError fit_gaussian_gllvm(
            Y; K = _K, X = X, lambda_constraint = fill(NaN, _P, _K))
        # Wrong-shaped pin matrix.
        @test_throws ArgumentError fit_gaussian_gllvm(
            Y; K = _K, lambda_constraint = fill(NaN, _P + 1, _K))
    end

    @testset "loading_profile refuses a non-confirmatory (exploratory) fit" begin
        base = fit_gaussian_gllvm(Y; K = _K)
        @test_throws ArgumentError loading_profile(base; y = Y)
    end

    @testset "loading_profile refuses when no free entries remain" begin
        allfixed = loading_profile_fixture_mask_b_allfixed()
        fit_allfixed = fit_gaussian_gllvm(Y; K = _K, lambda_constraint = allfixed)
        @test_throws ArgumentError loading_profile(fit_allfixed; y = Y)
    end

    @testset "one pin-and-refit grid cell, internal consistency (MASK-B-PINS)" begin
        pins = loading_profile_fixture_mask_b_pins()
        fit = fit_gaussian_gllvm(Y; K = _K, lambda_constraint = pins)
        result = loading_profile(fit; y = Y, n_grid = 3, entries = [2 1])
        @test length(result.table) == 3
        @test all(row -> row.trait == 2 && row.axis == 1, result.table)
        @test all(row -> row.converged, result.table)
        @test all(row -> row.objective >= -fit.logLik - 1e-6, result.table)
        @test all(row -> row.delta_deviance >= -1e-6, result.table)
        # The grid point nearest the confirmatory MLE has ~zero deviance.
        mle_row = argmin(row -> abs(row.profile_value - row.estimate), result.table)
        @test mle_row.delta_deviance < 1e-4
    end

    @testset "frozen R oracle match: packing convention (masks-known-contract MASK-B-PINS-P1)" begin
        # Direct comparison against the frozen R oracle named by the runbook:
        # docs/dev-log/core070/masks-known-contract.json case
        # CORE070-MASKS-KNOWN-MASK-B-PINS-PAIRED-CONTROL, point MASK-B-PINS-P1,
        # r_nll = 65.5136777950417, tolerance abs_nll_delta = 1e-06 (R @ b4d5fee64,
        # docs/dev-log/core070/masks-known-evidence.json).
        #
        # An initial search for this point's raw inputs missed them (only
        # SHA-256 provenance is in masks-known-evidence.json's
        # `retained_artifacts`); they were later found preserved at
        # ~/local-scratch/preservation/core070-execution-20260831T155501Z-delta/
        # runtime-delta/masks-known-points-01/attempt1/out/MASK-B-PINS-P1/
        # {observations,parameters,source}.tsv, SHA-256-verified byte-for-byte
        # against `retained_artifacts` before use. The values below (Y, X, β,
        # σ_eps, and the packed Λ) are transcribed from those files so this test
        # is self-contained and does not read that external, ephemeral path.
        #
        # `maps.tsv` for MASK-B-PINS records `pins = 0.8,0` — L11 pinned to
        # **+0.8**, not the -0.8 this repo's own (pre-existing, Stage 0)
        # `loading_profile_fixture_mask_b_pins()` uses. Those are two
        # independently authored synthetic fixtures for different purposes
        # (this masks-known-points campaign vs. the D3 substrate); this test
        # uses the R reference's own pin value since it targets the frozen R
        # number, not the Stage 0 fixture.
        #
        # This validates `unpack_lambda`'s diagonal-first/strict-lower-column
        # packing convention and `gaussian_marginal_loglik`/`gaussian_nll_packed`
        # against the real R oracle — the shared kernel that
        # `_lambda_b_theta_index` (hence `_confirmatory_lambda_constraint_theta_fixes`
        # and the σ_eps pin-scaling bug fixed above) packs into and reads from.
        # It does NOT call `fit_gaussian_gllvm(...; lambda_constraint = ...)`
        # end-to-end: that path is `X = nothing`-only by Stage 1 design, and this
        # R call has per-trait fixed intercepts (`X != nothing`, β below), so the
        # full wrapper cannot reach this exact model regardless.
        Y_oracle = [
            0.527194696796152 0.00943203712451463 -0.301277048588345 -0.556802495307928 -0.729014501270762 -0.798954917097928 -0.758924274663138 -0.61332939156758 -0.378198241744309 -0.0794154981989258 0.818369803069737 1.0414709848079 1.17193790136331 1.19540795775176 1.10929742682568 0.923085881738325 0.657272626635812 0.341120008059867
            0.450127009882173 0.491317235554749 0.160463226869684 -0.14402111088937 -0.388616282223224 -0.546395756838108 -0.599990206550703 -0.543499627015485 -0.38314284623659 -0.136572918000435 0.77415123057122 1.05698659871879 1.26749686961881 1.38250778698637 1.38935824662338 1.28729410809469 1.08755121511306 0.812118485241757
            0.369066234098832 0.965928155352931 0.641284864348666 0.312096683334935 0.0146026577519254 -0.218447253157945 -0.361397491879557 -0.398511223084363 -0.32570274057403 -0.150987246771676 0.700127985546573 1.02016703682664 1.29395153457706 1.49134160918208 1.59060735569487 1.5808209944866 1.46305986796802 1.25028784015712
        ]
        p_o, n_o = 3, 18
        # source.tsv is the 18x18 identity (verified separately): "Ordinary
        # source I" per masks-known-leaf.md, so the ordinary Sigma = sigma_eps^2 I
        # + I_18 * (Lambda Lambda')[trait,trait] applies without a kron-style
        # site covariance, matching gaussian_marginal_loglik's own assumption.
        X_oracle = zeros(p_o, n_o, p_o)
        for t in 1:p_o, s in 1:n_o
            X_oracle[t, s, t] = 1.0   # 0+trait: per-trait fixed intercept
        end
        beta_oracle = [0.2, -0.1, 0.3]
        sigma_eps_oracle = exp(-0.22314355131421)
        # Packed order [L11,L22,L21,L31,L32]: L11=.8 and L32=0 pinned (maps.tsv);
        # L22=.7, L21=.1, L31=.2 free (parameters.tsv theta_rr_B).
        theta_oracle = [0.8, 0.7, 0.1, 0.2, 0.0]
        Lambda_oracle = GLLVModels.unpack_lambda(theta_oracle, p_o, 2)

        r_nll = 65.5136777950417
        julia_nll = -GLLVModels.gaussian_marginal_loglik(
            Y_oracle, Lambda_oracle, sigma_eps_oracle; X = X_oracle, β = beta_oracle)
        @test abs(julia_nll - r_nll) <= 1e-6   # contract's own abs_nll_delta tolerance

        # Cross-check via the packed-θ entry point `_confirmatory_lambda_pin_theta_fixes`
        # itself depends on (gaussian_nll_packed + spec), not just the direct kernel.
        spec = (q = 3, p = p_o, K_B = 2, K_W = 0, has_diag = false)
        theta_packed = vcat(beta_oracle, log(sigma_eps_oracle), GLLVModels.pack_lambda(Lambda_oracle))
        julia_nll_packed = GLLVModels.gaussian_nll_packed(theta_packed, Y_oracle; spec = spec, X = X_oracle)
        @test julia_nll_packed == julia_nll
    end

    @testset "entries filter profiles only the requested pair" begin
        pins = loading_profile_fixture_mask_b_pins()
        fit = fit_gaussian_gllvm(Y; K = _K, lambda_constraint = pins)
        result = loading_profile(fit; y = Y, n_grid = 3, entries = [3 1])
        @test result.entries == [(3, 1)]
        @test length(result.table) == 3
    end

    @testset "old 3-positional-arg shim still dispatches to loading_profile_exploratory" begin
        base = fit_gaussian_gllvm(Y; K = _K)
        r = loading_profile(base, 1, 1; y = Y)
        @test r.method == :profile
        @test isfinite(r.estimate)
    end
end
