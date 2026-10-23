# `predictor::Symbol` mode on fit_delta_lognormal_gllvm / fit_delta_gamma_gllvm
# (2026-08-28) — the gllvmTMB twin-identity MODE: ONE shared linear predictor
# η = β + Λz drives both the occurrence (logit) and positive (log) parts,
# mirroring gllvmTMB.cpp:2816-2844 (delta_lognormal fid 12 / delta_gamma
# fid 13). `:separate` (default) stays bit-identical to the pre-existing
# two-predictor behaviour; `:shared` is new. See
# docs/dev-log/decisions/2026-08-28-delta-shared-predictor-identity.md.
#
# Seeds 170-176 — fresh, outside the ranges already claimed by
# test_delta_fit.jl (140-141), test_delta_gamma.jl (73-74, 160-163, 808),
# test_twopart_substrate.jl (130-131), and the parity-ladder brief's
# reserved 42-49/52/53/58.

using GLLVModels, Test, Random, Distributions, Statistics

@testset "delta family: predictor mode (:separate / :shared)" begin

    @testset ":separate ≡ omitted — bit-identical (compat safety net)" begin
        Random.seed!(170)
        p, K, n = 6, 2, 150
        βz_true = 0.5 .* randn(p) .+ 0.4
        βc_true = 0.5 .* randn(p)
        Λc_true = 0.5 .* randn(p, K)
        σ_true = 0.5
        Z = randn(K, n)
        ηc = βc_true .+ Λc_true * Z
        π = inv.(1 .+ exp.(-βz_true))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t] && (Y[t, s] = exp(ηc[t, s] + σ_true * randn()))
        end

        f_omit = fit_delta_lognormal_gllvm(Y; K = K)
        f_sep = fit_delta_lognormal_gllvm(Y; K = K, predictor = :separate)
        @test f_omit.predictor == :separate
        @test f_sep.predictor == :separate
        @test f_omit.loglik == f_sep.loglik
        @test f_omit.βz == f_sep.βz
        @test f_omit.βc == f_sep.βc
        @test f_omit.Λc == f_sep.Λc
        @test f_omit.σ == f_sep.σ

        g_omit = fit_delta_gamma_gllvm(Y; K = K)
        g_sep = fit_delta_gamma_gllvm(Y; K = K, predictor = :separate)
        @test g_omit.predictor == :separate
        @test g_sep.predictor == :separate
        @test g_omit.loglik == g_sep.loglik
        @test g_omit.βz == g_sep.βz
        @test g_omit.βc == g_sep.βc
        @test g_omit.Λc == g_sep.Λc
        @test g_omit.α == g_sep.α
    end

    @testset "invalid predictor throws ArgumentError (both fitters)" begin
        Random.seed!(171)
        p, K, n = 4, 1, 40
        Y = abs.(randn(p, n)) .* (rand(p, n) .< 0.6)
        @test_throws ArgumentError fit_delta_lognormal_gllvm(Y; K = K, predictor = :bogus)
        @test_throws ArgumentError fit_delta_gamma_gllvm(Y; K = K, predictor = :bogus)
    end

    @testset ":shared recovery on data generated under the tied DGP (lognormal)" begin
        Random.seed!(172)
        p, K, n = 5, 2, 80
        β_true = 0.5 .* randn(p) .+ 0.3
        Λ_true = 0.5 .* randn(p, K)
        σ_true = 0.4
        Z = randn(K, n)
        η = β_true .+ Λ_true * Z
        π = inv.(1 .+ exp.(-η))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t, s] && (Y[t, s] = exp(η[t, s] + σ_true * randn()))
        end

        # disp_group=:shared pinned: σ_true is a single shared scalar, and f.σ is
        # compared below as a scalar (chained inequality).
        f = fit_delta_lognormal_gllvm(Y; K = K, predictor = :shared, disp_group = :shared,
                                       iterations = 300)
        @test f isa DeltaLogNormalFit
        @test f.predictor == :shared
        @test f.converged
        @test isfinite(f.loglik)
        @test cor(f.βc, β_true) > 0.7
        @test cor(vec(f.Λc * f.Λc'), vec(Λ_true * Λ_true')) > 0.5
        @test 0.4 * σ_true < f.σ < 2.5 * σ_true
    end

    @testset ":shared recovery on data generated under the tied DGP (gamma)" begin
        Random.seed!(173)
        p, K, n = 5, 2, 80
        β_true = 0.4 .* randn(p) .+ 0.2
        Λ_true = 0.4 .* randn(p, K)
        α_true = 4.0
        Z = randn(K, n)
        η = β_true .+ Λ_true * Z
        π = inv.(1 .+ exp.(-η))
        μ = exp.(η)
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            if rand() < π[t, s]
                Y[t, s] = rand(Gamma(α_true, μ[t, s] / α_true))
            end
        end

        # disp_group=:shared pinned: α_true is a single shared scalar (the "tied
        # DGP" this testset is named for).
        f = fit_delta_gamma_gllvm(Y; K = K, predictor = :shared, disp_group = :shared,
                                   iterations = 300)
        @test f isa DeltaGammaFit
        @test f.predictor == :shared
        @test f.converged
        @test isfinite(f.loglik)
        @test cor(f.βc, β_true) > 0.6
        @test cor(vec(f.Λc * f.Λc'), vec(Λ_true * Λ_true')) > 0.4
    end

    @testset "the tie is real: βz≡βc, Λz≡Λc≡Λc, and loglik == direct kernel eval at θ̂" begin
        Random.seed!(174)
        p, K, n = 5, 2, 80
        β_true = 0.5 .* randn(p) .+ 0.3
        Λ_true = 0.5 .* randn(p, K)
        σ_true = 0.4
        Z = randn(K, n)
        η = β_true .+ Λ_true * Z
        π = inv.(1 .+ exp.(-η))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t, s] && (Y[t, s] = exp(η[t, s] + σ_true * randn()))
        end

        f = fit_delta_lognormal_gllvm(Y; K = K, predictor = :shared, iterations = 300)
        @test f.βz === f.βc || f.βz == f.βc   # identical vectors (materialised as the same object)
        ll_direct = GLLVModels.delta_lognormal_marginal_loglik_laplace(Y, f.Λc, f.βz, f.βc, f.σ;
                                                                   Λz = f.Λc)
        @test ll_direct ≈ f.loglik atol = 1e-8
    end

    @testset "hessian × predictor composition" begin
        Random.seed!(175)
        p, K, n = 5, 2, 80
        β_true = 0.4 .* randn(p) .+ 0.2
        Λ_true = 0.4 .* randn(p, K)
        α_true = 4.0
        Z = randn(K, n)
        η = β_true .+ Λ_true * Z
        π = inv.(1 .+ exp.(-η))
        μ = exp.(η)
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            if rand() < π[t, s]
                Y[t, s] = rand(Gamma(α_true, μ[t, s] / α_true))
            end
        end

        # DeltaGamma: the only two-part family whose observed count-part weight is
        # implemented (TWOPART_KNOWN_OPEN), so :observed and :fisher genuinely differ.
        g_obs = fit_delta_gamma_gllvm(Y; K = K, predictor = :shared, hessian = :observed,
                                       iterations = 250)
        g_fis = fit_delta_gamma_gllvm(Y; K = K, predictor = :shared, hessian = :fisher,
                                       iterations = 250)
        @test g_obs.loglik != g_fis.loglik

        # DeltaLogNormal: observed count-part weight not yet specialised (still
        # TWOPART_KNOWN_OPEN for this family) — :observed and :fisher coincide
        # bit-for-bit, exactly as documented for :separate in
        # test_twopart_hessian_kwarg.jl.
        Ylog = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t, s] && (Ylog[t, s] = exp(η[t, s] + 0.4 * randn()))
        end
        l_obs = fit_delta_lognormal_gllvm(Ylog; K = K, predictor = :shared, hessian = :observed,
                                           iterations = 250)
        l_fis = fit_delta_lognormal_gllvm(Ylog; K = K, predictor = :shared, hessian = :fisher,
                                           iterations = 250)
        @test l_obs.loglik == l_fis.loglik
    end

    @testset "offset symmetry under :shared" begin
        Random.seed!(176)
        p, K, n = 4, 1, 60
        β_true = 0.3 .* randn(p) .+ 0.2
        Λ_true = 0.4 .* randn(p, K)
        σ_true = 0.4
        Z = randn(K, n)
        η = β_true .+ Λ_true * Z
        π = inv.(1 .+ exp.(-η))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t, s] && (Y[t, s] = exp(η[t, s] + σ_true * randn()))
        end

        # A constant per-species offset under :shared is exactly absorbable into β
        # (offset-absorption identity, same as the :separate/offsetc docstring note)
        # PROVIDED the offset hits both parts symmetrically — that's what :shared
        # implements (offsetz = offsetc = offset). Verify: fitting with a constant
        # offset c added to every cell of species t, then reading off β̂, must give
        # (up to optimiser noise) β̂_offset ≈ β̂_no_offset − c for that species, and
        # critically the achieved loglik must match the no-offset fit (the offset is
        # fully absorbable, so optimisation should reach the same maximum).
        c = 0.7
        offset = zeros(p, n)
        offset[1, :] .= c
        f_noshift = fit_delta_lognormal_gllvm(Y; K = K, predictor = :shared, iterations = 300)
        f_shift = fit_delta_lognormal_gllvm(Y; K = K, predictor = :shared, offset = offset,
                                             iterations = 300)
        @test f_shift.converged
        @test f_shift.loglik ≈ f_noshift.loglik atol = 1e-3
        @test f_shift.βc[1] ≈ f_noshift.βc[1] - c atol = 1e-2
        # The tie itself survives the offset: βz≡βc still holds under :shared.
        @test f_shift.βz == f_shift.βc

        # Direct proof the offset hit BOTH parts: evaluate the kernel with the
        # offset applied only to offsetc (the OLD :separate-style wiring) at the
        # :shared fit's own θ̂ — if offset were asymmetric this would differ from
        # the fitted loglik; since :shared applies it symmetrically, recompute with
        # Λz = Λc, offsetz = offset, offsetc = offset and confirm it reproduces
        # f_shift.loglik (the wiring this test is pinned against).
        ll_symmetric = GLLVModels.delta_lognormal_marginal_loglik_laplace(
            Y, f_shift.Λc, f_shift.βz, f_shift.βc, f_shift.σ;
            Λz = f_shift.Λc, offsetz = offset, offsetc = offset)
        @test ll_symmetric ≈ f_shift.loglik atol = 1e-8
    end
end
