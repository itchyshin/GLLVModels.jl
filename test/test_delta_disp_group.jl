# `disp_group::Symbol` on fit_delta_lognormal_gllvm / fit_delta_gamma_gllvm
# (2026-08-28) — per-trait dispersion, matching gllvmTMB's per-trait
# `log_sigma_lognormal_delta` / `log_phi_gamma_delta` (gllvmTMB.cpp:1195-1196,
# length n_traits). `:species` is now the DEFAULT (`accept delta dispersion A`,
# maintainer paste 2026-09-24 — see
# docs/dev-log/decisions/2026-09-15-delta-dispersion-alignment-pending.md);
# `:shared` (previous default, before 2026-09-24) stays bit-identical to the
# pre-existing single-scalar-dispersion behaviour and remains available as an
# explicit opt-in. See also
# docs/dev-log/decisions/2026-08-28-per-trait-dispersion-synthesis.md and
# the naming precedent `disp_group` in fit_gllvm.jl / grouped_dispersion.jl.
#
# Seeds 180-187 — fresh, outside the ranges claimed by test_delta_fit.jl
# (140-141), test_delta_gamma.jl (73-74, 160-163, 808),
# test_twopart_substrate.jl (130-131), test_delta_shared_predictor.jl
# (170-176), and the parity-ladder brief's reserved 42-49/52/53/58.

using GLLVModels, Test, Random, Distributions, Statistics

@testset "delta family: disp_group mode (:shared / :species)" begin

    @testset ":species ≡ omitted — bit-identical (default since 2026-09-24)" begin
        Random.seed!(180)
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
        f_species = fit_delta_lognormal_gllvm(Y; K = K, disp_group = :species)
        @test f_omit.disp_group == :species
        @test f_species.disp_group == :species
        @test f_omit.loglik == f_species.loglik
        @test f_omit.βz == f_species.βz
        @test f_omit.βc == f_species.βc
        @test f_omit.Λc == f_species.Λc
        @test f_omit.σ == f_species.σ

        g_omit = fit_delta_gamma_gllvm(Y; K = K)
        g_species = fit_delta_gamma_gllvm(Y; K = K, disp_group = :species)
        @test g_omit.disp_group == :species
        @test g_species.disp_group == :species
        @test g_omit.loglik == g_species.loglik
        @test g_omit.βz == g_species.βz
        @test g_omit.βc == g_species.βc
        @test g_omit.Λc == g_species.Λc
        @test g_omit.α == g_species.α
    end

    @testset ":shared — previous default, still available as an explicit opt-in" begin
        Random.seed!(187)
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

        f_shared = fit_delta_lognormal_gllvm(Y; K = K, disp_group = :shared)
        @test f_shared.disp_group == :shared
        @test f_shared.σ isa Real

        g_shared = fit_delta_gamma_gllvm(Y; K = K, disp_group = :shared)
        @test g_shared.disp_group == :shared
        @test g_shared.α isa Real
    end

    @testset "invalid disp_group throws ArgumentError (both fitters)" begin
        Random.seed!(181)
        p, K, n = 4, 1, 40
        Y = abs.(randn(p, n)) .* (rand(p, n) .< 0.6)
        @test_throws ArgumentError fit_delta_lognormal_gllvm(Y; K = K, disp_group = :bogus)
        @test_throws ArgumentError fit_delta_gamma_gllvm(Y; K = K, disp_group = :bogus)
    end

    @testset ":species recovers per-trait dispersion (lognormal)" begin
        Random.seed!(182)
        p, K, n = 8, 1, 150
        βz_true = 0.6 .* randn(p) .+ 0.5
        βc_true = 0.4 .* randn(p)
        Λc_true = 0.4 .* randn(p, K)
        σ_true = 0.2 .+ 0.7 .* rand(p)     # genuinely different per-trait sdlogs
        Z = randn(K, n)
        ηc = βc_true .+ Λc_true * Z
        π = inv.(1 .+ exp.(-βz_true))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t] && (Y[t, s] = exp(ηc[t, s] + σ_true[t] * randn()))
        end

        f = fit_delta_lognormal_gllvm(Y; K = K, disp_group = :species, iterations = 400)
        @test f isa DeltaLogNormalFit
        @test f.disp_group == :species
        @test f.σ isa Vector{Float64}
        @test length(f.σ) == p
        @test f.converged
        @test isfinite(f.loglik)
        @test cor(f.σ, σ_true) > 0.5
    end

    @testset ":species recovers per-trait dispersion (gamma)" begin
        Random.seed!(183)
        p, K, n = 5, 1, 120
        βz_true = 0.5 .* randn(p) .+ 0.4
        βc_true = 0.3 .* randn(p)
        Λc_true = 0.3 .* randn(p, K)
        α_true = 1.5 .+ 4.0 .* rand(p)      # genuinely different per-trait shapes
        Z = randn(K, n)
        ηc = βc_true .+ Λc_true * Z
        π = inv.(1 .+ exp.(-βz_true))
        μ = exp.(ηc)
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t] && (Y[t, s] = rand(Gamma(α_true[t], μ[t, s] / α_true[t])))
        end

        f = fit_delta_gamma_gllvm(Y; K = K, disp_group = :species, iterations = 400)
        @test f isa DeltaGammaFit
        @test f.disp_group == :species
        @test f.α isa Vector{Float64}
        @test length(f.α) == p
        @test f.converged
        @test isfinite(f.loglik)
        @test cor(f.α, α_true) > 0.4
    end

    @testset ":species logLik >= :shared logLik (nesting)" begin
        Random.seed!(184)
        p, K, n = 5, 1, 120
        βz_true = 0.5 .* randn(p) .+ 0.4
        βc_true = 0.3 .* randn(p)
        Λc_true = 0.3 .* randn(p, K)
        σ_true = 0.2 .+ 0.5 .* rand(p)
        Z = randn(K, n)
        ηc = βc_true .+ Λc_true * Z
        π = inv.(1 .+ exp.(-βz_true))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t] && (Y[t, s] = exp(ηc[t, s] + σ_true[t] * randn()))
        end

        f_shared = fit_delta_lognormal_gllvm(Y; K = K, disp_group = :shared, iterations = 400)
        f_species = fit_delta_lognormal_gllvm(Y; K = K, disp_group = :species, iterations = 400)
        @test f_shared.converged
        @test f_species.converged
        # :shared is the :species model with all p sdlogs tied — a constrained
        # special case, so the (better-optimised) :species logLik should not be
        # meaningfully below the :shared logLik. Small negative slack tolerates
        # optimiser noise (LBFGS + finite-difference gradient on two different
        # objective shapes), not a systematic ordering violation.
        @test f_species.loglik >= f_shared.loglik - 1e-3

        g_shared = fit_delta_gamma_gllvm(Y .+ 0.0; K = K, disp_group = :shared, iterations = 400)
        g_species = fit_delta_gamma_gllvm(Y .+ 0.0; K = K, disp_group = :species, iterations = 400)
        @test g_shared.converged
        @test g_species.converged
        @test g_species.loglik >= g_shared.loglik - 1e-3
    end

    @testset "composition with predictor = :shared and hessian = :fisher" begin
        Random.seed!(185)
        p, K, n = 5, 1, 100
        β_true = 0.4 .* randn(p) .+ 0.3
        Λ_true = 0.4 .* randn(p, K)
        σ_true = 0.2 .+ 0.4 .* rand(p)
        Z = randn(K, n)
        η = β_true .+ Λ_true * Z
        π = inv.(1 .+ exp.(-η))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t, s] && (Y[t, s] = exp(η[t, s] + σ_true[t] * randn()))
        end

        f = fit_delta_lognormal_gllvm(Y; K = K, predictor = :shared, disp_group = :species,
                                       hessian = :fisher, iterations = 400)
        @test f isa DeltaLogNormalFit
        @test f.predictor == :shared
        @test f.disp_group == :species
        @test f.σ isa Vector{Float64}
        @test length(f.σ) == p
        @test f.converged
        @test isfinite(f.loglik)
        @test f.βz == f.βc   # the :shared tie survives disp_group = :species

        α_true = 1.5 .+ 3.0 .* rand(p)
        μ = exp.(η)
        Ygamma = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t, s] && (Ygamma[t, s] = rand(Gamma(α_true[t], μ[t, s] / α_true[t])))
        end
        g = fit_delta_gamma_gllvm(Ygamma; K = K, predictor = :shared, disp_group = :species,
                                   hessian = :fisher, iterations = 400)
        @test g isa DeltaGammaFit
        @test g.predictor == :shared
        @test g.disp_group == :species
        @test g.α isa Vector{Float64}
        @test length(g.α) == p
        @test g.converged
        @test isfinite(g.loglik)
        @test g.βz == g.βc
    end

    @testset "direct kernel eval at θ̂ reproduces fitted loglik (per-trait σ)" begin
        Random.seed!(186)
        p, K, n = 5, 1, 100
        βz_true = 0.5 .* randn(p) .+ 0.4
        βc_true = 0.3 .* randn(p)
        Λc_true = 0.3 .* randn(p, K)
        σ_true = 0.2 .+ 0.5 .* rand(p)
        Z = randn(K, n)
        ηc = βc_true .+ Λc_true * Z
        π = inv.(1 .+ exp.(-βz_true))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t] && (Y[t, s] = exp(ηc[t, s] + σ_true[t] * randn()))
        end

        f = fit_delta_lognormal_gllvm(Y; K = K, disp_group = :species, iterations = 400)
        ll_direct = GLLVModels.delta_lognormal_marginal_loglik_laplace(Y, f.Λc, f.βz, f.βc, f.σ;
                                                                    offsetc = nothing)
        @test ll_direct ≈ f.loglik atol = 1e-8
    end
end
