# test_tweedie_parity.jl — frozen-R Tweedie power contracts (twin fid 6)
#
# Included by runparity.jl, never by test/runtests.jl. The three cases below
# are deliberately distinct: fixed common power, a reference-engine constrained
# shared power, and gllvmTMB's public per-species default. None substitutes for
# another. See docs/dev-log/decisions/2026-08-30-core070-tweedie-power.md.

using GLLVModels, RCall, Test, Random, Distributions

const _TW_SEED = 82
const _TW_LOGLIK_ATOL = 1e-6

@testset "Tweedie GLLVModels parity: frozen gllvmTMB fid 6" begin
    Random.seed!(_TW_SEED)
    p, K, n = 5, 1, 150
    β_true = [0.5, -0.2, 0.3, -0.4, 0.1]
    Λ_true = 0.5 .* parity_loadings_p5k2()[:, 1:K]
    φ_true = [0.8, 1.0, 1.2, 0.9, 1.1]
    p_true = 1.5
    Z = randn(K, n)
    μ = exp.(β_true .+ Λ_true * Z)
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = GLLVModels._tweedie_sample(μ[t, s], φ_true[t], p_true, Random.default_rng())
    end

    @testset "fixed common power is the same public model on both engines" begin
        r_fit = fit_gllvmtmb_parity_tweedie(Y, K; p_fixed = p_true)
        jl_fit = fit_tweedie_gllvm_grouped(Y; K = K, power = p_true, iterations = 400)
        println("TWEEDIE_PARITY fixed julia_logLik=$(jl_fit.loglik) r_logLik=$(r_fit.logLik)")
        @test r_fit.health.passed && r_fit.converged && jl_fit.converged
        @test r_fit.health.n_free == GLLVModels._nparams(jl_fit)
        println("TWEEDIE_R_HEALTH ", r_fit.health)
        @test isfinite(r_fit.logLik) && isfinite(jl_fit.loglik)
        @test r_fit.power_group == "species" && !r_fit.reference_constraint_adapter
        @test all(≈(p_true; atol = 1e-8), r_fit.p_vec)
        @test r_fit.health.n_power_free == 0
        @test jl_fit.power_fixed && jl_fit.power == p_true
        @test GLLVModels._nparams(jl_fit) == p + GLLVModels.rr_theta_len(p, K) + p
        core070_record_values!("logLik, fixed common power"; julia = jl_fit.loglik, r = r_fit.logLik, atol = _TW_LOGLIK_ATOL,
                               test = "@test isapprox(jl_fit.loglik, r_fit.logLik; atol = _TW_LOGLIK_ATOL)")
        @test isapprox(jl_fit.loglik, r_fit.logLik; atol = _TW_LOGLIK_ATOL)
    end

    @testset "estimated shared power uses the explicit reference constraint adapter" begin
        r_fit = fit_gllvmtmb_parity_tweedie(Y, K; power_group = :shared)
        jl_fit = fit_tweedie_gllvm_grouped(Y; K = K, power_group = :shared, iterations = 400)
        println("TWEEDIE_PARITY shared julia_logLik=$(jl_fit.loglik) r_logLik=$(r_fit.logLik)")
        @test r_fit.health.passed && r_fit.converged && jl_fit.converged
        @test r_fit.health.n_free == GLLVModels._nparams(jl_fit)
        println("TWEEDIE_R_HEALTH ", r_fit.health)
        @test isfinite(r_fit.logLik) && isfinite(jl_fit.loglik)
        @test r_fit.power_group == "shared" && r_fit.reference_constraint_adapter
        @test all(≈(r_fit.p_vec[1]; atol = 1e-10), r_fit.p_vec)
        @test r_fit.health.n_power_free == 1
        @test !jl_fit.power_fixed && 1.0 < jl_fit.power < 2.0
        @test GLLVModels._nparams(jl_fit) == p + GLLVModels.rr_theta_len(p, K) + p + 1
        core070_record_values!("logLik, estimated shared power"; julia = jl_fit.loglik, r = r_fit.logLik, atol = _TW_LOGLIK_ATOL,
                               test = "@test isapprox(jl_fit.loglik, r_fit.logLik; atol = _TW_LOGLIK_ATOL)")
        @test isapprox(jl_fit.loglik, r_fit.logLik; atol = _TW_LOGLIK_ATOL)
    end

    @testset "estimated per-species power matches the public R default" begin
        r_fit = fit_gllvmtmb_parity_tweedie(Y, K; power_group = :species)
        jl_fit = fit_tweedie_gllvm_grouped(Y; K = K, power_group = :species, iterations = 400)
        println("TWEEDIE_PARITY species julia_logLik=$(jl_fit.loglik) r_logLik=$(r_fit.logLik)")
        @test r_fit.health.passed && r_fit.converged && jl_fit.converged
        @test r_fit.health.n_free == GLLVModels._nparams(jl_fit)
        println("TWEEDIE_R_HEALTH ", r_fit.health)
        @test isfinite(r_fit.logLik) && isfinite(jl_fit.loglik)
        @test r_fit.power_group == "species" && !r_fit.reference_constraint_adapter
        @test r_fit.health.n_power_free == p
        @test jl_fit isa GLLVModels.TweediePerTraitPowerFit
        @test all(pw -> 1.0 < pw < 2.0, jl_fit.power)
        @test all(pw -> 1.0 < pw < 2.0, r_fit.p_vec)
        @test GLLVModels.StatsAPI.dof(jl_fit) == p + GLLVModels.rr_theta_len(p, K) + p + p
        core070_record_values!("logLik, estimated per-species power"; julia = jl_fit.loglik, r = r_fit.logLik, atol = _TW_LOGLIK_ATOL,
                               test = "@test isapprox(jl_fit.loglik, r_fit.logLik; atol = _TW_LOGLIK_ATOL)")
        @test isapprox(jl_fit.loglik, r_fit.logLik; atol = _TW_LOGLIK_ATOL)
    end
end
