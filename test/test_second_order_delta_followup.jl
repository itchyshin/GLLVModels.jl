# Second-order follow-up: Delta shared-η Wald (#347) + Option A species packing scaffold.
# R↔Julia SE receipts run only when GLLVM_PARITY_TESTS=1 (needs RCall + gllvmTMB).
# Decision `2026-09-15-delta-dispersion-alignment-pending.md` stays PENDING until paste
# `accept delta dispersion A` — this file does not claim acceptance or D1 pass.

using GLLVModels, Test, Random, Distributions

function _sim_delta_lognormal(p, K, n; seed = 171)
    Random.seed!(seed)
    β = 0.3 .* randn(p)
    Λ = 0.4 .* randn(p, K)
    σ = 0.45
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        π = 1 / (1 + exp(-η[t, s]))
        rand() < π && (Y[t, s] = exp(η[t, s] + σ * randn()))
    end
    return Y
end

@testset "second-order delta follow-up (shared predictor Wald)" begin
    @testset "Delta-lognormal: shared θ packing + public Wald" begin
        Y = _sim_delta_lognormal(5, 1, 100; seed = 171)
        fit = fit_delta_lognormal_gllvm(Y; K = 1, predictor = :shared,
                                        disp_group = :shared, iterations = 400)
        @test fit.predictor === :shared
        @test fit.disp_group === :shared
        ad = GLLVModels._family_ci(fit, Y)
        @test length(ad.θ) == 5 + GLLVModels.rr_theta_len(5, 1) + 1
        @test count(startswith("beta["), ad.names) == 5
        ci = confint(fit, Y; method = :wald, parm = "beta")
        @test length(ci.term) == 5
        fin = isfinite.(ci.se)
        @test count(fin) ≥ 3
        @test all(ci.lower[fin] .< ci.estimate[fin] .< ci.upper[fin])
    end

    @testset "Delta-lognormal: species θ packing + public Wald (Option A scaffold)" begin
        Y = _sim_delta_lognormal(5, 1, 100; seed = 173)
        fit = fit_delta_lognormal_gllvm(Y; K = 1, predictor = :shared,
                                        disp_group = :species, iterations = 400)
        @test fit.disp_group === :species
        @test fit.σ isa AbstractVector && length(fit.σ) == 5
        ad = GLLVModels._family_ci(fit, Y)
        @test length(ad.θ) == 5 + GLLVModels.rr_theta_len(5, 1) + 5
        @test count(startswith("sigma["), ad.names) == 5
        ci = confint(fit, Y; method = :wald, parm = "beta")
        @test length(ci.term) == 5
        fin = isfinite.(ci.se)
        @test count(fin) ≥ 3
        @test all(ci.lower[fin] .< ci.estimate[fin] .< ci.upper[fin])
        Z = getLV(fit, Y)
        @test size(Z) == (100, 1)
        μ = predict(fit, Y; type = :response)
        @test size(μ) == size(Y)
        @test all(isfinite, μ)
        R = residuals(fit, Y; rng = MersenneTwister(42))
        @test size(R) == size(Y)
        @test all(isfinite, R)
    end

    @testset "Delta-Gamma: shared θ packing + public Wald" begin
        Random.seed!(172)
        p, K, n = 5, 1, 100
        β = 0.3 .* randn(p)
        Λ = 0.4 .* randn(p, K)
        α = 3.5
        Z = randn(K, n)
        η = β .+ Λ * Z
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            π = 1 / (1 + exp(-η[t, s]))
            if rand() < π
                μ = exp(η[t, s])
                Y[t, s] = rand(Gamma(α, μ / α))
            end
        end
        fit = fit_delta_gamma_gllvm(Y; K = K, predictor = :shared,
                                    disp_group = :shared, iterations = 400)
        ad = GLLVModels._family_ci(fit, Y)
        @test count(startswith("beta["), ad.names) == 5
        ci = confint(fit, Y; method = :wald, parm = "beta[1]")
        @test ci.term == ["beta[1]"]
        @test isfinite(ci.se[1])
    end

    @testset "Delta-Gamma: species θ packing + public Wald (Option A scaffold)" begin
        Random.seed!(174)
        p, K, n = 5, 1, 100
        β = 0.3 .* randn(p)
        Λ = 0.4 .* randn(p, K)
        α = 2.5 .+ rand(p)
        Z = randn(K, n)
        η = β .+ Λ * Z
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            π = 1 / (1 + exp(-η[t, s]))
            if rand() < π
                μ = exp(η[t, s])
                Y[t, s] = rand(Gamma(α[t], μ / α[t]))
            end
        end
        fit = fit_delta_gamma_gllvm(Y; K = K, predictor = :shared,
                                    disp_group = :species, iterations = 400)
        @test fit.disp_group === :species
        @test fit.α isa AbstractVector && length(fit.α) == p
        ad = GLLVModels._family_ci(fit, Y)
        @test length(ad.θ) == p + GLLVModels.rr_theta_len(p, K) + p
        @test count(startswith("alpha["), ad.names) == p
        ci = confint(fit, Y; method = :wald, parm = "beta")
        @test length(ci.term) == p
        fin = isfinite.(ci.se)
        @test count(fin) ≥ 3
        Z = getLV(fit, Y)
        @test size(Z) == (n, K)
        μ = predict(fit, Y; type = :response)
        @test all(isfinite, μ)
        R = residuals(fit, Y; rng = MersenneTwister(43))
        @test all(isfinite, R)
    end

    @testset "fit_gllvm disp_group routing (explicit :species, default still :shared)" begin
        Y = _sim_delta_lognormal(4, 1, 80; seed = 175)
        fit_def = fit_gllvm(Y; family = DeltaLogNormal(), K = 1, iterations = 300)
        @test fit_def.disp_group === :shared
        fit_sp = fit_gllvm(Y; family = DeltaLogNormal(), K = 1,
                           disp_group = :species, iterations = 300)
        @test fit_sp.disp_group === :species
        @test fit_sp.σ isa AbstractVector && length(fit_sp.σ) == 4
    end

    @testset "R paired each-own-optimum cells (live Δ)" begin
        if get(ENV, "GLLVM_PARITY_TESTS", "0") != "1"
            @test_skip "set GLLVM_PARITY_TESTS=1 with R + gllvmTMB for live second-order Δ"
        else
            using RCall
            using Distributions: Gamma
            include(joinpath(@__DIR__, "..", "tools", "core070_second_order", "common.jl"))
            include(joinpath(@__DIR__, "..", "tools", "core070_second_order", "cells.jl"))
            include(joinpath(@__DIR__, "..", "tools", "core070_second_order", "eoo_assess.jl"))
            for cell_id in ("delta_lognormal", "delta_gamma")
                d = run_one_cell(cell_id)
                @test get(d, "skip_reason", nothing) === nothing
                # Option A scaffold: cells use Julia :species; gap flag cleared.
                # Still ≠ D1 pass / ACCEPTED disposition until paste + remeasure.
                @test get(d, "parameterisation_gap", true) == false
                se_rel = d["se_max_relative_delta"]
                @test se_rel !== nothing && isfinite(se_rel)
            end
        end
    end
end
