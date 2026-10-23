using GLLVModels, Test, Random, LinearAlgebra, Statistics, Distributions

@testset "post-fit Delta-lognormal" begin
    Random.seed!(150)
    p, K, n = 6, 2, 200
    βz = 0.5 .* randn(p) .+ 0.4
    βc = 0.5 .* randn(p)
    Λc = 0.5 .* randn(p, K)
    σ = 0.5
    Z = randn(K, n)
    ηc = βc .+ Λc * Z
    π = inv.(1 .+ exp.(-βz))
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        rand() < π[t] && (Y[t, s] = exp(ηc[t, s] + σ * randn()))
    end
    # disp_group=:shared pinned: `GLLVModels.DeltaLogNormal(fit.σ)` below needs a
    # scalar σ (the marker's field is Float64), and `k` assumes one dispersion term.
    fit = fit_delta_lognormal_gllvm(Y; K = K, disp_group = :shared)

    @testset "getLV matches per-site two-part mode" begin
        Zh = GLLVModels.getLV(fit, Y; rotate = false)
        @test size(Zh) == (n, K)
        for s in 1:n
            ẑ = GLLVModels._twopart_mode(GLLVModels.DeltaLogNormal(fit.σ), view(Y, :, s),
                                    zeros(p, K), fit.Λc, fit.βz, fit.βc)
            @test Zh[s, :] ≈ ẑ atol = 1e-7
        end
        @test GLLVModels.rotation(fit)' * GLLVModels.rotation(fit) ≈ I(K) atol = 1e-10
    end

    @testset "predict / residuals / AIC / show" begin
        Em = GLLVModels.predict(fit, Y; type = :response)
        @test size(Em) == (p, n) && all(Em .≥ 0)
        occ = GLLVModels.predict(fit, Y; type = :occurrence)
        @test all(0 .< occ .< 1)
        @test size(GLLVModels.predict(fit, Y; type = :positive)) == (p, n)
        @test size(GLLVModels.predict(fit, Y; type = :link)) == (p, n)
        @test GLLVModels.fitted(fit, Y) == Em
        @test_throws ArgumentError GLLVModels.predict(fit, Y; type = :bogus)

        r1 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(1))
        r2 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(1))
        @test r1 == r2 && all(isfinite, r1)

        k = 2p + (p * K - div(K * (K - 1), 2)) + 1
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.loglik
        @test GLLVModels.bic(fit, n) ≈ k * log(n) - 2 * fit.loglik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Delta-lognormal", s) && occursin("AIC", s)
    end
end

# Postfit under `disp_group = :species` (the DEFAULT since `accept delta
# dispersion A`, maintainer paste 2026-09-24): getLV/predict/residuals/show/
# _nparams all branch on `fit.disp_group` / `fit.σ isa Real`, but until now no
# test actually exercised those vector-σ/α branches — every prior postfit test
# pinned `disp_group = :shared`. This testset closes that gap for both fitters.
@testset "post-fit Delta-lognormal / Delta-Gamma under :species (default)" begin
    @testset "Delta-lognormal :species" begin
        Random.seed!(950)
        p, K, n = 5, 2, 180
        βz = 0.5 .* randn(p) .+ 0.4
        βc = 0.5 .* randn(p)
        Λc = 0.5 .* randn(p, K)
        σ_true = 0.2 .+ 0.5 .* rand(p)     # genuinely different per-trait sdlogs
        Z = randn(K, n)
        ηc = βc .+ Λc * Z
        π = inv.(1 .+ exp.(-βz))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t] && (Y[t, s] = exp(ηc[t, s] + σ_true[t] * randn()))
        end
        fit = fit_delta_lognormal_gllvm(Y; K = K)   # omitted: default is :species
        @test fit.disp_group === :species
        @test fit.σ isa Vector{Float64} && length(fit.σ) == p

        Zh = GLLVModels.getLV(fit, Y; rotate = false)
        @test size(Zh) == (n, K)
        for s in 1:n
            ẑ = GLLVModels._twopart_mode(GLLVModels.DeltaLogNormal.(fit.σ), view(Y, :, s),
                                    zeros(p, K), fit.Λc, fit.βz, fit.βc)
            @test Zh[s, :] ≈ ẑ atol = 1e-7
        end

        Em = GLLVModels.predict(fit, Y; type = :response)
        @test size(Em) == (p, n) && all(Em .>= 0)
        occ = GLLVModels.predict(fit, Y; type = :occurrence)
        @test all(0 .< occ .< 1)
        @test size(GLLVModels.predict(fit, Y; type = :positive)) == (p, n)
        @test size(GLLVModels.predict(fit, Y; type = :link)) == (p, n)
        @test GLLVModels.fitted(fit, Y) == Em

        r1 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(1))
        r2 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(1))
        @test r1 == r2 && all(isfinite, r1)

        k = 2p + (p * K - div(K * (K - 1), 2)) + p    # βz + βc + Λc + p sdlogs
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.loglik
        @test GLLVModels.bic(fit, n) ≈ k * log(n) - 2 * fit.loglik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Delta-lognormal", s) && occursin("per-trait", s) &&
              occursin("disp_group=:species", s)
    end

    @testset "Delta-Gamma :species" begin
        Random.seed!(951)
        p, K, n = 5, 2, 180
        βz = 0.5 .* randn(p) .+ 0.4
        βc = 0.4 .* randn(p)
        Λc = 0.4 .* randn(p, K)
        α_true = 1.5 .+ 4.0 .* rand(p)     # genuinely different per-trait shapes
        Z = randn(K, n)
        ηc = βc .+ Λc * Z
        π = inv.(1 .+ exp.(-βz))
        μ = exp.(ηc)
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            rand() < π[t] && (Y[t, s] = rand(Gamma(α_true[t], μ[t, s] / α_true[t])))
        end
        fit = fit_delta_gamma_gllvm(Y; K = K)   # omitted: default is :species
        @test fit.disp_group === :species
        @test fit.α isa Vector{Float64} && length(fit.α) == p

        Zh = GLLVModels.getLV(fit, Y; rotate = false)
        @test size(Zh) == (n, K)
        for s in 1:n
            ẑ = GLLVModels._twopart_mode(GLLVModels.DeltaGamma.(fit.α), view(Y, :, s),
                                    zeros(p, K), fit.Λc, fit.βz, fit.βc)
            @test Zh[s, :] ≈ ẑ atol = 1e-7
        end

        Em = GLLVModels.predict(fit, Y; type = :response)
        @test size(Em) == (p, n) && all(Em .>= 0)
        occ = GLLVModels.predict(fit, Y; type = :occurrence)
        @test all(0 .< occ .< 1)
        @test size(GLLVModels.predict(fit, Y; type = :positive)) == (p, n)
        @test size(GLLVModels.predict(fit, Y; type = :link)) == (p, n)
        @test GLLVModels.fitted(fit, Y) == Em

        r1 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(1))
        r2 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(1))
        @test r1 == r2 && all(isfinite, r1)

        k = 2p + (p * K - div(K * (K - 1), 2)) + p    # βz + βc + Λc + p shapes
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.loglik
        @test GLLVModels.bic(fit, n) ≈ k * log(n) - 2 * fit.loglik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Delta-Gamma", s) && occursin("per-trait", s) &&
              occursin("disp_group=:species", s)
    end
end
