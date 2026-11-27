using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra

# Truncated CMP pmf (same tolerance/cap as the likelihood) for sampler checks.
function _cmp_pmf_probs(η, ν)
    masses = Float64[]
    total = 0.0
    y = 0
    tol = GLLVModels._CMP_LOGZ_TOL
    cap = GLLVModels._CMP_LOGZ_CAP
    while y < cap
        mass = exp(GLLVModels.compoisson_logpdf(y, η, ν))
        push!(masses, mass)
        total += mass
        y += 1
        mass < tol * total && break
    end
    return masses ./ total
end

@testset "COMPoisson simulate / sampler (#1533)" begin
    rng = MersenneTwister(1533)
    n_samp = 80_000
    η = 1.1
    λ = exp(GLLVModels._clamp_eta(η))

    for ν in (0.6, 1.0, 1.8)
        @testset "pmf check ν=$ν" begin
            probs = _cmp_pmf_probs(η, ν)
            ymax = length(probs) - 1
            counts = zeros(Int, length(probs))
            for _ in 1:n_samp
                y = GLLVModels._compoisson_sample(rng, η, ν)
                @test 0 ≤ y ≤ ymax
                counts[y + 1] += 1
            end
            emp = counts ./ n_samp
            if ν == 1.0
                pois = [pdf(Poisson(λ), y) for y in 0:ymax]
                @test maximum(abs.(emp[1:(ymax + 1)] .- pois)) < 0.01
            else
                @test maximum(abs.(emp .- probs)) < 0.01
            end
        end
    end

    @testset "simulate(fit, n) vs predict rates" begin
        Random.seed!(42)
        p, n0, K = 3, 80, 1
        β = [0.5, 0.2, 0.8]
        Λ = reshape([0.15, -0.1, 0.12], p, K)
        Z = randn(K, n0)
        Y = [rand(Poisson(exp(β[t] + dot(view(Λ, t, :), view(Z, :, s))))) for t in 1:p, s in 1:n0]
        fit = fit_compoisson_gllvm(Y; K = K, iterations = 80)
        n_sim = 25_000
        Ysim = simulate(fit, n_sim; rng = MersenneTwister(77))
        @test Ysim isa Matrix{Int}
        @test size(Ysim) == (p, n_sim)
        @test all(Ysim .>= 0)
        ref_rng = MersenneTwister(88)
        rates = predict(fit, Y; type = :rate)
        for t in 1:p
            sim_mean = mean(Ysim[t, :])
            marg_mean = mean(GLLVModels._compoisson_sample(ref_rng, fit.β[t] + dot(fit.Λ[t, :], randn(ref_rng, K)), fit.ν) for _ in 1:20_000)
            @test sim_mean ≈ marg_mean rtol = 0.08
            # Mean in-sample rate (Laplace λ per site) is on the same scale as simulate counts.
            pred_mean = mean(rates[t, :])
            @test abs(log1p(sim_mean) - log1p(pred_mean)) < 3.0
        end
        @test simulate(fit, 10; rng = MersenneTwister(1)) ==
              simulate(fit, 10; rng = MersenneTwister(1))
    end
end