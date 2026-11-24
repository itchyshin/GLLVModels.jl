using GLLVModels, Test, Random, LinearAlgebra, Statistics, Distributions

# Issue #1210: bootstrap loading percentiles must Procrustes-align each replicate
# to the point estimate (gllvmTMB loading-ci-bootstrap.R).

if !isdefined(@__MODULE__, :bootstrap_ci)
    include(joinpath(@__DIR__, "..", "src", "confint_bootstrap.jl"))
end

@testset "bootstrap loading Procrustes alignment (#1210)" begin
    @testset "Gaussian bootstrap_ci — aligned intervals bracket Wald for weak K=2" begin
        for seed in 1:2
            rng = MersenneTwister(seed)
            p, n, K = 8, 100, 2
            b = 0.2 .* randn(rng, p)
            L = [i < k ? 0.0 : 0.5 * (i == k ? 1.0 :
                 (iseven(i + k) ? 1 : -1) * (0.4 + 0.6 * (((i * 7 + k * 3) % 5) / 5))) for i in 1:p, k in 1:K]
            Y = zeros(p, n)
            for s in 1:n
                z = randn(rng, K)
                for t in 1:p
                    Y[t, s] = b[t] + dot(L[t, :], z) + 0.7 * randn(rng)
                end
            end
            fit = fit_gaussian_gllvm(Y; K = K)
            n_boot = seed == 1 ? 120 : 80
            ci = bootstrap_ci(fit; y = Y, n_boot = n_boot, seed = seed)
            w = confint(fit; y = Y)
            i0 = findfirst(startswith("Lambda_B"), ci.term)
            @test !isnothing(i0)
            rr = GLLVModels.rr_theta_len(p, K)
            Lh = fit.pars.Λ
            conv = findall(ci.converged)
            Lb = [GLLVModels.unpack_lambda(ci.replicates[b, i0:(i0 + rr - 1)], p, K) for b in conv]
            n_flip = count(x -> dot(x[:, 2], Lh[:, 2]) < 0, Lb)
            @test n_flip == 0
            j = findfirst(==("Lambda_B[5,2]"), ci.term)
            jw = findfirst(==("Lambda_B[5,2]"), w.term)
            @test !isnothing(j) && !isnothing(jw)
            est = Lh[5, 2]
            # Aligned bootstrap CI should not straddle zero when Wald lower and estimate share sign.
            if w.lower[jw] * est > 0
                @test ci.lower[j] * est > 0
            end
            if w.upper[jw] * est > 0
                @test ci.upper[j] * est > 0
            end
        end
    end

    @testset "Poisson confint(:bootstrap) — aligned loading replicates" begin
        seed = 1
        rng = MersenneTwister(seed)
        p, n, K = 6, 60, 1
        β = 0.15 .* randn(rng, p)
        Λ = 0.3 .* randn(rng, p, K)
        η = β .+ Λ * randn(rng, K, n)
        Y = rand.(rng, Poisson.(exp.(η)))
        fit = fit_poisson_gllvm(Y; K = K)
        ci = confint(fit, Y; method = :bootstrap, n_boot = 100, seed = seed)
        w = confint(fit, Y; method = :wald)
        j = findfirst(==("Lambda[3,1]"), ci.term)
        jw = findfirst(==("Lambda[3,1]"), w.term)
        @test !isnothing(j) && !isnothing(jw)
        est = fit.Λ[3, 1]
        @test isfinite(ci.lower[j]) && isfinite(ci.upper[j])
        if w.lower[jw] * est > 0 && w.upper[jw] * est > 0
            @test ci.lower[j] * est > 0
            @test ci.upper[j] * est > 0
        end
    end
end