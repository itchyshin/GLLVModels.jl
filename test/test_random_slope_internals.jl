using GLLVModels, Test, Random, LinearAlgebra
using Distributions: Poisson, MvNormal, logpdf

# Independent-reference tests for the grouped random-slope internals. The existing
# test_random_slopes*.jl files cover the q=1 reduction, recovery and FD gradients; this
# file adds dense-covariance references, permutation invariance, the augmented-loading
# layout and the zero-slope-variance limit.

# Dense n·p × n·p covariance of vec(y) (site-major stacking): I_n⊗(ΛΛ'+σ²I) plus, for
# every pair of sites in one group, (Z_i' Σ_b Z_j)·1ₚ1ₚ'.
function _dense_slope_cov(Λ, σ_eps, Z, Σ_b, codes)
    p = size(Λ, 1); n = length(codes)
    S = kron(Matrix(I, n, n), Λ * Λ' + σ_eps^2 * I)
    for i in 1:n, j in 1:n
        codes[i] == codes[j] || continue
        S[((i - 1) * p + 1):(i * p), ((j - 1) * p + 1):(j * p)] .+= dot(Z[i, :], Σ_b * Z[j, :])
    end
    return Symmetric(S)
end

_gidx(grouping) = (c = GLLVModels._code_grouping(grouping)[1];
                   (c, [findall(==(g), c) for g in 1:maximum(c)]))

@testset "random-slope internals" begin
    # (1) _unpack_chol_cov: symmetric PSD, Cholesky factor round-trips the packed theta.
    @testset "_unpack_chol_cov round trip, q = 1..3" begin
        Random.seed!(81001)
        for q in 1:3
            nc = GLLVModels._chol_cov_npar(q)
            @test nc == q * (q + 1) ÷ 2
            θ = 0.5 .* randn(nc)
            Σ, Lb = GLLVModels._unpack_chol_cov(θ, q)
            @test issymmetric(Σ)
            @test isposdef(Symmetric(Σ))
            @test Σ ≈ Lb * Lb'
            @test istril(Lb)
            C = cholesky(Symmetric(Σ)).L
            @test C ≈ Lb atol = 1e-10
            # re-pack from the Cholesky factor: log-diagonal, then below-diagonal, by column
            θ2 = Float64[]
            for j in 1:q
                push!(θ2, log(C[j, j]))
                for i in (j + 1):q
                    push!(θ2, C[i, j])
                end
            end
            @test θ2 ≈ θ atol = 1e-10
        end
    end

    # (2) dense MvNormal references on a small UNBALANCED grouping (sizes 1, 2, 3, 4).
    grouping0 = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    @testset "_grouped_intercept_loglik == dense logpdf(MvNormal)" begin
        Random.seed!(81002)
        p, K, n = 4, 2, length(grouping0)
        Λ = 0.5 .* randn(p, K); σ_eps, σ_u = 0.6, 0.8
        y = randn(p, n)
        codes, gi = _gidx(grouping0)
        S = _dense_slope_cov(Λ, σ_eps, ones(n, 1), fill(σ_u^2, 1, 1), codes)
        ref = logpdf(MvNormal(zeros(n * p), S), vec(y))
        @test GLLVModels._grouped_intercept_loglik(y, gi, Λ, σ_eps, σ_u) ≈ ref rtol = 1e-10
        @test GLLVModels.gaussian_grouped_intercept_loglik(y, grouping0, Λ, σ_eps, σ_u) ≈ ref rtol = 1e-10
    end

    @testset "_grouped_slope_loglik == dense logpdf(MvNormal), intercept + slope" begin
        Random.seed!(81003)
        p, K, n = 4, 1, length(grouping0)
        Λ = 0.5 .* randn(p, K); σ_eps = 0.6
        Z = hcat(ones(n), randn(n))
        Σ_b = [0.5 0.15; 0.15 0.4]
        y = randn(p, n)
        codes, gi = _gidx(grouping0)
        S = _dense_slope_cov(Λ, σ_eps, Z, Σ_b, codes)
        ref = logpdf(MvNormal(zeros(n * p), S), vec(y))
        @test GLLVModels._grouped_slope_loglik(y, gi, Z, Λ, σ_eps, Σ_b) ≈ ref rtol = 1e-10
    end

    # (3) invariance to the order of observations and of group labels.
    @testset "invariant to permuting observations and groups" begin
        Random.seed!(81004)
        p, K, n = 4, 1, length(grouping0)
        Λ = 0.5 .* randn(p, K); σ_eps, σ_u = 0.6, 0.8
        Z = hcat(ones(n), randn(n)); Σ_b = [0.5 0.15; 0.15 0.4]
        y = randn(p, n)
        _, gi = _gidx(grouping0)
        base_int = GLLVModels._grouped_intercept_loglik(y, gi, Λ, σ_eps, σ_u)
        base_slp = GLLVModels._grouped_slope_loglik(y, gi, Z, Λ, σ_eps, Σ_b)
        # permute observations (columns of y, rows of Z, entries of grouping)
        π = randperm(n)
        _, gip = _gidx(grouping0[π])
        @test GLLVModels._grouped_intercept_loglik(y[:, π], gip, Λ, σ_eps, σ_u) ≈ base_int rtol = 1e-10
        @test GLLVModels._grouped_slope_loglik(y[:, π], gip, Z[π, :], Λ, σ_eps, Σ_b) ≈ base_slp rtol = 1e-10
        # permute the order in which groups are visited
        gr = gi[randperm(length(gi))]
        @test GLLVModels._grouped_intercept_loglik(y, gr, Λ, σ_eps, σ_u) ≈ base_int rtol = 1e-10
        @test GLLVModels._grouped_slope_loglik(y, gr, Z, Λ, σ_eps, Σ_b) ≈ base_slp rtol = 1e-10
        # relabel groups
        relabel = [4, 3, 2, 1][grouping0]
        @test GLLVModels.gaussian_grouped_intercept_loglik(y, relabel, Λ, σ_eps, σ_u) ≈ base_int rtol = 1e-10
    end

    # (4) _slope_augment_loadings: size (n_g·p) × (n_g·K + q), hand-built reference.
    @testset "_slope_augment_loadings layout, K = 1 and K = 2" begin
        Random.seed!(81005)
        p, q, n = 3, 2, 6
        Z = hcat(ones(n), randn(n))
        Lb = [0.7 0.0; 0.2 0.5]
        for K in 1:2
            Λ = randn(p, K)
            sites = [5, 2, 4]; ng = length(sites)
            Λ̃ = GLLVModels._slope_augment_loadings(Λ, Z, Lb, sites)
            @test size(Λ̃) == (ng * p, ng * K + q)
            ref = zeros(ng * p, ng * K + q)
            for (j, i) in enumerate(sites)
                ref[((j - 1) * p + 1):(j * p), ((j - 1) * K + 1):(j * K)] = Λ
                zb = Lb' * Z[i, :]
                for t in 1:p
                    ref[(j - 1) * p + t, (ng * K + 1):(ng * K + q)] = zb
                end
            end
            @test Λ̃ == ref
            # singleton group: [Λ | 1ₚ (Lb'Z_i)']
            Λ1 = GLLVModels._slope_augment_loadings(Λ, Z, Lb, [3])
            @test Λ1 == hcat(Λ, repeat((Lb' * Z[3, :])', p))
        end
    end

    # (5) slope variance -> 0: group super-site Laplace -> plain per-site Poisson Laplace.
    @testset "random_slope_marginal_loglik_laplace -> plain Poisson Laplace as Σ_b -> 0" begin
        Random.seed!(81006)
        p, K, n = 4, 1, 12
        Λ = 0.4 .* randn(p, K); β = 0.3 .* randn(p)
        Y = float.(rand(0:5, p, n)); N = ones(Int, p, n)
        grouping = repeat(1:4, 3)
        Z = hcat(ones(n), randn(n))
        _, gi = _gidx(grouping)
        plain = GLLVModels.marginal_loglik_laplace(Poisson(), Y, N, Λ, β, GLLVModels.LogLink())
        ll(s) = GLLVModels.random_slope_marginal_loglik_laplace(
            Poisson(), Y, N, Z, Λ, β, Matrix(s * I, 2, 2), gi; link = GLLVModels.LogLink())
        d_small = abs(ll(1e-4) - plain)
        @test d_small < 1e-5                       # tolerance: O(σ²) gap, measured ≈ O(1e-8)
        @test abs(ll(1e-2) - plain) > d_small      # gap shrinks as the slope variance does
        @test ll(0.6) != plain                     # a real slope variance changes the value
    end

    # (6) fit_poisson_random_slope: finite fields, Bool converged, recovers slope sign.
    @testset "fit_poisson_random_slope smoke + slope sign" begin
        Random.seed!(81007)
        p, K, n, L = 5, 1, 400, 40
        Λt = 0.3 .* randn(p, K); βt = log.(2 .+ rand(p))
        x = randn(n); grouping = vcat(collect(1:L), rand(1:L, n - L))
        Z = hcat(ones(n), x)
        Σ_b = [0.3 0.2; 0.2 0.4]                  # positive intercept-slope covariance
        Lb = cholesky(Symmetric(Σ_b)).L
        bd = [Lb * randn(2) for _ in 1:L]
        re = [Z[i, 1] * bd[grouping[i]][1] + Z[i, 2] * bd[grouping[i]][2] for i in 1:n]
        η = βt .+ Λt * randn(K, n) .+ reshape(re, 1, n)
        Y = float.([rand(Poisson(exp(clamp(η[t, i], -20, 20)))) for t in 1:p, i in 1:n])
        fit = fit_poisson_random_slope(Y, grouping, Z; K = K, iterations = 150)
        @test fit isa GLLVModels.PoissonRandomSlopeFit
        @test all(isfinite, fit.β) && all(isfinite, fit.Λ) && all(isfinite, fit.Σ_b)
        @test isfinite(fit.loglik)
        @test fit.converged isa Bool
        @test fit.q == 2 && fit.nlevels == L && fit.iterations ≥ 1
        @test isposdef(Symmetric(fit.Σ_b))
        @test fit.Σ_b[2, 2] > 0                   # slope variance positive
        @test fit.Σ_b[1, 2] > 0                   # sign of the intercept-slope covariance
    end

    # fit_gaussian_random_slope: dimensions, finite fields, loglik == the marginal at the fit.
    @testset "fit_gaussian_random_slope returns its own maximised marginal" begin
        Random.seed!(81008)
        p, K, n, L = 5, 1, 200, 20
        x = randn(n); grouping = vcat(collect(1:L), rand(1:L, n - L))
        Z = hcat(ones(n), x)
        y = 0.5 .* randn(p, 1) * randn(1, n) .+ 0.5 .* randn(p, n)
        fit = fit_gaussian_random_slope(y, grouping, Z; K = K, iterations = 200)
        @test fit.converged isa Bool
        @test size(fit.Λ) == (p, K) && fit.nlevels == L && fit.q == 2
        @test all(isfinite, fit.Λ) && isfinite(fit.σ_eps) && fit.σ_eps > 0
        _, gi = _gidx(grouping)
        @test GLLVModels._grouped_slope_loglik(y, gi, Z, fit.Λ, fit.σ_eps, fit.Σ_b) ≈ fit.loglik rtol = 1e-8
    end
end
