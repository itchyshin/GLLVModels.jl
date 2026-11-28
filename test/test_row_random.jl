using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra

@testset "Random row effects" begin
    # ------------------------------------------------------------------
    # (a) EXACT anchor — σ_row = 0 reduces to the base K-LV marginal.
    # The augmenting column is the zero vector, so Λ̃ = [Λ | 0]; the augmented
    # Hessian is block-diagonal [[Λ'WΛ+I, 0],[0, 1]] with a zero mode
    # component, so the random-row marginal is IDENTICALLY the plain K-LV
    # marginal. Test Poisson AND a dispersion family (NB).
    # ------------------------------------------------------------------
    @testset "σ_row=0 exact reduction" begin
        Random.seed!(20260606)
        p, n, K = 5, 7, 2
        β = randn(p)
        Λ = 0.4 .* randn(p, K)
        # Poisson
        Yp = rand(0:5, p, n)
        ll_rr = GLLVModels.row_random_marginal_loglik_laplace(Poisson(), Yp, ones(Int, p, n),
                                                         Λ, β, 0.0)
        ll_base = GLLVModels.marginal_loglik_laplace(Poisson(), Yp, ones(Int, p, n),
                                                Λ, β, GLLVModels.LogLink())
        @test ll_rr ≈ ll_base atol = 1e-10
        # Negative binomial (dispersion family)
        fam = NegativeBinomial(5.0, 0.5)
        Yn = rand(0:8, p, n)
        ll_rr_nb = GLLVModels.row_random_marginal_loglik_laplace(fam, Yn, ones(Int, p, n),
                                                            Λ, β, 0.0)
        ll_base_nb = GLLVModels.marginal_loglik_laplace(fam, Yn, ones(Int, p, n),
                                                   Λ, β, GLLVModels.LogLink())
        @test ll_rr_nb ≈ ll_base_nb atol = 1e-10
    end

    # ------------------------------------------------------------------
    # (b) CONTINUITY — σ_row = ε tiny ≈ no-row-effect.
    # ------------------------------------------------------------------
    @testset "σ_row→0 continuity" begin
        Random.seed!(424242)
        p, n, K = 4, 6, 1
        β = randn(p)
        Λ = 0.5 .* randn(p, K)
        Y = rand(0:5, p, n)
        ll_eps = GLLVModels.row_random_marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n),
                                                          Λ, β, 1e-7)
        ll_base = GLLVModels.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n),
                                                Λ, β, GLLVModels.LogLink())
        @test ll_eps ≈ ll_base atol = 1e-6
    end

    # ------------------------------------------------------------------
    # (c) FIT smoke + loose recovery — simulate Poisson data WITH a real
    # row effect ρ_s ~ N(0, σ_row_true²), fit, and check the fit is
    # well-formed with σ̂_row within a factor of ~3 of truth and the
    # loadings Gram positively correlated with truth.
    # ------------------------------------------------------------------
    @testset "fit smoke + recovery" begin
        Random.seed!(777)
        p, n, K = 6, 120, 1
        σ_row_true = 0.7
        β = 0.5 .* randn(p)
        Λ = 0.6 .* randn(p, K)
        Z = randn(K, n)
        ρ = σ_row_true .* randn(n)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n, t in 1:p
            η = β[t] + ρ[s] + dot(Λ[t, :], Z[:, s])
            Y[t, s] = rand(Poisson(exp(η)))
        end

        fit = fit_row_random_gllvm(Y; family = Poisson(), K = K, iterations = 80)
        @test fit isa RowRandomFit
        @test isfinite(fit.loglik)
        @test fit.σ_row > 0
        @test isnan(fit.dispersion)
        # variance component — loose factor-of-~3 band
        @test σ_row_true / 3 ≤ fit.σ_row ≤ σ_row_true * 3

        # loadings Gram correlation (sign/rotation invariant for K=1)
        G_true = Λ * Λ'
        G_hat = fit.Λ * fit.Λ'
        c = cor(vec(G_true), vec(G_hat))
        @test c > 0.3

        # post-fit getLV + row-effect BLUPs are well-formed
        LV = getLV(fit, Y)
        @test size(LV) == (n, K)
        @test all(isfinite, LV)
        ρ̂ = row_effects(fit, Y)
        @test length(ρ̂) == n
        @test all(isfinite, ρ̂)
        # BLUPs track the simulated row effects (loose)
        @test cor(ρ̂, ρ) > 0.3
        # predict :roweffect agrees with row_effects
        @test predict(fit, Y; type = :roweffect) ≈ ρ̂
        @test size(predict(fit, Y; type = :response)) == (p, n)
    end

    # ------------------------------------------------------------------
    # Second small-σ_row start: Bernoulli d=2 (r6c seed 2, 8x250) has a local mode at
    # σ_row ≈ 0.66 (ll -1293.1906) reached from σ_row_init = 1.0; the better mode is
    # σ_row ≈ 0 (ll -1293.0028, matches gllvmTMB). The default fit must find it.
    # ------------------------------------------------------------------
    @testset "Bernoulli d=2 false-convergence (second start)" begin
        using DelimitedFiles
        Y = Matrix{Float64}(readdlm(joinpath(@__DIR__, "fixtures", "row_random_bern_d2_s2_Y.csv"), ','))
        fit = fit_row_random_gllvm(Y; family = Binomial(), K = 2)
        @test fit.converged
        @test fit.loglik >= -1293.01
        @test fit.σ_row < 0.1
    end
end
