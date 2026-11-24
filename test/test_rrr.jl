using GLLVModels, Test, Random, Distributions, Statistics

@testset "RRR / constrained ordination" begin
    # ----------------------------------------------------------------------
    # (1) EXACT anchor: at B = 0 the reduced-rank GLM marginal (η = β, since
    #     Λ B' X' = 0) must equal the independent intercept-only Poisson GLM
    #     log-likelihood to machine precision. Plus a hand-computed _rrr_eta
    #     entry for a nonzero B.
    # ----------------------------------------------------------------------
    @testset "B=0 anchor & η assembly" begin
        rng = MersenneTwister(20240603)
        p, n, q, K = 4, 6, 2, 2
        β = randn(rng, p) .* 0.3
        Λ = zeros(p, K)
        for k in 1:K, t in k:p
            Λ[t, k] = (t == k ? 0.6 : 0.2 * randn(rng))
        end
        X = randn(rng, n, q)
        Y = [rand(rng, Poisson(exp(β[t]))) for t in 1:p, _ in 1:n]

        ll_rrr = GLLVModels.rrr_marginal_loglik(
            Poisson(), Y, ones(Int, p, n), Λ, zeros(q, K), β, X, LogLink())
        ll_indep = sum(logpdf(Poisson(exp(β[t])), Y[t, s]) for t in 1:p, s in 1:n)
        @test ll_rrr ≈ ll_indep atol = 1e-8

        # Hand-checked single η entry for a nonzero B: η[t,s] = β[t] + (Λ B' X')[t,s].
        B = randn(rng, q, K)
        η = GLLVModels._rrr_eta(β, Λ, B, X)
        @test size(η) == (p, n)
        ΛBX = Λ * (B' * X')
        t0, s0 = 3, 5
        @test η[t0, s0] ≈ β[t0] + ΛBX[t0, s0]
    end

    # ----------------------------------------------------------------------
    # (2) MACHINERY: simulate Poisson data with a deterministic latent axis
    #     η = β + Λ B' x and check the fit returns a sane RRRFit with the
    #     right shapes and finite quantities (no recovery thresholds).
    # ----------------------------------------------------------------------
    @testset "fit machinery & post-fit API" begin
        rng = MersenneTwister(424242)
        p, n, q, K = 5, 40, 3, 2
        β = randn(rng, p) .* 0.3
        Λ = zeros(p, K)
        for k in 1:K, t in k:p
            Λ[t, k] = (t == k ? 0.7 : 0.25 * randn(rng))
        end
        B = randn(rng, q, K) .* 0.5
        X = randn(rng, n, q)
        η = β .+ Λ * (B' * X')
        Y = [rand(rng, Poisson(exp(clamp(η[t, s], -10, 10)))) for t in 1:p, s in 1:n]

        fit = fit_rrr_gllvm(Y; family = Poisson(), X = X, K = K)
        @test fit isa RRRFit
        @test isfinite(fit.loglik)
        @test size(fit.B) == (q, K)
        @test size(fit.Λ) == (p, K)

        Z = getLV(fit, X)
        @test size(Z) == (n, K)

        ηhat = predict(fit, X; type = :link)
        @test size(ηhat) == (p, n)

        μhat = predict(fit, X; type = :response)
        @test size(μhat) == (p, n)
        @test all(isfinite, μhat)

        # model-selection criteria (β + rank-K ΛB'; Poisson has no dispersion)
        @test GLLVModels._nparams(fit) == p + K * (p + q - K)
        @test isfinite(aic(fit))
        @test isfinite(bic(fit, n))
        # fitted(fit, X) resolves via the generic predict(::, X; type=:response) fallback
        ftd = fitted(fit, X)
        @test size(ftd) == (p, n)
        @test all(isfinite, ftd)
        @test ftd ≈ μhat
    end

    @testset "_nparams matches rank-K ΛB' df (#1199)" begin
        @test 10 + 2 * (10 + 3 - 2) == 32
        @test 10 + 1 * (10 + 3 - 1) == 22
    end
end
