using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra

@testset "Community row effects" begin
    # ------------------------------------------------------------------
    # (a) EXACT anchor 1 — offset build: every row equals ρ'.
    # ------------------------------------------------------------------
    @testset "offset build" begin
        O = GLLVModels._build_offset_row([0.0, 0.5, -0.3], 4)
        @test size(O) == (4, 3)
        # every row equals [0.0 0.5 -0.3]
        for t in 1:4
            @test O[t, 1] == 0.0
            @test O[t, 2] == 0.5
            @test O[t, 3] == -0.3
        end
        # column s is the constant ρ_s
        @test all(O[:, 2] .== 0.5)
        @test all(O[:, 3] .== -0.3)
    end

    # ------------------------------------------------------------------
    # (b) EXACT anchor 2 — ρ = 0 reduces to the base Poisson model.
    # A zero offset changes nothing, so the offset marginal must equal the
    # plain Poisson Laplace marginal to machine precision.
    # ------------------------------------------------------------------
    @testset "ρ=0 reduces to base model" begin
        Random.seed!(20260602)
        p, n, K = 5, 6, 2
        β = randn(p)
        Λ = 0.4 .* randn(p, K)          # nonzero loadings
        Y = rand(0:5, p, n)
        O0 = GLLVModels._build_offset_row(zeros(n), p)
        ll_off = GLLVModels._marginal_loglik_offset(Poisson(), Y, ones(Int, p, n),
                                               Λ, β, O0, GLLVModels.LogLink())
        ll_base = GLLVModels.poisson_marginal_loglik_laplace(Y, Λ, β)
        @test ll_off ≈ ll_base atol = 1e-8
    end

    # ------------------------------------------------------------------
    # (c) MACHINERY ONLY — simulate Poisson data WITH row effects, fit,
    # and assert the fit object is well-formed (no recovery thresholds).
    # ------------------------------------------------------------------
    @testset "fit machinery" begin
        Random.seed!(987)
        p, n, K = 6, 8, 1
        β = 0.5 .* randn(p)
        ρ = vcat(0.0, 0.4 .* randn(n - 1))   # ρ_1 = 0 reference
        Λ = 0.5 .* randn(p, K)
        Z = randn(K, n)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n, t in 1:p
            η = β[t] + ρ[s] + dot(Λ[t, :], Z[:, s])
            Y[t, s] = rand(Poisson(exp(η)))
        end

        fit = fit_roweffect_gllvm(Y; family = Poisson(), K = K, iterations = 80)
        @test fit isa RowEffectFit
        @test isfinite(fit.loglik)
        @test length(fit.ρ) == n
        @test fit.ρ[1] == 0.0
        @test all(isfinite, fit.ρ)
        @test length(fit.β) == p
        @test size(fit.Λ) == (p, K)
    end

    # ------------------------------------------------------------------
    # (d) MACHINERY ONLY — post-fit getLV / predict / ordination.
    # ------------------------------------------------------------------
    @testset "post-fit: getLV/predict" begin
        Random.seed!(988)
        p, n, K = 6, 10, 1
        β = 0.5 .* randn(p)
        ρ = vcat(0.0, 0.4 .* randn(n - 1))
        Λ = 0.5 .* randn(p, K)
        Z = randn(K, n)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n, t in 1:p
            η = β[t] + ρ[s] + dot(Λ[t, :], Z[:, s])
            Y[t, s] = rand(Poisson(exp(η)))
        end
        fit = fit_roweffect_gllvm(Y; family = Poisson(), K = K, iterations = 80)

        S = getLV(fit, Y)
        @test size(S) == (n, K)
        @test all(isfinite, S)

        ηhat = predict(fit, Y; type = :link)
        @test size(ηhat) == (p, n)
        μhat = predict(fit, Y; type = :response)
        @test size(μhat) == (p, n)
        @test all(isfinite, μhat)

        ord = ordination(fit, Y)
        @test size(ord.sites) == (n, K)
    end
end
