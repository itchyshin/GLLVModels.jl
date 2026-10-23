using GLLVModels, Test, Random, LinearAlgebra, Statistics, Distributions

if !isdefined(GLLVModels, :getLoadings)
    include(joinpath(@__DIR__, "..", "src", "postfit.jl"))
end

@testset "post-fit ordination core" begin
    @testset "rotation + getLoadings (Gaussian)" begin
        Random.seed!(0)
        p, K, n = 5, 2, 120
        Λt = 0.8 .* randn(p, K)
        y = Λt * randn(K, n) .+ 0.5 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)

        R = GLLVModels.rotation(fit)
        @test size(R) == (K, K)
        @test R' * R ≈ I(K) atol = 1e-10            # orthogonal

        Lr = GLLVModels.getLoadings(fit; rotate = true)
        L0 = GLLVModels.getLoadings(fit; rotate = false)
        @test size(Lr) == (p, K)
        @test L0 ≈ fit.pars.Λ                         # raw == stored Λ
        @test Lr ≈ L0 * R                             # rotated == Λ·R
        @test Lr * Lr' ≈ L0 * L0' atol = 1e-9         # rotation-invariant ΛΛ'
        # canonical: rotated columns ordered by decreasing norm
        nrm = [norm(@view Lr[:, k]) for k in 1:K]
        @test issorted(nrm; rev = true)
        # sign-fix: largest-magnitude entry of each rotated column is ≥ 0
        for k in 1:K
            @test Lr[argmax(abs.(@view Lr[:, k])), k] ≥ 0
        end
    end

    @testset "_laplace_mode matches the marginal's inner solve" begin
        Random.seed!(7)
        p, K, n = 4, 1, 1
        Λ = reshape([1.0, 0.8, -0.6, 0.4], p, K)
        β = [0.2, -0.1, 0.0, 0.3]
        y = reshape([1, 0, 1, 1], p, n)
        N = ones(Int, p, n)
        ẑ = GLLVModels._laplace_mode(view(y, :, 1), view(N, :, 1), Λ, β, LogitLink())
        @test length(ẑ) == K
        # At the mode the penalised-score stationarity holds: Λ'(working
        # residual) − ẑ ≈ 0 (the inner Newton step is ~0).
        η = β .+ Λ * ẑ
        μ = inv.(1 .+ exp.(-η))
        me = μ .* (1 .- μ)
        s = (vec(y) .- vec(N) .* μ) ./ (μ .* (1 .- μ)) .* me
        @test maximum(abs.(Λ' * s .- ẑ)) < 1e-6
    end

    @testset "getLV (Gaussian) matches the factor-analysis posterior" begin
        Random.seed!(1)
        p, K, n = 5, 2, 150
        Λt = 0.9 .* randn(p, K)
        y = Λt * randn(K, n) .+ 0.5 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)

        Z = GLLVModels.getLV(fit, y; rotate = false)
        @test size(Z) == (n, K)

        # Independent reference: m_s = (I + Λ'Ψ⁻¹Λ)⁻¹ Λ'Ψ⁻¹ y_s, Ψ = Σ_y − ΛΛ'.
        Λ = fit.pars.Λ
        Σ = GLLVModels.sigma_y_site(fit)
        Ψ = Σ - Λ * Λ'
        ΨiΛ = Ψ \ Λ
        M = Symmetric(I(K) + Λ' * ΨiΛ)
        Zref = (M \ (ΨiΛ' * y))'              # n×K
        @test Z ≈ Zref atol = 1e-8

        # Rotation consistency: Λ_rot Z_rotᵀ == Λ Z_rawᵀ.
        Zr = GLLVModels.getLV(fit, y; rotate = true)
        Lr = GLLVModels.getLoadings(fit; rotate = true)
        @test Lr * Zr' ≈ Λ * Z' atol = 1e-8
    end

    @testset "getLV (Binomial) matches per-site Laplace mode" begin
        Random.seed!(3)
        p, K, n = 6, 2, 80
        Λt = 0.9 .* randn(p, K)
        β  = 0.3 .* randn(p)
        η  = β .+ Λt * randn(K, n)
        μ  = inv.(1 .+ exp.(-η))
        Y  = Int.(rand(p, n) .< μ)
        fit = fit_binomial_gllvm(Y; K = K)

        Z = GLLVModels.getLV(fit, Y; rotate = false)
        @test size(Z) == (n, K)
        # Each row equals the per-site Laplace mode.
        N = ones(Int, p, n)
        for s in 1:n
            ẑ = GLLVModels._laplace_mode(view(Y, :, s), view(N, :, s), fit.Λ, fit.β, fit.link)
            @test Z[s, :] ≈ ẑ atol = 1e-7
        end
        # Rotation consistency.
        Zr = GLLVModels.getLV(fit, Y; rotate = true)
        @test GLLVModels.getLoadings(fit; rotate = true) * Zr' ≈ fit.Λ * Z' atol = 1e-7
    end
end

@testset "post-fit predict/fitted" begin
    @testset "predict (Gaussian): link == response, η = Λẑ" begin
        Random.seed!(2)
        p, K, n = 5, 2, 120
        Λt = 0.8 .* randn(p, K)
        y = Λt * randn(K, n) .+ 0.5 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)

        η = GLLVModels.predict(fit, y; type = :link)
        μ = GLLVModels.predict(fit, y; type = :response)
        @test size(η) == (p, n)
        @test η ≈ μ                                   # identity link
        Z = GLLVModels.getLV(fit, y; rotate = false)
        @test η ≈ fit.pars.Λ * Z' atol = 1e-10        # no fixed-effect mean
        @test GLLVModels.fitted(fit, y) ≈ μ
        @test_throws ArgumentError GLLVModels.predict(fit, y; type = :bogus)
    end

    @testset "predict (Binomial): probabilities in [0,1], logit-consistent" begin
        Random.seed!(4)
        p, K, n = 6, 2, 80
        η0 = 0.3 .* randn(p) .+ (0.9 .* randn(p, K)) * randn(K, n)
        Y  = Int.(rand(p, n) .< inv.(1 .+ exp.(-η0)))
        fit = fit_binomial_gllvm(Y; K = K)

        ηp = GLLVModels.predict(fit, Y; type = :link)
        pr = GLLVModels.predict(fit, Y; type = :response)
        @test size(pr) == (p, n)
        @test all(0 .≤ pr .≤ 1)
        @test pr ≈ inv.(1 .+ exp.(-ηp))               # logit link
        @test GLLVModels.fitted(fit, Y) ≈ pr
    end
end

@testset "post-fit residuals" begin
    @testset "residuals (Gaussian): standardized, DS == Pearson" begin
        Random.seed!(11)
        p, K, n = 5, 2, 300
        Λt = 0.8 .* randn(p, K)
        y = Λt * randn(K, n) .+ 0.5 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        rDS = GLLVModels.residuals(fit, y; type = :dunnsmyth)
        rP  = GLLVModels.residuals(fit, y; type = :pearson)
        @test size(rDS) == (p, n)
        @test rDS ≈ rP                                   # continuous CDF
        μ = GLLVModels.predict(fit, y; type = :response)
        @test rDS ≈ (y .- μ) ./ fit.pars.σ_eps atol = 1e-10
        @test_throws ArgumentError GLLVModels.residuals(fit, y; type = :bogus)
    end

    @testset "residuals (Binomial): DS reproducible + finite, Pearson formula" begin
        Random.seed!(13)
        p, K, n = 12, 1, 200
        η0 = 0.2 .* randn(p) .+ (0.9 .* randn(p, K)) * randn(K, n)
        Y  = Int.(rand(p, n) .< inv.(1 .+ exp.(-η0)))
        fit = fit_binomial_gllvm(Y; K = K)
        r1 = GLLVModels.residuals(fit, Y; type = :dunnsmyth, rng = MersenneTwister(1))
        r2 = GLLVModels.residuals(fit, Y; type = :dunnsmyth, rng = MersenneTwister(1))
        @test size(r1) == (p, n)
        @test r1 == r2                                    # reproducible with fixed rng
        @test all(isfinite, r1)
        # Loose sanity: roughly centered with real spread.
        @test abs(mean(r1)) < 0.3
        @test 0.3 < std(r1) < 2.0
        # Pearson formula (N = 1).
        μ = GLLVModels.predict(fit, Y; type = :response)
        rP = GLLVModels.residuals(fit, Y; type = :pearson)
        @test rP ≈ (Y .- μ) ./ sqrt.(μ .* (1 .- μ)) atol = 1e-10
    end
end

@testset "post-fit AIC/BIC + show" begin
    @testset "param count + AIC/BIC (Gaussian J1)" begin
        Random.seed!(21)
        p, K, n = 5, 2, 200
        Λt = 0.8 .* randn(p, K)
        y = Λt * randn(K, n) .+ 0.5 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        k = p * K - div(K * (K - 1), 2) + 1          # loadings + σ_eps (no intercepts, X=nothing)
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.logLik
        @test GLLVModels.bic(fit, n) ≈ k * log(n) - 2 * fit.logLik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Gaussian", s) && occursin("logLik", s) && occursin("AIC", s)
    end

    @testset "param count + AIC/BIC (Binomial)" begin
        Random.seed!(22)
        p, K, n = 6, 2, 120
        η0 = 0.3 .* randn(p) .+ (0.9 .* randn(p, K)) * randn(K, n)
        Y  = Int.(rand(p, n) .< inv.(1 .+ exp.(-η0)))
        fit = fit_binomial_gllvm(Y; K = K)
        k = p + (p * K - div(K * (K - 1), 2))        # intercepts + loadings
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.loglik
        @test GLLVModels.bic(fit, n) ≈ k * log(n) - 2 * fit.loglik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Binomial", s) && occursin("AIC", s)
    end
end

@testset "post-fit Poisson fits" begin
    Random.seed!(50)
    p, K, n = 6, 2, 150
    β = log.(fill(5.0, p))
    Λt = 0.4 .* randn(p, K)
    η = β .+ Λt * randn(K, n)
    Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
    fit = fit_gllvm(Y; family = Poisson(), K = K)

    @testset "getLV / getLoadings / rotation" begin
        Z = GLLVModels.getLV(fit, Y; rotate = false)
        @test size(Z) == (n, K)
        for s in 1:n
            ẑ = GLLVModels._laplace_mode(Poisson(), view(Y, :, s), ones(Int, p), fit.Λ, fit.β, fit.link)
            @test Z[s, :] ≈ ẑ atol = 1e-7
        end
        @test size(GLLVModels.getLoadings(fit)) == (p, K)
        R = GLLVModels.rotation(fit)
        @test R' * R ≈ I(K) atol = 1e-10
        Zr = GLLVModels.getLV(fit, Y; rotate = true)
        @test GLLVModels.getLoadings(fit; rotate = true) * Zr' ≈ fit.Λ * Z' atol = 1e-7
    end

    @testset "predict (rates) + residuals + AIC/BIC + show" begin
        η_hat = GLLVModels.predict(fit, Y; type = :link)
        μ_hat = GLLVModels.predict(fit, Y; type = :response)
        @test size(μ_hat) == (p, n)
        @test all(μ_hat .≥ 0)
        @test μ_hat ≈ exp.(η_hat)                          # log link

        r1 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(2))
        r2 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(2))
        @test r1 == r2 && all(isfinite, r1)
        rp = GLLVModels.residuals(fit, Y; type = :pearson)
        @test rp ≈ (Y .- μ_hat) ./ sqrt.(μ_hat) atol = 1e-10

        k = p + (p * K - div(K * (K - 1), 2))
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.loglik
        @test GLLVModels.bic(fit, n) ≈ k * log(n) - 2 * fit.loglik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Poisson", s) && occursin("AIC", s)
    end
end

@testset "post-fit NB fits" begin
    Random.seed!(80)
    p, K, n = 6, 2, 150
    β = log.(fill(5.0, p))
    Λt = 0.4 .* randn(p, K)
    r_true = 6.0
    μ = exp.(β .+ Λt * randn(K, n))
    Y = [rand(NegativeBinomial(r_true, r_true / (r_true + μ[t, s]))) for t in 1:p, s in 1:n]
    # Shared-φ postfit surface: named fitter (public fit_gllvm(NB) → grouped).
    fit = fit_nb_gllvm(Y; K = K)

    @testset "getLV / getLoadings / rotation" begin
        Z = GLLVModels.getLV(fit, Y; rotate = false)
        @test size(Z) == (n, K)
        for s in 1:n
            ẑ = GLLVModels._laplace_mode(NegativeBinomial(fit.r, 0.5), view(Y, :, s),
                                    ones(Int, p), fit.Λ, fit.β, fit.link)
            @test Z[s, :] ≈ ẑ atol = 1e-7
        end
        @test size(GLLVModels.getLoadings(fit)) == (p, K)
        @test GLLVModels.rotation(fit)' * GLLVModels.rotation(fit) ≈ I(K) atol = 1e-10
    end

    @testset "predict (means) + residuals + AIC/BIC + show" begin
        η_hat = GLLVModels.predict(fit, Y; type = :link)
        μ_hat = GLLVModels.predict(fit, Y; type = :response)
        @test all(μ_hat .≥ 0)
        @test μ_hat ≈ exp.(η_hat)
        r1 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(3))
        r2 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(3))
        @test r1 == r2 && all(isfinite, r1)
        rp = GLLVModels.residuals(fit, Y; type = :pearson)
        @test rp ≈ (Y .- μ_hat) ./ sqrt.(μ_hat .+ μ_hat .^ 2 ./ fit.r) atol = 1e-9
        k = p + (p * K - div(K * (K - 1), 2)) + 1            # + dispersion r
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.loglik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Negative-binomial", s) && occursin("AIC", s)
    end
end

@testset "post-fit Beta fits" begin
    Random.seed!(110)
    p, K, n = 6, 2, 200
    β = zeros(p)
    Λt = 0.5 .* randn(p, K)
    φ_true = 12.0
    μ = inv.(1 .+ exp.(-(β .+ Λt * randn(K, n))))
    Y = [rand(Beta(μ[t, s] * φ_true, (1 - μ[t, s]) * φ_true)) for t in 1:p, s in 1:n]
    # Shared-φ postfit surface: named fitter (public fit_gllvm(Beta) → grouped).
    fit = fit_beta_gllvm(Y; K = K)

    @testset "getLV / getLoadings / rotation" begin
        Z = GLLVModels.getLV(fit, Y; rotate = false)
        @test size(Z) == (n, K)
        for s in 1:n
            ẑ = GLLVModels._laplace_mode(Beta(fit.φ, 1.0), view(Y, :, s),
                                    ones(Int, p), fit.Λ, fit.β, fit.link)
            @test Z[s, :] ≈ ẑ atol = 1e-7
        end
        @test size(GLLVModels.getLoadings(fit)) == (p, K)
        @test GLLVModels.rotation(fit)' * GLLVModels.rotation(fit) ≈ I(K) atol = 1e-10
        Zr = GLLVModels.getLV(fit, Y; rotate = true)
        @test GLLVModels.getLoadings(fit; rotate = true) * Zr' ≈ fit.Λ * Z' atol = 1e-7
    end

    @testset "predict (proportions) + residuals + AIC/BIC + show" begin
        η_hat = GLLVModels.predict(fit, Y; type = :link)
        μ_hat = GLLVModels.predict(fit, Y; type = :response)
        @test size(μ_hat) == (p, n)
        @test all(0 .< μ_hat .< 1)
        @test μ_hat ≈ inv.(1 .+ exp.(-η_hat))                # logit link
        # Continuous CDF ⇒ Dunn–Smyth residual is deterministic (no rng arg).
        rDS = GLLVModels.residuals(fit, Y; type = :dunnsmyth)
        @test size(rDS) == (p, n)
        @test all(isfinite, rDS)
        rp = GLLVModels.residuals(fit, Y; type = :pearson)
        @test rp ≈ (Y .- μ_hat) ./ sqrt.(μ_hat .* (1 .- μ_hat) ./ (1 + fit.φ)) atol = 1e-9
        @test_throws ArgumentError GLLVModels.residuals(fit, Y; type = :bogus)
        k = p + (p * K - div(K * (K - 1), 2)) + 1            # + precision φ
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.loglik
        @test GLLVModels.bic(fit, n) ≈ k * log(n) - 2 * fit.loglik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Beta", s) && occursin("AIC", s)
    end
end

@testset "post-fit Gamma fits" begin
    Random.seed!(170)
    p, K, n = 6, 2, 200
    β = 0.5 .* randn(p)
    Λt = 0.4 .* randn(p, K)
    α_true = 5.0
    μ = exp.(β .+ Λt * randn(K, n))
    Y = [rand(Gamma(α_true, μ[t, s] / α_true)) for t in 1:p, s in 1:n]
    fit = fit_gllvm(Y; family = Gamma(), K = K)

    @testset "getLV / getLoadings / rotation" begin
        Z = GLLVModels.getLV(fit, Y; rotate = false)
        @test size(Z) == (n, K)
        ones_p = ones(Int, p)
        for s in 1:n
            ẑ = GLLVModels._laplace_mode(Gamma(fit.α, 1.0), view(Y, :, s),
                                    ones_p, fit.Λ, fit.β, fit.link)
            @test Z[s, :] ≈ ẑ atol = 1e-7
        end
        @test size(GLLVModels.getLoadings(fit)) == (p, K)
        @test GLLVModels.rotation(fit)' * GLLVModels.rotation(fit) ≈ I(K) atol = 1e-10
        Zr = GLLVModels.getLV(fit, Y; rotate = true)
        @test GLLVModels.getLoadings(fit; rotate = true) * Zr' ≈ fit.Λ * Z' atol = 1e-7
    end

    @testset "predict (positive) + residuals + AIC/BIC + show" begin
        η_hat = GLLVModels.predict(fit, Y; type = :link)
        μ_hat = GLLVModels.predict(fit, Y; type = :response)
        @test size(μ_hat) == (p, n)
        @test all(μ_hat .> 0)
        @test μ_hat ≈ exp.(η_hat)                              # log link
        # Continuous CDF ⇒ Dunn–Smyth residual is deterministic (no rng arg).
        rDS = GLLVModels.residuals(fit, Y; type = :dunnsmyth)
        @test size(rDS) == (p, n)
        @test all(isfinite, rDS)
        rp = GLLVModels.residuals(fit, Y; type = :pearson)
        @test rp ≈ (Y .- μ_hat) ./ sqrt.(μ_hat .^ 2 ./ fit.α) atol = 1e-9
        @test_throws ArgumentError GLLVModels.residuals(fit, Y; type = :bogus)
        k = p + (p * K - div(K * (K - 1), 2)) + 1            # + shape α
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.loglik
        @test GLLVModels.bic(fit, n) ≈ k * log(n) - 2 * fit.loglik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Gamma", s) && occursin("AIC", s)
    end
end

@testset "post-fit Ordinal fits" begin
    Random.seed!(140)
    p, K, n = 6, 2, 200
    C = 4
    τ = [-1.0, 0.1, 1.2]
    Λt = 0.5 .* randn(p, K)
    η = Λt * randn(K, n)
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n, t in 1:p
        pr = [GLLVModels._ord_prob(c, η[t, s], τ) for c in 1:C]
        Y[t, s] = rand(Categorical(pr))
    end
    # Shared-cutpoint postfit surface: named fitter (public fit_gllvm(Ordinal)
    # → per-trait parity route).
    fit = fit_ordinal_gllvm(Y; K = K)

    @testset "getLV / getLoadings / rotation" begin
        Z = GLLVModels.getLV(fit, Y; rotate = false)
        @test size(Z) == (n, K)
        for s in 1:n
            ẑ = GLLVModels._ordinal_laplace_mode(view(Y, :, s), fit.Λ, fit.τ)
            @test Z[s, :] ≈ ẑ atol = 1e-7
        end
        @test size(GLLVModels.getLoadings(fit)) == (p, K)
        @test GLLVModels.rotation(fit)' * GLLVModels.rotation(fit) ≈ I(K) atol = 1e-10
    end

    @testset "predict (class/prob/link) + residuals + AIC/BIC + show" begin
        cls = GLLVModels.predict(fit, Y; type = :class)
        @test size(cls) == (p, n)
        @test all(c -> 1 ≤ c ≤ C, cls)
        P = GLLVModels.predict(fit, Y; type = :prob)
        @test size(P) == (p, n, C)
        @test all(≥(0), P)
        @test all(isapprox.(sum(P; dims = 3), 1.0; atol = 1e-8))   # probs sum to 1
        @test GLLVModels.fitted(fit, Y) == cls                          # response == modal class
        @test size(GLLVModels.predict(fit, Y; type = :link)) == (p, n)
        r1 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(5))
        r2 = GLLVModels.residuals(fit, Y; rng = MersenneTwister(5))
        @test r1 == r2 && all(isfinite, r1)
        @test_throws ArgumentError GLLVModels.residuals(fit, Y; type = :pearson)
        k = (p * K - div(K * (K - 1), 2)) + (C - 1)
        @test GLLVModels._nparams(fit) == k
        @test GLLVModels.aic(fit) ≈ 2k - 2 * fit.loglik
        @test GLLVModels.bic(fit, n) ≈ k * log(n) - 2 * fit.loglik
        s = sprint(show, MIME("text/plain"), fit)
        @test occursin("Ordinal", s) && occursin("AIC", s)
    end
end

@testset "GllvmFit postfit requires X when β was estimated" begin
    # Regression: _fitted_mean returned a zero mean when X was omitted, so
    # getLV/predict/fitted/residuals on a fit with fixed effects were silently
    # wrong. They must now throw, and still agree when X is supplied.
    Random.seed!(7)
    p, K, n = 6, 2, 80
    x = randn(n)
    X = zeros(p, n, 2); X[:, :, 1] .= 1.0; X[:, :, 2] .= x'
    Y = 3.0 .+ 2.0 .* x' .+ 0.6 .* randn(p, K) * randn(K, n) .+ 0.3 .* randn(p, n)
    fit = fit_gaussian_gllvm(Y; K = K, X = X)
    @test fit.converged
    @test length(fit.pars.β) == 2
    @test !GLLVModels._has_gaussian_record(fit)

    @testset "missing X throws" begin
        @test_throws ArgumentError getLV(fit, Y)
        @test_throws ArgumentError predict(fit, Y)
        @test_throws ArgumentError fitted(fit, Y)
        @test_throws ArgumentError residuals(fit, Y)
    end

    @testset "supplied X uses the fitted mean" begin
        μ = [dot(X[t, s, :], fit.pars.β) for t in 1:p, s in 1:n]
        Z = getLV(fit, Y; X = X, rotate = false)
        η = predict(fit, Y; X = X)
        @test η ≈ μ .+ fit.pars.Λ * Z' atol = 1e-10
        @test fitted(fit, Y; X = X) ≈ η atol = 1e-10
        @test residuals(fit, Y; X = X) ≈ (Y .- η) ./ fit.pars.σ_eps atol = 1e-10
        # Residuals centre near zero only when the fixed-effect mean is used.
        @test abs(mean(residuals(fit, Y; X = X))) < 0.1
    end

    @testset "the X_lv-only :mean component does not need X" begin
        @test getLV(fit, Y; component = :mean, rotate = false) == zeros(n, K)
    end

    @testset "a fit without X still works without X" begin
        f0 = fit_gaussian_gllvm(Y .- mean(Y; dims = 2); K = K)
        @test isempty(f0.pars.β)
        @test size(predict(f0, Y .- mean(Y; dims = 2))) == (p, n)
    end
end
