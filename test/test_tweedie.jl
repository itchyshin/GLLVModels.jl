using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra, SpecialFunctions

@testset "Tweedie family" begin
    # ---------------------------------------------------------------------
    # 1. y = 0 exact point mass: log f(0) = -μ^{2-p} / (φ (2-p)).
    # ---------------------------------------------------------------------
    @testset "y=0 point mass" begin
        for (μ, φ, p) in [(1.0, 1.0, 1.5), (3.0, 0.5, 1.3), (0.7, 2.0, 1.7)]
            @test isapprox(tweedie_logpdf(0.0, μ, φ, p),
                           -μ^(2 - p) / (φ * (2 - p)); atol = 1e-10)
        end
    end

    # ---------------------------------------------------------------------
    # 2. p → 2 limit ⇒ Gamma. The series gate. At p → 2 the Tweedie is a
    #    Gamma with shape 1/φ, scale μφ (mean μ, Var φ μ²).
    # ---------------------------------------------------------------------
    @testset "p->2 Gamma limit (series gate)" begin
        μ = 2.0; φ = 0.7; p = 1.99
        for y in [0.3, 1.0, 2.5, 5.0]
            got = tweedie_logpdf(y, μ, φ, p)
            ref = logpdf(Gamma(1 / φ, μ * φ), y)
            @test isapprox(got, ref; atol = 5e-2)
        end
    end

    # ---------------------------------------------------------------------
    # 3. Λ = 0 ⇒ the marginal reduces to a sum of independent logpdfs at
    #    μ_t = exp(β_t).
    # ---------------------------------------------------------------------
    @testset "Lambda=0 marginal reduction" begin
        Random.seed!(11)
        p_sp = 4; n = 6; K = 2; φ = 0.8; p = 1.4
        β = randn(p_sp) .* 0.3
        Y = zeros(p_sp, n)
        for t in 1:p_sp, s in 1:n
            Y[t, s] = rand() < 0.3 ? 0.0 : rand() * 3.0 + 0.1
        end
        got = tweedie_marginal_loglik_laplace(Y, zeros(p_sp, K), β, φ, p)
        ref = sum(tweedie_logpdf(Y[t, s], exp(β[t]), φ, p)
                  for t in 1:p_sp, s in 1:n)
        @test isapprox(got, ref; atol = 1e-8)
    end

    # ---------------------------------------------------------------------
    # 4. Machinery: fit runs and returns sane structure (no recovery thresholds).
    # ---------------------------------------------------------------------
    @testset "fit machinery" begin
        Random.seed!(2024)
        p_sp = 5; n = 40; K = 2
        β = log.(rand(p_sp) .* 2 .+ 0.5)
        Λ = randn(p_sp, K) .* 0.4
        Y = zeros(p_sp, n)
        for s in 1:n
            z = randn(K)
            for t in 1:p_sp
                μ = exp(β[t] + dot(Λ[t, :], z))
                # compound Poisson–Gamma draw (true zeros + positive part)
                λ = μ                       # rough Poisson intensity
                k = rand(Poisson(λ))
                Y[t, s] = k == 0 ? 0.0 : sum(rand(Gamma(2.0, μ / (2.0 * λ + 1e-9)), k))
            end
        end
        fit = fit_tweedie_gllvm(Y; K = K)
        @test fit isa TweedieFit
        @test isfinite(fit.loglik)
        @test 1 < fit.p < 2
        @test fit.φ > 0
        @test size(fit.Λ) == (p_sp, K)
        # `converged` must mean converged: see test_tweedie_engine_health.jl for
        # the full G-a…G-d gates this cell used to pass while stalled.
        @test fit.converged
    end

    # ---------------------------------------------------------------------
    # 8. Small dispersion (#1177): the normalising series must not be cut at a
    #    fixed number of terms. At p = 3/2 the density has a closed form,
    #    log f = -(2/φ)(√μ + y/√μ) + log(√x I₁(2√x) / y),  x = 4y/φ²,
    #    with log I₁(z) = z + log(besselix(1, z)).
    # ---------------------------------------------------------------------
    @testset "small dispersion: series not truncated (#1177)" begin
        closed(y, μ, φ) = begin
            x = 4y / φ^2; z = 2sqrt(x)
            -(2 / φ) * (sqrt(μ) + y / sqrt(μ)) + 0.5 * log(x) + z + log(besselix(1, z)) - log(y)
        end
        for (y, μ, φ) in [(2.0, 2.0, 1e-4), (2.0, 2.0, 1e-6), (8.0, 8.0, 1e-6),
                          (2.0, 2.0, 1e-7), (8.0, 8.0, 1e-7), (0.5, 0.5, 1e-7),
                          (2.0, 1.3, 1e-7)]
            @test isapprox(tweedie_logpdf(y, μ, φ, 1.5), closed(y, μ, φ); atol = 1e-6)
        end
        # the AD copy of the series used by the AGHQ Tweedie fitter has to agree
        for (y, φ, pw) in [(2.0, 1e-7, 1.5), (0.57, 1.5e-6, 1.9496), (1.2, 3e-7, 1.2936)]
            @test GLLVModels._tweedie_logA_ad(y, φ, pw) ≈ GLLVModels._tweedie_logA(y, φ, pw) atol = 1e-8
        end
        # where the old fixed cap was not reached the cap does not change anything
        @test GLLVModels._tweedie_series_cap(1) == 5000
        @test GLLVModels._tweedie_series_cap(10^12) == 400_000
    end
end
