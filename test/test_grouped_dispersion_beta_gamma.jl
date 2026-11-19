using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra

# Grouped / species-specific dispersion for Beta (precision φ) and Gamma (shape α),
# mirroring the NB disp.group path. Anchors:
#   1. EXACT REDUCTION: a constant per-species dispersion equals the shared-dispersion
#      scalar marginal to machine precision — the grouped path reduces exactly;
#   2. a one-group fit runs, is finite, and its single dispersion ≈ the scalar fit's;
#   3. a per-species (group = 1:p) smoke fit runs and returns a positive length-G vector.

@testset "Grouped / species-specific Beta & Gamma dispersion (disp.group)" begin

    @testset "Beta: constant φvec == shared-φ marginal (exact)" begin
        Random.seed!(601)
        p, K, n = 5, 1, 60
        β = 0.3 .* randn(p)
        Λ = 0.4 .* randn(p, K)
        φ = 8.0
        Y = Matrix{Float64}(undef, p, n)
        for s in 1:n
            η = β .+ Λ * randn(K)
            μ = 1 ./ (1 .+ exp.(-η))
            for t in 1:p
                Y[t, s] = rand(Beta(μ[t] * φ, (1 - μ[t]) * φ))
            end
        end
        ll_shared  = GLLVModels.beta_marginal_loglik_laplace(Y, Λ, β, φ)
        ll_grouped = GLLVModels.beta_grouped_marginal_loglik_laplace(Y, Λ, β, fill(φ, p))
        @test ll_grouped ≈ ll_shared atol = 1e-10
        # mixed per-species precision also evaluates finitely.
        @test isfinite(GLLVModels.beta_grouped_marginal_loglik_laplace(Y, Λ, β, [4.0, 8.0, 12.0, 20.0, 30.0]))
    end

    @testset "Gamma: constant αvec == shared-α marginal (exact)" begin
        Random.seed!(602)
        p, K, n = 5, 1, 60
        β = 0.3 .* randn(p)
        Λ = 0.4 .* randn(p, K)
        α = 3.0
        Y = Matrix{Float64}(undef, p, n)
        for s in 1:n
            η = β .+ Λ * randn(K)
            μ = exp.(η)
            for t in 1:p
                Y[t, s] = rand(Gamma(α, μ[t] / α))
            end
        end
        # `tol` here is the MODE tolerance, not the assertion tolerance, and it
        # is tightened rather than the atol loosened. Reason, measured:
        # Gamma/log's FISHER log-det weight is the constant α, so it is entirely
        # mode-INSENSITIVE and this identity held at any mode tolerance. The
        # OBSERVED weight is α·y/μ(ẑ), which IS mode-sensitive, so the two
        # Newton loops stopping within the default tol=1e-9 of the same fixed
        # point — but at slightly different points — leave a ~7e-11 residual.
        # Sweeping the mode tolerance: 1e-9 → 7.08e-11, 1e-11 → 9.10e-13,
        # 1e-13 → 0.000e+00 EXACTLY. The identity is exact; the default mode
        # tolerance was simply not tight enough to observe it once the log-det
        # became mode-sensitive. Fixing the cause, not the tolerance.
        ll_shared  = GLLVModels.gamma_marginal_loglik_laplace(Y, Λ, β, α; tol = 1e-13)
        ll_grouped = GLLVModels.gamma_grouped_marginal_loglik_laplace(Y, Λ, β, fill(α, p); tol = 1e-13)
        @test ll_grouped ≈ ll_shared atol = 1e-10
        # mixed per-species shape also evaluates finitely.
        @test isfinite(GLLVModels.gamma_grouped_marginal_loglik_laplace(Y, Λ, β, [1.5, 3.0, 5.0, 8.0, 12.0]))
    end

    @testset "fit_beta_gllvm_grouped: one group ≈ fit_beta_gllvm" begin
        Random.seed!(603)
        p, K, n = 5, 1, 80
        βtrue = 0.4 .* randn(p)
        Λtrue = 0.4 .* randn(p, K)
        φtrue = 12.0
        Z = randn(K, n)
        η = βtrue .+ Λtrue * Z
        μ = 1 ./ (1 .+ exp.(-η))
        Y = [rand(Beta(μ[t, i] * φtrue, (1 - μ[t, i]) * φtrue)) for t in 1:p, i in 1:n]

        fg = fit_beta_gllvm_grouped(Y; K = K, group = ones(Int, p), iterations = 40)
        @test fg isa GLLVModels.BetaGroupedFit
        @test length(fg.φ) == 1
        @test all(fg.φ .> 0)
        @test isfinite(fg.loglik)
        fs = fit_beta_gllvm(Y; K = K, iterations = 40)
        # one shared group ⇒ same precision to within ~20%.
        @test isapprox(fg.φ[1], fs.φ; rtol = 0.2)
    end

    @testset "fit_gamma_gllvm_grouped: one group ≈ fit_gamma_gllvm" begin
        Random.seed!(604)
        p, K, n = 5, 1, 80
        βtrue = 0.4 .* randn(p)
        Λtrue = 0.4 .* randn(p, K)
        αtrue = 4.0
        Z = randn(K, n)
        η = βtrue .+ Λtrue * Z
        μ = exp.(η)
        Y = [rand(Gamma(αtrue, μ[t, i] / αtrue)) for t in 1:p, i in 1:n]

        fg = fit_gamma_gllvm_grouped(Y; K = K, group = ones(Int, p), iterations = 40)
        @test fg isa GLLVModels.GammaGroupedFit
        @test length(fg.α) == 1
        @test all(fg.α .> 0)
        @test isfinite(fg.loglik)
        fs = fit_gamma_gllvm(Y; K = K, iterations = 40)
        # one shared group ⇒ same shape to within ~20%.
        @test isapprox(fg.α[1], fs.α; rtol = 0.2)
    end

    @testset "fit_beta_gllvm_grouped: per-species smoke (group = 1:p)" begin
        Random.seed!(605)
        p, K, n = 4, 1, 60
        β = 0.3 .* randn(p)
        Λ = 0.4 .* randn(p, K)
        Z = randn(K, n)
        η = β .+ Λ * Z
        μ = 1 ./ (1 .+ exp.(-η))
        Y = [rand(Beta(μ[t, i] * 10.0, (1 - μ[t, i]) * 10.0)) for t in 1:p, i in 1:n]

        fg = fit_beta_gllvm_grouped(Y; K = K, iterations = 40)  # default group = 1:p
        @test fg isa GLLVModels.BetaGroupedFit
        @test length(fg.φ) == p
        @test all(fg.φ .> 0)
        @test isfinite(fg.loglik)
        @test fg.group == collect(1:p)
    end

    @testset "fit_gamma_gllvm_grouped: per-species smoke (group = 1:p)" begin
        Random.seed!(606)
        p, K, n = 4, 1, 60
        β = 0.3 .* randn(p)
        Λ = 0.4 .* randn(p, K)
        Z = randn(K, n)
        η = β .+ Λ * Z
        μ = exp.(η)
        Y = [rand(Gamma(3.0, μ[t, i] / 3.0)) for t in 1:p, i in 1:n]

        fg = fit_gamma_gllvm_grouped(Y; K = K, iterations = 40)  # default group = 1:p
        @test fg isa GLLVModels.GammaGroupedFit
        @test length(fg.α) == p
        @test all(fg.α .> 0)
        @test isfinite(fg.loglik)
        @test fg.group == collect(1:p)
    end

    # #1156: the observed Laplace curvature exists only for Beta with the logit link. The
    # grouped fitters used to default to :observed for every link, the objective's bare
    # `catch` turned the resulting ArgumentError into the 1e12 sentinel, and the fit came
    # back with loglik -Inf, converged = false, iterations = 0. The default is now the
    # link's own default curvature (`_default_hessian`), as in the shared-precision fitter.
    @testset "Beta grouped: probit / cloglog links fit under the default curvature (#1156)" begin
        rng = MersenneTwister(11); p, n = 4, 40
        L = [0.8, 0.4, -0.5, 0.3]; z = randn(rng, n); a = rand(rng, p) .* 0.7 .+ 0.3
        Y = [(μ = cdf(Normal(), 0.5 * a[j] - 0.3 + 0.5 * L[j] * z[i]); φ = 8.0;
              clamp(rand(rng, Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)) for j in 1:p, i in 1:n]
        X = reshape(repeat(z', p, 1), p, n, 1)
        for lk in (ProbitLink(), CLogLogLink())
            f = fit_gllvm(Y; family = Beta(), K = 1, link = lk)
            g = fit_beta_gllvm_grouped(Y; K = 1, link = lk, hessian = :fisher)
            @test isfinite(f.loglik)
            @test f.iterations > 0
            @test f.converged
            @test f.hessian === :fisher
            @test f.loglik ≈ g.loglik atol = 1e-6
            fc = fit_beta_gllvm_grouped_cov(Y; X = X, K = 1, link = lk)
            @test isfinite(fc.loglik)
            @test fc.iterations > 0
            @test fc.hessian === :fisher
            m = beta_grouped_marginal_loglik_laplace(Y, reshape([0.3, 0.2, -0.2, 0.1], p, 1),
                                                     zeros(p), fill(10.0, p); link = lk)
            @test isfinite(m)
        end
        # the logit default is unchanged
        @test fit_beta_gllvm_grouped(Y; K = 1, iterations = 5).hessian === :observed
        # an explicit :observed with a non-logit link keeps refusing
        @test_throws ArgumentError beta_grouped_marginal_loglik_laplace(Y,
            reshape([0.3, 0.2, -0.2, 0.1], p, 1), zeros(p), fill(10.0, p);
            link = ProbitLink(), hessian = :observed)
    end
end
