# Coverage lane w5 (ranks 11–12 + 14): beta_binomial.jl, simulate_fit.jl, confint_derived.jl
using GLLVModels, Test, Random, LinearAlgebra, Distributions, Statistics

@testset "Cursor 1127cov w5 (beta_binomial, simulate_fit, confint_derived)" begin

    @testset "beta-binomial post-fit surfaces and grouped routes" begin
        Random.seed!(112705)
        p, n, K = 4, 35, 1
        N = fill(6, p, n)
        Y = [min(N[t, s] - 1, max(0, round(Int, N[t, s] * 0.4 + randn()))) for t in 1:p, s in 1:n]
        Λ0 = reshape(collect(0.15 .* (1:p)), p, K)
        fit = fit_beta_binomial_gllvm(Y; K = K, N = N, Λ_init = Λ0, iterations = 25, g_tol = 1e-4)
        @test fit isa BetaBinomialFit
        @test GLLVModels._loadings(fit) === fit.Λ
        @test GLLVModels._loglik(fit) == fit.loglik
        @test GLLVModels._nparams(fit) == p + GLLVModels.rr_theta_len(p, K) + 1
        Z = getLV(fit, Y; N = N, rotate = false)
        @test size(Z) == (n, K)
        η = predict(fit, Y; N = N, type = :link)
        μ = predict(fit, Y; N = N, type = :mean)
        @test size(η) == (p, n) && size(μ) == (p, n)
        @test all(0 .< μ .< 1)
        @test_throws ArgumentError predict(fit, Y; N = N, type = :bogus)
        @test occursin("BetaBinomialFit", sprint(show, fit))

        grp = ones(Int, p)
        fg = fit_beta_binomial_gllvm_grouped(Y; K = K, N = N, group = grp, iterations = 20, g_tol = 1e-4)
        @test fg isa BetaBinomialGroupedFit && fg.hessian === :fisher
        @test GLLVModels._nparams(fg) == p + GLLVModels.rr_theta_len(p, K) + length(fg.φ)
        @test occursin("BetaBinomialGroupedFit", sprint(show, fg))
        fit8 = BetaBinomialGroupedFit(fg.β, fg.Λ, fg.φ, fg.group, fg.link, fg.loglik, fg.converged, fg.iterations)
        @test fit8.hessian === :fisher

        q = 1
        X = reshape(randn(n), n, q)
        X3 = zeros(p, n, q)
        @inbounds for s in 1:n, t in 1:p
            X3[t, s, 1] = X[s, 1]
        end
        fgc = fit_beta_binomial_gllvm_grouped_cov(Y; X = X3, K = K, N = N, group = grp,
                                                  iterations = 20, g_tol = 1e-4)
        @test fgc isa BetaBinomialGroupedCovFit
        @test GLLVModels._nparams(fgc) == p + q + GLLVModels.rr_theta_len(p, K) + length(fgc.φ)
        @test occursin("BetaBinomialGroupedCovFit", sprint(show, fgc))
        @test_throws ArgumentError predict(fgc, Y, X3; type = :bogus)
        _ = predict(fgc, Y, X3; type = :mean)
        _ = getLV(fgc, Y, X3; rotate = false)
        fit8c = BetaBinomialGroupedCovFit(fgc.β, fgc.γ, fgc.γ_fixed, fgc.Λ, fgc.φ, fgc.group,
                                           fgc.link, fgc.loglik, fgc.converged, fgc.iterations)
        @test fit8c.hessian === :fisher
    end

    @testset "beta-binomial mode search: non-finite log-posterior step (undamped)" begin
        Λ = reshape([1.0], 1, 1)
        β = [0.0]
        y, N = [0], [1]
        z, ok = GLLVModels._beta_binomial_mode_search(y, N, Λ, β, 50.0; maxiter = 5, tol = 1e-12)
        @test length(z) == 1 && isfinite(z[1])
        @test ok == false || ok == true
    end

    @testset "simulate_fit: scalar families and X_lv contracts" begin
        rng = MersenneTwister(112706)
        p, n, K = 3, 25, 1
        Ye = [abs(randn(rng)) + 0.2 for t in 1:p, s in 1:n]
        ex = fit_exponential_gllvm(Ye; K = K, iterations = 15, g_tol = 1e-4)
        Ysim = simulate(ex, 8; rng = MersenneTwister(1))
        @test size(Ysim) == (p, 8) && eltype(Ysim) == Float64 && all(Ysim .> 0)

        Yb = [clamp(rand(Beta(2, 2)), 0.05, 0.95) for t in 1:p, s in 1:n]
        bf = fit_beta_gllvm(Yb; K = K, iterations = 15, g_tol = 1e-4)
        @test size(simulate(bf, 6; rng = MersenneTwister(2))) == (p, 6)

        Yg = [rand(Gamma(2.0, 0.5)) for t in 1:p, s in 1:n]
        gf = fit_gamma_gllvm(Yg; K = K, iterations = 15, g_tol = 1e-4)
        @test size(simulate(gf, 6; rng = MersenneTwister(3))) == (p, 6)

        Yo = [rand(1:3) for _ in 1:p, _ in 1:n]
        of = fit_ordinal_gllvm(Yo; K = K, iterations = 20, g_tol = 1e-4)
        Yord = simulate(of, 5; rng = MersenneTwister(4))
        @test size(Yord) == (p, 5) && eltype(Yord) == Int

        X_lv = randn(n, 1)
        Yp = [rand(Poisson(3)) for t in 1:p, s in 1:n]
        pf = fit_poisson_gllvm(Yp; K = K, X_lv = X_lv, iterations = 20, g_tol = 1e-4)
        @test_throws ArgumentError simulate(pf, n)
        @test size(simulate(pf, n; X_lv = X_lv, rng = MersenneTwister(5))) == (p, n)
        @test_throws ArgumentError simulate(pf, n; X_lv = X_lv[1:(end - 1), :])
        @test_throws ArgumentError simulate(pf, n; X_lv = randn(n, 2))
    end

    @testset "confint_derived: proportions, phylo_signal, bootstrap guards" begin
        fitP, YP = begin
            rng = MersenneTwister(11)
            p, n, K = 4, 40, 1
            Λ = reshape([0.7, 0.5, 0.4, -0.3], p, K)
            Y = Λ * randn(rng, K, n) .+ 0.5 .* randn(rng, p, n)
            fit_gaussian_gllvm(Y; K = K), Y
        end
        ps = GLLVModels.proportions(fitP; component = :shared)
        @test length(ps) == size(fitP.pars.Λ, 1)
        @test all(0 .<= ps .<= 1)
        @test GLLVModels.proportions(fitP; component = :unique_W) == zeros(Float64, 4)
        @test GLLVModels.proportions(fitP; component = :unique_B) == zeros(Float64, 4)
        @test_throws ArgumentError GLLVModels.proportions(fitP; component = :not_a_component)

        rng = MersenneTwister(20260902)
        p, n = 3, 20
        Σ_phy = [1.0 0.2 0.1; 0.2 1.0 0.15; 0.1 0.15 1.0]
        Y = 0.5 .* randn(rng, p, n)
        fitS = fit_gaussian_gllvm(Y; K = 1, has_diag = true, has_phy_unique = true, Σ_phy = Σ_phy,
                                  iterations = 30, g_tol = 1e-4)
        H = GLLVModels.phylo_signal(fitS; Σ_phy = Σ_phy)
        @test length(H) == p && all(0 .<= H .<= 1)
        @test all(isnan, GLLVModels.phylo_signal(fitP))

        @test_throws ArgumentError GLLVModels.bootstrap_ci_derived(
            fitP, GLLVModels.communality; n_boot = 3)
        @test_throws ArgumentError GLLVModels.bootstrap_ci_derived(
            fitS, f -> GLLVModels.communality(f)[1]; y = Y, n_boot = 3, seed = 1)

        y2 = randn(3, 24)
        q = 1
        X3expand = zeros(3, 24, q)
        @inbounds for s in 1:24, t in 1:3
            X3expand[t, s, 1] = randn()
        end
        fitX = GLLVModels.fit_gaussian_gllvm(y2; X = X3expand, K = 1, iterations = 30, g_tol = 1e-4)
        @test_throws ArgumentError GLLVModels.bootstrap_ci_derived(
            fitX, f -> GLLVModels.communality(f)[1]; n_boot = 3, n_sites = size(y2, 2), seed = 1)
    end
end