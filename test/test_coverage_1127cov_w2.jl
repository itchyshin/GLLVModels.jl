# Second-pass coverage lane w2 (27 Nov): exercise unwired phylo xLV validation /
# profile helpers and postfit StatsAPI methods with small, fast fits (<60 s total).
using GLLVModels, Test, Random, LinearAlgebra, Statistics, Distributions

@testset "coverage 1127cov w2" begin

    @testset "postfit StatsAPI — coef / dof / loglikelihood / nobs cluster" begin
        Random.seed!(1127001)
        y = rand(1:3, 48)
        Ymn = reshape(Int.(y), 1, length(y))
        fit_mn = fit_multinomial_gllvm(Ymn)
        @test fit_mn isa MultinomialFit
        @test dof(fit_mn) > 0
        @test isfinite(loglikelihood(fit_mn))
        @test loglikelihood(fit_mn) ≈ fit_mn.loglik
        @test nobs(fit_mn, Ymn) == length(Ymn)
        yvec = vec(Ymn)
        @test nobs(fit_mn, yvec) == length(yvec)

        p, K, n, q = 4, 1, 30, 1
        X = reshape(collect(range(-1.0, 1.0; length = n)), n, q)
        Yp = [rand(Poisson(exp(0.2 + 0.15 * X[s, 1]))) for _ in 1:p, s in 1:n]
        fit_rrr = fit_rrr_gllvm(Yp; family = Distributions.Poisson(), X = X, K = K, iterations = 80)
        @test coef(fit_rrr) ≈ fit_rrr.B
        @test dof(fit_rrr) > length(coef(fit_rrr))

        p, K, n, q, r = 4, 1, 25, 2, 2
        Xenv = randn(n, q)
        TR = randn(p, r)
        C0 = 0.1 .* randn(q, r)
        Λ0 = 0.2 .* randn(p, K)
        O = GLLVModels._build_offset_fourthcorner(Xenv, TR, C0)
        Yfc = [rand(Poisson(exp(0.1 + O[t, s] + dot(Λ0[t, :], randn(K))))) for t in 1:p, s in 1:n]
        fit_fc = fit_fourthcorner_gllvm(Yfc; family = Distributions.Poisson(), Xenv = Xenv, TR = TR, K = K,
                                        iterations = 80)
        @test coef(fit_fc) == fit_fc.C
        @test isfinite(deviance(fit_fc))

        Yz = [rand(Poisson(2.0)) for _ in 1:3, _ in 1:20]
        fit_zip = fit_zip_gllvm(Yz; K = 1, iterations = 80)
        @test dof(fit_zip) >= size(fit_zip.Λc, 1)
        @test GLLVModels.rotation(fit_zip) ≈
            GLLVModels._svd_rotation(fit_zip.Λc) atol = 1e-12

        fit_po = fit_poisson_gllvm(Yp; K = 1, iterations = 60)
        @test_throws ArgumentError nobs(fit_po)
    end

    @testset "phylo Gamma xLV — validation, profile_indices, constrained refit" begin
        phy = augmented_phy("(A:0.1,B:0.1);")
        p = phy.n_leaves
        n = 6
        K = 1
        X_lv = reshape(collect(range(-0.4, 0.4; length = n)), n, 1)
        Y = fill(2.5, p, n)
        theta0 = GLLVModels._phylo_gamma_xlv_theta0(Y, K, X_lv)
        @test length(theta0) > 0

        @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_theta0(Y, 0, X_lv)
        @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_theta0(Y, K, zeros(n, 0))
        Ybad = copy(Y); Ybad[1, 1] = 0.0
        @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_validate(
            Ybad, fill(0.0, p), ones(p, K), zeros(1, K), 2.0, phy, X_lv, LogLink())

        nb = p * size(X_lv, 2)
        @test GLLVModels._phylo_gamma_xlv_profile_indices([1, 2], nb) == [1, 2]
        @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_profile_indices(Int[], nb)
        @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_profile_indices([1, nb + 1], nb)

        nll_const = theta -> 0.5 * sum(abs2, theta)
        refit = GLLVModels._phylo_gamma_xlv_constrained_refit(
            nll_const, theta0, p, K, 1, 1, 0.0;
            profile_iterations = 30, penalty_weights = (1e1, 1e2))
        @test isfinite(refit.nll)
        @test isfinite(refit.effect)

        _, Lambda_hat, _, alpha_shape, _ =
            GLLVModels._phylo_gamma_xlv_unpack_packed(theta0, p, K, 1)
        fit_stub = (Lambda = Lambda_hat, theta_packed = theta0)
        eta_hat = GLLVModels._phylo_gamma_xlv_effects_from_packed(theta0, p, K, 1)
        prof = GLLVModels._phylo_gamma_xlv_profile_eta_realized(
            fit_stub, Y, phy, X_lv, [1], eta_hat;
            level = 0.90, profile_iterations = 25, profile_endpoints = false,
            newton_maxiter = 40, newton_tol = 1e-8)
        @test prof.method == :profile_eta_realized
        @test prof.endpoint_status == [:not_requested]
        @test length(prof.term) == 1
        @test isfinite(prof.estimate[1])
        @test prof.alpha_shape ≈ alpha_shape
    end

    @testset "phylo Ordinal xLV — theta0 and profile validation" begin
        phy = augmented_phy("(A:0.1,B:0.1);")
        p = phy.n_leaves
        n = 8
        K = 1
        X_lv = reshape(collect(range(-0.5, 0.5; length = n)), n, 1)
        Y = fill(2, p, n)
        theta0, C = GLLVModels._phylo_ordinal_xlv_theta0(Y, K, X_lv)
        @test C >= 2
        @test length(theta0) > 0

        @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_theta0(Y, 0, X_lv)
        Ybad = copy(Y); Ybad[2, 3] = 0
        @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_theta0(Ybad, K, X_lv)

        nb = p * size(X_lv, 2)
        @test GLLVModels._phylo_ordinal_xlv_profile_indices([1], nb) == [1]
        @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_profile_indices([1, 1], nb)

        Lambda_hat, _, tau, _ = GLLVModels._phylo_ordinal_xlv_unpack_packed(theta0, p, K, 1, C)
        fit_stub = (Lambda = Lambda_hat, theta_packed = theta0, C = C, tau = tau)
        eta_hat = GLLVModels._phylo_ordinal_xlv_effects_from_packed(theta0, p, K, 1, C)
        prof = GLLVModels._phylo_ordinal_xlv_profile_eta_realized(
            fit_stub, Y, phy, X_lv, [1], eta_hat;
            level = 0.90, profile_iterations = 25, profile_endpoints = false,
            newton_maxiter = 40, newton_tol = 1e-8)
        @test prof.method == :profile_eta_realized
        @test prof.endpoint_status == [:not_requested]
        @test isfinite(prof.estimate[1])
    end
end