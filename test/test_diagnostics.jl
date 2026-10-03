using GLLVModels, Test, Random, LinearAlgebra, Statistics

# Diagnostics/compare cluster (src/diagnostics.jl) — TDD red-first fixtures.
# Each block cites the R source it ports (see the function docstrings for
# exact file/line references into
# .unlazy/core070-aghq/oracle-source/readback/R/).

@testset "diagnostics" begin

    @testset "sanity_multi — healthy Gaussian fit passes" begin
        Random.seed!(10)
        p, K, n = 5, 1, 400
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3, 0.2], p, K)
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        s = GLLVModels.sanity_multi(fit; y = y)
        @test s.pass
        @test s.loadings_finite
        @test s.pd_hessian === true
        @test s.gradient_ok
        @test s.gradient_norm < 1e-3
    end

    @testset "sanity_multi — non-finite loadings fail" begin
        Random.seed!(11)
        p, K, n = 4, 1, 100
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3], p, K)
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        badpars = merge(fit.pars, (Λ = fill(NaN, size(fit.pars.Λ)),))
        badfit = GLLVModels.GllvmFit(fit.model, badpars, fit.logLik, fit.n_iter, fit.converged,
                                 fit.optim_result, fit.cputime)
        s = GLLVModels.sanity_multi(badfit)
        @test !s.pass
        @test !s.loadings_finite
    end

    @testset "check_auto_residual — coherent for logit ordinal, flags probit" begin
        Random.seed!(12)
        p, K, n = 4, 1, 300
        Λ_true = reshape([1.0, 0.8, -0.6, 0.5], p, K)
        η = Λ_true * randn(K, n)
        τ = [-1.0, 0.0, 1.0]
        Y = Matrix{Int}(undef, p, n)
        for t in 1:p, s in 1:n
            Y[t, s] = 1 + sum(η[t, s] .> τ)
        end
        fit_logit = fit_ordinal_gllvm(Y; K = K, link = GLLVModels.LogitLink())
        r_logit = GLLVModels.check_auto_residual(fit_logit)
        @test r_logit.coherent
        @test !r_logit.ordinal_probit

        fit_probit = fit_ordinal_gllvm(Y; K = K, link = GLLVModels.ProbitLink())
        r_probit = GLLVModels.check_auto_residual(fit_probit)
        @test !r_probit.coherent
        @test r_probit.ordinal_probit
        @test !isempty(r_probit.messages)
    end

    @testset "gllvmTMB_diagnose / check_gllvmTMB — boundary flags on a degenerate fit" begin
        Random.seed!(13)
        p, K, n = 5, 1, 400
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3, 0.2], p, K)
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        d = GLLVModels.gllvmTMB_diagnose(fit; y = y)
        @test d.pass
        @test isempty(d.boundary_flags)

        # Force a near-zero σ_eps to trigger the variance boundary flag.
        tiny_pars = merge(fit.pars, (σ_eps = 1e-8,))
        tiny_fit = GLLVModels.GllvmFit(fit.model, tiny_pars, fit.logLik, fit.n_iter, fit.converged,
                                   fit.optim_result, fit.cputime)
        d2 = GLLVModels.gllvmTMB_diagnose(tiny_fit)
        @test !d2.pass
        @test any(f -> startswith(f, "variance_near_zero"), d2.boundary_flags)

        c = GLLVModels.check_gllvmTMB(fit; y = y)
        @test c.pass
        @test c.residual.coherent
    end

    @testset "gllvmTMB_diagnose — var_tol applied on a consistent (variance) scale" begin
        # Regression for the post-M2 review finding: var_tol was compared
        # directly against σ_eps/σ_phy (SD-scale parameters) AND σ²_B/σ²_W
        # (variance-scale parameters) with the SAME threshold, mixing scales
        # by orders of magnitude. Convention adopted here: everything is
        # compared on the VARIANCE scale — SD parameters are squared first.
        # σ_eps = 0.005 has SD = 0.005 (> var_tol on the old SD-scale
        # comparison, so it would NOT have flagged) but variance = 2.5e-5,
        # which IS within var_tol = 1e-4 of the zero boundary.
        Random.seed!(152)
        p, K, n = 4, 1, 200
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3], p, K)
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        fit0 = fit_gaussian_gllvm(y; K = K)
        pars = merge(fit0.pars, (σ_eps = 0.005,))
        fit = GLLVModels.GllvmFit(fit0.model, pars, fit0.logLik, fit0.n_iter, fit0.converged,
                              fit0.optim_result, fit0.cputime)
        d = GLLVModels.gllvmTMB_diagnose(fit; var_tol = 1e-4)
        @test any(f -> startswith(f, "variance_near_zero:σ_eps"), d.boundary_flags)
    end

    @testset "gllvmTMB_diagnose — implied Σ uses ALL tiers (sigma_y_site), not just Λ_B + σ_eps" begin
        # Regression for the post-M2 review finding: the old implied-Σ was
        # Λ*Λ' + σ_eps²*I, ignoring the W-tier (Λ_W) contribution entirely.
        # A strong shared Λ_B factor with a tiny σ_eps but a LARGE W-tier
        # variance is a genuinely low-correlation fit once the W tier is
        # accounted for, but the naive Λ_B-only Σ reports it as near-boundary
        # correlated. Since #135, sigma_y_site adds the full Λ_W Λ_W' block
        # (W-tier scores shared across traits, as in gllvmTMB), so Λ_W is
        # chosen to pull traits 1 and 2 apart (opposite signs) and leave
        # trait 3 alone: all-tier |r| <= 0.8, Λ_B-only r ≈ 1.
        Random.seed!(151)
        p, K, n = 3, 1, 100
        y = 0.5 * randn(p, n)
        fit0 = fit_gaussian_gllvm(y; K = K, K_W = 1)
        pars = merge(fit0.pars, (Λ = fill(1.0, p, 1), Λ_W = reshape([3.0, -3.0, 0.0], p, 1),
                                 σ_eps = 0.01))
        fit = GLLVModels.GllvmFit(fit0.model, pars, fit0.logLik, fit0.n_iter, fit0.converged,
                              fit0.optim_result, fit0.cputime)

        Σfull = GLLVModels.sigma_y_site(fit)
        d = sqrt.(diag(Σfull))
        Rfull = Σfull ./ (d * d')
        @test maximum(abs, Rfull[1, 2]) < 0.995  # the true (all-tier) correlation is small

        diagres = GLLVModels.gllvmTMB_diagnose(fit)
        @test !any(f -> startswith(f, "correlation_near_boundary"), diagres.boundary_flags)
    end

    @testset "fit_diagnostic_table — one row per named check, matching column lengths" begin
        Random.seed!(14)
        p, K, n = 4, 1, 200
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3], p, K)
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        tbl = GLLVModels.fit_diagnostic_table(fit; y = y)
        @test length(tbl.check) == length(tbl.status) == length(tbl.message)
        @test "converged" in tbl.check
        @test "pd_hessian" in tbl.check
        @test "auto_residual" in tbl.check
        @test all(s -> s in (:pass, :fail, :unavailable), tbl.status)
    end

    @testset "diagnose_kernel_separability — single-tier fit reports missing" begin
        Random.seed!(15)
        p, K, n = 4, 1, 150
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3], p, K)
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        r = GLLVModels.diagnose_kernel_separability(fit)
        @test r.separable === missing
    end

    @testset "diagnose_kernel_separability — identical (non-orthonormal) column spaces are NOT separable" begin
        # Regression for the post-M2 review finding: svd(Λ_B'Λ_W) on raw
        # (non-orthonormal) loadings is not cos(principal angle). Λ_B == Λ_W
        # here (identical 1-D column space) but scaled to 0.2 so the naive
        # M = Λ_B'Λ_W = 0.16·(sum of squares) product is well below 1 — the
        # buggy computation reported this as "separable" (angle ≈ 1.41 rad)
        # when the true principal angle between identical spaces is 0.
        Random.seed!(150)
        p, K, n = 4, 1, 100
        y = 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K, K_W = 1)
        badpars = merge(fit.pars, (Λ = fill(0.2, p, 1), Λ_W = fill(0.2, p, 1)))
        badfit = GLLVModels.GllvmFit(fit.model, badpars, fit.logLik, fit.n_iter, fit.converged,
                                 fit.optim_result, fit.cputime)
        r = GLLVModels.diagnose_kernel_separability(badfit)
        @test r.min_principal_angle ≈ 0.0 atol = 1e-8
        @test r.separable === false
    end

    @testset "compare_fits_Sigma_table / compare_loadings — identical fits agree exactly" begin
        Random.seed!(16)
        p, K, n = 5, 2, 300
        Λ_true = randn(p, K)
        y = Λ_true * randn(K, n) + 0.4 * randn(p, n)
        fit1 = fit_gaussian_gllvm(y; K = K)
        fit2 = fit1

        sc = GLLVModels.compare_fits_Sigma_table(fit1, fit2)
        @test sc.frobenius_norm ≈ 0.0 atol = 1e-10
        @test sc.max_abs_diff ≈ 0.0 atol = 1e-10

        cl = GLLVModels.compare_loadings(fit1, fit2)
        @test cl.frobenius_norm_LLt ≈ 0.0 atol = 1e-10
        @test all(a -> isapprox(a, 0.0; atol = 1e-6), cl.principal_angles)
    end

    @testset "compare_loadings — proper principal angles for a small-magnitude identical subspace" begin
        # Same non-orthonormal-basis trap as the diagnose_kernel_separability
        # regression above: Λ1 == Λ2 = 0.2·ones(p,1) (identical 1-D column
        # space) but small enough in magnitude that the raw dot product
        # ‖Λ1‖‖Λ2‖ < 1 — the buggy `svd(Λ1'Λ2).S` computation reports this as
        # a nonzero principal angle even though the true angle is 0.
        Random.seed!(151)
        p, K, n = 4, 1, 100
        y = 0.5 * randn(p, n)
        fit0 = fit_gaussian_gllvm(y; K = K)
        Λ = fill(0.2, p, 1)
        pars = merge(fit0.pars, (Λ = Λ,))
        fit1 = GLLVModels.GllvmFit(fit0.model, pars, fit0.logLik, fit0.n_iter, fit0.converged,
                               fit0.optim_result, fit0.cputime)
        fit2 = fit1
        cl = GLLVModels.compare_loadings(fit1, fit2)
        @test all(a -> isapprox(a, 0.0; atol = 1e-8), cl.principal_angles)
    end

    @testset "compare_fits_Sigma_table / compare_loadings — a genuinely different fit disagrees" begin
        Random.seed!(17)
        p, K, n = 5, 1, 300
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3, 0.2], p, K)
        y1 = Λ_true * randn(K, n) + 0.3 * randn(p, n)
        y2 = 3.0 .* Λ_true * randn(K, n) .+ 2.0 * randn(p, n)
        fit1 = fit_gaussian_gllvm(y1; K = K)
        fit2 = fit_gaussian_gllvm(y2; K = K)

        sc = GLLVModels.compare_fits_Sigma_table(fit1, fit2)
        @test sc.frobenius_norm > 1.0

        cl = GLLVModels.compare_loadings(fit1, fit2)
        @test cl.frobenius_norm_LLt > 0.1

        @test_throws ArgumentError GLLVModels.compare_fits_Sigma_table(fit1, fit_gaussian_gllvm(y1[1:3, :]; K = K))
    end

    @testset "compare_fits_dep_vs_two_psi / compare_fits_indep_vs_two_psi — bridge shape and self-comparison" begin
        Random.seed!(18)
        p, K, n = 4, 1, 250
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3], p, K)
        y = Λ_true * randn(K, n) + 0.4 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        r = GLLVModels.compare_fits_dep_vs_two_psi(fit, fit, n)
        @test r.aic_delta ≈ 0.0 atol = 1e-8
        @test r.bic_delta ≈ 0.0 atol = 1e-8
        @test r.loglik_dep ≈ r.loglik_alt

        r2 = GLLVModels.compare_fits_indep_vs_two_psi(fit, fit, n)
        @test r2.aic_delta ≈ 0.0 atol = 1e-8
    end

    @testset "deprecated names forward to their renamed target (maintainer decision round2-3 #5)" begin
        Random.seed!(18)
        p, K, n = 4, 1, 250
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3], p, K)
        y = Λ_true * randn(K, n) + 0.4 * randn(p, n)
        fit1 = fit_gaussian_gllvm(y; K = K)
        fit2 = fit1

        tbl_new = GLLVModels.fit_diagnostic_table(fit1; y = y)
        tbl_old = @test_deprecated GLLVModels.diagnostic_table(fit1; y = y)
        @test tbl_old == tbl_new

        sc_new = GLLVModels.compare_fits_Sigma_table(fit1, fit2)
        sc_old = @test_deprecated GLLVModels.compare_Sigma_table(fit1, fit2)
        @test sc_old == sc_new

        r_new = GLLVModels.compare_fits_dep_vs_two_psi(fit1, fit2, n)
        r_old = @test_deprecated GLLVModels.compare_dep_vs_two_psi(fit1, fit2, n)
        @test r_old == r_new

        r2_new = GLLVModels.compare_fits_indep_vs_two_psi(fit1, fit2, n)
        r2_old = @test_deprecated GLLVModels.compare_indep_vs_two_psi(fit1, fit2, n)
        @test r2_old == r2_new
    end

    @testset "predictive_check — a well-fitting Poisson model does not flag" begin
        Random.seed!(19)
        rng = MersenneTwister(19)
        p, K, n = 4, 1, 300
        Λ_true = reshape([0.6, 0.5, -0.4, 0.3], p, K)
        β_true = fill(0.5, p)
        Z = randn(rng, K, n)
        η = β_true .+ Λ_true * Z
        Y = [rand(rng, GLLVModels.Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
        fit = fit_poisson_gllvm(Y; K = K, iterations = 100)

        pc = GLLVModels.predictive_check(fit, Y; nsim = 100, rng = MersenneTwister(1))
        @test length(pc.stat) == length(pc.trait) == length(pc.observed) == length(pc.p_value)
        @test all(0.0 .<= pc.p_value .<= 1.0)
        # A correctly specified model should rarely flag every trait/stat at once.
        @test !all(pc.p_value .< 0.01)
    end

    @testset "predictive_check — a well-fitting Gaussian model does not flag" begin
        Random.seed!(20)
        p, K, n = 4, 1, 300
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3], p, K)
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        pc = GLLVModels.predictive_check(fit, y; nsim = 100, rng = MersenneTwister(2))
        @test length(pc.stat) == length(pc.trait) == length(pc.observed) == length(pc.p_value)
        @test all(0.0 .<= pc.p_value .<= 1.0)
        @test !all(pc.p_value .< 0.01)
    end

    @testset "predictive_check — a fit type with no simulate() method documents the gap" begin
        Random.seed!(24)
        p, K, n = 4, 1, 200
        Λ_true = reshape([1.0, 0.8, -0.6, 0.5], p, K)
        η = Λ_true * randn(K, n)
        τ = [-1.0, 0.0, 1.0]
        Y = Matrix{Int}(undef, p, n)
        for t in 1:p, s in 1:n
            Y[t, s] = 1 + sum(η[t, s] .> τ)
        end
        fit = fit_ordinal_gllvm_pertrait(Y; K = K)
        @test_throws ArgumentError GLLVModels.predictive_check(fit, Y)
    end

    @testset "confint_inspect — Wald and profile roughly agree on a clean fixture" begin
        Random.seed!(21)
        p, K, n = 3, 1, 400
        Λ_true = reshape([0.7, 0.5, 0.4], p, K)
        σ_true = 0.5
        y = Λ_true * randn(K, n) + σ_true * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        ci = GLLVModels.confint_inspect(fit, y; parm = "sigma_eps")
        @test ci.term == ["sigma_eps"]
        @test ci.wald_lower[1] < σ_true < ci.wald_upper[1]
        @test isfinite(ci.profile_lower[1])
        @test !ci.disagree[1]
    end

    @testset "gllvmTMB_check_consistency — correctly specified fit does not flag score non-centring" begin
        Random.seed!(22)
        p, K, n = 4, 1, 300
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3], p, K)
        σ_true = 0.5
        y = Λ_true * randn(K, n) + σ_true * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        cc = GLLVModels.gllvmTMB_check_consistency(fit, y; n_sim = 60, seed = 42)
        @test cc.n_sim == 60
        @test length(cc.marginal_bias) == length(fit.pars.θ_packed)
        # Omnibus Hotelling test should not report the score significantly
        # off-centre for a correctly specified model (per-parameter flags can
        # still fire occasionally from multiple-comparison noise, matching R's
        # own per-parameter bias flag, which the omnibus p-value guards against).
        @test !isnan(cc.marginal_p_value)
        @test cc.marginal_p_value > 0.01
    end

    @testset "gllvmTMB_check_consistency — rejects unsupported structure" begin
        Random.seed!(23)
        p, K, n = 4, 1, 100
        y = 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K, K_W = 1, has_diag = true)
        @test_throws ArgumentError GLLVModels.gllvmTMB_check_consistency(fit, y)

        # A unique phylogenetic effect (has_phy_unique, K_phy = 0) is not
        # re-simulated either, so the check must refuse it rather than score
        # phylo-free replicates against the phylo likelihood.
        phy = GLLVModels.random_balanced_tree(p; branch_length = 0.5)
        Σ_phy = Matrix(Symmetric(GLLVModels.sigma_phy_dense(phy; σ²_phy = 1.0)))
        fit_phy = fit_gaussian_gllvm(y; K = K, has_phy_unique = true, Σ_phy = Σ_phy)
        @test fit_phy.model.K_phy == 0 && fit_phy.model.has_phy_unique
        @test_throws ArgumentError GLLVModels.gllvmTMB_check_consistency(fit_phy, y;
                                                                         Σ_phy = Σ_phy)
    end

    # ---------------------------------------------------------------------
    # src/twolevel.jl regression (no dedicated owned twolevel test file for
    # this repair pass; kept here since test_diagnostics.jl is the owned
    # test file closest in scope).
    # ---------------------------------------------------------------------

    @testset "repeatability_bootstrap_ci — boundary Σ_B (PSD, not PD) does not abort the call" begin
        # Regression for the post-M2 review finding: cholesky(Symmetric(Σ_B)).L
        # was called with no PosDefException guard, but a boundary fit's
        # Σ_B = Λ_B Λ_B' + diag(σ²_B) is only PSD (rank-deficient at σ²_B == 0
        # with K_B < p), so plain Cholesky throws and aborts the whole call —
        # inconsistent with the NaN-row convention used everywhere else in
        # this file (e.g. repeatability_wald_ci's `failed` row).
        p, K_B, K_W = 3, 1, 1
        Λ_B = reshape([1.0, 0.0, 0.0], p, K_B)   # rank-1, boundary σ²_B == 0
        σ²_B = zeros(p)
        Λ_W = reshape([0.5, 0.4, 0.3], p, K_W)
        σ²_W = fill(0.2, p)
        Σ_B = Λ_B * Λ_B' .+ Diagonal(σ²_B)
        Σ_W = Λ_W * Λ_W' .+ Diagonal(σ²_W)
        @test_throws LinearAlgebra.PosDefException cholesky(Symmetric(Σ_B))

        nindiv, nobs = 20, 3
        individual = repeat(1:nindiv, inner = nobs)
        fit = GLLVModels.TwoLevelFit(Λ_B, σ²_B, Λ_W, σ²_W, Σ_B, Σ_W, nindiv, -100.0, true, 5)

        out = GLLVModels.repeatability_bootstrap_ci(fit, individual; nsim = 5, seed = 1)
        @test length(out) == p
        @test all(r -> r.estimate isa Real && isfinite(r.estimate), out)
        @test all(r -> r.method === :bootstrap, out)
    end
end
