using GLLVModels, Test, Random, LinearAlgebra, Statistics, Distributions
using GLLVModels: StatsAPI, coef, vcov, nobs, dof, loglikelihood, aic, bic, stderror, coeftable

@testset "StatsAPI extractors" begin
    Random.seed!(42)
    p, K, n = 4, 1, 60
    # nobs(fit, Y) is R's p·n observed-cell count, not the site count n
    # (docs/dev-log/decisions/2026-09-01-maintainer-decisions-round1.md #1).
    pn = p * n

    # 1. Gaussian GllvmFit
    Λt = 0.6 .* randn(p, K)
    y_gauss = Λt * randn(K, n) .+ 0.4 .* randn(p, n)
    fit_gauss = fit_gaussian_gllvm(y_gauss; K = K)

    @testset "GllvmFit (Gaussian)" begin
        @test StatsAPI.dof(fit_gauss) == GLLVModels._nparams(fit_gauss)
        @test StatsAPI.loglikelihood(fit_gauss) ≈ fit_gauss.logLik
        @test StatsAPI.aic(fit_gauss) ≈ 2 * StatsAPI.dof(fit_gauss) - 2 * StatsAPI.loglikelihood(fit_gauss)
        @test StatsAPI.bic(fit_gauss, n) ≈ StatsAPI.dof(fit_gauss) * log(n) - 2 * StatsAPI.loglikelihood(fit_gauss)
        @test StatsAPI.bic(fit_gauss, y_gauss) ≈ StatsAPI.bic(fit_gauss, pn)
        @test StatsAPI.nobs(fit_gauss, y_gauss) == pn
        @test isempty(StatsAPI.coef(fit_gauss)) # intercept-only / no fixed effects β

        V = StatsAPI.vcov(fit_gauss, y_gauss)
        @test size(V, 1) == size(V, 2)
        @test isposdef(V) || tr(V) > 0
        se = StatsAPI.stderror(fit_gauss, y_gauss)
        @test length(se) == size(V, 1)
        @test all(≥(0), se)

        ct = StatsAPI.coeftable(fit_gauss, y_gauss)
        @test ct isa GllvmCoefTable

        s = summary(fit_gauss)
        @test occursin("Gaussian GLLVM fit", s)
        @test occursin("logLik", s)
    end

    @testset "nobs/bic masked-cell counting (p·n convention)" begin
        # docs/dev-log/decisions/2026-09-01-maintainer-decisions-round1.md #1:
        # nobs(fit, Y) counts observed p×n response cells, not sites — with a
        # `missing`-carrying Y or an explicit `mask`, masked-out cells must be
        # excluded from both nobs and bic.
        mask = trues(p, n)
        mask[1, 3] = false
        mask[2, 10] = false
        n_masked = count(mask)
        @test n_masked == pn - 2

        @test StatsAPI.nobs(fit_gauss, y_gauss; mask = mask) == n_masked
        @test StatsAPI.bic(fit_gauss, y_gauss; mask = mask) ≈
              StatsAPI.dof(fit_gauss) * log(n_masked) - 2 * StatsAPI.loglikelihood(fit_gauss)
        @test !isapprox(StatsAPI.bic(fit_gauss, y_gauss; mask = mask),
                        StatsAPI.bic(fit_gauss, y_gauss))

        y_missing = Matrix{Union{Float64, Missing}}(y_gauss)
        y_missing[1, 3] = missing
        y_missing[2, 10] = missing
        @test StatsAPI.nobs(fit_gauss, y_missing) == n_masked
        @test StatsAPI.bic(fit_gauss, y_missing) ≈ StatsAPI.bic(fit_gauss, y_gauss; mask = mask)
    end

    # 2. BinomialFit
    η_bin = 0.2 .+ Λt * randn(K, n)
    μ_bin = inv.(1 .+ exp.(-η_bin))
    Y_bin = Int.(rand(p, n) .< μ_bin)
    fit_bin = fit_binomial_gllvm(Y_bin; K = K)

    @testset "BinomialFit" begin
        @test StatsAPI.dof(fit_bin) == GLLVModels._nparams(fit_bin)
        @test StatsAPI.loglikelihood(fit_bin) ≈ fit_bin.loglik
        @test StatsAPI.aic(fit_bin) ≈ 2 * StatsAPI.dof(fit_bin) - 2 * fit_bin.loglik
        @test StatsAPI.bic(fit_bin, n) ≈ StatsAPI.dof(fit_bin) * log(n) - 2 * fit_bin.loglik
        @test StatsAPI.nobs(fit_bin, Y_bin) == pn
        @test length(StatsAPI.coef(fit_bin)) == p # β intercepts

        s = summary(fit_bin)
        @test occursin("Binomial GLLVM fit", s)
    end

    # 3. PoissonFit
    η_pois = 0.5 .+ Λt * randn(K, n)
    Y_pois = [rand(Poisson(exp(η_pois[t, s]))) for t in 1:p, s in 1:n]
    fit_pois = fit_poisson_gllvm(Y_pois; K = K)

    @testset "PoissonFit" begin
        @test StatsAPI.dof(fit_pois) == GLLVModels._nparams(fit_pois)
        @test StatsAPI.loglikelihood(fit_pois) ≈ fit_pois.loglik
        @test StatsAPI.aic(fit_pois) ≈ 2 * StatsAPI.dof(fit_pois) - 2 * fit_pois.loglik
        @test StatsAPI.bic(fit_pois, n) ≈ StatsAPI.dof(fit_pois) * log(n) - 2 * fit_pois.loglik
        @test StatsAPI.nobs(fit_pois, Y_pois) == pn
        @test length(StatsAPI.coef(fit_pois)) == p

        s = summary(fit_pois)
        @test occursin("Poisson GLLVM fit", s)
    end

    # 4. NBFit
    fit_nb = fit_nb_gllvm(Y_pois; K = K)
    @testset "NBFit" begin
        @test StatsAPI.dof(fit_nb) == GLLVModels._nparams(fit_nb)
        @test StatsAPI.loglikelihood(fit_nb) ≈ fit_nb.loglik
        @test StatsAPI.aic(fit_nb) ≈ 2 * StatsAPI.dof(fit_nb) - 2 * fit_nb.loglik
        @test StatsAPI.nobs(fit_nb, Y_pois) == pn

        s = summary(fit_nb)
        @test occursin("NegativeBinomial", s)
    end

    # 5. BetaFit
    Y_beta = clamp.(inv.(1 .+ exp.(-η_bin)), 0.05, 0.95)
    fit_beta = fit_beta_gllvm(Y_beta; K = K)
    @testset "BetaFit" begin
        @test StatsAPI.dof(fit_beta) == GLLVModels._nparams(fit_beta)
        @test StatsAPI.loglikelihood(fit_beta) ≈ fit_beta.loglik
        @test StatsAPI.aic(fit_beta) ≈ 2 * StatsAPI.dof(fit_beta) - 2 * fit_beta.loglik
        @test StatsAPI.nobs(fit_beta, Y_beta) == pn

        s = summary(fit_beta)
        @test occursin("Beta GLLVM fit", s)
    end

    # 6. GammaFit
    Y_gamma = exp.(η_pois) .+ 0.1
    fit_gamma = fit_gamma_gllvm(Y_gamma; K = K)
    @testset "GammaFit" begin
        @test StatsAPI.dof(fit_gamma) == GLLVModels._nparams(fit_gamma)
        @test StatsAPI.loglikelihood(fit_gamma) ≈ fit_gamma.loglik
        @test StatsAPI.aic(fit_gamma) ≈ 2 * StatsAPI.dof(fit_gamma) - 2 * fit_gamma.loglik
        @test StatsAPI.nobs(fit_gamma, Y_gamma) == pn

        s = summary(fit_gamma)
        @test occursin("Gamma GLLVM fit", s)
    end

    # 7. Covariates fit (GllvmCovFit)
    q = 2
    X_3d = randn(p, n, q)
    fit_cov = fit_gllvm_cov(Y_pois; family = Poisson(), X = X_3d, K = K)
    @testset "GllvmCovFit" begin
        @test StatsAPI.dof(fit_cov) == GLLVModels._nparams(fit_cov)
        @test StatsAPI.loglikelihood(fit_cov) ≈ fit_cov.loglik
        @test length(StatsAPI.coef(fit_cov)) == length(fit_cov.γ)
        @test StatsAPI.nobs(fit_cov, Y_pois) == pn
        s = summary(fit_cov)
        @test occursin("GLLVM Covariates fit", s)
    end

    # 8. OrdinalFit
    C = 3
    τ = [-0.5, 0.5]
    Y_ord = Matrix{Int}(undef, p, n)
    for s in 1:n, t in 1:p
        pr = [GLLVModels._ord_prob(c, η_bin[t, s], τ) for c in 1:C]
        Y_ord[t, s] = rand(Categorical(pr))
    end
    fit_ord = fit_ordinal_gllvm(Y_ord; K = K)
    @testset "OrdinalFit" begin
        @test StatsAPI.dof(fit_ord) == GLLVModels._nparams(fit_ord)
        @test StatsAPI.loglikelihood(fit_ord) ≈ fit_ord.loglik
        @test StatsAPI.aic(fit_ord) ≈ 2 * StatsAPI.dof(fit_ord) - 2 * fit_ord.loglik
        @test StatsAPI.nobs(fit_ord, Y_ord) == pn
        @test length(StatsAPI.coef(fit_ord)) == length(fit_ord.τ)
        s = summary(fit_ord)
        @test occursin("Ordinal GLLVM fit", s)
    end

    # 9. TweedieFit
    Y_tw = [rand() < 0.3 ? 0.0 : rand(Gamma(2.0, 1.0)) for _ in 1:p, _ in 1:n]
    fit_tw = fit_tweedie_gllvm(Y_tw; K = K)
    @testset "TweedieFit" begin
        @test StatsAPI.dof(fit_tw) == GLLVModels._nparams(fit_tw)
        @test StatsAPI.loglikelihood(fit_tw) ≈ fit_tw.loglik
        @test StatsAPI.aic(fit_tw) ≈ 2 * StatsAPI.dof(fit_tw) - 2 * fit_tw.loglik
        @test StatsAPI.nobs(fit_tw, Y_tw) == pn
        s = summary(fit_tw)
        @test occursin("Tweedie GLLVM fit", s)
    end

    @testset "IC accessors warn when not converged (#1465)" begin
        Random.seed!(7)
        Y_nc = rand(Poisson(2), 4, 10)
        fit_nc = fit_poisson_gllvm(Y_nc; K = 2, iterations = 2)
        @test !fit_nc.converged
        ll = fit_nc.loglik
        ll_out, aic_out, bic_out = @test_logs (:warn, r"converged == false") begin
            loglikelihood(fit_nc)
            loglikelihood(fit_nc)
            a = aic(fit_nc)
            b = bic(fit_nc, Y_nc)
            (loglikelihood(fit_nc), a, b)
        end
        @test isfinite(ll_out) && isfinite(aic_out) && isfinite(bic_out)
        @test ll_out ≈ ll
    end

    # 10. Re-exported symbols check
    @testset "Re-exports in GLLVModels" begin
        @test isdefined(GLLVModels, :coef)
        @test isdefined(GLLVModels, :vcov)
        @test isdefined(GLLVModels, :nobs)
        @test isdefined(GLLVModels, :dof)
        @test isdefined(GLLVModels, :loglikelihood)
        @test isdefined(GLLVModels, :aic)
        @test isdefined(GLLVModels, :bic)
        @test isdefined(GLLVModels, :stderror)
        @test isdefined(GLLVModels, :coeftable)
    end
end
