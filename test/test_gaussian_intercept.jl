# fit_gllvm(Y; family = Normal(), K) with no X estimates per-trait intercepts,
# as every other family on this entry point does and as gllvmTMB's
# `value ~ 0 + trait + latent(...)` does. Before the fix it fitted a zero mean,
# so shifting Y moved logLik (2026-09-27 lane auto-d cross-check).
using Test, GLLVModels, Random, Distributions, Statistics, LinearAlgebra

@testset "Gaussian no-X fit estimates trait intercepts" begin
    Random.seed!(3)
    p, n = 6, 100
    Y0 = 0.8 .* randn(p, 1) * randn(1, n) .+ randn(p, n)
    shift = collect(1.0:p) .+ 5.0
    Y1 = Y0 .+ shift

    f0 = fit_gllvm(Y0; family = Normal(), K = 1)
    f1 = fit_gllvm(Y1; family = Normal(), K = 1)

    @testset "shift equivariance" begin
        @test length(f0.pars.β) == p
        @test length(f1.pars.β) == p
        @test f0.converged && f1.converged
        @test isapprox(f0.logLik, f1.logLik; atol = 1e-6)
        @test isapprox(f1.pars.β, f0.pars.β .+ shift; atol = 1e-4)
        @test isapprox(f0.pars.σ_eps, f1.pars.σ_eps; atol = 1e-5)
        @test isapprox(abs.(f0.pars.Λ), abs.(f1.pars.Λ); atol = 1e-4)
    end

    @testset "complete-data ML intercepts are the trait means" begin
        # Every site shares the design I_p, so the GLS mean is the sample mean
        # for any covariance.
        @test isapprox(f1.pars.β, vec(mean(Y1; dims = 2)); atol = 1e-4)
    end

    @testset "parameter count includes the p intercepts" begin
        f_zero = fit_gaussian_gllvm(Y0; K = 1)
        @test GLLVModels._nparams(f1) == GLLVModels._nparams(f_zero) + p
    end

    @testset "post-fit helpers apply the intercepts without X" begin
        z0 = getLV(f0, Y0; rotate = false)
        z1 = getLV(f1, Y1; rotate = false)
        @test isapprox(abs.(z0), abs.(z1); atol = 1e-3)
        @test isapprox(predict(f1, Y1) .- shift, predict(f0, Y0); atol = 1e-3)
        @test isapprox(residuals(f1, Y1), residuals(f0, Y0); atol = 1e-3)
        @test abs(mean(residuals(f1, Y1))) < 0.05
        ysim = simulate(f1, n; rng = MersenneTwister(1))
        @test size(ysim) == (p, n)
        @test isapprox(vec(mean(ysim; dims = 2)), f1.pars.β; atol = 1.0)
    end

    @testset "interval routines apply the intercepts without X" begin
        XI = GLLVModels._trait_intercept_design(p, n)
        a = confint(f1, Y1)
        b = confint(f1, Y1; X = XI)
        @test a.lower == b.lower && a.upper == b.upper
        @test count(startswith("beta"), a.term) == p
        @test vcov(f1, Y1) == vcov(f1; y = Y1, X = XI)
        pa = profile_ci(f1, 1; y = Y1)
        pb = profile_ci(f1, 1; y = Y1, X = XI)
        @test (pa.lower, pa.upper) == (pb.lower, pb.upper)
        ba = bootstrap_ci(f1; y = Y1, n_boot = 10, seed = 1)
        bb = bootstrap_ci(f1; y = Y1, X = XI, n_boot = 10, seed = 1)
        @test ba.lower == bb.lower && ba.upper == bb.upper
    end

    @testset "cv_gllvm is shift equivariant" begin
        for split in (:random, :site)
            c0 = cv_gllvm(Y0; family = Normal(), K = 1, k_folds = 3, split = split,
                          rng = MersenneTwister(7))
            c1 = cv_gllvm(Y1; family = Normal(), K = 1, k_folds = 3, split = split,
                          rng = MersenneTwister(7))
            @test isapprox(c0.loglik, c1.loglik; rtol = 1e-4)
            @test isapprox(c0.mse, c1.mse; rtol = 1e-3)
        end
    end

    @testset "@formula(y ~ 1) matches the no-X route" begin
        ff = gllvm(@formula(y ~ 1), Y1, (; temp = randn(MersenneTwister(5), n));
                   family = Normal(), K = 1)
        @test length(ff.pars.β) == p
        @test isapprox(ff.logLik, f1.logLik; atol = 1e-8)
    end

    @testset "@formula(y ~ 0) stays zero mean" begin
        # Documented contract, and the R `value ~ 0 + latent(...)` pairing in
        # test/parity/core070_aghq_admission_cases.toml.
        fz = gllvm(@formula(y ~ 0), Y1, (; temp = randn(MersenneTwister(5), n));
                   family = Normal(), K = 1)
        @test isempty(fz.pars.β)
        @test fz.logLik == fit_gaussian_gllvm(Y1; K = 1).logLik
    end

    @testset "explicit X is unchanged" begin
        X = zeros(p, n, 2)
        X[:, :, 1] .= 1.0
        X[:, :, 2] .= randn(MersenneTwister(4), 1, n)
        a = fit_gllvm(Y1; family = Normal(), K = 1, X = X)
        b = fit_gaussian_gllvm(Y1; K = 1, X = X)
        @test a.logLik == b.logLik
        @test a.pars.β == b.pars.β
        @test !haskey(a.pars, :mean_design)
    end

    @testset "fit_gaussian_gllvm keeps its documented zero mean" begin
        f = fit_gaussian_gllvm(Y1; K = 1)
        @test isempty(f.pars.β)
    end
end
