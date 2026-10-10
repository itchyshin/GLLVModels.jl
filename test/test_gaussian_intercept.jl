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

    @testset "check_consistency runs on the public Gaussian route" begin
        fc = fit_gllvm(Y1; family = Normal(), K = 1)
        cc = GLLVModels.gllvmTMB_check_consistency(fc, Y1; n_sim = 60, seed = 42)
        @test length(cc.marginal_bias) == length(fc.pars.θ_packed)
        @test cc.marginal_p_value > 0.01
    end
end

# The phylogenetic effect J_n ⊗ B is a per-species constant across sites, so a
# free per-species intercept absorbs it and drives σ_phy to zero. Phylo fits on
# the public route therefore estimate one intercept shared by all species.
@testset "Gaussian phylo fit uses one common intercept" begin
    Random.seed!(21)
    p, n = 6, 200
    Λ = reshape(0.3 .+ 0.4 .* abs.(randn(p)), p, 1)
    Λ[2:2:end] .*= -1.0
    phy = GLLVModels.random_balanced_tree(p; branch_length = 0.5)
    Σ_phy = Matrix(Symmetric(GLLVModels.sigma_phy_dense(phy; σ²_phy = 1.0)))
    y = Λ * randn(1, n)
    y .+= 0.8 .* (cholesky(Symmetric(Σ_phy)).L * randn(p))
    y .+= 0.5 .* randn(p, n)
    f = fit_gllvm(y; family = Normal(), K = 1, has_phy_unique = true, Σ_phy = Σ_phy)
    ref = fit_gaussian_gllvm(y; K = 1, has_phy_unique = true, Σ_phy = Σ_phy,
                             X = ones(p, n, 1))
    @test length(f.pars.β) == 1
    @test isapprox(f.logLik, ref.logLik; atol = 1e-8)
    @test maximum(abs.(f.pars.σ_phy)) > 0.1

    # A common shift moves only the intercept.
    f3 = fit_gllvm(y .+ 3.0; family = Normal(), K = 1, has_phy_unique = true, Σ_phy = Σ_phy)
    @test isapprox(f3.logLik, f.logLik; atol = 1e-6)
    @test isapprox(f3.pars.β[1], f.pars.β[1] + 3.0; atol = 1e-4)

    # Post-fit helpers apply the common intercept when X is omitted.
    @test predict(f, y) ≈ predict(f, y; X = ones(p, n, 1))
    @test residuals(f, y) ≈ residuals(f, y; X = ones(p, n, 1))
end

# #577: on psych::bfi (items scored 1–6) the public Normal route reported
# `converged = true` thousands of logLik units below the centred fit, with and
# without a response mask. The intercept design above fixed the unmasked path; this
# pins the masked path too: shifting the observed values by a per-trait constant
# moves only the intercepts. (Observed-mean centring is not the ML intercept under
# missingness, so the comparison is shift against shift, not raw against centred.)
@testset "masked Gaussian fit is shift equivariant (#577)" begin
    rng = MersenneTwister(577)
    p, n = 6, 120
    Y0 = 0.9 .* randn(rng, p, 1) * randn(rng, 1, n) .+ randn(rng, p, n)
    mk = rand(rng, p, n) .> 0.1
    shift = [2.3, 4.9, 4.6, 4.7, 4.5, 3.1]
    Y1 = Y0 .+ shift
    Y0[.!mk] .= 0.0; Y1[.!mk] .= 0.0
    f0 = fit_gllvm(Y0; family = Normal(), K = 1, mask = mk)
    f1 = fit_gllvm(Y1; family = Normal(), K = 1, mask = mk)
    @test f0.converged && f1.converged
    @test length(f1.pars.β) == p
    @test isapprox(f0.logLik, f1.logLik; atol = 1e-5)
    @test isapprox(f1.pars.β, f0.pars.β .+ shift; atol = 1e-3)
end
