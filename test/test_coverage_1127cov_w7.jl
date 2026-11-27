# Coverage lane w7 (27 Nov 2026): ranks 16–18 on main @ Pkg.test(coverage=true).
#   src/grouped_nongaussian_fit.jl, src/confint_profile.jl, src/lowrank_cholesky.jl
using Test
using Random
using LinearAlgebra
using Distributions
using GLLVModels

@testset "coverage 1127cov w7 (grouped_nongaussian, confint_profile, lowrank cholesky)" begin
    @testset "LowRankPlusDiagChol mul! errors and matvec" begin
        F = GLLVModels.low_rank_chol([1.0 0.0; 0.5 0.5; 0.2 0.1], [1.0, 0.1, 2.0])
        @test_throws DimensionMismatch LinearAlgebra.mul!(zeros(2), F, ones(3))
        @test_throws DimensionMismatch LinearAlgebra.mul!(ones(3), F, ones(2))
        X = randn(3, 5)
        @test_throws DimensionMismatch LinearAlgebra.mul!(zeros(2, 5), F, X)
        x = ones(3)
        y1 = F * x
        y2 = similar(x)
        LinearAlgebra.mul!(y2, F, x)
        @test y2 ≈ y1
        Xm = hcat(x, 2x)
        Ym = F * Xm
        Y2 = similar(Xm)
        LinearAlgebra.mul!(Y2, F, Xm)
        @test Y2 ≈ Ym
    end

    @testset "grouped_nongaussian_fit validation and targets" begin
        term = GLLVModels.GroupingTerm(:unit; mode = :indep, common = true)
        unit = repeat(1:4; inner = 2)
        Ypoisson = reshape(Float64[1, 2, 9, 10, 11, 12, 8, 7], 1, :)
        @test_throws ArgumentError GLLVModels._grouped_nongaussian_kind(Gamma(2.0, 1.0))
        @test_throws ArgumentError GLLVModels.fit_grouped_nongaussian(
            Ypoisson; family = GLLVModels.Poisson(), terms = [term], unit = unit,
            N = fill("5", 1, length(unit)))
        poisson_fit = GLLVModels.fit_grouped_nongaussian(
            Ypoisson; family = GLLVModels.Poisson(), terms = [term], unit = unit,
            moment_start = true, iterations = 80, nelder_mead = true)
        @test poisson_fit.converged || isfinite(poisson_fit.loglik)
        targets = GLLVModels._grouped_nongaussian_primary_targets(poisson_fit)
        @test any(t -> String(t.name) == "unit.sd[1]", targets)
        p, n = 2, 8
        Ylat = [2 3 1 4 2 3 2 4; 3 2 4 1 3 2 4 2]
        Nbin = fill(5.0, p, n)
        latent_fit = GLLVModels.fit_grouped_nongaussian(
            Ylat; family = GLLVModels.Binomial(), N = Nbin, unit = unit,
            terms = [GLLVModels.GroupingTerm(:unit; mode = :latent, rank = 1, unique = true)],
            iterations = 80)
        targets_latent = GLLVModels._grouped_nongaussian_primary_targets(latent_fit)
        @test any(t -> occursin("unique_variance", String(t.name)), targets_latent)
        @test any(t -> occursin("covariance", String(t.name)), targets_latent)
        @test_throws ArgumentError GLLVModels._grouped_nongaussian_trials(
            ones(1, 2), fill("3", 1, 2), :binomial)
        bridge_terms = [term]
        bridge_theta = [log(2.5), log(0.35)]
        bridge_D = GLLVModels._trait_mean_design(1, n)
        bridge_inc = [GLLVModels._grouped_incidence(unit, n)]
        warm_obj = GLLVModels._grouped_nongaussian_objective(
            Ypoisson, ones(size(Ypoisson)), bridge_D, bridge_terms, bridge_inc, :poisson;
            inner_maxiter = 40, inner_tol = 1e-6, warm_start_inner = true)
        v1 = warm_obj(bridge_theta)
        v2 = warm_obj(bridge_theta .+ [1e-5, -1e-5])
        @test isfinite(v1)
        @test isfinite(v2) || v2 >= 1e11
    end

    @testset "confint_profile term lists and Symbol routes" begin
        rng = MersenneTwister(1127)
        p, n = 5, 40
        Λ = 0.35 .* ones(p, 1)
        y = Λ * randn(rng, 1, n) .+ 0.4 .* randn(rng, p, n)
        fit_ww = fit_gaussian_gllvm(y; K = 1, K_W = 1, has_diag = true)
        names, kinds = GLLVModels._profile_all_term_names(fit_ww)
        @test any(==("Lambda_W[1,1]"), names)
        @test any(==("sigma_W[1]"), names)
        @test kinds[findfirst(==("sigma_eps"), names)] === :log_sd
        y_phy = copy(y)
        Σ = Matrix{Float64}(I, p, p)
        fit_phy = fit_gaussian_gllvm(y_phy; K = 1, K_phy = 1, Σ_phy = Σ)
        names_phy, _ = GLLVModels._profile_all_term_names(fit_phy)
        @test any(startswith("Lambda_phy[1,"), names_phy)
        @test GLLVModels._profile_parm_index(fit_ww, "Lambda:1,1") ==
            GLLVModels._profile_parm_index(fit_ww, "Lambda_B[1,1]")
        @test GLLVModels._profile_parm_index(fit_ww, "Lambda_W:1,1") ==
            GLLVModels._profile_parm_index(fit_ww, "Lambda_W[1,1]")
        prof = GLLVModels.profile_ci(fit_ww, :sigma_eps; y = y, max_expand = 6, max_bisect = 4)
        @test prof.method in (:profile, :partial, :failed)
        @test hasproperty(prof, :lower) && hasproperty(prof, :upper)
        curve = GLLVModels.tmbprofile_wrapper(fit_ww, :sigma_eps;
            y = y, max_expand = 6, max_bisect = 4)
        @test length(curve.theta) >= 1 && length(curve.nll) == length(curve.theta)
        @test curve.method == prof.method
        idx_bad = length(fit_ww.pars.θ_packed) + 99
        @test_throws ArgumentError GLLVModels.tmbprofile_wrapper(fit_ww, idx_bad; y = y)
    end
end