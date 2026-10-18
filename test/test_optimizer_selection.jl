using Test
using GLLVModels
using Optim
using Distributions: Normal, Poisson, NegativeBinomial
using StatsModels: @formula

@testset "public outer optimizer selection" begin
    Yg = [sin(0.31t + 0.17s) + 0.1cos(0.13t*s) for t in 1:3, s in 1:14]
    Yc = [round(Int, abs(1.3sin(0.27t + 0.23s)) + 0.1) for t in 1:3, s in 1:14]

    @testset "Gaussian requested method and default regression" begin
        selected = fit_gaussian_gllvm(Yg; K = 1, optimizer = :bfgs)
        info = optimizer_info(selected)
        @test info.requested == :bfgs
        @test info.actual == Optim.BFGS
        @test info.controls.optim_options isa Optim.Options
        @test !isempty(info.stages)
        @test all(stage -> stage.method isa Optim.BFGS, info.stages)
        @test info.stages[end].result === selected.optim_result
        @test_throws ArgumentError confint(selected, Yg; method = :profile)

        default = fit_gaussian_gllvm(Yg; K = 1)
        explicit_default = fit_gaussian_gllvm(Yg; K = 1,
            optimizer = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)))
        @test isapprox(default.logLik, explicit_default.logLik; atol = 1e-7, rtol = 1e-8)
        @test default.converged
        @test optimizer_info(default).requested === nothing
        @test optimizer_info(default).actual == Optim.LBFGS
    end

    @testset "count Laplace routes and default per-trait NB2" begin
        pois = fit_poisson_gllvm(Yc; K = 1, optimizer = Optim.BFGS(), iterations = 3)
        @test optimizer_info(pois).requested isa Optim.BFGS
        @test optimizer_info(pois).actual == Optim.BFGS
        @test all(stage -> stage.method isa Optim.BFGS, optimizer_info(pois).stages)
        @test_throws ArgumentError confint(pois, Yc; method = :profile)
        default_pois = fit_poisson_gllvm(Yc; K = 1, optimizer = nothing,
            iterations = 1, g_tol = 1e-30)
        explicit_lbfgs = fit_poisson_gllvm(Yc; K = 1, optimizer = :lbfgs,
            iterations = 1, g_tol = 1e-30)
        @test optimizer_info(default_pois).requested === nothing
        @test optimizer_info(default_pois).actual in (Optim.LBFGS, Optim.BFGS)
        @test any(stage -> stage.name === :legacy_bfgs_continuation,
            optimizer_info(default_pois).stages)
        @test optimizer_info(explicit_lbfgs).actual == Optim.LBFGS
        @test !explicit_lbfgs.converged
        @test all(stage -> stage.method isa Optim.LBFGS, optimizer_info(explicit_lbfgs).stages)
        @test !any(stage -> stage.name === :legacy_bfgs_continuation,
            optimizer_info(explicit_lbfgs).stages)

        nb_shared = fit_nb_gllvm(Yc; K = 1, optimizer = :bfgs, iterations = 3)
        @test optimizer_info(nb_shared).actual == Optim.BFGS
        @test all(stage -> stage.method isa Optim.BFGS, optimizer_info(nb_shared).stages)

        nb_species = fit_nb_gllvm_grouped(Yc; K = 1, group = 1:3,
            optimizer = :lbfgs, iterations = 3)
        @test optimizer_info(nb_species).actual == Optim.LBFGS
        @test all(stage -> stage.method isa Optim.LBFGS, optimizer_info(nb_species).stages)
        nb_default = fit_nb_gllvm_grouped(Yc; K = 1, group = 1:3,
            iterations = 1, g_tol = 1e-30)
        @test optimizer_info(nb_default).requested === nothing
        @test optimizer_info(nb_default).actual in (Optim.LBFGS, Optim.BFGS)
    end

    @testset "fit_gllvm, formula, and select_lv forward" begin
        selected = fit_gllvm(Yg; family = Normal(), K = 1, optimizer = :bfgs)
        @test optimizer_info(selected).actual == Optim.BFGS
        pois = fit_gllvm(Yc; family = Poisson(), K = 1, optimizer = :bfgs, iterations = 3)
        @test optimizer_info(pois).actual == Optim.BFGS

        selection = select_lv(Yg; family = Normal(), Kmax = 1, optimizer = :bfgs)
        @test optimizer_info(selection.best).actual == Optim.BFGS

        formula_pois = gllvm(@formula(y ~ 1), Yc, (y = Yc[1, :],);
            family = Poisson(), K = 1, optimizer = :bfgs, iterations = 3)
        @test optimizer_info(formula_pois).actual == Optim.BFGS
        formula_nb = gllvm(@formula(y ~ 1), Yc, (y = Yc[1, :],);
            family = NegativeBinomial(), K = 1, optimizer = :lbfgs, iterations = 2)
        @test optimizer_info(formula_nb).actual == Optim.LBFGS
        @test all(stage -> stage.method isa Optim.LBFGS, optimizer_info(formula_nb).stages)
    end

    @testset "unsupported routes reject before fitting" begin
        @test_throws ArgumentError fit_poisson_gllvm(Yc; K = 1, aghq = true, optimizer = :bfgs)
        @test_throws ArgumentError fit_gllvm(Yc; family = Poisson(), K = 1,
            row_eff = :random, optimizer = :bfgs)
        @test_throws ArgumentError fit_nb_gllvm(Yc; K = 1, optimizer = :nelder_mead)
        @test_throws ArgumentError fit_poisson_gllvm(Yc; K = 1,
            optimizer = :bfgs, aghq = true, iterations = 1)
        @test_throws ArgumentError fit_gaussian_sources(Yg; sources = SourceCovariance[],
            X = zeros(3, 14, 0), sigma_eps_fixed = 1.0, optimizer = :bfgs)
        @test_throws ArgumentError gllvm(@formula(y ~ 1 + x), Yc,
            (y = Yc[1, :], x = collect(1:14));
            family = Poisson(), K = 1, optimizer = :bfgs)
    end

    @testset "legacy positional constructor compatibility" begin
        gf = fit_gaussian_gllvm(Yg; K = 1)
        pois = fit_poisson_gllvm(Yc; K = 1, iterations = 3)
        nb_shared = fit_nb_gllvm(Yc; K = 1, iterations = 3)
        nb_species = fit_nb_gllvm_grouped(Yc; K = 1, group = 1:3, iterations = 3)
        sf = fit_gaussian_sources(Yg; sources = SourceCovariance[], iterations = 3)
        @test optimizer_info(GllvmFit(gf.model, gf.pars, gf.logLik, gf.n_iter,
            gf.converged, gf.optim_result, gf.cputime)) === nothing
        @test optimizer_info(GllvmFit(gf.model, gf.pars, gf.logLik, gf.n_iter,
            gf.converged, gf.optim_result, gf.cputime, gf.integration)) === nothing
        @test optimizer_info(PoissonFit(pois.β, pois.Λ, pois.link, pois.loglik,
            pois.converged, pois.iterations, pois.alpha_lv, pois.theta_packed,
            pois.hessian, pois.integration, pois.offset, pois.weights)) === nothing
        @test optimizer_info(NBFit(nb_shared.β, nb_shared.Λ, nb_shared.r, nb_shared.link,
            nb_shared.loglik, nb_shared.converged, nb_shared.iterations, nb_shared.alpha_lv,
            nb_shared.theta_packed, nb_shared.hessian, nb_shared.offset)) === nothing
        @test optimizer_info(NBGroupedFit(nb_species.β, nb_species.Λ, nb_species.r_group,
            nb_species.group, nb_species.link, nb_species.loglik, nb_species.converged,
            nb_species.iterations, nb_species.hessian, nb_species.dispersion_boundary,
            nb_species.offset)) === nothing
        @test optimizer_info(GaussianSourcesFit(sf.beta, sf.sigma_eps, sf.trait_covariances,
            sf.sources, sf.parameters, sf.loglik, sf.converged, sf.gradient_norm,
            sf.hessian_min_eigenvalue, sf.hessian_positive_definite, sf.iterations,
            sf.stopping_reason, sf.observations, sf.residual_fixed, sf.mean_design,
            sf.response_shape, sf.coefficient_names)) === nothing
    end
end

println("OPTIMIZER_SELECTION_PASS")
