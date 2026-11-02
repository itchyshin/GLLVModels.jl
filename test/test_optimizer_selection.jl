using Test
using GLLVModels
using Optim
using TOML
using Distributions: Normal, Poisson, NegativeBinomial
using StatsModels: @formula

# Observe the actual public Optim boundary. The method-specific function
# overloads record the algorithm and then invoke Optim's original generic
# dispatcher; the objective-object overloads are therefore not counted twice.
const OPTIMIZER_BOUNDARY_CALLS = Any[]
@eval Optim begin
    function optimize(f::Function, x0::AbstractArray, method::Optim.BFGS,
                      options::Optim.Options; kwargs...)
        push!(Main.OPTIMIZER_BOUNDARY_CALLS, method)
        invoke(Optim.optimize,
            Tuple{Any,AbstractArray,Optim.AbstractOptimizer,Optim.Options},
            f, x0, method, options; kwargs...)
    end
    function optimize(f::Function, g::Function, x0::AbstractArray,
                      method::Optim.BFGS, options::Optim.Options; kwargs...)
        push!(Main.OPTIMIZER_BOUNDARY_CALLS, method)
        invoke(Optim.optimize,
            Tuple{Any,Any,AbstractArray,Optim.AbstractOptimizer,Optim.Options},
            f, g, x0, method, options; kwargs...)
    end
    function optimize(f::Function, x0::AbstractArray, method::Optim.LBFGS,
                      options::Optim.Options; kwargs...)
        push!(Main.OPTIMIZER_BOUNDARY_CALLS, method)
        invoke(Optim.optimize,
            Tuple{Any,AbstractArray,Optim.AbstractOptimizer,Optim.Options},
            f, x0, method, options; kwargs...)
    end
    function optimize(f::Function, g::Function, x0::AbstractArray,
                      method::Optim.LBFGS, options::Optim.Options; kwargs...)
        push!(Main.OPTIMIZER_BOUNDARY_CALLS, method)
        invoke(Optim.optimize,
            Tuple{Any,Any,AbstractArray,Optim.AbstractOptimizer,Optim.Options},
            f, g, x0, method, options; kwargs...)
    end
    function optimize(f::Optim.NLSolversBase.InplaceObjective, x0::AbstractArray,
                      method::Optim.BFGS, options::Optim.Options; kwargs...)
        push!(Main.OPTIMIZER_BOUNDARY_CALLS, method)
        invoke(Optim.optimize,
            Tuple{Any,AbstractArray,Optim.AbstractOptimizer,Optim.Options},
            f, x0, method, options; kwargs...)
    end
    function optimize(f::Optim.NLSolversBase.InplaceObjective, x0::AbstractArray,
                      method::Optim.LBFGS, options::Optim.Options; kwargs...)
        push!(Main.OPTIMIZER_BOUNDARY_CALLS, method)
        invoke(Optim.optimize,
            Tuple{Any,AbstractArray,Optim.AbstractOptimizer,Optim.Options},
            f, x0, method, options; kwargs...)
    end
end
const OPTIMIZER_BOUNDARY_SPY_METHODS = Method[
    method for method in methods(Optim.optimize)
    if method.module === Optim && endswith(string(method.file), "test_optimizer_selection.jl")]

@testset "public outer optimizer selection" begin
    Yg = [sin(0.31t + 0.17s) + 0.1cos(0.13t*s) for t in 1:3, s in 1:14]
    Yc = [round(Int, abs(1.3sin(0.27t + 0.23s)) + 0.1) for t in 1:3, s in 1:14]

    @testset "Gaussian requested method and default regression" begin
        empty!(OPTIMIZER_BOUNDARY_CALLS)
        selected = fit_gaussian_gllvm(Yg; K = 1, optimizer = :bfgs)
        info = optimizer_info(selected)
        @test info.requested == :bfgs
        @test info.actual == Optim.BFGS
        @test info.controls.optim_options isa Optim.Options
        @test !isempty(info.stages)
        @test all(stage -> stage.method isa Optim.BFGS, info.stages)
        @test info.stages[end].result === selected.optim_result
        @test length(OPTIMIZER_BOUNDARY_CALLS) == length(info.stages)
        @test all(method -> method isa Optim.BFGS, OPTIMIZER_BOUNDARY_CALLS)
        @test_throws ArgumentError confint(selected, Yg; method = :profile)

        default = fit_gaussian_gllvm(Yg; K = 1)
        explicit_default = fit_gaussian_gllvm(Yg; K = 1,
            optimizer = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)))
        @test isapprox(default.logLik, explicit_default.logLik; atol = 1e-7, rtol = 1e-8)
        @test default.converged
        @test optimizer_info(default).requested === nothing
        @test optimizer_info(default).actual == Optim.LBFGS
    end

    @testset "Gaussian mask and offset use selected optimizer at every solve" begin
        partial_mask = trues(size(Yg))
        partial_mask[1, end] = false
        empty!(OPTIMIZER_BOUNDARY_CALLS)
        masked = fit_gaussian_gllvm(Yg; K = 1,
            mask = partial_mask, optimizer = :bfgs, iterations = 1)
        masked_info = optimizer_info(masked)
        @test count(stage -> stage.name === :mask_offset_refit, masked_info.stages) == 1
        @test length(OPTIMIZER_BOUNDARY_CALLS) == length(masked_info.stages)
        @test all(method -> method isa Optim.BFGS, OPTIMIZER_BOUNDARY_CALLS)
        empty!(OPTIMIZER_BOUNDARY_CALLS)
        offset_fit = fit_gaussian_gllvm(Yg; K = 1,
            offset = zeros(size(Yg)), optimizer = :bfgs, iterations = 1)
        offset_info = optimizer_info(offset_fit)
        @test count(stage -> stage.name === :mask_offset_refit, offset_info.stages) == 1
        @test length(OPTIMIZER_BOUNDARY_CALLS) == length(offset_info.stages)
        @test all(method -> method isa Optim.BFGS, OPTIMIZER_BOUNDARY_CALLS)
    end

    @testset "count Laplace routes and default per-trait NB2" begin
        empty!(OPTIMIZER_BOUNDARY_CALLS)
        pois = fit_poisson_gllvm(Yc; K = 1, optimizer = Optim.BFGS(), iterations = 3)
        @test optimizer_info(pois).requested isa Optim.BFGS
        @test optimizer_info(pois).actual == Optim.BFGS
        @test all(stage -> stage.method isa Optim.BFGS, optimizer_info(pois).stages)
        @test length(OPTIMIZER_BOUNDARY_CALLS) == length(optimizer_info(pois).stages)
        @test all(method -> method isa Optim.BFGS, OPTIMIZER_BOUNDARY_CALLS)
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

        empty!(OPTIMIZER_BOUNDARY_CALLS)
        nb_species = fit_nb_gllvm_grouped(Yc; K = 1, group = 1:3,
            optimizer = :lbfgs, iterations = 3)
        @test optimizer_info(nb_species).actual == Optim.LBFGS
        @test all(stage -> stage.method isa Optim.LBFGS, optimizer_info(nb_species).stages)
        @test length(OPTIMIZER_BOUNDARY_CALLS) == length(optimizer_info(nb_species).stages)
        @test all(method -> method isa Optim.LBFGS, OPTIMIZER_BOUNDARY_CALLS)

        # A one-step fit is deliberately unconverged, so the per-trait NB2
        # route enters its Poisson-ridge continuation with the explicit BFGS.
        empty!(OPTIMIZER_BOUNDARY_CALLS)
        nb_species_bfgs = fit_nb_gllvm_grouped(Yc; K = 1, group = 1:3,
            optimizer = :bfgs, iterations = 1, g_tol = 1e-30)
        nb_bfgs_info = optimizer_info(nb_species_bfgs)
        @test any(stage -> stage.name === :ridge_polish, nb_bfgs_info.stages)
        @test length(OPTIMIZER_BOUNDARY_CALLS) == length(nb_bfgs_info.stages)
        @test all(method -> method isa Optim.BFGS, OPTIMIZER_BOUNDARY_CALLS)
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

@testset "optimizer boundary spy cleanup" begin
    foreach(Base.delete_method, OPTIMIZER_BOUNDARY_SPY_METHODS)
    installed = collect(methods(Optim.optimize))
    @test all(method -> !(method in installed), OPTIMIZER_BOUNDARY_SPY_METHODS)
end

println("OPTIMIZER_SELECTION_PASS")
