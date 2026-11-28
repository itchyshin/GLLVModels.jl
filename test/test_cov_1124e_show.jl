# Coverage: show methods, parameter counts, and argument refusals that
# return before a fit or a Laplace mode search.
using Test
using GLLVModels
using Distributions
using LinearAlgebra

grab(f) = try
    f()
    nothing
catch e
    e
end

@testset "show methods and early refusals (1124e)" begin
    β = [0.1, -0.2]
    Λ = [0.3 0.0; 0.1 0.4]
    D = [-0.2 0.0; 0.0 -0.1]
    Y = [1.0 0.0; 2.0 1.0]

    @testset "quadratic" begin
        fit = QuadraticFit(Poisson(), β, Λ, D, NaN, LogLink(), -12.3456789, true, 7)
        shown = sprint(show, fit)
        @test occursin("QuadraticFit(Poisson, p=2, K=2", shown)
        @test occursin(string(round(-12.3456789; sigdigits = 7)), shown)
        @test !occursin("disp=", shown)
        @test !occursin("NOT CONVERGED", shown)
        @test GLLVModels._loadings(fit) === fit.Λ
        @test GLLVModels._loglik(fit) == -12.3456789
        @test GLLVModels._nparams(fit) == 2 + (4 - 1) + 4
        stopped = QuadraticFit(NegativeBinomial(), β, Λ[:, 1:1], D[:, 1:1], 1.5,
                               LogLink(), -3.0, false, 1)
        shown2 = sprint(show, stopped)
        @test occursin("disp=", shown2)
        @test occursin(string(round(1.5; sigdigits = 4)), shown2)
        @test occursin("NOT CONVERGED", shown2)
        @test GLLVModels._nparams(stopped) == 2 + 2 + 2 + 1
        e = grab(() -> predict(fit, Y; type = :mean))
        @test e isa ArgumentError
        @test occursin("type must be :link or :response; got :mean", e.msg)
        e = grab(() -> GLLVModels.quadratic_loglik_site(Poisson(), [1.0, 0.0], [1, 1], Λ, D, β,
                                                       LogLink(); hessian = :foo))
        @test e isa ArgumentError
        @test occursin("hessian must be :fisher or :observed; got :foo", e.msg)
    end

    @testset "row effect" begin
        fit = RowEffectFit(Poisson(), β, [0.0], Λ[:, 1:1], NaN, LogLink(), -1.25, true, 3)
        shown = sprint(show, fit)
        @test occursin("RowEffectFit(Poisson, p=2, n=1, K=1", shown)
        @test !occursin("NOT CONVERGED", shown)
        @test GLLVModels._nparams(fit) == 2 + 0 + 2
        wide = RowEffectFit(Gamma(), β, [0.0, 0.2, -0.1], Λ, 0.8, LogLink(), -8.0, false, 2)
        @test occursin("NOT CONVERGED", sprint(show, wide))
        @test occursin("disp=", sprint(show, wide))
        @test GLLVModels._nparams(wide) == 2 + 2 + (4 - 1) + 1
        O = GLLVModels._build_offset_row([0.0, 0.5, -0.2], 2)
        @test O == [0.0 0.5 -0.2; 0.0 0.5 -0.2]
        e = grab(() -> predict(fit, Y; type = :mean))
        @test e isa ArgumentError
        @test occursin("type must be :link or :response; got :mean", e.msg)
        e = grab(() -> fit_roweffect_gllvm(zeros(2, 0); family = Poisson(), K = 1))
        @test e isa ArgumentError
        @test e.msg == "Y must have at least one site (column)"
    end

    @testset "random row" begin
        fit = RowRandomFit(Poisson(), β, Λ[:, 1:1], 0.25, NaN, LogLink(), -2.0, true, 4)
        shown = sprint(show, fit)
        @test occursin("RowRandomFit(Poisson, p=2, K=1", shown)
        @test occursin("σ_row=" * string(round(0.25; sigdigits = 4)), shown)
        @test !occursin("NOT CONVERGED", shown)
        @test GLLVModels._loadings(fit) === fit.Λ
        @test GLLVModels._loglik(fit) == -2.0
        @test GLLVModels._nparams(fit) == 2 + 2 + 1
        stopped = RowRandomFit(Beta(), β, Λ, 1.0, 2.5, LogitLink(), -6.0, false, 1)
        @test occursin("NOT CONVERGED", sprint(show, stopped))
        @test GLLVModels._nparams(stopped) == 2 + (4 - 1) + 1 + 1
        aug = GLLVModels._augment_lambda(Λ[:, 1:1], 0.4)
        @test size(aug) == (2, 2)
        @test aug[:, 1] == Λ[:, 1]
        @test aug[:, 2] == [0.4, 0.4]
        e = grab(() -> predict(fit, Y; type = :latent))
        @test e isa ArgumentError
        @test occursin("type must be :link, :response, or :roweffect; got :latent", e.msg)
        e = grab(() -> fit_row_random_gllvm(zeros(2, 0); K = 1))
        @test e isa ArgumentError
        @test e.msg == "Y must have at least one site (column)"
        e = grab(() -> fit_row_random_gllvm(ones(2, 3); K = 1, σ_row_init = 0.0))
        @test e isa ArgumentError
        @test e.msg == "σ_row_init must be positive (it is fit on the log scale)"
        e = grab(() -> fit_row_random_gllvm(ones(2, 3); K = 1, σ_row_init = -0.2))
        @test e isa ArgumentError
        @test occursin("σ_row_init must be positive", e.msg)
    end

    @testset "species covariates and constrained ordination" begin
        Bsp = [0.2 0.0; -0.1 0.3]
        species = GllvmSpeciesCovFit(Poisson(), β, Bsp, Λ[:, 1:1], NaN, LogLink(), -4.5, true, 2)
        shown = sprint(show, species)
        @test occursin("GllvmSpeciesCovFit(Poisson, p=2, q=2, K=1", shown)
        @test !occursin("disp=", shown)
        @test GLLVModels._loglik(species) == -4.5
        @test GLLVModels._nparams(species) == 2 + 4 + 2
        stopped = GllvmSpeciesCovFit(Gamma(), β, Bsp[:, 1:1], Λ, 1.25, LogLink(), -9.0, false, 1)
        @test occursin("NOT CONVERGED", sprint(show, stopped))
        @test GLLVModels._nparams(stopped) == 2 + 2 + (4 - 1) + 1
        X3 = zeros(2, 2, 2)
        e = grab(() -> predict(species, Y, X3; type = :latent))
        @test e isa ArgumentError
        @test occursin("type must be :response, :mean, or :link; got :latent", e.msg)

        B = [0.5 0.0; 0.1 -0.2; 0.0 0.3]
        constrained = ConstrainedOrdinationFit(Poisson(), β, Λ, B, NaN, LogLink(), -1.0, false, 5)
        shown = sprint(show, constrained)
        @test occursin("ConstrainedOrdinationFit(Poisson, p=2, q=3, K=2", shown)
        @test occursin("NOT CONVERGED", shown)
        @test GLLVModels._loglik(constrained) == -1.0
        @test GLLVModels._nparams(constrained) == 2 + (4 - 1) + 6
        with_disp = ConstrainedOrdinationFit(Beta(), β, Λ[:, 1:1], B[:, 1:1], 3.0, LogitLink(),
                                             -2.0, true, 1)
        @test occursin("disp=", sprint(show, with_disp))
        @test !occursin("NOT CONVERGED", sprint(show, with_disp))
        @test GLLVModels._nparams(with_disp) == 2 + 2 + 3 + 1
        e = grab(() -> predict(constrained, Y, zeros(2, 3); type = :latent))
        @test e isa ArgumentError
        @test occursin("type must be :response, :mean, or :link; got :latent", e.msg)
    end

    @testset "two-level show and early refusals" begin
        Λb = [0.2 0.0; 0.1 0.3]
        Λw = reshape([0.4, 0.2], 2, 1)
        σb = [0.5, 0.4]
        σw = [0.3, 0.2]
        Σb = Λb * Λb' + Matrix(Diagonal(σb))
        Σw = Λw * Λw' + Matrix(Diagonal(σw))
        fit = TwoLevelFit(Λb, σb, Λw, σw, Σb, Σw, 3, -4.0, false, 2)
        shown = sprint(show, fit)
        @test occursin("TwoLevelFit(p=2, K_B=2, K_W=1, nindiv=3", shown)
        @test occursin("NOT CONVERGED", shown)
        y = ones(2, 3)
        indiv = [1, 1, 2]
        e = grab(() -> repeatability_ci(fit, y, indiv; method = :other))
        @test e isa ArgumentError
        @test occursin("method must be :wald, :bootstrap, or :profile; got other", e.msg)
        e = grab(() -> GLLVModels.repeatability_wald_ci(fit, y, indiv; level = 0))
        @test e isa ArgumentError
        @test occursin("level must be in (0, 1); got 0", e.msg)
        e = grab(() -> GLLVModels.repeatability_wald_ci(fit, y, indiv; level = 1))
        @test e isa ArgumentError
        @test occursin("got 1", e.msg)
        e = grab(() -> GLLVModels.repeatability_bootstrap_ci(fit, indiv; level = 1.5))
        @test e isa ArgumentError
        @test occursin("got 1.5", e.msg)
        e = grab(() -> GLLVModels.repeatability_wald_ci(fit, y, [1]; level = 0.95))
        @test e isa DimensionMismatch
        @test occursin("individual length must equal n_obs = 3", e.msg)
        e = grab(() -> fit_twolevel_gaussian(ones(2, 4), [1, 1]; K_B = 1, K_W = 1))
        @test e isa DimensionMismatch
        @test occursin("individual length must equal n_obs = 4", e.msg)
        e = grab(() -> fit_twolevel_gaussian(ones(2, 4), [1, 1, 2, 2]; K_B = 0, K_W = 1))
        @test e isa ArgumentError
        @test e.msg == "K_B and K_W must be ≥ 1"
        Ym = Matrix{Union{Missing, Float64}}(ones(2, 3))
        Ym[1, 2] = missing
        e = grab(() -> fit_twolevel_gaussian(Ym, [1, 1, 2]; K_B = 1, K_W = 1))
        @test e isa ArgumentError
        @test occursin("fit_twolevel_gaussian does not accept `missing` in `Y`: 1 missing", e.msg)
        @test occursin("(1, 2)", e.msg)
    end

    @testset "Destination-B fixed-effect summary" begin
        row = (name = "b1", estimate = 0.2, se = 0.1, lower = 0.0, upper = 0.4,
               status = :available, method = :observed_marginal_wald)
        report = GLLVModels.DestinationBFixedEffectSummary([row], :available, 1.0e-6, -10.5,
                                                          true, 1.0e-6, true, 0.01, 4, :converged)
        shown = sprint(show, report)
        @test occursin("Destination-B fixed-effect summary", shown)
        @test occursin("inference = available", shown)
        @test occursin("convergence = true", shown)
        @test occursin("observed curvature PD = true", shown)
        @test occursin("fixed effects = 1", shown)
        @test occursin("logLik = -10.5", shown)
        @test occursin("inference gradient = 1.0e-6", shown)
        e = grab(() -> GLLVModels._destination_b_fixed_effect_information(x -> 0.0, [0.1, 0.2], 0;
                                                                        converged = true))
        @test e isa ArgumentError
        @test e.msg == "fixed-effect coordinate count is invalid"
        e = grab(() -> GLLVModels._destination_b_fixed_effect_information(x -> 0.0, [0.1], 2;
                                                                        converged = false))
        @test e isa ArgumentError
        @test e.msg == "fixed-effect coordinate count is invalid"
    end
end
