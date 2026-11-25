# Lane cursor-cloud-1125y15 — grouped_profile_refit/inversion extras, exponential VA marginal
using GLLVModels, Test, Random, Distributions, LinearAlgebra

const G = GLLVModels

@testset "cov 1125y15 miscellaneous guards and marginals" begin
    @testset "_grouped_profile_evaluator_vector" begin
        v = G._grouped_profile_evaluator_vector([1.0, 2.0, 3.0], 4, 2, 0.25)
        @test v == [1.0, log(0.25) / 2, 2.0, 3.0]
        v2 = G._grouped_profile_evaluator_vector([5.0, 6.0], 3, 1, 0.0)
        @test v2 == [0.0, 5.0, 6.0]
        @test G._grouped_profile_evaluator_vector([1.0], 4, 2, 1.0) === nothing
    end

    @testset "_grouped_profile_refit_provenance" begin
        prov = G._grouped_profile_refit_provenance(:evaluate, 0.25, :upper, 1, 2, 3,
            "unit", [1.0 2.0], [1.0 0.0; 0.0 1.0],
            [G.GroupingTerm(:unit; mode = :indep)], [[1, 1, 2, 2]])
        @test prov.stage === :evaluate
        @test prov.fixed_variance ≈ 0.25
        @test prov.side === :upper
        @test prov.selected == (1, 2)
    end

    @testset "_profile_lr_refit accepted path" begin
        refit(v, _) = (accepted = true, objective_nll = 1.0 + 0.1 * v, status = :accepted,
            provenance = (model = :x, fixed_variance = v))
        pt = G._profile_lr_refit(1.0, 0.5, refit)
        @test pt.refit_accepted
        @test isfinite(pt.lr)
        @test pt.lr >= 0
        @test pt.fixed_variance ≈ 0.5
    end

    @testset "exponential_marginal_loglik_va tiny" begin
        Y = abs.(rand(Poisson(1.2), 4, 8)) .+ 0.05
        ll = G.exponential_marginal_loglik_va(Y, fill(0.05, 4, 1), zeros(4); maxiter = 30)
        @test isfinite(ll)
        Λ0 = zeros(4, 1)
        ll0 = G.exponential_marginal_loglik_va(Y, Λ0, zeros(4); maxiter = 20)
        @test isfinite(ll0)
    end

    @testset "fit_phylo_glm tiny Poisson" begin
        phy = G.augmented_phy("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
        Y = rand(Poisson(1.5), phy.n_leaves, 6)
        fit = G.fit_phylo_glm(Y, phy; family = Poisson(), iterations = 40, g_tol = 1e-3)
        @test isfinite(fit.loglik)
        @test fit.converged in (true, false)
        @test length(fit.β) == phy.n_leaves
    end

    @testset "_isdm_parse_formula offset extraction" begin
        f = :(y ~ 0 + trait + trait & env + offset(log_area) +
            latent(0 + trait | cell, d = 1))
        parts = G._isdm_parse_formula(f; trait = :trait, unit = :cell)
        @test parts.response === :y
        @test parts.K == 1
        @test parts.offset == :(log_area)
        @test :trait in parts.fixed_symbols
    end

    @testset "delta gamma support check" begin
        Y = abs.(randn(2, 5)) .+ 0.5
        @test_throws ArgumentError fit_delta_gamma_gllvm(-Y; K = 1, iterations = 5)
    end
end