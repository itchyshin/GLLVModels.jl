# Lane cursor-cloud-1125y11 — destination_b_postfit.jl
using GLLVModels, Test, LinearAlgebra, SparseArrays

function _ggf()
    p, n = 2, 3
    D = zeros(p * n, p)
    for obs in 1:n, trait in 1:p
        D[(obs - 1) * p + trait, trait] = 1.0
    end
    beta = [1.0, 2.0]
    response = [1.1 2.2 3.3; 2.0 3.1 4.2]
    term = GroupingTerm(:unit; mode = :indep)
    params = vcat(beta, log(0.5), log(0.6), log(0.5))
    return GroupedGaussianFit(beta, 0.5, [Matrix(Diagonal([0.25, 0.36]))], [term],
        params, -5.0, true, 1e-8, 0.1, true, 10, :converged, D,
        Union{String, Symbol}["trait1", "trait2"],
        ["trait1", "trait2", "unit.log_sd[1]", "unit.log_sd[2]", "log_sigma_eps"],
        (p, n), response, [sparse(collect(1:n), collect(1:n), ones(n), n, n)])
end

@testset "cov 1125y11 destination B postfit" begin
    fit = _ggf()
    p = 2

    @testset "population predict shapes" begin
        μ = GLLVModels.destination_b_population_predict(fit)
        @test size(μ) == (p, 3)
        @test GLLVModels.predict(fit) == μ
        @test GLLVModels.fitted(fit) == μ
        @test GLLVModels.residuals(fit) == fit.response .- μ
        D3 = reshape(fit.mean_design, p, 3, p)
        μ3 = GLLVModels.destination_b_population_predict(fit, D3)
        @test μ3 == μ
    end

    @testset "type aliases for Gaussian" begin
        for typ in (:link, :response, :mean)
            @test GLLVModels.destination_b_population_predict(fit; type = typ) ==
                GLLVModels.destination_b_population_predict(fit)
        end
    end

    @testset "StatsAPI on GroupedGaussianFit" begin
        @test GLLVModels.coef(fit) == fit.beta
        @test GLLVModels.loglikelihood(fit) == fit.loglik
        @test GLLVModels.nobs(fit) == length(vec(fit.response))
    end

    @testset "design validation" begin
        @test_throws DimensionMismatch GLLVModels.destination_b_population_predict(
            fit, fit.mean_design[1:(end - 1), :])
        bad = fill(NaN, 6, 2)
        @test_throws ArgumentError GLLVModels.destination_b_population_predict(fit, bad)
        @test_throws ArgumentError GLLVModels.destination_b_population_predict(fit, "not a matrix")
        @test_throws ArgumentError GLLVModels.destination_b_population_predict(fit; type = :blup)
    end
end