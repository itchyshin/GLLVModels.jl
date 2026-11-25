# Lane cursor-cloud-1125y11 — grouped_nongaussian_postfit.jl prediction surface
using GLLVModels, Test, LinearAlgebra, SparseArrays

function _gnp_mini(kind::Symbol)
    p, n = 2, 4
    D = zeros(p * n, p)
    for obs in 1:n, trait in 1:p
        D[(obs - 1) * p + trait, trait] = 1.0
    end
    beta = [0.5, -0.2]
    trials = fill(5.0, p, n)
    response = ones(p, n)
    family, disp = if kind === :poisson
        GLLVModels.Poisson(), nothing
    elseif kind === :binomial
        GLLVModels.Binomial(), nothing
    else
        GLLVModels.Beta(2.0, 1.0), 2.0
    end
    params = kind === :beta ? vcat(beta, [0.0, 0.0, log(2.0)]) : vcat(beta, [0.0, 0.0])
    labels = kind === :beta ? ["t1", "t2", "u1", "u2", "phi"] : ["t1", "t2", "u1", "u2"]
    return GroupedNonGaussianFit(beta, family, disp, [Matrix(Diagonal([0.1, 0.2]))], [GroupingTerm(:unit; mode = :indep)],
        params, -3.0, true, 1e-7, 0.1, true, 5, :converged, D,
        Union{String, Symbol}["trait1", "trait2"], labels, (p, n), response, trials,
        [sparse(collect(1:n), collect(1:n), ones(n), n, n)], kind, :none, :ok)
end

@testset "cov 1125y11 grouped nongaussian predict" begin
    η2 = reshape(_gnp_mini(:poisson).mean_design * _gnp_mini(:poisson).beta, 2, 4)
    for kind in (:poisson, :binomial, :beta)
        fit = _gnp_mini(kind)
        link = GLLVModels.grouped_nongaussian_zero_effect_predict(fit; type = :link)
        resp = GLLVModels.grouped_nongaussian_zero_effect_predict(fit; type = :response)
        @test size(link) == (2, 4)
        @test all(isfinite, link)
        @test all(isfinite, resp)
        if kind === :poisson
            @test resp ≈ exp.(link) rtol = 1e-12
        elseif kind === :binomial
            @test resp ≈ fit.trials .* (1.0 ./ (1.0 .+ exp.(-link))) rtol = 1e-12
        else
            @test resp ≈ 1.0 ./ (1.0 .+ exp.(-link)) rtol = 1e-12
        end
    end

    fitB = _gnp_mini(:binomial)
    Dnew = zeros(8, 2)
    for obs in 1:4, trait in 1:2
        Dnew[(obs - 1) * 2 + trait, trait] = 1.0
    end
    Nnew = fill(4.0, 2, 4)
    μ = GLLVModels.grouped_nongaussian_zero_effect_predict(fitB, Dnew; N = Nnew)
    @test size(μ) == (2, 4)

    @test_throws ArgumentError GLLVModels.grouped_nongaussian_zero_effect_predict(
        _gnp_mini(:poisson); type = :blup)
    @test_throws ArgumentError GLLVModels.grouped_nongaussian_zero_effect_predict(
        fitB, Dnew)
    @test_throws DimensionMismatch GLLVModels.grouped_nongaussian_zero_effect_predict(
        fitB, Dnew; N = fill(1.0, 2, 3))
    @test_throws ArgumentError GLLVModels._grouped_nongaussian_prediction_design(
        fitB, "not a matrix")
end