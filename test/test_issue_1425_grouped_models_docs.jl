# Issue #1425: grouped-models.md must not imply bare `predict(fit)` on ungrouped
# Laplace family fits, and its runnable grouped examples must execute.

using Test
using GLLVModels
using Distributions
using Random

const _GROUPED_MODELS_DOC = joinpath(@__DIR__, "..", "docs", "src", "grouped-models.md")

@testset "issue 1425 grouped-models postfit documentation" begin
    doc = read(_GROUPED_MODELS_DOC, String)

    @test occursin("GroupedGaussianFit", doc)
    @test occursin("GroupedNonGaussianFit", doc)
    @test occursin("predict(fit, Y)", doc)
    @test occursin("fitted(fit, Y)", doc)
    @test occursin("residuals(fit, Y)", doc)

    # Issue repro: ungrouped Laplace Poisson requires the training matrix.
    Random.seed!(112_601)
    Y = rand(0:8, 4, 40)
    fit = fit_gllvm(Y; family = Poisson(), K = 1, iterations = 50)
    @test fit isa PoissonFit
    @test_throws MethodError predict(fit)
    @test_throws MethodError fitted(fit)
    @test_throws MethodError residuals(fit)
    @test predict(fit, Y) isa AbstractMatrix
    @test fitted(fit, Y) isa AbstractMatrix
    @test residuals(fit, Y) isa AbstractMatrix

    # Non-Gaussian grouped doc snippet: one-argument in-sample postfit.
    Yg = [1.0 3 2 4 1 2; 2 1 3 2 4 1]
    unit = repeat([:a, :b, :c], inner = 2)
    terms = [GroupingTerm(:unit; mode = :indep, common = true)]
    fitg = fit_gllvm(Yg; family = Poisson(), grouping = terms, unit = unit, iterations = 50)
    @test fitg isa GroupedNonGaussianFit
    @test predict(fitg) isa AbstractMatrix
    @test fitted(fitg) isa AbstractMatrix
    @test residuals(fitg) isa AbstractMatrix

    # Gaussian grouped interface snippet from the doc page.
    Yn = reshape([1.0, 1.2, 0.9, 1.1, 3.0, 2.8, 3.1, 2.9], 1, :)
    unit_n = repeat(1:4, inner = 2)
    fitn = fit_gllvm(Yn; grouping = [GroupingTerm(:unit; mode = :indep, common = true)],
        unit = unit_n, iterations = 80)
    report = summary(fitn, Yn)
    @test report.inference_status isa Symbol
    @test predict(fitn) isa AbstractMatrix
    @test vcov(fitn) isa AbstractMatrix
end