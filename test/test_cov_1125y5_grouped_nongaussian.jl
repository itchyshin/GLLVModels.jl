# Lane 1125y5: grouped_nongaussian_fit.jl helpers and small fits.
using Test, GLLVModels, Random, LinearAlgebra, SparseArrays
const G = GLLVModels

@testset "1125y5 grouped non-Gaussian helpers" begin
    @test G._grouped_nongaussian_kind(G.Poisson()) === :poisson
    @test G._grouped_nongaussian_kind(G.Binomial()) === :binomial
    @test G._grouped_nongaussian_kind(G.Beta(3.0, 1.0)) === :beta
    @test G._grouped_nongaussian_kind(G.NegativeBinomial(2.0, 0.5)) === :nb2
    @test_throws ArgumentError G._grouped_nongaussian_kind(G.Beta(2.0, 2.0))
    @test_throws ArgumentError G._grouped_nongaussian_kind(G.NegativeBinomial(2.0, 0.4))
    @test_throws ArgumentError G._grouped_nongaussian_kind(G.Gamma(2.0, 1.0))

    @test G._grouped_nongaussian_dispersion_mode(:beta, :shared) === :shared
    @test G._grouped_nongaussian_dispersion_mode(:poisson, :shared) === :none
    @test_throws ArgumentError G._grouped_nongaussian_dispersion_mode(:beta, :foo)

    @test G._grouped_nongaussian_internal_dispersion_mode(:poisson, :trait) === :none
    @test G._grouped_nongaussian_internal_dispersion_mode(:nb2, :trait) === :trait
    @test_throws ArgumentError G._grouped_nongaussian_internal_dispersion_mode(:nb2, :bogus)

    @test G._grouped_nongaussian_dispersion_count(:beta, :shared, 3) == 1
    @test G._grouped_nongaussian_dispersion_count(:poisson, :none, 5) == 0
    idx = G._grouped_nongaussian_dispersion_indices(2, 1, :nb2, :trait, 2)
    @test first(idx) == 4 && last(idx) == 5

    θ = [0.0, 0.1, 0.2, log(2.0), log(3.0)]
    fams = G._grouped_nongaussian_family(:nb2, θ, 4:5, 2, 4, :trait)
    @test length(fams) == 8
    @test fams[1] isa G.NegativeBinomial

    labels = G._grouped_nongaussian_parameter_labels(["b"], 2,
        [G.GroupingTerm(:unit; mode = :indep, common = true)], :beta, :trait)
    @test "log_phi[1]" in labels && "log_phi[2]" in labels

    loads = G._grouped_laplace_trait_factors(reshape([1.0 2.0; 3.0 4.0], 2, 2), false)
    @test size(loads, 1) == 2
    uniq = G._grouped_laplace_trait_factors(reshape([1.0; 2.0], 2, 1), [0.0, 0.04])
    @test uniq isa Matrix

    ms = G._grouped_moment_log_sd([1.0, 2.0, 2.0, 3.0, 3.0], [1, 1, 2, 2, 3])
    @test isfinite(ms)
end

@testset "1125y5 grouped non-Gaussian tiny fit" begin
    Random.seed!(1125)
    unit = repeat([1, 2]; inner = 3)
    Y = reshape([0, 1, 2, 1, 3, 2], 1, :)
    term = G.GroupingTerm(:unit; mode = :indep, common = true)
    fit = G.fit_grouped_nongaussian(Y; family = G.Poisson(), terms = [term], unit = unit,
        iterations = 25)
    @test fit isa G.GroupedNonGaussianFit
    @test isfinite(fit.loglik) || fit.loglik == -Inf
    @test length(fit.beta) == 1
    buf = sprint(show, "text/plain", fit)
    @test buf isa String && !isempty(buf)
    ci = G.grouped_nongaussian_intervals(Y, fit; unit = unit)
    @test ci.status in (:available, :unavailable)
end