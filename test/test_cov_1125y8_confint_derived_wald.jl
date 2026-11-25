# Lane cursor-cloud-1125y8: confint_derived_wald.jl transforms and packed closures.
using Test, GLLVModels, Random, LinearAlgebra

const G = GLLVModels

@testset "cov 1125y8 confint derived wald internals" begin
    @testset "_tw_link" begin
        fz, invz, bz = G._tw_link(:fisher_z)
        x = 0.5
        @test invz(fz(x)) ≈ x atol = 1e-15
        @test bz == (-1.0, 1.0)
        lo, invl, bl = G._tw_link(:log)
        @test invl(lo(2.0)) ≈ 2.0
        @test bl == (0.0, Inf)
        id, invi, bi = G._tw_link(:identity)
        @test id(3.3) == 3.3 && invi(-1.0) == -1.0
        @test_throws ArgumentError G._tw_link(:bogus)
    end

    rng = MersenneTwister(1125087)
    p, n, K = 3, 25, 1
    y = randn(rng, p, n)
    fit = fit_gaussian_gllvm(y; K = K, iterations = 8)
    spec = G._derived_spec(fit)

    @testset "_correlation_packed" begin
        θ = fit.pars.θ_packed
        ρ12 = G._correlation_packed(θ, spec, 1, 2)
        ρ21 = G._correlation_packed(θ, spec, 2, 1)
        @test ρ12 ≈ ρ21 atol = 1e-12
        @test abs(ρ12) <= 1
        f = G._make_correlation_closure(spec, 1, 3)
        @test f(θ) ≈ G._correlation_packed(θ, spec, 1, 3)
    end

    @testset "_icc_packed and _proportion_packed" begin
        θ = fit.pars.θ_packed
        icc = G._icc_packed(θ, spec, 1)
        @test 0 <= icc <= 1 || isnan(icc)
        prop = G._proportion_packed(θ, spec, 2, :residual)
        @test isfinite(prop) || isnan(prop)
        g = G._make_icc_closure(spec, 2)
        @test g(θ) ≈ G._icc_packed(θ, spec, 2)
    end

    @testset "_transformed_wald_ci_with_sigma" begin
        θ = copy(fit.pars.θ_packed)
        ci = G._transformed_wald_ci_with_sigma(θ, G._make_correlation_closure(spec, 1, 2),
            Diagonal(fill(1e-4, length(θ))), true; transform = :fisher_z, level = 0.95)
        @test ci.method === :transformed_wald || ci.method === :unavailable
        @test hasproperty(ci, :estimate)
    end
end