# Lane cursor-cloud-1125y8: source_covariance.jl _gaussian_source_loglik gates.
using Test, GLLVModels, LinearAlgebra, Random

const G = GLLVModels

@testset "cov 1125y8 gaussian source loglik" begin
    rng = MersenneTwister(1125086)
    p, n = 3, 6
    Y = randn(rng, p, n)
    beta = randn(rng, p)
    loading = randn(rng, p, 1)
    C = [1.0 0.2; 0.2 1.0]
    groups = repeat(1:2; inner = 3)
    ll = G._gaussian_source_loglik(Y, beta, loading, [C], reshape(groups, n, 1), 0.5)
    @test isfinite(ll) && ll < 0

    @testset "validation errors" begin
        @test_throws ArgumentError G._gaussian_source_loglik(Y, beta, loading, [C], reshape(groups, n, 1), 0.0)
        @test_throws DimensionMismatch G._gaussian_source_loglik(Y, beta[1:end-1], loading, [C], reshape(groups, n, 1), 0.5)
        @test_throws Exception G._gaussian_source_loglik(Y, beta, loading, [C], zeros(n - 1, 1), 0.5)
        @test_throws ArgumentError G._gaussian_source_loglik(Y, beta, loading, [C], fill(99, n, 1), 0.5)
        @test_throws ArgumentError G._gaussian_source_loglik(fill(NaN, p, n), beta, loading, [C], reshape(groups, n, 1), 0.5)
        bad = [1.0 3.0; 2.0 1.0]
        @test_throws ArgumentError G._gaussian_source_loglik(Y, beta, loading, [bad], reshape(groups, n, 1), 0.5)
        @test_throws ArgumentError G._gaussian_source_loglik(Y, beta, randn(p, 3), [C, C, C], hcat(groups, groups, groups), 0.5)
    end

    @testset "two-source additive density" begin
        C2 = [1.5 0.1; 0.1 1.1]
        load2 = hcat(loading, 0.3 .* randn(rng, p, 1))
        g2 = hcat(groups, repeat(2:-1:1; inner = 3))
        ll2 = G._gaussian_source_loglik(Y, beta, load2, [C, C2], g2, 0.4)
        @test isfinite(ll2)
        @test ll2 != ll
    end
end