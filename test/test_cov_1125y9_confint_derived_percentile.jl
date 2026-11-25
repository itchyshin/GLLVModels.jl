# Lane cursor-cloud-1125y9: confint_derived.jl percentile helper and unpack slices.
using Test, GLLVModels, Random, LinearAlgebra, Distributions

const G = GLLVModels

@testset "1125y9 confint derived internals" begin
    @testset "_derived_percentile" begin
        v = [10.0, 20.0, 30.0, 40.0]
        @test G._derived_percentile(v, 0.0) ≈ 10.0
        @test G._derived_percentile(v, 1.0) ≈ 40.0
        @test G._derived_percentile(v, 0.5) ≈ 25.0
        @test G._derived_percentile([7.0], 0.25) ≈ 7.0
        @test_throws ArgumentError G._derived_percentile(Int[], 0.5)
        @test_throws ArgumentError G._derived_percentile(v, 1.1)
    end

    rng = MersenneTwister(1125092)
    p, n, K = 4, 18, 1
    Y = randn(rng, p, n)
    fit = fit_gllvm(Y; family = Normal(), K = K, has_diag = true, iterations = 10)
    spec = G._derived_spec(fit)
    θ = fit.pars.θ_packed
    u = G._derived_unpack(θ, spec)

    @test u.β !== nothing && length(u.β) == p
    @test u.σ_eps > 0
    @test u.σ²_B !== nothing && length(u.σ²_B) == p
    @test all(>(0), u.σ²_B)
    @test size(u.Λ_B) == (p, K)

    @testset "_sigma_y_site_from_unpacked symmetry" begin
        Σ = G._sigma_y_site_from_unpacked(u, spec)
        @test Σ ≈ (Σ + Σ') / 2 atol = 1e-12
        @test all(x -> x > 0, diag(Σ))
    end

    @testset "proportions sum to one" begin
        parts = [:shared, :unique_W, :unique_B, :unique_Wd, :residual]
        acc = zeros(p)
        for part in parts
            acc .+= G.proportions(fit; component = part)
        end
        @test acc ≈ ones(p) rtol = 1e-10
    end

    @testset "communality in unit interval" begin
        c = G.communality(fit)
        @test length(c) == p
        @test all(x -> 0 <= x <= 1 || isnan(x), c)
    end
end