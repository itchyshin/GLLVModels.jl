# Lane cursor-cloud-1125y9: aghq_grid.jl GH rule and logsumexp helper.
using Test, GLLVModels, LinearAlgebra

const G = GLLVModels

@testset "1125y9 aghq grid misc" begin
    @testset "_aghq_gh_normal moments" begin
        nodes, w = G._aghq_gh_normal(3)
        @test sum(w) ≈ 1.0 atol = 1e-14
        @test sum(w .* nodes .^ 2) ≈ 1.0 atol = 1e-12
        n1, w1 = G._aghq_gh_normal(1)
        @test n1 == [0.0] && w1 == [1.0]
        @test_throws ArgumentError G._aghq_gh_normal(0)
    end

    @testset "aghq_grid sizes" begin
        g = G.aghq_grid(2, 3)
        @test g.d == 2 && g.k == 3
        @test size(g.nodes) == (9, 2)
        @test length(g.logw) == 9
        @test G.aghq_grid_ok(g)
    end

    @testset "_aghq_logsumexp" begin
        x = [-1000.0, -1000.0, 0.0]
        @test isfinite(G._aghq_logsumexp(x))
        @test G._aghq_logsumexp([-Inf, -Inf]) == -Inf
        @test G._aghq_logsumexp([1.0, 2.0]) ≈ log(exp(1) + exp(2)) rtol = 1e-12
    end
end