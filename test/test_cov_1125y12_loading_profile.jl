# Lane cursor-cloud-1125y12 — loading_profile_confirmatory.jl guards
using GLLVModels, Test, Random

@testset "cov 1125y12 loading_profile confirmatory guards" begin
    rng = MersenneTwister(11251207)
    Y = randn(rng, 4, 10)
    p, K = 4, 2
    M = fill(NaN, p, K)
    M[1, 1] = 1.0
    fit_expl = fit_gaussian_gllvm(Y; K, iterations = 60, g_tol = 1e-4)
    @test_throws ArgumentError loading_profile(fit_expl; y = Y)
    fit_conf = fit_gaussian_gllvm(Y; K, lambda_constraint = M, iterations = 80, g_tol = 1e-4)
    @test_throws ArgumentError loading_profile(fit_conf; level = :W, y = Y)
    @test_throws ArgumentError loading_profile(fit_conf; n_grid = 2, y = Y)
    @test_throws ArgumentError loading_profile(fit_conf; conf_level = 1.5, y = Y)
    @test_throws ArgumentError loading_profile(fit_conf; grid_extent = 0.0, y = Y)
    # all pinned — no free entries
    Mall = ones(p, K)
    fit_pin = fit_gaussian_gllvm(Y; K, lambda_constraint = Mall, iterations = 40, g_tol = 1e-3)
    @test_throws ArgumentError loading_profile(fit_pin; y = Y, n_grid = 5, grid_extent = 2.0)
end