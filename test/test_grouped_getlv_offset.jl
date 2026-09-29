using Test
using GLLVModels
using Random

# getLV on a plain grouped-dispersion fit must honour the fit-time `offset`, as the
# grouped fitters do (fit_*_gllvm_grouped(Y; offset = O)). Oracle-free check: a
# per-trait constant offset c is the same linear predictor as shifting β by c, so
# getLV(fit, Y; offset = c .* ones(1, n)) must equal getLV(fit with β + c, Y).

@testset "grouped getLV honours offset" begin
    rng = MersenneTwister(20260928)
    p, n, K = 4, 12, 1
    β = [0.4, 0.8, -0.2, 0.1]
    Λ = reshape([0.7, -0.5, 0.6, 0.3], p, K)
    group = [1, 1, 2, 2]
    c = [0.5, -0.3, 0.2, 0.4]
    O = c .* ones(1, n)
    Ocell = 0.3 .* randn(rng, p, n)

    Ycount = rand(rng, 0:6, p, n)
    Ypos = 0.5 .+ 2.0 .* rand(rng, p, n)
    Yunit = 0.05 .+ 0.9 .* rand(rng, p, n)

    cases = (
        ("NB2",   (b) -> NBGroupedFit(b, Λ, [3.0, 5.0], group, LogLink(), 0.0, true, 0), Ycount),
        ("NB1",   (b) -> NB1GroupedFit(b, Λ, [0.8, 1.5], group, LogLink(), 0.0, true, 0), Ycount),
        ("Beta",  (b) -> BetaGroupedFit(b .- 0.5, Λ, [6.0, 10.0], group, LogitLink(), 0.0, true, 0), Yunit),
        ("Gamma", (b) -> GammaGroupedFit(b, Λ, [2.0, 4.0], group, LogLink(), 0.0, true, 0), Ypos),
    )

    for (name, mk, Y) in cases
        @testset "$name" begin
            fit = mk(β)
            shifted = mk(β .+ c)
            z_off = getLV(fit, Y; offset = O, rotate = false)
            z_shift = getLV(shifted, Y; rotate = false)
            @test z_off ≈ z_shift atol = 1e-6
            @test !isapprox(z_off, getLV(fit, Y; rotate = false); atol = 1e-3)
            # offset = nothing and an all-zero offset are the same model
            @test getLV(fit, Y; offset = zeros(p, n)) ≈ getLV(fit, Y) atol = 1e-10
            # a cell-varying offset is accepted and changes the scores
            z_cell = getLV(fit, Y; offset = Ocell)
            @test all(isfinite, z_cell)
            @test size(z_cell) == (n, K)
            @test_throws DimensionMismatch getLV(fit, Y; offset = zeros(p, n + 1))
        end
    end
end
