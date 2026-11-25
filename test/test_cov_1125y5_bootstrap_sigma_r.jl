# Lane 1125y5: bootstrap_sigma_r.jl show, guards, tiny replicate.
using Test, GLLVModels, Random, LinearAlgebra, Statistics, Logging
const G = GLLVModels

@testset "1125y5 bootstrap_Sigma_r guards and show" begin
    @test G._bsig_normalise_level(:unit) == [:B]
    @test G._bsig_normalise_level(:unit_obs) == [:W]
    lv = G._bsig_normalise_level([:B, :unit, :W])
    @test lv == [:B, :W]
    @test_throws ArgumentError G._bsig_normalise_level(:bogus)
    @test_throws ArgumentError G._bsig_normalise_level(Symbol[])

    wh = G._bsig_check_what(:Sigma)
    @test wh == [:Sigma]
    @test G._bsig_check_what([:R, :ICC]) == [:R, :ICC]
    @test_throws ArgumentError G._bsig_check_what(:nope)
    @test_throws ArgumentError G._bsig_check_what(Symbol[])

    ΛB = reshape([0.8, 0.6, 0.4], 3, 1)
    f2 = G.TwoLevelFit(ΛB, [0.1, 0.2, 0.3], zeros(3, 1), fill(0.5, 3),
        Matrix(ΛB * ΛB' + Diagonal([0.1, 0.2, 0.3])),
        zeros(3, 3), 10, -1.0, true, 1)
    @test G._bsig_available_tiers(f2) == [:B, :W]
    Σ, shared = G._bsig_pair(f2, :B)
    @test size(Σ) == (3, 3)
    @test shared !== nothing
    @test G._bsig_icc(f2.Σ_B, f2.Σ_W) isa Vector{Float64}

    Random.seed!(5)
    p, n, K = 4, 40, 1
    y = randn(p, n)
    fit = fit_gaussian_gllvm(y; K = K, iterations = 8)
    refit = (yb; kwargs...) -> fit_gaussian_gllvm(yb; K = K, iterations = 5)
    out = with_logger(NullLogger()) do
        G.bootstrap_Sigma_r(fit; n_boot = 6, _refit = refit, what = :communality, level = :unit,
            progress = false, y = y, n_sites = n)
    end
    @test out isa G.BootstrapSigmaR
    @test out.n_boot == 6
    @test !isempty(keys(out.point_est))
    @test occursin("BootstrapSigmaR", sprint(show, out))
    @test occursin("BootstrapSigmaR", sprint(show, "text/plain", out))
end