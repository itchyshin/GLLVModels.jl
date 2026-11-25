# Lane 1125y5: postfit_tables.jl rotation, cross-rho, tidy/summary, predict_missing.
using Test, GLLVModels, Random, LinearAlgebra, Statistics, Distributions
const G = GLLVModels

@testset "1125y5 postfit_tables helpers" begin
    @test G._cross_rho_interp(0.0, 0.0, 1.0, 1.0, 0.5) ≈ 0.5
    r = collect(-1.0:0.2:1.0)
    dd = (r .- 0.1) .^ 2
    out = profile_cross_rho_ci(r, dd; level = 0.95)
    @test out.estimate ≈ 0.1 atol = 0.15
    @test out.level == 0.95

    Λ = [0.9 0.1; 0.2 0.8; -0.3 0.4]
    Rv, Tr = G._varimax_rotation(Λ; normalize = true)
    @test size(Rv) == size(Λ)
    @test all(isfinite, Rv)
    Pv, Tp = G._promax_rotation(Λ; m = 3)
    @test size(Pv) == size(Λ)

    sim = G.simulate_unit_trait(MersenneTwister(3); n_units = 4, n_obs_per_unit = 2,
        n_traits = 3, K_B = 1, K_W = 1)
    @test size(sim.Y) == (3, 8)
    @test length(sim.individual) == 8
end

@testset "1125y5 postfit_tables accessors" begin
    Random.seed!(11)
    p, K, n = 4, 1, 30
    y = randn(p, n)
    fit = fit_gaussian_gllvm(y; K = K, iterations = 12)
    rows = tidy(fit, y; effects = (:fixed, :ran_pars))
    @test rows isa Vector
    sm = summary(fit, y)
    @test sm isa G.GllvmSummary
    @test occursin("GllvmSummary", sprint(show, "text/plain", sm))
    st = G._gllvm_se_status(fit, y)
    @test st in (:ok, :sdreport_error, :incomplete)

    rot = rotate_loadings(fit, y; method = :varimax)
    @test size(rot.Lambda) == (p, K)
    ext = extract_rotated_loadings_table(fit, y; method = :none)
    @test ext isa NamedTuple

    Σs = fit.pars.Λ * fit.pars.Λ'
    mods = extract_coevolution_modules(Σs; row_traits = 1:p, col_traits = 1:p)
    @test haskey(mods, :modules)

    @test_throws DimensionMismatch predict_missing(fit, y[1:(end - 1), :])

    @test_throws ArgumentError G.imputed(fit, y[:, 1])
    @test G._tidy_fixed_rows(fit, y; conf_int = false, conf_level = 0.95) isa Vector
    @test G._tidy_ran_pars_rows(fit, y; conf_int = false, conf_level = 0.95) isa Vector
end