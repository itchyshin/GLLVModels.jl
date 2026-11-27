# refs coverage 27 Nov (w4 lane): ranks 7–9 uncovered — grouped_dispersion.jl, profile.jl, twopart.jl
using Test, GLLVModels, LinearAlgebra, Random, Distributions

const G = GLLVModels

@testset "profile: Σ_phy guard and β GLS profile (#coverage-1127-w4)" begin
    @testset "missing Σ_phy when phy block present" begin
        spec = (q = 0, p = 2, K_B = 1, K_W = 0, has_diag = false, K_phy = 1, has_phy_unique = false)
        θ = G.pack_lambda([0.2 0.15]')
        y = randn(2, 6)
        @test_throws ArgumentError G.gaussian_profile_nll(θ, y; spec = spec, Σ_phy = nothing)
    end

    @testset "β profile-out (Woodbury GLS) on stacked X" begin
        rng = MersenneTwister(20261127)
        p, n, q = 3, 10, 2
        y = randn(rng, p, n)
        X = zeros(p, n, q)
        for j in 1:q
            X[:, :, j] .= randn(rng, p, n)
        end
        L_B = reshape([0.25, -0.15, 0.12], p, 1)
        spec = (q = q, p = p, K_B = 1, K_W = 0, has_diag = true)
        τB = fill(0.05, p)
        τW = fill(0.02, p)
        params = vcat(0.5 .* log.(τB), 0.5 .* log.(τW), G.pack_lambda(L_B))
        @test length(params) == G.profile_nparams(spec; profile_beta = true)
        nll_prof = G.gaussian_profile_nll(params, y; spec = spec, X = X, profile_beta = true)
        @test isfinite(nll_prof)
        rec = G.profile_recover(params, y; spec = spec, X = X, profile_beta = true)
        @test isfinite(rec.logLik)
        @test length(rec.β) == q
        params_full = vcat(rec.β, params)
        nll_full = G.gaussian_profile_nll(params_full, y; spec = spec, X = X, profile_beta = false)
        @test nll_prof ≈ nll_full rtol = 1e-10 atol = 1e-8
    end
end

@testset "grouped_dispersion: post-fit introspection and site-mode (#coverage-1127-w4)" begin
    β = [0.2, -0.1]
    Λ = reshape([0.35, -0.2], 2, 1)
    g = [1, 1]
    fams = fill(Poisson(), 2)

    f_nb = G.NBGroupedFit(β, Λ, [3.0], g, G.LogLink(), -12.0, true, 4, :observed,
                          [true], nothing)
    s_nb = sprint(show, f_nb)
    @test occursin("NBGroupedFit", s_nb)
    @test occursin("dispersion at boundary", s_nb)
    @test G._loglik(f_nb) == -12.0
    @test G._nparams(f_nb) == 2 + G.rr_theta_len(2, 1) + 1

    f_beta = G.BetaGroupedFit(β, Λ, [8.0, 9.0], [1, 2], G.LogitLink(), -8.0, false, 2)
    s_beta = sprint(show, f_beta)
    @test occursin("NOT CONVERGED", s_beta)

    γ = [0.1]
    γ_fixed = [false]
    Xstub = reshape([1.0, 0.5], 2, 1, 1)
    f_cov = G.NBGroupedCovFit(β, γ, γ_fixed, Λ, [2.5], g, G.LogLink(), -9.5, true, 3)
    s_cov = sprint(show, f_cov)
    @test occursin("NBGroupedCovFit", s_cov)
    @test G._loglik(f_cov) == -9.5
    @test G._nparams(f_cov) == 2 + 1 + G.rr_theta_len(2, 1) + 1

    Y = [2 1; 0 3]
    @test_throws ArgumentError G._grouped_getLV(Y, Λ, β, G.LogLink(), fams; N = ones(Int, 3, 2))

    z, ok = G._grouped_site_mode(fams, Y[:, 1], fill(1, 2), Λ, β, G.LogLink())
    @test ok
    @test length(z) == 1

    # Non-NB/Gamma/Beta family vector falls through to generic Laplace mode (line 302).
    z2 = G._grouped_laplace_mode(fams, Y[:, 1], fill(1, 2), Λ, β, G.LogLink(); maxiter = 5)
    @test length(z2) == 1
end

@testset "twopart: per-trait dispatch and fit shows (#coverage-1127-w4)" begin
    fams = [G.ZIPoisson(), G.HurdleNB(2.0)]
    ηz, ηc, Wc = -0.5, 1.2, 2.5
    @test G._tp_observed_Wc_at(fams, 2, 1, ηc, Wc) == G._tp_observed_Wc(fams[2], 1, ηc, Wc)
    @test isfinite(G._tp_observed_Wc_at(fams, 1, 0, ηz, ηc, Wc))

    βz = [0.1, -0.2]
    βc = [0.05, 0.15]
    Λc = reshape([0.4, -0.3], 2, 1)
    σ = 0.7
    f_dln = G.DeltaLogNormalFit(βz, βc, Λc, σ, -3.2, false, 7)
    @test sprint(show, f_dln) == sprint(show, G.DeltaLogNormalFit(βz, βc, Λc, σ, -3.2, false, 7, :separate))
    s_dln = sprint(show, f_dln)
    @test occursin("DeltaLogNormalFit", s_dln) && occursin("NOT CONVERGED", s_dln)

    f_hpois = G.HurdlePoissonFit(βz, βc, Λc, -4.1, true, 5)
    @test occursin("HurdlePoissonFit", sprint(show, f_hpois))

    f_hnb = G.HurdleNBFit(βz, βc, Λc, 1.5, -5.0, false, 9)
    @test occursin("HurdleNBFit", sprint(show, f_hnb))

    f_dg = G.DeltaGammaFit(βz, βc, Λc, 2.0, -2.8, true, 3, :shared)
    @test occursin("DeltaGammaFit", sprint(show, f_dg))

    f_zip = G.ZIPFit(βz, βc, Λc, -6.0, false, 2)
    @test occursin("NOT CONVERGED", sprint(show, f_zip))
end