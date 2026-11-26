# Lane cursor-cloud-1125y8: profile.jl sufficient-stats, profile_nparams, _ss_quad.
using Test, GLLVModels, LinearAlgebra, Random, ForwardDiff

const G = GLLVModels

@testset "cov 1125y8 profile internals" begin
    rng = MersenneTwister(1125081)
    p, n = 4, 20

    @testset "profile_nparams counts" begin
        base = (q = 2, p = p, K_B = 1, K_W = 0, has_diag = true, K_phy = 0, has_phy_unique = false)
        @test G.profile_nparams(base; profile_beta = true) ==
            2 * p + G.rr_theta_len(p, 1)
        @test G.profile_nparams(base; profile_beta = false) ==
            base.q + 2 * p + G.rr_theta_len(p, 1)
        phy = merge(base, (K_phy = 1, has_phy_unique = true))
        @test G.profile_nparams(phy; profile_beta = true) ==
            phy.q + 2 * p + G.rr_theta_len(p, 1) + p + G.rr_theta_len(p, 1)
        wide = (q = 0, p = p, K_B = 2, K_W = 1, has_diag = false, K_phy = 0, has_phy_unique = false)
        @test G.profile_nparams(wide) == G.rr_theta_len(p, 2) + G.rr_theta_len(p, 1)
    end

    @testset "_gaussian_suff" begin
        y = randn(rng, p, n)
        s0 = G._gaussian_suff(y, nothing, 0)
        @test s0.q == 0 && s0.C === nothing && size(s0.Syy) == (p, p)
        @test s0.Syy ≈ y * y'
        q = 2
        X = randn(rng, p, n, q)
        s1 = G._gaussian_suff(y, X, q)
        @test s1.q == q
        @test size(s1.C) == (p * p, q)
        @test size(s1.Sxx) == (p * p, q * q)
        @test s1.Syy ≈ s0.Syy
    end

    @testset "_ss_quad matches explicit trace quadratic" begin
        Λ = 0.25 .* randn(rng, p, 1)
        d = 0.4 .+ rand(rng, p)
        M = Λ * Λ' + Diagonal(d)
        Ainv = inv(Symmetric(M))
        q = 2
        X = randn(rng, p, n, q)
        y = randn(rng, p, n)
        β = randn(rng, q)
        suff = G._gaussian_suff(y, X, q)
        direct = sum(Ainv .* suff.Syy)
        for j in 1:q
            Cj = reshape(suff.C[:, j], p, p)
            direct -= 2 * β[j] * sum(Ainv .* Cj')
        end
        Mm = reshape(suff.Sxx' * vec(Ainv), q, q)
        direct += dot(β, Mm * β)
        @test G._ss_quad(Ainv, β, suff) ≈ direct atol = 1e-12
        @test G._ss_quad(Ainv, zeros(q), suff) ≈ sum(Ainv .* suff.Syy) atol = 1e-12
    end

    @testset "gaussian_profile_nll suff keyword agrees" begin
        q = 1
        X = reshape(ones(p, n), p, n, 1)
        y = randn(rng, p, n) .+ 0.5
        spec = (q = q, p = p, K_B = 1, K_W = 0, has_diag = false, K_phy = 0, has_phy_unique = false)
        np = G.profile_nparams(spec; profile_beta = false)
        θ = 0.05 .* randn(rng, np)
        suff = G._gaussian_suff(y, X, q)
        v1 = G.gaussian_profile_nll(θ, y; spec = spec, X = X, profile_beta = false)
        v2 = G.gaussian_profile_nll(θ, y; spec = spec, X = X, profile_beta = false, suff = suff)
        @test v1 ≈ v2 rtol = 0 atol = 1e-10
        g1 = ForwardDiff.gradient(θ -> G.gaussian_profile_nll(θ, y; spec = spec, X = X, profile_beta = false), θ)
        g2 = ForwardDiff.gradient(θ -> G.gaussian_profile_nll(θ, y; spec = spec, X = X, profile_beta = false, suff = suff), θ)
        @test g1 ≈ g2 atol = 1e-10
    end

    @testset "profile_recover returns finite outputs" begin
        y = randn(rng, p, n)
        spec = (q = 0, p = p, K_B = 1, K_W = 0, has_diag = true, K_phy = 0, has_phy_unique = false)
        θ0 = G.init_theta_rr(p, 1)
        log_τ = fill(log(0.3), 2 * p)
        θ = vcat(log_τ, θ0)
        rec = G.profile_recover(θ, y; spec = spec)
        @test isfinite(rec.logLik)
        @test rec.σ_eps > 0
        @test size(rec.Λ_B) == (p, 1)
        @test length(rec.σ²_B) == p
        @test all(>(0), rec.σ²_B)
    end

    @testset "parameter length errors" begin
        y = randn(rng, p, n)
        spec = (q = 0, p = p, K_B = 1, K_W = 0, has_diag = false, K_phy = 0, has_phy_unique = false)
        @test_throws ArgumentError G.gaussian_profile_nll(zeros(1), y; spec = spec)
        @test_throws Exception G.profile_recover(zeros(1), y; spec = spec)
    end
end