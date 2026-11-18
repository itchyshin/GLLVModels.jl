# Regression (wave11 scratch): the Gaussian profile quadratic form r'Ã⁻¹r must stay
# nonnegative and accurate when the loadings nearly span a large, unabsorbed mean
# (no X, data mean 1e6, spread 1e-3). The subtractive Woodbury form went negative
# in Float64 and log() threw a DomainError at src/profile.jl (gaussian_profile_nll).
using Test, GLLVModels, LinearAlgebra, Random

@testset "profile quad: no cancellation with a large unabsorbed mean" begin
    G = GLLVModels
    p, n = 6, 30
    rng = MersenneTwister(20261108)
    λt = randn(rng, p); z = randn(rng, n)
    Y = 1e-3 .* (λt * z' .+ 0.5 .* randn(rng, p, n)) .+ 1e6
    spec = (q = 0, p = p, K_B = 1, K_W = 0, has_diag = false, K_phy = 0, has_phy_unique = false)
    # BigFloat reference of the same profile NLL (closed form, K = 1, d̃ = 1)
    function nll_big(θ)
        λ = big.(θ); y = big.(Y); A = 1 + dot(λ, λ)
        quad = sum(abs2, y) - sum(abs2, λ' * y) / A
        np = n * p
        return Float64(big(0.5) * (np * log(2 * big(π)) + np + np * log(quad / np) + n * log(A)))
    end
    for lα in (6.0, 7.0, 8.0, 9.0)
        θ = fill(10.0^lα, p)
        v = G.gaussian_profile_nll(θ, Y; spec = spec, profile_beta = false)
        @test isfinite(v)
        @test isapprox(v, nll_big(θ); rtol = 1e-8)
    end
    @test isfinite(G.fit_gaussian_gllvm(Y; K = 1).logLik)
end

@testset "profile quad guard: healthy values pass through bit-for-bit" begin
    G = GLLVModels
    rng = MersenneTwister(1)
    resid = randn(rng, 4, 5); L = randn(rng, 4, 1); d = fill(1.0, 4)
    z = [0.3]
    # healthy: quad_sub well above 1e-4*T -> returned unchanged (===)
    @test G._quad_cancel_guard(2.5, 3.0, resid, L, z, d) === 2.5
    # cancellation: quad_sub tiny or negative relative to T -> stable form, nonnegative
    for qs in (1e-9, 0.0, -1e-3)
        v = G._quad_cancel_guard(qs, 3.0, resid, L, z, d)
        e = resid .- L * z
        @test v == sum(d .* e .* e) + sum(abs2, z)
        @test v >= 0
    end
end
