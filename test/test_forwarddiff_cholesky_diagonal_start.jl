# Regression for issue #1215: ForwardDiff 0.10.x on Julia ≥ 1.12 mis-differentiates
# `cholesky(Symmetric(A)) \ b` when A is exactly diagonal at the start (init_theta_rr).
using Test, GLLVModels, LinearAlgebra, ForwardDiff, Random

function _central_grad(f, x::AbstractVector; h = 1e-6)
    g = similar(x)
    @inbounds for i in eachindex(x)
        e = zero(x)
        e[i] = h
        g[i] = (f(x .+ e) - f(x .- e)) / (2h)
    end
    return g
end

@testset "ForwardDiff Cholesky diagonal start (issue #1215)" begin
    p, n, K = 8, 120, 2
    rng = MersenneTwister(1215)
    y = randn(rng, p, n)
    θ_rr = GLLVModels.init_theta_rr(p, K)
    σ = 0.9

    densenll(θ) =
        -GLLVModels.gaussian_marginal_loglik(y, GLLVModels.unpack_lambda(θ, p, K), σ)
    profnll(θ) = GLLVModels.gaussian_profile_nll(θ, y;
        spec = (q = 0, p = p, K_B = K, K_W = 0, has_diag = false,
                K_phy = 0, has_phy_unique = false),
        profile_beta = false)

    for f in (densenll, profnll)
        gfd = ForwardDiff.gradient(f, θ_rr)
        gcd = _central_grad(f, θ_rr)
        scale = max(1.0, maximum(abs.(gfd)))
        @test maximum(abs.(gfd .- gcd)) / scale < 1e-6
    end
end