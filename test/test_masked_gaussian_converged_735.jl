# Masked Gaussian fits report converged = true only at a stationary point (refs #735).
#
# The masked/offset route of fit_gaussian_gllvm took `converged` from Optim.converged(res),
# which is also true for an x- or f-only stop. A loose `x_tol` forces that stop early, with
# the gradient far above `g_tol` and the log-likelihood short of the optimum. The fit must
# now continue on the gradient test and report converged only if the final gradient meets
# `g_tol`. No seeded magnitudes are asserted: only structural properties and agreement with
# a tightly converged fit of the same data.

using GLLVModels, Test, Random, Distributions, Optim, StableRNGs

@testset "masked Gaussian converged needs the gradient test (#735)" begin
    rng = StableRNG(735)
    p, n, K = 6, 80, 1
    Λ = reshape(collect(range(0.9, 0.3; length = p)), p, K)
    Y = Λ * randn(rng, K, n) .+ 0.5 .* randn(rng, p, n) .+ collect(1.0:p)
    mask = rand(rng, p, n) .> 0.15
    for t in 1:p; mask[t, 1:5] .= true; end
    Y[.!mask] .= 0.0
    g_tol = 1e-6

    ref = fit_gaussian_gllvm(Y; K = K, mask = mask, x_tol = 1e-12, g_tol = g_tol)
    loose = fit_gaussian_gllvm(Y; K = K, mask = mask, x_tol = 0.5, g_tol = g_tol)

    for f in (ref, loose)
        @test f.converged
        @test Optim.g_residual(f.optim_result) <= g_tol
    end
    @test isapprox(loose.logLik, ref.logLik; atol = 1e-6)
end
