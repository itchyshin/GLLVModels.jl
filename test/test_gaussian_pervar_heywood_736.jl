# Per-variable Gaussian fit near a Heywood boundary reports a converged fit (refs #736).
#
# With `pervar = true` the default route is EM, which converges only linearly as a unique
# variance tends to 0 and so exhausted its iteration cap with `converged = false`, short of the
# optimum. The EM result is now polished by L-BFGS on the log-variance scale, where the gradient
# vanishes at the boundary. No seeded magnitudes are asserted: only structural properties and
# that the polish never loses likelihood against the EM point.

using GLLVModels, Test, Random, Distributions, LinearAlgebra, StableRNGs

@testset "pervar Heywood boundary converges (#736)" begin
    rng = StableRNG(736)
    p, n, K = 6, 77, 1
    λ = [0.9, 0.5, 1.0, 0.7, 0.6, 0.4]
    ψtrue = [0.3, 0.9, 1e-6, 0.3, 0.4, 0.6]   # trait 3 is a Heywood case
    η = randn(rng, 1, n)
    Y = λ * η .+ sqrt.(ψtrue) .* randn(rng, p, n)

    fit = fit_gaussian_pervar_gllvm(Y; K = K)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test all(>(0), fit.ψ²)
    @test minimum(fit.ψ²) < 1e-2          # the boundary trait is recovered

    # The polish must be at least as good as the EM point it started from.
    Yc = Y .- sum(Y, dims = 2) ./ n
    Λ0, _ = ppca_init(Yc, K)
    col_var = vec(sum(abs2, Yc, dims = 2)) ./ (n - 1)
    _, _, ll_em, _, _ = em_fa(Yc, K; λ_init = Λ0, ψ_init = max.(0.5 .* col_var, 1e-3),
                              tol = 1e-8, max_iter = 2000)
    @test fit.loglik >= ll_em - 1e-8

    # Direct check: likelihood at the returned parameters matches the reported one.
    ll = gaussian_pervar_marginal_loglik(Yc, fit.Λ, fit.φ²)
    @test isapprox(ll, fit.loglik; atol = 1e-6)
end
