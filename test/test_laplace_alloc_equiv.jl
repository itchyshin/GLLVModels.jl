# Focused consistency guard for the buffer-reuse refactor of the Laplace core
# (src/families/laplace.jl). The full existing suite is the real bit-exactness
# anchor (it pins these marginals to machine precision); this file adds two cheap
# checks that specifically catch the failure modes introduced by reusing buffers:
#   1. idempotence — repeated calls to `marginal_loglik_laplace` return EXACTLY the
#      same value (catches a buffer aliased/leaked across calls), and
#   2. site additivity — the total marginal equals the sum of per-site
#      `laplace_loglik_site` values (catches buffer reuse corrupting per-site work
#      within the loop).
# Kept tiny/fast: small p, K, n; fixed (Λ, β); no optimisation.

using GLLVModels, Test, Distributions

@testset "laplace buffer-reuse equivalence" begin
    # Small fixed Poisson problem.
    p, K, n = 4, 2, 5
    Λ = [ 0.6 -0.2;
         -0.3  0.5;
          0.1  0.4;
          0.7 -0.1]
    β = [0.2, -0.4, 0.1, 0.3]
    Yp = [ 1  0  2  1  3;
           0  1  1  0  2;
           2  3  0  1  1;
           1  1  2  4  0]
    Np = ones(Int, size(Yp))

    # (1) idempotence: exact equality across repeated calls (== not ≈).
    m1 = GLLVModels.marginal_loglik_laplace(Poisson(), Yp, Np, Λ, β, LogLink())
    m2 = GLLVModels.marginal_loglik_laplace(Poisson(), Yp, Np, Λ, β, LogLink())
    @test m1 == m2
    @test isfinite(m1)
    # Top-level wrapper agrees with the generic call, also exactly.
    @test GLLVModels.poisson_marginal_loglik_laplace(Yp, Λ, β) == m1

    # (2) site additivity: total == Σ per-site (atol 1e-12). A buffer corrupting a
    # site's result would break this even though each call is internally clean.
    site_sum = sum(
        GLLVModels.laplace_loglik_site(Poisson(), view(Yp, :, i), view(Np, :, i),
                                  Λ, β, LogLink())
        for i in axes(Yp, 2)
    )
    @test isapprox(m1, site_sum; atol = 1e-12)

    # Same two checks for a small NB problem (exercises a different family path
    # through the shared buffers).
    r = 3.0
    Nnb = ones(Int, size(Yp))
    nb1 = GLLVModels.nb_marginal_loglik_laplace(Yp, Λ, β, r)
    nb2 = GLLVModels.nb_marginal_loglik_laplace(Yp, Λ, β, r)
    @test nb1 == nb2
    @test isfinite(nb1)
    nb_site_sum = sum(
        GLLVModels.laplace_loglik_site(NegativeBinomial(float(r), 0.5),
                                  view(Yp, :, i), view(Nnb, :, i), Λ, β, LogLink())
        for i in axes(Yp, 2)
    )
    @test isapprox(nb1, nb_site_sum; atol = 1e-12)
end
