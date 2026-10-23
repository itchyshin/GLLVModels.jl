# Focused consistency guard for the buffer-reuse refactor of the two-part Laplace
# mode-finder (src/families/twopart.jl). The full existing suite is the real
# bit-exactness anchor (the per-family two-part anchors pin these marginals to
# machine precision); this file adds two cheap checks that specifically catch the
# failure modes introduced by reusing buffers:
#   1. idempotence — repeated calls to the two-part marginal return EXACTLY the
#      same value (catches a buffer aliased/leaked across calls), and
#   2. site additivity — the total marginal equals the sum of per-site
#      `twopart_loglik_site` values (catches buffer reuse corrupting per-site work
#      within the loop).
# Kept tiny/fast: small p, K, n; fixed (Λc, βz, βc); no optimisation. A non-zero Λc
# is used so the Newton mode-finder actually iterates (the reused buffers are
# exercised) rather than terminating at z = 0.

using Test, GLLVModels

@testset "two-part buffer-reuse equivalence" begin
    # Small fixed problem; non-zero Λc so the shared-z Newton loop iterates.
    p, K, n = 4, 2, 5
    Λc = [ 0.6 -0.2;
          -0.3  0.5;
           0.1  0.4;
           0.7 -0.1]
    Λz = zeros(p, K)                  # v1 convention: occurrence intercept-only
    βz = [0.2, -0.4, 0.1, 0.3]
    βc = [0.5, 0.0, 0.3, -0.2]
    # Integer count matrix with zeros (absences) and positive counts (the count part).
    Yc = [ 1  0  2  1  3;
           0  1  1  0  2;
           2  3  0  1  1;
           1  1  2  4  0]
    Yc = Float64.(Yc)

    # ----- ZIP (mixture) -----
    z1 = GLLVModels.zip_marginal_loglik_laplace(Yc, Λc, βz, βc)
    z2 = GLLVModels.zip_marginal_loglik_laplace(Yc, Λc, βz, βc)
    @test z1 == z2                    # idempotence: exact equality (== not ≈)
    @test isfinite(z1)
    zip_site_sum = sum(
        GLLVModels.twopart_loglik_site(GLLVModels.ZIPoisson(), view(Yc, :, s),
                                  Λz, Λc, βz, βc)
        for s in axes(Yc, 2)
    )
    @test isapprox(z1, zip_site_sum; atol = 1e-12)

    # ----- Hurdle-NB (different family path through the shared buffers) -----
    r = 3.0
    h1 = GLLVModels.hurdle_nb_marginal_loglik_laplace(Yc, Λc, βz, βc, r)
    h2 = GLLVModels.hurdle_nb_marginal_loglik_laplace(Yc, Λc, βz, βc, r)
    @test h1 == h2
    @test isfinite(h1)
    hnb_site_sum = sum(
        GLLVModels.twopart_loglik_site(GLLVModels.HurdleNB(float(r)), view(Yc, :, s),
                                  Λz, Λc, βz, βc)
        for s in axes(Yc, 2)
    )
    @test isapprox(h1, hnb_site_sum; atol = 1e-12)
end
