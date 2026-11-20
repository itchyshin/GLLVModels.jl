# #1154: the Student-t Laplace log-likelihood is location-scale, so
# loglik(c*Y) + length(Y)*log(c) must not depend on c. Under the identity link the
# mean must not be capped at |eta| = 30 inside the grouped mode search.
using GLLVModels, Test, Random

@testset "Student-t identity link: response-scale invariance (#1154)" begin
    Random.seed!(20261112)
    p, K, n, nu = 6, 1, 60, 4.0
    beta = [0.3, -0.2, 0.5, 0.0, -0.4, 0.1]
    Lam = reshape([0.7, 0.5, -0.4, 0.3, 0.6, -0.2], p, K)
    Z = randn(K, n)
    noise = [randn() / sqrt(sum(abs2, randn(4)) / nu) for _ in 1:p, _ in 1:n]
    Y1 = beta .+ Lam * Z .+ 0.7 .* noise
    base = fit_studentt_gllvm(Y1; K = K, nu = nu, disp_group = :species)
    # c = 1e4 additionally needs a scale-aware sigma start (an estimator-default change
    # held for a decision; see #1154), so it is not asserted here.
    for c in (1e-2, 1e2)
        f = fit_studentt_gllvm(c .* Y1; K = K, nu = nu, disp_group = :species)
        @test f.converged
        @test isfinite(f.loglik)
        @test abs(f.loglik + length(Y1) * log(c) - base.loglik) < 1e-2
    end
end
