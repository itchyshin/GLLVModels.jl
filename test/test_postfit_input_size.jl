using GLLVModels, Test, Random, Distributions

# Post-fit methods must refuse a mask, trials matrix N or design X that is not the shape of Y
# (refs #1198, the post-fit sibling of the simulate N check). The per-site loops read
# `view(mask, :, s)`, `view(N, :, s)` and the X offset under `@inbounds`, so a too-small argument was
# read past its end: wrong values with no error in a default build (for example `predict(fp, Y;
# mask = trues(p, 3))`), a `BoundsError` under `--check-bounds=yes`. A silent out-of-bounds read
# cannot be asserted by value, so each case asserts the throw.
@testset "post-fit: mask, N and X must match Y" begin
    rng = MersenneTwister(2); p, n = 5, 60
    Y = [rand(rng, Binomial(5, 0.3 + 0.1 * randn(rng)^2 / 4)) for t in 1:p, s in 1:n]
    Nfull = fill(5, p, n)
    badmask = (trues(p, 3), trues(p, n + 3), trues(3, n))
    badN = (fill(5, p, 3), fill(5, p, n + 3), fill(5, 3, n))

    fp = fit_poisson_gllvm(Y; K = 1)
    for m in badmask
        @test_throws DimensionMismatch getLV(fp, Y; mask = m)
        @test_throws DimensionMismatch predict(fp, Y; mask = m)
        @test_throws DimensionMismatch latent_score_sd(fp, Y; mask = m)
    end
    for N in badN
        @test_throws DimensionMismatch getLV(fp, Y; N = N)
        @test_throws DimensionMismatch latent_score_sd(fp, Y; N = N)
    end
    # a correctly sized mask / N gives the same answer as before the check
    ok = trues(p, n); ok[1, 1] = false
    @test getLV(fp, Y; mask = ok) == getLV(fp, Y; mask = copy(ok))
    @test size(predict(fp, Y; mask = ok)) == (p, n)

    fb = fit_binomial_gllvm(Y; K = 1, N = Nfull)
    for m in badmask
        @test_throws DimensionMismatch getLV(fb, Y; N = Nfull, mask = m)
        @test_throws DimensionMismatch predict(fb, Y; N = Nfull, mask = m)
    end
    for N in badN
        @test_throws DimensionMismatch getLV(fb, Y; N = N)
        @test_throws DimensionMismatch predict(fb, Y; N = N)
        @test_throws DimensionMismatch residuals(fb, Y; N = N, rng = MersenneTwister(1))
        @test_throws DimensionMismatch latent_score_sd(fb, Y; N = N)
    end

    X = reshape(randn(rng, n), 1, n, 1) .* ones(p, 1, 1)
    fc = fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1, N = Nfull)
    for N in badN
        @test_throws DimensionMismatch getLV(fc, Y, X; N = N)
        @test_throws DimensionMismatch predict(fc, Y, X; N = N)
        @test_throws DimensionMismatch fitted(fc, Y, X; N = N)
    end
    @test_throws DimensionMismatch getLV(fc, Y, X[:, 1:10, :]; N = Nfull)
    @test_throws DimensionMismatch predict(fc, Y, X[:, 1:10, :]; N = Nfull)
    @test_throws DimensionMismatch getLV(fc, Y, cat(X, X[:, 1:3, :]; dims = 2); N = Nfull)
    @test size(predict(fc, Y, X; N = Nfull)) == (p, n)

    # other Laplace fit types that share the pattern
    Yc = [rand(rng, Poisson(exp(0.7 + 0.5 * randn(rng)))) for t in 1:p, s in 1:n]
    fnb = fit_nb_gllvm(Yc; K = 1)
    for m in badmask
        @test_throws DimensionMismatch getLV(fnb, Yc; mask = m)
        @test_throws DimensionMismatch latent_score_sd(fnb, Yc; mask = m)
    end
    for N in badN
        @test_throws DimensionMismatch getLV(fnb, Yc; N = N)
        @test_throws DimensionMismatch predict(fnb, Yc; N = N)
    end
    Yo = clamp.([rand(rng, 1:3) for t in 1:p, s in 1:n], 1, 3)
    fo = fit_ordinal_gllvm(Yo; K = 1)
    for m in badmask
        @test_throws DimensionMismatch getLV(fo, Yo; mask = m)
    end
    Yg = [rand(rng, Gamma(2.0, 1.0)) + 0.1 for t in 1:p, s in 1:n]
    fg = fit_gamma_gllvm(Yg; K = 1)
    for m in badmask
        @test_throws DimensionMismatch getLV(fg, Yg; mask = m)
    end
    fng = fit_nb_gllvm_grouped(Yc; K = 1, group = [1, 1, 2, 2, 2])
    for m in badmask
        @test_throws DimensionMismatch getLV(fng, Yc; mask = m)
        @test_throws DimensionMismatch predict(fng, Yc; mask = m)
    end
    fngc = fit_nb_gllvm_grouped_cov(Yc; X = X, K = 1, group = [1, 1, 2, 2, 2])
    @test_throws DimensionMismatch getLV(fngc, Yc, X[:, 1:10, :])
    @test_throws DimensionMismatch getLV(fngc, Yc, X; mask = trues(p, 3))

    # trials-matrix and design-matrix variants
    fsc = fit_gllvm_speciescov(Y; family = Binomial(), X = X, K = 1, N = Nfull)
    for N in badN
        @test_throws DimensionMismatch getLV(fsc, Y, X; N = N)
    end
    @test_throws DimensionMismatch getLV(fsc, Y, X[:, 1:10, :]; N = Nfull)
    fre = fit_roweffect_gllvm(Y; family = Binomial(), K = 1, N = Nfull)
    frr = fit_row_random_gllvm(Y; family = Binomial(), K = 1, N = Nfull)
    fmx = fit_mixed_gllvm(Y; families = fill(Binomial(), p), K = 1, N = Nfull)
    fbb = fit_beta_binomial_gllvm(Y; K = 1, N = Nfull)
    for N in badN
        @test_throws DimensionMismatch getLV(fre, Y; N = N)
        @test_throws DimensionMismatch getLV(frr, Y; N = N)
        @test_throws DimensionMismatch getLV(fmx, Y; N = N)
        @test_throws DimensionMismatch getLV(fbb, Y; N = N)
    end
    Xe = randn(rng, n, 1); TR = randn(rng, p, 1)
    ffc = fit_fourthcorner_gllvm(Y; family = Binomial(), Xenv = Xe, TR = TR, K = 1, N = Nfull)
    @test_throws DimensionMismatch getLV(ffc, Y, Xe[1:10, :], TR; N = Nfull)
    @test_throws DimensionMismatch getLV(ffc, Y, vcat(Xe, Xe[1:3, :]), TR; N = Nfull)
    for N in badN
        @test_throws DimensionMismatch getLV(ffc, Y, Xe, TR; N = N)
    end

    fco = fit_constrained_gllvm(Y; family = Binomial(), X = Xe, K = 1, N = Nfull)
    @test_throws DimensionMismatch getLV(fco, Y, Xe[1:10, :]; N = Nfull)
    @test_throws DimensionMismatch predict(fco, Y, vcat(Xe, Xe[1:3, :]); N = Nfull)
    for N in badN
        @test_throws DimensionMismatch getLV(fco, Y, Xe; N = N)
    end

    # nobs counts a mask: a mask of another shape gave a wrong observation count and BIC
    @test nobs(fp, Y; mask = ok) == count(ok)
    for m in badmask
        @test_throws DimensionMismatch nobs(fp, Y; mask = m)
    end
end
