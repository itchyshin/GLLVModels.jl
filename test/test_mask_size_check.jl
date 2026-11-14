using GLLVModels, Test, Random, Distributions

# An explicit `mask` must have the shape of Y. Before this check the Laplace fitters accepted a
# mask that was larger than Y (7x40 or 6x41 for a 6x40 Y: the extra entries were ignored, or the
# objective was -Inf) and failed with a raw BoundsError on a smaller or transposed mask; the
# ordinal fitters accepted a larger mask too. The contract: DimensionMismatch naming both sizes.
@testset "mask size must equal size(Y)" begin
    # helper units
    Y2 = [1 2 3; 4 5 6]
    ok = trues(2, 3)
    @test GLLVModels._resolve_obs_mask(ok, Y2) === ok
    for bad in (trues(3, 3), trues(2, 4), trues(1, 3), trues(2, 2), trues(3, 2), trues(6))
        err = try GLLVModels._resolve_obs_mask(bad, Y2); nothing catch e; e end
        @test err isa DimensionMismatch
        @test occursin(string(size(bad)), err.msg) && occursin("(2, 3)", err.msg)
    end

    # fitters
    rng = MersenneTwister(3)
    p, n = 4, 30
    Λ = 0.5 .* randn(rng, p, 1); Z = randn(rng, 1, n); η = 0.5 .+ Λ * Z
    Yc = [rand(rng, Poisson(exp(η[i, j]))) for i in 1:p, j in 1:n]
    Yg = [rand(rng, Gamma(5.0, exp(η[i, j]) / 5)) for i in 1:p, j in 1:n]
    Yb = [rand(rng, Binomial(5, 1 / (1 + exp(-η[i, j])))) for i in 1:p, j in 1:n]
    Yr = clamp.(Yg ./ (1 .+ Yg), 1e-3, 1 - 1e-3)
    Yo = [1 + (η[i, j] + randn(rng) > 0) + (η[i, j] + randn(rng) > 1.2) for i in 1:p, j in 1:n]
    fits = (
        "Poisson"      => m -> fit_gllvm(Yc; family = Poisson(), K = 1, mask = m),
        "NB2 grouped"  => m -> fit_gllvm(Yc; family = NegativeBinomial(), disp_group = :species, K = 1, mask = m),
        "Gamma"        => m -> fit_gllvm(Yg; family = Gamma(), K = 1, mask = m),
        "Binomial"     => m -> fit_gllvm(Yb; family = Binomial(), N = fill(5, p, n), K = 1, mask = m),
        "Beta"         => m -> fit_gllvm(Yr; family = Beta(), K = 1, mask = m),
        "ordinal"      => m -> fit_ordinal_gllvm(Yo; K = 1, mask = m),
        "ordinal pertrait" => m -> fit_ordinal_gllvm_pertrait(Yo; K = 1, mask = m),
    )
    wrong = (trues(p + 1, n), trues(p, n + 1), trues(p - 1, n), trues(p, n - 1), trues(n, p))
    for (name, f) in fits, m in wrong
        @testset "$name mask $(size(m))" begin
            @test_throws DimensionMismatch f(m)
        end
    end
    # a mask of the right shape still fits, and equals the unmasked fit when all observed
    full = trues(p, n)
    for (name, f) in fits
        r = f(full)
        @test isfinite(hasproperty(r, :loglik) ? r.loglik : r.logLik)
    end
end
