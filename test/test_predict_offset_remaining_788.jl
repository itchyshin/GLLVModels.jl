using GLLVModels, Test, Random, Distributions
const GM = GLLVModels

# Known gap, #788: fit types that accept `offset` at fit time but do not keep it, so predict /
# getLV / residuals read a zero offset. #787 and the grouped-dispersion follow-up fixed the Poisson,
# NB2, binomial, hurdle-Poisson and grouped fits (test_predict_offset*.jl). The zero-truncated
# fits below still keep none. These are @test_broken: they record the intended behaviour (the
# training offset is stored and `predict` on the training Y uses it) and flip to an
# "Unexpected Pass" error when the gap is closed, at which point change them to @test.
# No default or design is changed here.
#
# Still affected per the issue and not yet pinned here: NBGroupedCovFit and the other *CovFit
# types, TweedieGrouped, ZIPCovFit / ZINBCovFit / ZIBCovFit, TruncatedNegBin2PerTraitFit.

@testset "ignored offsets on remaining fit types (#788, known gap)" begin
    p, n, K = 5, 50, 1
    rng = MersenneTwister(788)
    O = 0.8 .* randn(rng, p, n)
    Λt = 0.5 .* randn(rng, p, K); Zt = randn(rng, n, K)
    η = 0.8 .+ O .+ Λt * Zt'
    function _ztpois(λ)                       # zero-truncated Poisson draw
        while true
            y = rand(rng, Poisson(λ)); y >= 1 && return y
        end
    end
    Y = [_ztpois(exp(η[t, s])) for t in 1:p, s in 1:n]

    f = fit_truncated_poisson_gllvm(Y; K = K, offset = O)
    @test f isa TruncatedPoissonFit

    # Intended: the fit keeps the offset it was trained with ...
    @test_broken hasproperty(f, :offset) && f.offset == O
    # ... so predicting on the training Y without `offset` equals predicting with it.
    @test_broken isapprox(GM.predict(f, Y; type = :link),
                          GM.predict(f, Y; type = :link, offset = O); atol = 1e-6)
end
