using GLLVModels, Test, Random, Distributions
const GM = GLLVModels

# #788: fit types that accept `offset` at fit time but do not keep it, so predict / getLV /
# residuals would read a zero offset. #787 and the grouped-dispersion follow-up fixed the Poisson,
# NB2, binomial, hurdle-Poisson and grouped fits (test_predict_offset*.jl); the zero-truncated
# count fits (TruncatedPoissonFit, TruncatedNegBin2Fit, TruncatedNegBin2PerTraitFit) now store
# the training offset and have getLV / predict / residuals (test_truncated_postfit_788.jl).
# The check below pins that the stored offset is the training offset.
#
# The grouped Tweedie fits (TweedieGroupedFit, TweediePerTraitPowerFit) now store their offset too
# and have the post-fit methods: test_tweedie_grouped_postfit_788.jl. CensoredPoissonFit and
# LognormalFit likewise: test_censored_poisson_postfit_788.jl, test_lognormal_postfit_788.jl.

@testset "truncated Poisson fit stores its training offset (#788)" begin
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

    # The fit keeps the offset it was trained with ...
    @test hasproperty(f, :offset) && f.offset == O
    # ... and predict on the training Y uses it by default (see test_truncated_postfit_788.jl
    # for the full link / response / residual checks).
    @test predict(f, Y; type = :link) == predict(f, Y; type = :link, offset = O)
end
