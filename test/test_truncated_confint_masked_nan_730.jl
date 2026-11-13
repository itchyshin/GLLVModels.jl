# #730: `confint` on a zero-truncated Poisson / NB2 (and NB1) fit made with a `mask` must not
# depend on the value held in a masked cell. The fitters drop masked cells
# (`_sanitize_unobserved`), but the truncated `confint` adapters ran `round.(Int, Y)` over the
# whole matrix, so a `NaN` in a masked cell threw `InexactError` in `confint` although the fit
# had succeeded; the NB1 adapter passed the `NaN` to its objective and got `ArgumentError`.
using Test
using GLLVModels
using Random
using Distributions: Poisson

@testset "#730 confint on a masked truncated fit ignores the masked cell's value" begin
    rng = MersenneTwister(730)
    p, n, K = 4, 40, 1
    Y = Float64.(max.([rand(rng, Poisson(exp(1.0 + 0.3 * randn(rng)))) for _ in 1:p, _ in 1:n], 1))
    m = trues(p, n); m[2, 5] = false; m[4, 17] = false
    cells = ((2, 5), (4, 17))
    withval(v) = (Z = copy(Y); for (t, s) in cells; Z[t, s] = v; end; Z)
    Ynan = withval(NaN)

    fits = (
        ("TruncatedPoissonFit", fit_truncated_poisson_gllvm, InexactError),
        ("TruncatedNegBin2Fit", fit_truncated_nbinom2_gllvm, InexactError),
        ("NB1Fit", fit_nb1_gllvm, ArgumentError),
    )
    for (name, fitter, refused) in fits
        @testset "$name" begin
            fit = fitter(Ynan; K = K, mask = m)
            @test string(nameof(typeof(fit))) == name
            ci = confint(fit, Ynan; method = :wald, mask = m)         # InexactError on main
            @test ci isa NamedTuple
            @test all(isfinite, ci.estimate)
            # the masked cell may hold any value: NaN, a valid count, or an invalid one
            for v in (0.0, 1.0, 3.0, 7.0)
                @test isequal(confint(fit, withval(v); method = :wald, mask = m), ci)
            end
            # an unmasked NaN is still refused, with the same error as before
            Ybad = copy(Y); Ybad[3, 7] = NaN
            @test_throws refused confint(fit, Ybad; method = :wald, mask = m)
        end
    end

    @testset "the placeholder cannot reach the objective" begin
        fp = fit_truncated_poisson_gllvm(Ynan; K = K, mask = m)
        fb = fit_truncated_nbinom2_gllvm(Ynan; K = K, mask = m)
        ll(ph) = begin
            Yi = GLLVModels._ci_count_matrix(Ynan, m, ph)
            (GLLVModels.truncated_poisson_marginal_loglik_laplace(Yi, fp.Λ, fp.β, fp.link; mask = m),
             GLLVModels.truncated_nbinom2_marginal_loglik_laplace(Yi, fb.Λ, fb.β, fb.r;
                                                                  link = fb.link, mask = m))
        end
        f1 = fit_nb1_gllvm(Ynan; K = K, mask = m)
        ll1(ph) = GLLVModels.nb1_marginal_loglik_laplace(
            GLLVModels._ci_fill_masked(Ynan, m, ph), f1.Λ, f1.β, f1.φ; link = f1.link, mask = m)
        @test isfinite(ll1(0))
        for ph in (1, 7)
            @test ll1(ph) == ll1(0)
        end
        base = ll(1)
        @test all(isfinite, base)
        for ph in (0, 7)
            @test ll(ph) == base
        end
        # no mask: the conversion is `round.(Int, Y)`, so an unmasked NaN still throws
        @test_throws InexactError GLLVModels._ci_count_matrix(Ynan, nothing)
    end
end
