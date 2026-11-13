# confint for the grouped Beta fits rebuilds the objective the fitter optimised (#1155).
# The fitter and the Beta grouped kernel clamp y at [1e-12, 1 - 1e-12] (gllvmTMB's y_safe); the
# confint adapter used to clamp Y at [1e-6, 1 - 1e-6] first, so a response below 1e-6 (or above
# 1 - 1e-6) gave a rebuilt objective that differs from the fit's logLik and `confint` refused a
# converged fit ("the objective rebuilt from Y ... gives log-likelihood 424.52 ... the fit reports 425.11").

using GLLVModels, Test, Random, Distributions
const _GM1155 = GLLVModels

function _beta_y_1155(p, n, seed)
    rng = MersenneTwister(seed); L = 0.7 .* randn(rng, p); z = randn(rng, n); a = 0.3 .* randn(rng, p)
    [(μ = 1 / (1 + exp(-(a[j] + L[j] * z[i]))); φ = 5.0 + j; rand(rng, Beta(μ * φ, (1 - μ) * φ))) for j in 1:p, i in 1:n]
end

@testset "grouped Beta confint rebuilds the fit's objective for y near 0 and 1 (#1155)" begin
    p, n = 4, 40
    Y = _beta_y_1155(p, n, 3)
    Y[2, 5] = 3e-8; Y[3, 7] = 1 - 3e-8          # outside [1e-6, 1 - 1e-6], inside [1e-12, 1 - 1e-12]
    @test minimum(Y) < 1e-6 && maximum(Y) > 1 - 1e-6

    @testset "BetaGroupedFit" begin
        fit = _GM1155.fit_beta_gllvm_grouped(Y; K = 1, group = collect(1:p))
        ad = _GM1155._family_ci(fit, Y)
        @test -ad.nll(ad.θ) ≈ fit.loglik rtol = 1e-9
        ci = confint(fit, Y; method = :wald)       # used to throw the objective-mismatch ArgumentError
        @test length(ci.term) == length(ad.θ)
    end

    @testset "BetaGroupedCovFit" begin
        X = reshape(0.5 .* randn(MersenneTwister(11), p * n), p, n, 1)
        fit = _GM1155.fit_beta_gllvm_grouped_cov(Y; X = X, K = 1, group = collect(1:p))
        ad = _GM1155._family_ci(fit, Y; X = X)
        @test -ad.nll(ad.θ) ≈ fit.loglik rtol = 1e-9
    end

    @testset "data inside [1e-6, 1 - 1e-6] is unchanged" begin
        Yin = _beta_y_1155(p, n, 1)
        @test minimum(Yin) > 1e-6 && maximum(Yin) < 1 - 1e-6
        fit = _GM1155.fit_beta_gllvm_grouped(Yin; K = 1, group = collect(1:p))
        ad = _GM1155._family_ci(fit, Yin)
        @test -ad.nll(ad.θ) ≈ fit.loglik rtol = 1e-9
    end
end
