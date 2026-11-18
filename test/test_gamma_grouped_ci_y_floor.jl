# confint for the grouped Gamma fits rebuilds the objective the fitter optimised (refs #1155).
# The fitter and the Gamma grouped kernel use Y as given (the fitter only refuses y <= 0); the confint
# adapters used to floor Y at 1e-9 first, so a response below 1e-9 gave a rebuilt objective that differs
# from the fit's logLik: `confint` on a GammaGroupedFit refused a converged fit, and on a
# GammaGroupedCovFit (no objective guard) silently computed intervals on the wrong objective.

using GLLVModels, Test, Random, Distributions
const _GM_GY = GLLVModels

function _gamma_y_gy(p, n, seed)
    rng = MersenneTwister(seed); L = 0.5 .* randn(rng, p); z = randn(rng, n); a = 1.0 .+ 0.3 .* randn(rng, p)
    [(μ = exp(a[j] + L[j] * z[i]); α = 3.0 + j; rand(rng, Gamma(α, μ / α))) for j in 1:p, i in 1:n]
end

@testset "grouped Gamma confint rebuilds the fit's objective for y below 1e-9 (refs #1155)" begin
    p, n = 4, 40
    Y = _gamma_y_gy(p, n, 3)
    Y[2, 5] = 3e-10                              # below the old 1e-9 floor
    @test minimum(Y) < 1e-9
    X = reshape(0.5 .* randn(MersenneTwister(11), p * n), p, n, 1)

    @testset "GammaGroupedFit" begin
        fit = _GM_GY.fit_gamma_gllvm_grouped(Y; K = 1, group = collect(1:p))
        ad = _GM_GY._family_ci(fit, Y)
        @test -ad.nll(ad.θ) ≈ fit.loglik rtol = 1e-9
        ci = confint(fit, Y; method = :wald)      # used to throw the objective-mismatch ArgumentError
        @test length(ci.term) == length(ad.θ)
    end

    @testset "GammaGroupedCovFit" begin
        fit = _GM_GY.fit_gamma_gllvm_grouped_cov(Y; X = X, K = 1, group = collect(1:p))
        ad = _GM_GY._family_ci(fit, Y; X = X)
        @test -ad.nll(ad.θ) ≈ fit.loglik rtol = 1e-9
    end

    @testset "data above 1e-9 is unchanged" begin
        Yin = _gamma_y_gy(p, n, 1)
        @test minimum(Yin) > 1e-9
        fit = _GM_GY.fit_gamma_gllvm_grouped(Yin; K = 1, group = collect(1:p))
        ad = _GM_GY._family_ci(fit, Yin)
        @test -ad.nll(ad.θ) ≈ fit.loglik rtol = 1e-9
        fitc = _GM_GY.fit_gamma_gllvm_grouped_cov(Yin; X = X, K = 1, group = collect(1:p))
        adc = _GM_GY._family_ci(fitc, Yin; X = X)
        @test -adc.nll(adc.θ) ≈ fitc.loglik rtol = 1e-9
    end
end
