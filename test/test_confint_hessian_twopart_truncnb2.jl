# confint rebuilds the fit's own curvature for DeltaGammaFit and the truncated NB2 fits (#778).
#
# These fit structs did not record the `hessian` the fitter used, so `_family_ci` rebuilt the
# `:observed` Laplace objective for a `hessian = :fisher` fit: a different function from the one
# that was maximised (p = 5, n = 70, K = 1: -logLik 532.669 against 532.823 for Delta-Gamma,
# 678.053 against 678.795 for truncated NB2). The fit now stores `hessian` and the adapters
# default to it. Contract, per fit type x selector: the objective `confint` rebuilds, evaluated at
# the fit's own packed theta-hat, equals -fit.loglik.

using GLLVModels, Test, Random, Distributions
const GMH = GLLVModels

function _h778_delta_gamma(rng, p, n, K; shared)
    β = 0.3 .* randn(rng, p) .+ 1.0
    Λ0 = 0.5 .* randn(rng, p, K)
    Y = zeros(Float64, p, n)
    for s in 1:n
        z = randn(rng, K)
        for t in 1:p
            ηc = β[t] + (Λ0 * z)[t]
            π = shared ? 1 / (1 + exp(-ηc)) : 0.7
            rand(rng) < π && (Y[t, s] = rand(rng, Gamma(2.0, exp(ηc) / 2.0)))
        end
    end
    return Y
end

function _h778_truncnb2(rng, p, n, K)
    β = 0.3 .* randn(rng, p) .+ 0.8
    Λ0 = 0.5 .* randn(rng, p, K)
    Y = zeros(Int, p, n)
    for s in 1:n
        η = β .+ Λ0 * randn(rng, K)
        for t in 1:p
            Y[t, s] = GMH._rand_ztnb(rng, 3.0, exp(η[t]))
        end
    end
    return Y
end

_h778_nll_at_hat(fit, Y) = (ad = GMH._family_ci(fit, Y); ad.nll(ad.θ))

@testset "confint rebuilds the fit's own curvature (DeltaGamma, truncated NB2; #778)" begin
    p, n, K = 5, 70, 1
    builders = Any[
        ("Delta-Gamma (separate)", rng -> _h778_delta_gamma(rng, p, n, K; shared = false),
            (Y; kw...) -> fit_delta_gamma_gllvm(Y; K = K, kw...)),
        ("Delta-Gamma (shared predictor)", rng -> _h778_delta_gamma(rng, p, n, K; shared = true),
            (Y; kw...) -> fit_delta_gamma_gllvm(Y; K = K, predictor = :shared, kw...)),
        ("Truncated NB2", rng -> _h778_truncnb2(rng, p, n, K),
            (Y; kw...) -> fit_truncated_nbinom2_gllvm(Y; K = K, kw...)),
        ("Truncated NB2 (per trait)", rng -> _h778_truncnb2(rng, p, n, K),
            (Y; kw...) -> fit_truncated_nbinom2_gllvm_pertrait(Y; K = K, kw...)),
    ]
    @testset "$(b[1])" for b in builders
        _, gen, fitter = b
        Y = gen(MersenneTwister(41))
        ff = fitter(Y; hessian = :fisher)
        fo = fitter(Y)
        # the fit records its selector
        @test ff.hessian === :fisher
        @test fo.hessian === :observed
        # the rebuilt objective at theta-hat is the fit's own, under both selectors
        @test isapprox(_h778_nll_at_hat(ff, Y), -ff.loglik; atol = 1e-6)
        @test isapprox(_h778_nll_at_hat(fo, Y), -fo.loglik; atol = 1e-6)
        # not decorative: the two curvatures give different objectives at the :fisher estimate
        ad = GMH._family_ci(ff, Y)
        ao = GMH._family_ci(ff, Y; hessian = :observed)
        @test abs(ao.nll(ad.θ) - ad.nll(ad.θ)) > 1e-6
        # confint runs on the :fisher fit and the :observed fit alike
        @test confint(ff, Y).pd_hessian isa Bool
        @test confint(fo, Y).pd_hessian isa Bool
    end
end

@testset "hessian field: positional-compat constructors default to :observed (#778)" begin
    β = [0.5, 0.5]; Λ = reshape([0.3, 0.2], 2, 1)
    @test DeltaGammaFit(β, β, Λ, 2.0, -1.0, true, 1).hessian === :observed
    @test DeltaGammaFit(β, β, Λ, 2.0, -1.0, true, 1, :separate, :shared, nothing).hessian === :observed
    @test DeltaGammaFit(β, β, Λ, 2.0, -1.0, true, 1, :separate, :shared, nothing, :fisher).hessian === :fisher
    θ = [0.5, 0.5, 0.3, 0.2, 0.0]
    @test TruncatedNegBin2Fit(β, Λ, 3.0, GMH.LogLink(), -1.0, true, 1, θ, 0.5).hessian === :observed
    @test TruncatedNegBin2PerTraitFit(β, Λ, [3.0, 3.0], GMH.LogLink(), -1.0, true, 1, θ).hessian === :observed
end
