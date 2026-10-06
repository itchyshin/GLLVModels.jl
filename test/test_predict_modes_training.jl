using GLLVModels, Test, Random, Distributions, LinearAlgebra

# predict(fit, Y; offset = O_new, modes = :training) on a Laplace offset fit
# (DATA-OFF-PREDICT, maintainer ruling 2026-10-05, vault D-319, option (b)).
#
# gllvmTMB's predict(fit, newdata) on the training units re-evaluates the stored offset
# expression on newdata and keeps the training units' latent modes:
#     η = β + O_new + Λ ẑ_train.
# `modes = :training` gives that quantity; the default `modes = :refit` keeps the earlier
# behaviour (re-solve the modes at the prediction offset). The reference here is built from
# getLV at the stored training offset, not from the code path under test.

function _sim(rng, p, n; family = :poisson)
    O = 0.8 .* randn(rng, p, n)
    Λ = 0.6 .* randn(rng, p, 1); β = fill(0.5, p)
    η = β .+ O .+ Λ * randn(rng, 1, n)
    Y = family === :poisson ? [rand(rng, Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n] :
        family === :nb2     ? [rand(rng, NegativeBinomial(5.0, 5.0 / (5.0 + exp(η[t, s])))) for t in 1:p, s in 1:n] :
        family === :binom   ? [rand(rng, Bernoulli(1 / (1 + exp(-η[t, s])))) for t in 1:p, s in 1:n] :
                              [rand(rng, Gamma(4.0, exp(η[t, s]) / 4.0)) for t in 1:p, s in 1:n]
    return Y, O
end

@testset "predict modes = :training keeps the training modes (offset fits)" begin
    p, n = 5, 60
    cases = (
        ("Poisson", :poisson, Poisson()),
        ("NB2", :nb2, NegativeBinomial(1.0, 0.5)),
        ("binomial", :binom, Binomial()),
        ("Gamma", :gamma, Gamma()))
    for (label, kind, fam) in cases
        @testset "$label" begin
            Y, O = _sim(MersenneTwister(20261005), p, n; family = kind)
            f = fit_gllvm(Y; family = fam, K = 1, offset = O)
            @test f.offset == O
            Onew = O .+ 0.7 .* randn(MersenneTwister(7), p, n)
            Ztr = getLV(f, Y; rotate = false)                    # training modes, stored offset
            ref = f.β .+ Onew .+ f.Λ * Ztr'
            got = predict(f, Y; type = :link, offset = Onew, modes = :training)
            @test isapprox(got, ref; atol = 1e-10, rtol = 0)
            # the default is unchanged: it re-solves the modes at the new offset, so it differs
            refit = predict(f, Y; type = :link, offset = Onew)
            @test refit == predict(f, Y; type = :link, offset = Onew, modes = :refit)
            @test maximum(abs, refit .- got) > 1e-3
            # without a new offset the two modes agree (both use the stored offset)
            @test predict(f, Y; type = :link, modes = :training) == predict(f, Y; type = :link)
            # the response scale is the inverse link of the same η
            @test isapprox(predict(f, Y; type = :response, offset = Onew, modes = :training),
                           GLLVModels.linkinv.(Ref(f.link), got); atol = 1e-12, rtol = 1e-12)
            # Y must be the training data (its size), and the keyword is checked
            @test_throws ArgumentError predict(f, Y[:, 1:10]; type = :link, offset = Onew[:, 1:10],
                                               modes = :training)
            @test_throws ArgumentError predict(f, Y; type = :link, offset = Onew, modes = :fresh)
        end
    end
    # a fit without an offset: :training keeps the offset-free modes and adds the new offset
    Y, _ = _sim(MersenneTwister(3), p, n)
    f0 = fit_gllvm(Y; family = Poisson(), K = 1)
    Onew = 0.5 .* randn(MersenneTwister(4), p, n)
    @test isapprox(predict(f0, Y; type = :link, offset = Onew, modes = :training),
                   f0.β .+ Onew .+ f0.Λ * getLV(f0, Y; rotate = false)'; atol = 1e-10, rtol = 0)
end
