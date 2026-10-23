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

# The same check on the other Laplace fits that store an offset, on an X_lv fit, and the refusals.
function _sim_draw(rng, p, n, draw)
    O = 0.6 .* randn(rng, p, n)
    Λ = 0.5 .* randn(rng, p, 1); β = fill(0.6, p)
    η = β .+ O .+ Λ * randn(rng, 1, n)
    return [draw(rng, η[t, s]) for t in 1:p, s in 1:n], O
end

# η = β + O_new + Λ ẑ_train, ẑ_train from getLV at the stored training offset.
function _check_training(f, Y, O; X_lv = nothing)
    @test f.offset == O
    kw = X_lv === nothing ? (;) : (; X_lv = X_lv)
    Onew = O .+ 0.7 .* randn(MersenneTwister(7), size(O)...)
    Ztr = X_lv === nothing ? getLV(f, Y; rotate = false) :
                             getLV(f, Y; rotate = false, X_lv = X_lv, component = :total)
    got = predict(f, Y; type = :link, offset = Onew, modes = :training, kw...)
    @test isapprox(got, f.β .+ Onew .+ f.Λ * Ztr'; atol = 1e-10, rtol = 0)
    @test maximum(abs, predict(f, Y; type = :link, offset = Onew, kw...) .- got) > 1e-3
    @test_throws ArgumentError predict(f, Y[:, 1:10]; type = :link, offset = Onew[:, 1:10],
                                       modes = :training)
end

@testset "predict modes = :training on more fit types" begin
    p, n = 5, 60
    @testset "NB1" begin
        Y, O = _sim_draw(MersenneTwister(41), p, n,
                         (r, η) -> rand(r, NegativeBinomial(exp(η) / 0.8, 1 / 1.8)))
        _check_training(fit_nb1_gllvm(Y; K = 1, offset = O), Y, O)
    end
    @testset "GP1" begin
        Y, O = _sim_draw(MersenneTwister(42), p, n, (r, η) -> rand(r, Poisson(exp(η))))
        _check_training(fit_gp1_gllvm(Y; K = 1, offset = O), Y, O)
    end
    @testset "Beta" begin
        Y, O = _sim_draw(MersenneTwister(43), p, n, (r, η) -> clamp(
            rand(r, Beta(6 / (1 + exp(-η)), 6 * (1 - 1 / (1 + exp(-η))))), 1e-6, 1 - 1e-6))
        _check_training(fit_beta_gllvm(Y; K = 1, offset = O), Y, O)
    end
    @testset "Exponential" begin
        Y, O = _sim_draw(MersenneTwister(44), p, n, (r, η) -> rand(r, Exponential(exp(η))))
        _check_training(fit_exponential_gllvm(Y; K = 1, offset = O), Y, O)
    end
    @testset "Poisson with X_lv" begin
        rng = MersenneTwister(45)
        X = reshape(randn(rng, n), n, 1)
        O = 0.6 .* randn(rng, p, n)
        Λ = 0.5 .* randn(rng, p, 1)
        Y = [rand(rng, Poisson(exp(0.5 + O[t, s] + Λ[t, 1] * (0.8 * X[s, 1] + randn(rng)))))
             for t in 1:p, s in 1:n]
        f = fit_poisson_gllvm(Y; K = 1, X_lv = X, offset = O)
        _check_training(f, Y, O; X_lv = X)
    end
    @testset "AGHQ fits refuse :training; an unknown keyword is named first" begin
        Y, O = _sim_draw(MersenneTwister(46), p, n, (r, η) -> rand(r, Poisson(exp(η))))
        fa = fit_poisson_gllvm(Y; K = 1, aghq = 3, offset = O)
        @test fa.integration.actual === :aghq
        err = try predict(fa, Y; type = :link, modes = :training); nothing catch e; e end
        @test err isa ArgumentError && occursin("Laplace path only", err.msg)
        err = try predict(fa, Y; type = :link, modes = :fresh); nothing catch e; e end
        @test err isa ArgumentError && occursin("must be :refit or :training", err.msg)
        @test predict(fa, Y; type = :link, modes = :refit) == predict(fa, Y; type = :link)
    end
    @testset "no-offset fit: :training checks the response count" begin
        Y, _ = _sim_draw(MersenneTwister(47), p, n, (r, η) -> rand(r, Poisson(exp(η))))
        f0 = fit_poisson_gllvm(Y; K = 1)
        @test f0.offset === nothing
        err = try predict(f0, Y[1:(p - 1), :]; type = :link, modes = :training); nothing catch e; e end
        @test err isa ArgumentError && occursin("training data", err.msg)
        # the unit count is not stored on a fit without an offset, so it cannot be checked there
        @test size(predict(f0, Y[:, 1:10]; type = :link, modes = :training)) == (p, 10)
    end
end
