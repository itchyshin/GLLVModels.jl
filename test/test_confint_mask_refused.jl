using GLLVModels, Test, Random
using Distributions: Poisson

# `confint(fit, Y; mask = M)` for fit types whose `_family_ci` adapter cannot use a mask
# must throw an ArgumentError that names the fit type. Before this, the adapters absorbed
# `mask` in `kwargs...` and returned intervals from the full-`Y` objective, which is not
# the objective that produced the masked fit (#775). Unmasked calls are unchanged.

@testset "confint refuses a mask the adapter cannot use (#775)" begin
    rng = MersenneTwister(775)
    p, n, K = 4, 40, 1
    mk = trues(p, n); mk[:, 1:10] .= false

    Yc = Float64.(rand(rng, 0:6, p, n)); Yc[rand(rng, p, n) .< 0.2] .= 0.0
    Yc[.!mk] .= 1e-3
    Nb = fill(5, p, n)
    Yb = rand(rng, 0:5, p, n); Yb[.!mk] .= 0
    Yu = clamp.(rand(rng, p, n), 0.0, 1.0); Yu[1, 1:5] .= 0.0; Yu[2, 1:5] .= 1.0
    Yu[.!mk] .= 0.5
    Yp = rand(rng, 0:5, p, n); Yp[.!mk] .= 0
    X = randn(rng, p, n, 1)
    Yo = rand(rng, 1:3, p, n); Yo[1, :] .= rand(rng, 1:3, n)

    cases = [
        ("TweedieFit", fit_tweedie_gllvm(Yc; K = K, mask = mk, iterations = 20), Yc, (;)),
        ("BetaBinomialFit", fit_beta_binomial_gllvm(Yb; K = K, N = Nb, mask = mk, iterations = 20), Yb, (N = Nb,)),
        ("OrderedBetaFit", fit_ordered_beta_gllvm(Yu; K = K, mask = mk, iterations = 20), Yu, (;)),
        ("RowEffectFit", fit_roweffect_gllvm(Yp; family = Poisson(), K = K, mask = mk, iterations = 20), Yp, (;)),
        ("GllvmCovFit", fit_gllvm_cov(Yp; family = Poisson(), X = X, K = K, mask = mk, iterations = 20), Yp, (X = X,)),
        ("OrdinalFit", fit_ordinal_gllvm(Yo; K = K, mask = mk, iterations = 20), Yo, (;)),
        ("OrdinalPerTraitFit", fit_ordinal_gllvm_pertrait(Yo; K = K, mask = mk, iterations = 20), Yo, (;)),
        ("OrdinalPerTraitCovFit", fit_ordinal_gllvm_pertrait_cov(Yo; X = X, K = K, mask = mk, iterations = 20), Yo, (X = X,)),
    ]

    for (name, fit, Y, kw) in cases
        @testset "$name" begin
            @test string(nameof(typeof(fit))) == name
            msg = try
                confint(fit, Y; method = :wald, mask = mk, kw...)
                nothing
            catch e
                e isa ArgumentError ? sprint(showerror, e) : rethrow()
            end
            @test msg !== nothing
            @test msg !== nothing && occursin(name, msg) && occursin("mask", msg)
            # a mask of the wrong shape is still refused with the shape message
            @test_throws ArgumentError confint(fit, Y; method = :wald, mask = trues(p + 1, n), kw...)
            if name in ("GllvmCovFit", "OrdinalPerTraitCovFit")
                # the fit was made on a masked objective; unmasked confint is refused (#1260)
                for call in (() -> confint(fit, Y; method = :wald, mask = trues(p, n), kw...),
                             () -> confint(fit, Y; method = :wald, kw...))
                    msg_u = try; call(); nothing catch e; e isa ArgumentError ? sprint(showerror, e) : rethrow() end
                    @test msg_u !== nothing && occursin("log-likelihood", msg_u)
                end
            else
                # an all-observed mask is the unmasked call
                @test isequal(confint(fit, Y; method = :wald, mask = trues(p, n), kw...).se,
                              confint(fit, Y; method = :wald, kw...).se)
                # unmasked calls keep working and are deterministic
                a = confint(fit, Y; method = :wald, mask = nothing, kw...)
                b = confint(fit, Y; method = :wald, kw...)
                @test isequal(a.se, b.se) && isequal(a.estimate, b.estimate)
                @test length(b.term) > 0
            end
        end
    end

    # masked covariate fits: unmasked confint must not return wrong SEs (#1260)
    @testset "masked GllvmCovFit objective guard (#1260)" begin
        rng = MersenneTwister(1260)
        p2, n2, K2 = 4, 60, 1
        x = randn(rng, n2)
        X = reshape(repeat(x', p2, 1), p2, n2, 1)
        Y = zeros(p2, n2)
        for s in 1:n2
            z = randn(rng)
            for t in 1:p2
                Y[t, s] = rand(rng, Poisson(exp(0.3 + 0.4 * x[s] + 0.7 * z * (t == 1 ? 1.0 : 0.5))))
            end
        end
        mk2 = trues(p2, n2); mk2[:, 1:20] .= false
        fm = fit_gllvm_cov(Y; family = Poisson(), X = X, K = K2, mask = mk2, iterations = 80)
        msg = try
            confint(fm, Y; method = :wald, X = X)
            nothing
        catch e
            e isa ArgumentError ? sprint(showerror, e) : rethrow()
        end
        @test msg !== nothing
        @test occursin("log-likelihood", msg) || occursin("objective", msg)
        fs = fit_gllvm_cov(Y[:, 21:end]; family = Poisson(), X = X[:, 21:end, :], K = K2, iterations = 80)
        ref = confint(fs, Y[:, 21:end]; method = :wald, X = X[:, 21:end, :])
        @test isfinite(ref.se[1])
    end

    # fit types that do use the mask are unaffected
    # (the exponential family needs a strictly positive response, #730, so it gets its own Y)
    Ye = max.(Yc, 0.1)
    fe = fit_exponential_gllvm(Ye; K = K, mask = mk, iterations = 20)
    @test length(confint(fe, Ye; method = :wald, mask = mk).term) > 0
end
