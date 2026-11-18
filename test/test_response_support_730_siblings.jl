# #730 (siblings of the Poisson / Binomial / Beta / Gamma guard): the remaining scalar
# Laplace fitters must reject out-of-support or non-finite OBSERVED responses with an
# ArgumentError naming the fitter and the first bad cell, instead of returning a fit with
# loglik = -Inf or failing late with an InexactError. Masked cells stay untouched.
using Test, GLLVModels, Random

@testset "#730 response-domain validation, scalar siblings" begin
    rng = MersenneTwister(7)
    p, n = 4, 40
    Ycnt  = rand(rng, 0:6, p, n)                    # counts, zeros allowed
    Ytrc  = rand(rng, 1:6, p, n)                    # zero-truncated counts
    Ypos  = exp.(randn(rng, p, n))                  # y > 0
    Ynn   = Ypos .* (rand(rng, p, n) .> 0.3)        # y >= 0 with exact zeros (Tweedie)
    Yunit = rand(rng, p, n); Yunit[1, 2] = 0.0; Yunit[2, 3] = 1.0   # [0, 1] closed
    Yreal = randn(rng, p, n)
    Nbb   = fill(6, p, n)

    function with_cell(Y, v)
        Z = Matrix{Any}(Y); Z[1, 1] = v
        return identity.(Z)
    end
    thrown(f) = try; f(); nothing; catch e; e; end
    msg_ok(e, name) = e isa ArgumentError && occursin(name, e.msg) && occursin("Y[1, 1]", e.msg)

    # (fitter name, fitter, clean data, bad values, extra kwargs)
    cases = [
        ("fit_nb_gllvm",               fit_nb_gllvm,               Ycnt,  (-1, 2.5, NaN), (;)),
        ("fit_nb1_gllvm",              fit_nb1_gllvm,              Ycnt,  (-1, 2.5, NaN), (;)),
        ("fit_gp1_gllvm",              fit_gp1_gllvm,              Ycnt,  (-1, 2.5, NaN), (;)),
        ("fit_compoisson_gllvm",       fit_compoisson_gllvm,       Ycnt,  (-1, 2.5, NaN), (;)),
        ("fit_beta_binomial_gllvm",    fit_beta_binomial_gllvm,    Ycnt,  (-1, 2.5, NaN, 7), (; N = Nbb)),
        ("fit_tweedie_gllvm",          fit_tweedie_gllvm,          Ynn,   (-1.0, NaN, Inf), (;)),
        ("fit_exponential_gllvm",      fit_exponential_gllvm,      Ypos,  (-1.0, 0.0, NaN), (;)),
        ("fit_truncated_poisson_gllvm",   fit_truncated_poisson_gllvm,   Ytrc, (-1, 2.5, NaN), (;)),
        ("fit_truncated_nbinom2_gllvm",   fit_truncated_nbinom2_gllvm,   Ytrc, (-1, 2.5, NaN), (;)),
        ("fit_censored_poisson_gllvm", fit_censored_poisson_gllvm, Ycnt,  (-1, 2.5, NaN), (;)),
        ("fit_ordered_beta_gllvm",     fit_ordered_beta_gllvm,     Yunit, (-0.1, 1.5, NaN), (;)),
    ]

    for (name, f, Y, bads, kw) in cases
        @testset "$name" begin
            fit = f(Y; K = 1, kw...)
            @test isfinite(fit.loglik)
            for bad in bads
                Z = with_cell(eltype(Y) <: Integer && bad isa Integer ? Y : float.(Y), bad)
                @test msg_ok(thrown(() -> f(Z; K = 1, kw...)), name)
            end
            # a bad value under mask = false is never read
            mask = trues(p, n); mask[1, 1] = false
            Zm = with_cell(float.(Y), NaN)
            fm = f(Zm; K = 1, mask = mask, kw...)
            @test isfinite(fm.loglik)
            Zb = with_cell(float.(Y), bads[1])
            @test isfinite(f(Zb; K = 1, mask = mask, kw...).loglik)
        end
    end

    @testset "zero-truncated fitters still reject y = 0 (dedicated check)" begin
        for f in (fit_truncated_poisson_gllvm, fit_truncated_nbinom2_gllvm)
            @test thrown(() -> f(with_cell(float.(Ytrc), 0.0); K = 1)) isa ArgumentError
        end
    end

    @testset "fit_studentt_gllvm" begin
        @test isfinite(fit_studentt_gllvm(Yreal; K = 1).loglik)
        for bad in (NaN, Inf, -Inf)
            Z = copy(Yreal); Z[1, 1] = bad
            @test msg_ok(thrown(() -> fit_studentt_gllvm(Z; K = 1)), "fit_studentt_gllvm")
        end
    end

    @testset "censored Poisson: the censored route is checked, explicit bounds are not" begin
        cen = falses(p, n); cen[2, 2] = true
        @test isfinite(fit_censored_poisson_gllvm(Ytrc; K = 1, censored = cen).loglik)
        Z = with_cell(float.(Ytrc), -1.0)
        @test msg_ok(thrown(() -> fit_censored_poisson_gllvm(Z; K = 1, censored = cen)),
                     "fit_censored_poisson_gllvm")
    end

    @testset "missing cells and zero-weight-free routes still fit" begin
        Zm = Matrix{Union{Missing, Int}}(Ycnt); Zm[1, 1] = missing
        @test isfinite(fit_nb_gllvm(Zm; K = 1).loglik)
        Zt = Matrix{Union{Missing, Float64}}(Ypos); Zt[2, 2] = missing
        @test isfinite(fit_exponential_gllvm(Zt; K = 1).loglik)
    end
end
