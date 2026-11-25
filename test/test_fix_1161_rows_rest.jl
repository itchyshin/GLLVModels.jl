# Up-front link refusal for Student-t, Binomial, and Beta-binomial (refs #1161).
using GLLVModels, Test, Random, Distributions
const GM = GLLVModels

function _issue1161_rows_rest_data()
    rng = MersenneTwister(11)
    p, n = 4, 40
    L = [0.8, 0.4, -0.5, 0.3]
    z = randn(rng, n)
    η = [rand(rng) * 0.7 + 0.3 + 0.8 * L[j] * z[i] for j in 1:p, i in 1:n]
    Ybin = [rand(rng, Binomial(10, 1 / (1 + exp(-(x - 1))))) for x in η]
    Ybb = [begin
               m = 1 / (1 + exp(-(x - 1)))
               rand(rng, Binomial(10, rand(rng, Beta(m * 6, (1 - m) * 6))))
           end for x in η]
    Ystu = [x + 0.5 * rand(rng, TDist(4)) for x in η]
    N10 = fill(10, p, n)
    return Matrix{Int}(Ybin), Ybb, Ystu, N10
end

const _BOUNDED_BINOMIAL_LINKS = (GM.LogitLink(), GM.ProbitLink(), GM.CLogLogLink())
const _BAD_STUDENT_LINKS = (GM.LogitLink(), GM.ProbitLink(), GM.CLogLogLink(), GM.LogLink())
const _BAD_BETABINOM_LINKS = (GM.IdentityLink(), GM.LogLink())

function _expect_refusal(f, family_label::AbstractString, bad_links)
    for bad in bad_links
        err = try
            f(bad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(family_label, err.msg)
        @test occursin(string(nameof(typeof(bad))), err.msg)
    end
end

@testset "issue 1161 rows rest: Student-t link check" begin
    _, _, Ystu, _ = _issue1161_rows_rest_data()
    _expect_refusal(bad -> fit_studentt_gllvm(Ystu; K = 1, disp_group = :species, link = bad),
        "Student-t", _BAD_STUDENT_LINKS)
    fit = fit_studentt_gllvm(Ystu; K = 1, disp_group = :species, nu = 4.0)
    @test isfinite(fit.loglik)
    err = try
        fit_gllvm(Ystu; family = GM.StudentTFamily(4.0), K = 1, link = GM.LogLink())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
end

@testset "issue 1161 rows rest: Binomial link check" begin
    Ybin, _, _, N10 = _issue1161_rows_rest_data()
    _expect_refusal(bad -> fit_binomial_gllvm(Ybin; K = 1, link = bad, N = N10),
        "Binomial", (GM.IdentityLink(), GM.LogLink()))
    for good in _BOUNDED_BINOMIAL_LINKS
        fit = fit_binomial_gllvm(Ybin; K = 1, link = good, N = N10)
        @test fit.converged
        @test fit.link isa typeof(good)
    end
end

@testset "issue 1161 rows rest: Beta-binomial link check" begin
    _, Ybb, _, N10 = _issue1161_rows_rest_data()
    _expect_refusal(bad -> fit_gllvm(Ybb; family = GM.BetaBinom(), K = 1, N = N10, link = bad),
        "Beta-binomial", _BAD_BETABINOM_LINKS)
    _expect_refusal(bad -> fit_beta_binomial_gllvm(Ybb; K = 1, N = N10, link = bad),
        "Beta-binomial", _BAD_BETABINOM_LINKS)
    for good in _BOUNDED_BINOMIAL_LINKS
        fit = fit_gllvm(Ybb; family = GM.BetaBinom(), K = 1, N = N10, link = good, iterations = 80)
        @test fit.link isa typeof(good)
    end
end