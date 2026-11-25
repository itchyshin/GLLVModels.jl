# Up-front link refusal for NB2, NB1, GP-1, Gamma, and Tweedie Laplace fitters (refs #1161).
using GLLVModels, Test, Random, Distributions
const GM = GLLVModels

function _issue1161_rest_data()
    rng = MersenneTwister(11)
    p, n = 4, 40
    L = [0.8, 0.4, -0.5, 0.3]
    z = randn(rng, n)
    η = [rand(rng) * 0.7 + 0.3 + 0.8 * L[j] * z[i] for j in 1:p, i in 1:n]
    Ynb2 = Matrix{Int}([rand(rng, NegativeBinomial(3, 3 / (3 + exp(x)))) for x in η])
    Ynb1 = Matrix{Int}([rand(rng, NegativeBinomial(exp(x), 0.5)) for x in η])
    Ygp1 = Matrix{Int}([rand(rng, Poisson(exp(x))) for x in η])
    Yγ = [rand(rng, Gamma(3, exp(x) / 3)) for x in η]
    tw = zeros(p, n)
    for j in 1:p, i in 1:n
        μ = exp(η[j, i])
        k = rand(rng, Poisson(μ^0.4 / 0.4))
        tw[j, i] = k > 0 ? sum(rand(rng, Gamma(0.4 / 0.6, 0.6 * μ^0.6), k)) : 0.0
    end
    return Ynb2, Ynb1, Ygp1, Yγ, tw
end

const _BOUNDED_BAD_LINKS = (GM.LogitLink(), GM.ProbitLink(), GM.CLogLogLink())
const _NON_LOG_LINKS = (
    GM.LogitLink(),
    GM.ProbitLink(),
    GM.CLogLogLink(),
    GM.IdentityLink(),
)

function _expect_log_link_only(f, family_label::AbstractString; bad_links = _NON_LOG_LINKS)
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
        @test occursin("LogLink", err.msg)
    end
end

@testset "issue 1161 rest: NB2 link check" begin
    Ynb2, _, _, _, _ = _issue1161_rest_data()
    _expect_log_link_only(bad -> fit_gllvm(Ynb2; family = GM.NegativeBinomial(), K = 1,
            disp_group = :species, link = bad), "NB2")
    @test fit_gllvm(Ynb2; family = GM.NegativeBinomial(), K = 1, disp_group = :species,
            link = GM.LogLink()).converged
    _expect_log_link_only(bad -> fit_nb_gllvm(Ynb2; K = 1, link = bad), "NB2"; bad_links = _BOUNDED_BAD_LINKS)
    @test fit_nb_gllvm(Ynb2; K = 1, link = GM.LogLink()).converged
    @test fit_nb_gllvm(Ynb2; K = 1, link = GM.IdentityLink()).converged
end

@testset "issue 1161 rest: NB1 link check" begin
    _, Ynb1, _, _, _ = _issue1161_rest_data()
    _expect_log_link_only(bad -> fit_gllvm(Ynb1; family = GM.NB1(), K = 1, link = bad), "NB1")
    @test fit_gllvm(Ynb1; family = GM.NB1(), K = 1, link = GM.LogLink()).converged
    _expect_log_link_only(bad -> fit_nb1_gllvm(Ynb1; K = 1, link = bad), "NB1")
end

@testset "issue 1161 rest: GP-1 link check" begin
    _, _, Ygp1, _, _ = _issue1161_rest_data()
    α = 0.1
    _expect_log_link_only(bad -> fit_gllvm(Ygp1; family = GM.GeneralizedPoisson1(α), K = 1, link = bad),
        "GP-1")
    @test fit_gllvm(Ygp1; family = GM.GeneralizedPoisson1(α), K = 1, link = GM.LogLink()).converged
    _expect_log_link_only(bad -> fit_gp1_gllvm(Ygp1; K = 1, link = bad), "GP-1")
end

@testset "issue 1161 rest: Gamma link check" begin
    _, _, _, Yγ, _ = _issue1161_rest_data()
    _expect_log_link_only(bad -> fit_gllvm(Yγ; family = GM.Gamma(), K = 1, disp_group = :species, link = bad),
        "Gamma")
    @test fit_gllvm(Yγ; family = GM.Gamma(), K = 1, disp_group = :species, link = GM.LogLink()).converged
    _expect_log_link_only(bad -> fit_gamma_gllvm(Yγ; K = 1, link = bad), "Gamma"; bad_links = _BOUNDED_BAD_LINKS)
    @test fit_gamma_gllvm(Yγ; K = 1, link = GM.IdentityLink()).converged
end

@testset "issue 1161 rest: Tweedie link check" begin
    _, _, _, _, Ytw = _issue1161_rest_data()
    _expect_log_link_only(bad -> fit_tweedie_gllvm_grouped(Ytw; K = 1, power_group = :species, link = bad),
        "Tweedie")
    @test fit_tweedie_gllvm_grouped(Ytw; K = 1, power_group = :species, link = GM.LogLink()).converged
    _expect_log_link_only(bad -> fit_tweedie_gllvm(Ytw; K = 1, link = bad), "Tweedie")
end