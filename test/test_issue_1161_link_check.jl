# Up-front link refusal for Poisson Laplace and Beta grouped-dispersion fitters (refs #1161).
using GLLVModels, Test, Random, Distributions
const GM = GLLVModels

function _issue1161_data()
    rng = MersenneTwister(11)
    p, n = 4, 40
    L = [0.8, 0.4, -0.5, 0.3]
    z = randn(rng, n)
    η = [rand(rng) * 0.7 + 0.3 + 0.8 * L[j] * z[i] for j in 1:p, i in 1:n]
    Yp = Matrix{Int}([rand(rng, Poisson(exp(x))) for x in η])
    clampu(x) = clamp(x, 1e-4, 1 - 1e-4)
    Yb = [(m = 1 / (1 + exp(-(x - 0.5))); clampu(rand(rng, Beta(m * 8, (1 - m) * 8)))) for x in η]
    return Yp, Yb
end

@testset "issue 1161: Poisson Laplace link check" begin
    Yp, _ = _issue1161_data()
    for bad in (GM.LogitLink(), GM.ProbitLink(), GM.CLogLogLink())
        err = try
            fit_poisson_gllvm(Yp; K = 1, link = bad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Poisson", err.msg)
        @test occursin(string(nameof(typeof(bad))), err.msg)
        @test occursin("LogLink", err.msg) && occursin("IdentityLink", err.msg)
    end
    @test fit_poisson_gllvm(Yp; K = 1, link = GM.LogLink()).converged
    @test fit_poisson_gllvm(Yp; K = 1, link = GM.IdentityLink()).converged
    err = try
        fit_gllvm(Yp; family = GM.Poisson(), K = 1, link = GM.ProbitLink())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
end

@testset "issue 1161: Beta grouped-dispersion link check" begin
    _, Yb = _issue1161_data()
    for bad in (GM.ProbitLink(), GM.CLogLogLink(), GM.IdentityLink(), GM.LogLink())
        err = try
            fit_gllvm(Yb; family = GM.Beta(), K = 1, link = bad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Beta", err.msg)
        @test occursin(string(nameof(typeof(bad))), err.msg)
        @test occursin("LogitLink", err.msg)
    end
    @test fit_gllvm(Yb; family = GM.Beta(), K = 1, link = GM.LogitLink()).converged
    # Ungrouped route keeps non-logit links that still optimise successfully.
    @test fit_beta_gllvm(Yb; K = 1, link = GM.ProbitLink()).converged
end