using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

# Delta-lognormal / Delta-Gamma fits made with `predictor = :shared` (#820), the grouped-fit
# `predict` methods (#832) and the non-finite stored offset guard (#788 item 3).
#
# #820: under `predictor = :shared` one linear predictor η = β + Λz (+ offset) drives BOTH
# parts (βz ≡ βc, Λz ≡ Λc). `getLV` searched the mode with Λz = 0, so it maximised a different
# posterior, and `predict(...; type = :occurrence)` returned logistic(βz), the same value for
# every unit, instead of logistic(βz + Λc ẑ + offset). `residuals` used the same constant π.
#
# The references are not produced by the code under test: the mode condition is the central
# finite-difference gradient of the shared two-part log posterior written out here from the
# Distributions.jl densities, and the occurrence probability is logistic of the fit's own
# β + O + Λ ẑ computed in the test.

_lg820(x) = 1 / (1 + exp(-x))
_sd820(σ, t) = σ isa Real ? σ : σ[t]

# log f(y | η) under the shared predictor: η drives the occurrence probability and the
# positive part. `pos(t, η)` is the positive-part distribution of trait t.
_ll820(pos, y, t, η) = y > 0 ? log(_lg820(η)) + logpdf(pos(t, η), y) : log1p(-_lg820(η))

function _logpost820(pos, y, β, o, Λ, z)
    η = β .+ o .+ Λ * z
    return sum(_ll820(pos, y[t], t, η[t]) for t in eachindex(y)) - dot(z, z) / 2
end

function _fdscore820(pos, y, β, o, Λ, z; h = 1e-5)
    g = similar(z)
    for k in eachindex(z)
        e = zeros(length(z)); e[k] = h
        g[k] = (_logpost820(pos, y, β, o, Λ, z .+ e) - _logpost820(pos, y, β, o, Λ, z .- e)) / (2h)
    end
    return g
end

# Largest finite-difference score norm over the sites (≈ 0 at the per-unit maximisers).
_worst820(pos, Y, β, Λ, O, Z) =
    maximum(norm(_fdscore820(pos, Y[:, s], β, O[:, s], Λ, Z[s, :])) for s in axes(Y, 2))

function _sim820(name; p = 5, n = 60, K = 1, offset = false)
    rng = MersenneTwister(820 + length(name) + offset)
    O = offset ? 0.5 .* randn(rng, p, n) : zeros(p, n)
    Λ = 0.9 .* randn(rng, p, K); Z = randn(rng, n, K)
    η = 0.4 .+ O .+ Λ * Z'
    Y = zeros(p, n)
    for s in 1:n, t in 1:p
        rand(rng) < _lg820(η[t, s]) || continue
        Y[t, s] = name == "LN" ? rand(rng, LogNormal(η[t, s], 0.5)) :
                                 rand(rng, Gamma(3.0, exp(η[t, s]) / 3))
    end
    return Y, (offset ? O : nothing)
end

@testset "Delta predictor = :shared: getLV, predict, residuals use the shared loadings (#820)" begin
    cases = (
        ("LN", (Y, O) -> fit_delta_lognormal_gllvm(Y; K = 1, predictor = :shared, offset = O),
         σ -> (t, η) -> LogNormal(η, _sd820(σ, t))),
        ("Gamma", (Y, O) -> fit_delta_gamma_gllvm(Y; K = 1, predictor = :shared, offset = O),
         α -> (t, η) -> Gamma(_sd820(α, t), exp(η) / _sd820(α, t))),
    )
    for (name, fitfun, mkpos) in cases, withoff in (false, true)
        @testset "$name, offset = $withoff" begin
            Y, O = _sim820(name; offset = withoff)
            p, n = size(Y)
            f = fitfun(Y, O)
            @test f.predictor === :shared
            disp = name == "LN" ? f.σ : f.α
            pos = mkpos(disp)
            Of = O === nothing ? zeros(p, n) : O
            Z = getLV(f, Y; rotate = false)
            # The mode is a zero of the SHARED two-part score (occurrence loads on z).
            @test _worst820(pos, Y, f.βc, f.Λc, Of, Z) < 1e-4
            @test size(Z) == (n, 1)
            # :occurrence = logistic(βz + Λc ẑ + offset), varying by unit.
            π = predict(f, Y; type = :occurrence)
            ηref = f.βz .+ Of .+ f.Λc * Z'
            @test π ≈ _lg820.(ηref)
            @test all(>(1e-6), vec(std(π; dims = 2)))
            # :link / :positive / :response are built on the same η.
            @test predict(f, Y; type = :link) ≈ ηref
            @test predict(f, Y; type = :response) ≈ π .* predict(f, Y; type = :positive)
            # residuals: a positive cell's residual uses the unit-specific π.
            R = residuals(f, Y; rng = MersenneTwister(1))
            @test all(isfinite, R)
            t, s = first(Tuple.(findall(>(0), Y)))
            u = (1 - π[t, s]) + π[t, s] * cdf(pos(t, ηref[t, s]), Y[t, s])
            @test R[t, s] ≈ quantile(Normal(), u) atol = 1e-8
        end
    end

    @testset "predictor = :separate is unchanged (constant occurrence probability)" begin
        Y, O = _sim820("LN"; offset = true)
        f = fit_delta_lognormal_gllvm(Y; K = 1, predictor = :separate, offset = O)
        π = predict(f, Y; type = :occurrence)
        @test all(π .== π[:, 1])                          # logistic(βz), per trait
        @test π[:, 1] == inv.(1 .+ exp.(-f.βz))
        # Mode: occurrence is intercept-only, so the score is the positive part only.
        Z = getLV(f, Y; rotate = false)
        pos = (t, η) -> LogNormal(η, _sd820(f.σ, t))
        function sepscore(y, o, z)
            h = 1e-5
            q(z) = (η = f.βc .+ o .+ f.Λc * z;
                    sum(y[t] > 0 ? logpdf(pos(t, η[t]), y[t]) : 0.0 for t in eachindex(y)) - dot(z, z) / 2)
            [(q(z .+ h) - q(z .- h)) / (2h)]
        end
        @test maximum(norm(sepscore(Y[:, s], O[:, s], Z[s, :])) for s in axes(Y, 2)) < 1e-4
    end
end

@testset "predict on the grouped-dispersion fits of the public route (#832)" begin
    p, n, K = 5, 50, 1
    rng = MersenneTwister(832)
    O = 0.6 .* randn(rng, p, n); Λ = 0.5 .* randn(rng, p, K); Z = randn(rng, n, K)
    η = 0.5 .+ O .+ Λ * Z'
    Yb = [rand(rng, Beta(exp(η[t, s]) / (1 + exp(η[t, s])) * 8, (1 - exp(η[t, s]) / (1 + exp(η[t, s]))) * 8))
          for t in 1:p, s in 1:n]
    Yg = [rand(rng, Gamma(3.0, exp(η[t, s]) / 3)) for t in 1:p, s in 1:n]
    Yn = [rand(rng, NegativeBinomial(4.0, 4.0 / (4.0 + exp(η[t, s])))) for t in 1:p, s in 1:n]

    for (name, Y, fam, T) in (("Beta", Yb, GM.Beta(), GM.BetaGroupedFit),
                              ("Gamma", Yg, GM.Gamma(), GM.GammaGroupedFit),
                              ("NB1", Yn, GM.NB1(), GM.NB1GroupedFit))
        @testset "$name" begin
            f = fit_gllvm(Y; family = fam, K = K, offset = O, disp_group = :species)
            @test f isa T
            Zf = getLV(f, Y; rotate = false)
            ηhat = f.β .+ O .+ f.Λ * Zf'
            @test predict(f, Y; type = :link) ≈ ηhat
            @test predict(f, Y) ≈ linkinv.(Ref(f.link), ηhat)
            # An explicit training offset is the stored one; the offset-free route differs.
            @test predict(f, Y; type = :link, offset = O) ≈ ηhat
            @test maximum(abs, predict(f, Y; type = :link, offset = zeros(p, n)) .- ηhat) > 0.1
            # New units from an offset fit need their own offset.
            @test_throws ArgumentError predict(f, Y[:, 1:10])
            @test size(predict(f, Y[:, 1:10]; offset = O[:, 1:10])) == (p, 10)
            # modes = :training keeps the training modes and swaps in the new offset.
            O2 = O .+ 0.3
            pt = predict(f, Y; type = :link, offset = O2, modes = :training)
            @test pt ≈ f.β .+ O2 .+ f.Λ * Zf'
            @test_throws ArgumentError predict(f, Y; modes = :bogus)
            @test_throws ArgumentError predict(f, Y; type = :bogus)
        end
    end

    @testset "NBGroupedFit gains modes = :training" begin
        f = fit_gllvm(Yn; family = GM.NegativeBinomial(), K = K, offset = O, disp_group = :species)
        @test f isa NBGroupedFit
        Zf = getLV(f, Yn; rotate = false)
        O2 = O .+ 0.3
        @test predict(f, Yn; type = :link, offset = O2, modes = :training) ≈
              f.β .+ O2 .+ f.Λ * Zf'
    end
end

@testset "stored non-finite offset without a mask is refused (#788)" begin
    rng = MersenneTwister(788)
    p, n, K = 5, 40, 1
    Λ = 0.6 .* randn(rng, p, K); Z = randn(rng, n, K)
    Y = [rand(rng, Poisson(exp(0.8 + dot(Λ[t, :], Z[s, :])))) for t in 1:p, s in 1:n]
    M = trues(p, n); M[1:2, 1:10] .= false
    O = 0.3 .* randn(rng, p, n); O[.!M] .= NaN
    Yp = copy(Y); Yp[.!M] .= 0
    f = fit_poisson_gllvm(Yp; K = K, mask = M, offset = O)
    @test_throws ArgumentError getLV(f, Yp)
    @test_throws ArgumentError predict(f, Yp)
    S = getLV(f, Yp; mask = M)                 # with the mask the stored offset is usable
    @test all(isfinite, S)
    # An explicit finite offset is always accepted.
    @test all(isfinite, getLV(f, Yp; offset = zeros(p, n)))
    # A finite stored offset is untouched by the guard.
    g = fit_poisson_gllvm(Y; K = K, offset = fill(0.2, p, n))
    @test all(isfinite, getLV(g, Y))
end
