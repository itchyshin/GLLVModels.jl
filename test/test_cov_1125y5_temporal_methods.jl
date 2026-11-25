# Lane 1125y5: temporal_methods.jl extract, profile helpers, small fit postfit.
using Test, GLLVModels, Random, LinearAlgebra, Statistics
const G = GLLVModels

function _tiny_temporal_data()
    rows = [(s, o, t) for t in ["a", "b", "c"] for o in 1:3 for s in 1:2]
    d = (
        series = [r[1] for r in rows],
        occasion = Float64[r[2] for r in rows],
        trait = [r[3] for r in rows],
    )
    merge(d, (value = randn(length(rows)),))
end

@testset "1125y5 temporal_methods internals" begin
    p = collect(0.0:0.1:1.0)
    v = (p .- 0.5) .^ 2
    @test G._temporal_terminus_status(p, v, :lower) in (:asymptotic, :truncated, :undecidable)
    @test G._temporal_terminus_status([1.0], [1.0], :upper) === :undecidable

    theta = collect(0.0:0.05:1.0)
    value = (theta .- 0.4) .^ 2
    b = G._temporal_profile_bounds(theta, value, minimum(value), 0.4, 1.0)
    @test haskey(b, :lower) && haskey(b, :upper)

    Λ = [0.9 0.0; 0.0 0.8]
    rep = G._temporal_report_sign(Λ; traits = ["t1", "t2"])
    @test rep.multiplier in (-1, 1)
    @test rep.anchor_trait in ("t1", "t2")
    @test isfinite(rep.anchor_loading)
end

@testset "1125y5 temporal_methods small fit surfaces" begin
    d = _tiny_temporal_data()
    term = temporal_indep(:(0 + trait | series), :occasion; structure = :ar1)
    fit = fit_temporal_gllvm(d; formula = @formula(value ~ 0 + trait), temporal = term,
        iterations = 12)
    @test fit isa G.TemporalGaussianFit
    ext = extract_temporal(fit)
    @test ext isa NamedTuple
    lv = getLV(fit)
    @test lv === nothing || lv isa NamedTuple
    ord = extract_ordination(fit; level = :unit)
    @test ord === nothing || ord isa NamedTuple
    pred = predict(fit)
    @test length(pred.est) == length(d.value)
    buf = sprint(show, "text/plain", fit)
    @test occursin("TemporalGaussianFit", buf) || occursin("loglik", buf)
    other = G._temporal_other_tiers(fit)
    @test other isa Vector
end