# refs #1283: complete numeric covariate stored as Union{Missing,Float64} matches Float64.
using GLLVModels, Test, Random, StatsModels

@testset "temporal Union Missing numeric covariate reads as continuous (refs #1283)" begin
    rng = MersenneTwister(1)
    rows = [(s, t, j) for s in 1:6 for t in 1:8 for j in 1:3]
    N = length(rows)
    xv = randn(rng, N)
    base = (value = randn(rng, N), trait = ["t$(r[3])" for r in rows],
            series = ["s$(r[1])" for r in rows], occasion = Float64[r[2] for r in rows])
    tt = temporal_indep(:(0 + trait | series), :occasion; structure = :ar1)
    f = @formula(value ~ 0 + trait + x)
    xu = Vector{Union{Missing, Float64}}(xv)

    fa = fit_temporal_gllvm(merge(base, (x = xv,)); formula = f, temporal = tt)
    fb = fit_temporal_gllvm(merge(base, (x = xu,)); formula = f, temporal = tt)
    @test length(coef(fa)) == length(coef(fb)) == 4
    @test dof(fa) == dof(fb) == 9
    @test loglikelihood(fa) == loglikelihood(fb)
end