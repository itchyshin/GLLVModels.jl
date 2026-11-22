# refs #1283: temporal formula covariates with Union{Missing,Number} and missing values.
using Test, GLLVModels, StatsModels, Random

@testset "fit_temporal_gllvm Union{Missing} covariates (refs #1283)" begin
    rng = MersenneTwister(1)
    rows = [(s, t, j) for s in 1:6 for t in 1:8 for j in 1:3]
    N = length(rows)
    xv = randn(rng, N)
    base = (
        value = randn(rng, N),
        trait = ["t$(r[3])" for r in rows],
        series = ["s$(r[1])" for r in rows],
        occasion = Float64[r[2] for r in rows],
    )
    tt = temporal_indep(:(0 + trait | series), :occasion; structure = :ar1)
    f = @formula(value ~ 0 + trait + x)

    fit_f = fit_temporal_gllvm(merge(base, (x = xv,)); formula = f, temporal = tt)
    fit_u = fit_temporal_gllvm(merge(base, (x = Vector{Union{Missing, Float64}}(xv),));
                               formula = f, temporal = tt)
    @test length(coef(fit_f)) == length(coef(fit_u)) == 4
    @test dof(fit_f) == dof(fit_u) == 9
    @test loglikelihood(fit_f) == loglikelihood(fit_u)

    xm = Vector{Union{Missing, Float64}}(xv)
    xm[5] = missing
    err = try
        fit_temporal_gllvm(merge(base, (x = xm,)); formula = f, temporal = tt)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("`x`", err.msg)
    @test occursin("missing", err.msg)
    @test occursin("fit_temporal_gllvm", err.msg)

    @testset "coalesce(x, 0.0) with incomplete Union column" begin
        xm = Vector{Union{Missing, Float64}}(xv)
        xm[5] = missing
        f_coalesce = @formula(value ~ 0 + trait + coalesce(x, 0.0))
        fit_c = fit_temporal_gllvm(merge(base, (x = xm,)); formula = f_coalesce, temporal = tt)
        fit_pre = fit_temporal_gllvm(
            merge(base, (x = coalesce.(xm, 0.0),)); formula = f, temporal = tt)
        @test isfinite(loglikelihood(fit_c))
        @test length(coef(fit_c)) == length(coef(fit_pre)) == 4
        @test loglikelihood(fit_c) == loglikelihood(fit_pre)
    end
end