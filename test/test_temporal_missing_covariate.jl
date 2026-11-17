using GLLVModels, Test, Random, StatsModels

# A temporal-fit covariate that holds a `missing` used to stop with `KeyError: key missing
# not found` inside `StatsModels.schema` (refs #1283). It now gets the same check as the
# `gllvm` formula route (refs #1190). How a `Union{Missing,T}` column with no `missing` is
# read is not changed here.
@testset "temporal fit covariate columns with a Missing in the element type" begin
    rng = MersenneTwister(1)
    rows = [(s, t, j) for s in 1:6 for t in 1:8 for j in 1:3]
    N = length(rows)
    xv = randn(rng, N)
    base = (value = randn(rng, N), trait = ["t$(r[3])" for r in rows],
            series = ["s$(r[1])" for r in rows], occasion = Float64[r[2] for r in rows])
    tt = temporal_indep(:(0 + trait | series), :occasion; structure = :ar1)
    f = @formula(value ~ 0 + trait + x)

    @testset "a missing value is an ArgumentError naming the column" begin
        xm = Vector{Union{Missing, Float64}}(xv); xm[5] = missing
        err = try fit_temporal_gllvm(merge(base, (x = xm,)); formula = f, temporal = tt); nothing catch e; e end
        @test err isa ArgumentError
        @test err.msg == "covariate `x` in the formula has 1 `missing` value(s) (first at row 5); " *
            "`fit_temporal_gllvm` does not drop or impute rows. Remove those rows from the data, or " *
            "fill the covariate, before fitting"
        g = Vector{Union{Missing, String}}(rand(rng, ["a", "b"], N)); g[9] = missing
        err = try fit_temporal_gllvm(merge(base, (x = xv, g = g)); formula = @formula(value ~ 0 + trait + x + g),
                                     temporal = tt); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("`g`", err.msg) && occursin("row 9", err.msg)
    end

    @testset "the `gllvm` route keeps its own wording" begin
        xm = Vector{Union{Missing, Float64}}(randn(rng, 10)); xm[3] = missing
        err = try gllvm(@formula(y ~ 1 + x), randn(rng, 3, 10), (x = xm,); K = 1); nothing catch e; e end
        @test err isa ArgumentError
        @test err.msg == "covariate `x` in the formula has 1 `missing` value(s) (first at site 3); " *
            "`gllvm` does not drop or impute sites. Remove those sites from `Y` and `data`, or fill " *
            "the covariate, before fitting"
    end

    @testset "a column used only inside a function is left to that function" begin
        xm = Vector{Union{Missing, Float64}}(xv); xm[5] = missing
        fa = fit_temporal_gllvm(merge(base, (x = xm,)); formula = @formula(value ~ 0 + trait + coalesce(x, 0.0)),
                                temporal = tt)
        fb = fit_temporal_gllvm(merge(base, (x = coalesce.(xm, 0.0),)); formula = f, temporal = tt)
        @test loglikelihood(fa) == loglikelihood(fb)
        @test length(coef(fa)) == length(coef(fb)) == 4
    end

    @testset "a column outside the formula may hold missing" begin
        other = Vector{Union{Missing, Float64}}(randn(rng, N)); other[2] = missing
        fa = fit_temporal_gllvm(merge(base, (x = xv,)); formula = f, temporal = tt)
        fb = fit_temporal_gllvm(merge(base, (x = xv, other = other)); formula = f, temporal = tt)
        @test loglikelihood(fb) == loglikelihood(fa)
        @test length(coef(fa)) == 4
    end
end
