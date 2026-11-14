using GLLVModels, Test, Random, Distributions, StatsModels

# A formula covariate that holds a `missing` used to stop with `KeyError: key missing not
# found` (refs #1190). How a `Union{Missing,T}` column with no `missing` is read is not
# changed here.
@testset "formula covariate columns with a Missing in the element type" begin
    rng = MersenneTwister(11)
    p, n = 4, 40
    x = randn(rng, n)
    xi = rand(rng, -2:2, n)
    g = rand(rng, ["a", "b", "c"], n)
    eta = [0.3 * j + 0.5 * x[s] for j in 1:p, s in 1:n]
    Yg = eta .+ 0.4 .* randn(rng, p, n)
    Yp = [Float64(rand(rng, Poisson(exp(0.3 * e)))) for e in eta]
    xu = Vector{Union{Missing, Float64}}(x)
    xiu = Vector{Union{Missing, Int}}(xi)

    @testset "a Union column with no missing value is not refused" begin
        @test gllvm(@formula(y ~ 1 + x), Yg, (x = xu,); K = 1) isa GLLVModels.AnyGllvmFit
        @test gllvm(@formula(y ~ 1 + x), Yp, (x = xiu,); family = Poisson(), K = 1) isa GLLVModels.AnyGllvmFit
    end

    @testset "a missing value is a clear ArgumentError naming the column ($(nameof(typeof(fam))))" for
            (fam, Y) in ((Normal(), Yg), (Poisson(), Yp))
        xm = copy(xu); xm[4] = missing
        err = try gllvm(@formula(y ~ 1 + x), Y, (x = xm,); family = fam, K = 1); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("`x`", err.msg) && occursin("site 4", err.msg)
        gm = Vector{Union{Missing, String}}(g); gm[7] = missing
        err = try gllvm(@formula(y ~ 1 + x + g), Y, (x = x, g = gm); family = fam, K = 1); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("`g`", err.msg) && occursin("site 7", err.msg)
        @test_throws ArgumentError gllvm(@formula(y ~ 1 + x), Y, (x = xm,); family = fam, K = 1, pervar = fam isa Normal)
    end

    @testset "a column outside the formula may hold missing" begin
        other = Vector{Union{Missing, Float64}}(randn(rng, n)); other[2] = missing
        fa = gllvm(@formula(y ~ 1 + x), Yg, (x = x,); K = 1)
        fb = gllvm(@formula(y ~ 1 + x), Yg, (x = x, other = other); K = 1)
        @test loglikelihood(fb) == loglikelihood(fa)
    end

    @testset "other column types are unchanged" begin
        fa = gllvm(@formula(y ~ 1 + g), Yg, (g = g,); K = 1)
        fb = gllvm(@formula(y ~ 1 + g), Yg, (g = Vector{Union{Missing, String}}(g),); K = 1)
        @test loglikelihood(fb) == loglikelihood(fa)
        @test dof(fa) == dof(fb)
    end
end
