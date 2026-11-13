using GLLVModels, Test, Random, Distributions, StatsModels

# A formula covariate typed `Union{Missing,T}` used to be read by StatsModels as categorical
# (one indicator per distinct value) even when it held no `missing`, and a column that did
# hold a `missing` stopped with `KeyError: key missing not found` (refs #1190).
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

    @testset "no missing value: same model as the plain column ($(nameof(typeof(fam))))" for
            (fam, Y) in ((Normal(), Yg), (Poisson(), Yp))
        fa = gllvm(@formula(y ~ 1 + x), Y, (x = x,); family = fam, K = 1)
        fb = gllvm(@formula(y ~ 1 + x), Y, (x = xu,); family = fam, K = 1)
        @test loglikelihood(fb) == loglikelihood(fa)
        @test dof(fb) == dof(fa)
        fc = gllvm(@formula(y ~ 1 + x), Y, (x = Float64.(xi),); family = fam, K = 1)
        fd = gllvm(@formula(y ~ 1 + x), Y, (x = xiu,); family = fam, K = 1)
        @test loglikelihood(fd) == loglikelihood(fc)
        @test dof(fd) == dof(fc)
    end
    @test length(gllvm(@formula(y ~ 1 + x), Yp, (x = xu,); family = Poisson(), K = 1).γ) == 1

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
