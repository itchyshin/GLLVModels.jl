# Lane 1125x round 3: formula.jl internal symbol extraction and design guards.
using Test, GLLVModels, StatsModels, Distributions

const G = GLLVModels

@testset "1125x formula internal helpers" begin
    @testset "_extract_formula_symbols and _formula_bare_symbols" begin
        f = @formula(y ~ 1 + x + log(z) + x * g)
        syms = G._extract_formula_symbols(f.rhs)
        @test :x in syms
        @test :z in syms
        @test :g in syms
        bare = G._formula_bare_symbols(f.rhs)
        @test :x in bare && :g in bare
        @test :z ∉ bare
    end

    @testset "_build_site_modelmatrix" begin
        n = 8
        data = (x = collect(1.0:n), habitat = fill("a", n))
        habitat = repeat(["a", "b"], outer = n ÷ 2)
        data = (x = collect(1.0:n), habitat = habitat)
        mm, cnames = G._build_site_modelmatrix((@formula(y ~ 1 + x)).rhs, data)
        @test size(mm) == (n, 1)
        @test cnames == ["x"]
        mm0, cn0 = G._build_site_modelmatrix(StatsModels.ConstantTerm(1), data)
        @test size(mm0) == (n, 0)
        @test cn0 == String[]
        @test_throws ArgumentError G._build_site_modelmatrix((@formula(y ~ 1 + missingcol)).rhs, data)
    end

    @testset "_formula_covariate_columns" begin
        n = 6
        cols = (x = Vector{Union{Missing, Float64}}(collect(1.0:n)),)
        cols.x[3] = missing
        rhs = (@formula(y ~ 1 + x)).rhs
        err = try
            G._formula_covariate_columns(rhs, cols)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("site 3", err.msg)
        err_long = try
            G._formula_covariate_columns(rhs, cols; long_route = "fit_rows")
            nothing
        catch e
            e
        end
        @test err_long isa ArgumentError
        @test occursin("fit_rows", err_long.msg)
        @test occursin("row 3", err_long.msg)
        cols2 = (x = Vector{Union{Missing, Float64}}(collect(1.0:n)),)
        G._formula_covariate_columns(rhs, cols2) === cols2
    end

    @testset "_default_route_no_intercept_error" begin
        f0 = @formula(y ~ 0 + x)
        @test_throws ArgumentError G._default_route_no_intercept_error(f0, Poisson())
        @test_throws ArgumentError G._default_route_no_intercept_error(f0, Binomial())
        G._default_route_no_intercept_error(f0, Normal()) === nothing
        G._default_route_no_intercept_error(@formula(y ~ 1 + x), Poisson()) === nothing
    end
end