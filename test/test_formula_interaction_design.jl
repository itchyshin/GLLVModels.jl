using GLLVModels, Test, Random, Distributions, StatsModels

@testset "formula interaction design matches R rank rules (#1236)" begin
    @testset "_build_site_modelmatrix: x & g without x main" begin
        n = 45
        rng = MersenneTwister(1)
        x = randn(rng, n)
        g = shuffle(rng, repeat(["a", "b", "c"], div(n, 3)))
        data = (x = x, g = g)
        mm_and, _ = GLLVModels._build_site_modelmatrix(
            (@formula(y ~ 1 + g + x & g)).rhs, data)
        mm_star, _ = GLLVModels._build_site_modelmatrix(
            (@formula(y ~ 1 + x * g)).rhs, data)
        @test size(mm_and, 2) == size(mm_star, 2)
        ref = FormulaTerm(ConstantTerm(0), (@formula(y ~ 1 + g + x & g)).rhs)
        sch = schema(ref, data)
        applied = apply_schema(ref, sch, StatisticalModel)
        mm_ref = Matrix{Float64}(modelmatrix(applied.rhs, data))
        names_ref = coefnames(applied.rhs)
        names_ref = names_ref isa AbstractVector ? string.(names_ref) : [string(names_ref)]
        keep = findall(!=("(Intercept)"), names_ref)
        @test size(mm_and, 2) == length(keep)
        @test mm_and ≈ mm_ref[:, keep]
    end

    @testset "Poisson: g + x&g and x&g span the same model as x * g" begin
        rng = MersenneTwister(1)
        p, n = 5, 45
        x = randn(rng, n)
        g = shuffle(rng, repeat(["a", "b", "c"], 15))
        Y = [rand(rng, Poisson(exp(0.5 + 0.3 * x[s] * (g[s] == "b")))) for t in 1:p, s in 1:n]
        D = (x = x, g = g)
        f_and = gllvm(@formula(y ~ 1 + g + x & g), Y, D; family = Poisson(), K = 1)
        f_star = gllvm(@formula(y ~ 1 + x * g), Y, D; family = Poisson(), K = 1)
        f_only = gllvm(@formula(y ~ 1 + x & g), Y, D; family = Poisson(), K = 1)
        @test length(f_and.γ) == length(f_star.γ)
        @test length(f_only.γ) == 3  # one x slope per level of g (R's x:g without mains)
        @test f_and.loglik ≈ f_star.loglik atol = 1e-4
    end

    @testset "g + g & x2: one x2 slope per level with g main" begin
        rng = MersenneTwister(2026)
        p, n = 6, 40
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        g = shuffle(rng, repeat(["high", "low", "mid"], div(n, 3) + 1))[1:n]
        Y = rand(rng, Poisson(3), p, n)
        D = (x1 = x1, x2 = x2, g = g)
        mm, cnames = GLLVModels._build_site_modelmatrix(
            (@formula(y ~ 1 + x1 + g + g & x2)).rhs, D)
        @test size(mm, 2) == 6
        @test count(nm -> contains(nm, "& x2"), cnames) == 3
        f_mixed = gllvm(@formula(y ~ 1 + x1 + g + g & x2), Y, D; family = Poisson(), K = 1)
        @test length(f_mixed.γ) == size(mm, 2)
    end
end