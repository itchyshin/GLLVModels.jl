using GLLVModels, Test, Random, Distributions, StatsModels

# The site design of the non-Gaussian covariate routes must follow StatsModels' rank rules, as
# the Gaussian routes do and as R does: `g + x:g` has one slope of x per level of g, so it spans
# the same model as `x * g`. Before the fix the schema was applied with no model context and no
# intercept, which dropped the first level of g from the interaction and forced its slope to 0.
@testset "formula site design: interaction without the main effect keeps every level" begin
    rng = MersenneTwister(20261113)
    p, n, K = 5, 45, 1
    x = randn(rng, n)
    g = shuffle(rng, repeat(["a", "b", "c"], n ÷ 3))
    slope = Dict("a" => 0.2, "b" => 0.6, "c" => -0.5)
    a = 0.3 .* randn(rng, p)
    lam = 0.6 .* randn(rng, p)
    z = randn(rng, n)
    eta = [a[t] + slope[g[s]] * x[s] + lam[t] * z[s] for t in 1:p, s in 1:n]
    Y = [rand(rng, Poisson(exp(0.5 + 0.5 * e))) for e in eta]
    D = (x = x, g = g)
    kw = (family = Poisson(), K = K)

    f_full = gllvm(@formula(y ~ 1 + x * g), Y, D; kw...)
    f_nested = gllvm(@formula(y ~ 1 + g + x & g), Y, D; kw...)
    f_only = gllvm(@formula(y ~ 1 + x & g), Y, D; kw...)

    @test length(f_full.γ) == 5
    @test length(f_nested.γ) == 5          # g: b, g: c, x & g: a, b, c
    @test length(f_only.γ) == 3            # one slope per level of g
    @test dof(f_nested) == dof(f_full)
    @test loglikelihood(f_nested) ≈ loglikelihood(f_full) atol = 1e-5
    @test isfinite(loglikelihood(f_only))

    # Main effects with an intercept keep their reference-level coding.
    f_main = gllvm(@formula(y ~ 1 + x + g), Y, D; kw...)
    @test length(f_main.γ) == 3

    # The Gaussian route and a non-Gaussian route now agree on the number of columns.
    Yg = eta .+ 0.5 .* randn(rng, p, n)
    fg = gllvm(@formula(y ~ 1 + g + x & g), Yg, D; family = Normal(), K = K)
    @test length(coef(fg)) == p + length(f_nested.γ)
end
