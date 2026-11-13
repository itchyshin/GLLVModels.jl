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

# Pin the columns (names, order, values) of the formula shapes that already fitted the right model,
# and the per-level slopes of the repaired shapes against a hand-built design.
@testset "formula site design: columns of unchanged shapes, slopes of repaired shapes by level" begin
    rng = MersenneTwister(20261114)
    n = 30
    x = randn(rng, n); z = randn(rng, n)
    g = shuffle(rng, repeat(["a", "b", "c"], n ÷ 3)); h = shuffle(rng, repeat(["u", "v"], n ÷ 2))
    cols = (x = x, z = z, g = g, h = h)
    build(f) = GLLVModels._build_site_modelmatrix(f.rhs, cols)
    ind(v, l) = Float64.(v .== l)

    # shapes that do not hit the rank rule: same names and order as before the fix
    for (f, names) in (
            (@formula(y ~ 1), String[]),
            (@formula(y ~ x + z), ["x", "z"]),
            (@formula(y ~ x * z), ["x", "z", "x & z"]),
            (@formula(y ~ g + x), ["g: b", "g: c", "x"]),
            (@formula(y ~ g * x), ["g: b", "g: c", "x", "g: b & x", "g: c & x"]),
            (@formula(y ~ x + x & g), ["x", "x & g: b", "x & g: c"]),
            (@formula(y ~ g * h), ["g: b", "g: c", "h: v", "g: b & h: v", "g: c & h: v"]),
            (@formula(y ~ log(x + 5) + g), ["log(x + 5)", "g: b", "g: c"]),
        )
        mm, nm = build(f)
        @test nm == names
        @test size(mm) == (n, length(names))
    end
    mm, nm = build(@formula(y ~ g + x))
    @test mm == hcat(ind(g, "b"), ind(g, "c"), x)

    # repaired shapes: one slope per level, each column labelled with its own level
    mm, nm = build(@formula(y ~ x & g))
    @test nm == ["x & g: a", "x & g: b", "x & g: c"]
    @test mm == hcat(x .* ind(g, "a"), x .* ind(g, "b"), x .* ind(g, "c"))
    mm, nm = build(@formula(y ~ g + x & g))
    @test nm == ["g: b", "g: c", "x & g: a", "x & g: b", "x & g: c"]
    @test mm == hcat(ind(g, "b"), ind(g, "c"), x .* ind(g, "a"), x .* ind(g, "b"), x .* ind(g, "c"))

    # the fit equals the fit on the hand-built design, slope by slope
    p, K = 5, 1
    slope = Dict("a" => 0.2, "b" => 0.6, "c" => -0.5)
    a = 0.3 .* randn(rng, p); lam = 0.6 .* randn(rng, p); lv = randn(rng, n)
    Y = [rand(rng, Poisson(exp(0.5 + 0.5 * (a[t] + slope[g[s]] * x[s] + lam[t] * lv[s])))) for t in 1:p, s in 1:n]
    Xh = Array{Float64, 3}(undef, p, n, 3)
    for k in 1:3, s in 1:n, t in 1:p
        Xh[t, s, k] = x[s] * (g[s] == ("a", "b", "c")[k])
    end
    f_hand = GLLVModels.fit_gllvm_cov(Y; family = Poisson(), X = Xh, K = K)
    f_form = gllvm(@formula(y ~ 1 + x & g), Y, (x = x, g = g); family = Poisson(), K = K)
    @test loglikelihood(f_form) ≈ loglikelihood(f_hand) atol = 1e-8
    @test f_form.γ ≈ f_hand.γ atol = 1e-6
end
