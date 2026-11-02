using GLLVModels, Test, Random, Distributions

# Edge cases for the four covariate-structural Wald entries (`confint_speciescov`,
# `confint_fourthcorner`, `confint_rrr`, `confint_constrained`) that the single
# Poisson smoke test in test_structural_confint.jl does not reach: the dispersion
# term, Binomial with/without `N`, `parm` selection, `level`, estimate-column
# ordering against the fit's own coefficients, and K = 2.

# Per model: fit, Wald CI (forwarding kwargs), and the expected `estimate` column
# in term order built from the fit's own fields.
function _sce_models(Y, family, q, r; K, N = nothing)
    p, n = size(Y)
    Nkw = N === nothing ? NamedTuple() : (N = N,)
    Xs = randn(p, n, q); Xenv = randn(n, q); TR = randn(p, r); X = randn(n, q)
    disp(f) = hasproperty(f, :dispersion) && f.dispersion !== nothing &&
              !(f.dispersion isa Real && isnan(f.dispersion)) ? [f.dispersion] : Float64[]
    fs = fit_gllvm_speciescov(Y; family = family, X = Xs, K = K, iterations = 60, Nkw...)
    ff = fit_fourthcorner_gllvm(Y; family = family, Xenv = Xenv, TR = TR, K = K, iterations = 60, Nkw...)
    fr = fit_rrr_gllvm(Y; family = family, X = X, K = K, iterations = 60, Nkw...)
    fc = fit_constrained_gllvm(Y; family = family, X = X, K = K, iterations = 60, Nkw...)
    return [
        (name = "speciescov", fit = fs, bprefix = "B[",
         ci = (; kw...) -> confint_speciescov(fs, Y, Xs; Nkw..., kw...),
         expected = vcat(fs.β, vec(fs.B), GLLVModels.pack_lambda(fs.Λ), disp(fs))),
        (name = "fourthcorner", fit = ff, bprefix = "C[",
         ci = (; kw...) -> confint_fourthcorner(ff, Y, Xenv, TR; Nkw..., kw...),
         expected = vcat(ff.β, vec(ff.C), GLLVModels.pack_lambda(ff.Λ), disp(ff))),
        (name = "rrr", fit = fr, bprefix = "B[",
         ci = (; kw...) -> confint_rrr(fr, Y, X; Nkw..., kw...),
         expected = vcat(fr.β, GLLVModels.pack_lambda(fr.Λ), vec(fr.B), disp(fr))),
        (name = "constrained", fit = fc, bprefix = "B[",
         ci = (; kw...) -> confint_constrained(fc, Y, X; Nkw..., kw...),
         expected = vcat(fc.β, GLLVModels.pack_lambda(fc.Λ), vec(fc.B), disp(fc))),
    ]
end

width(ci, i) = ci.upper[i] - ci.lower[i]

@testset "Structural confint edge cases" begin
    Random.seed!(4242)
    p, n, q, r = 3, 24, 2, 2

    @testset "dispersion term (NegativeBinomial, Gamma)" begin
        # clearly over-dispersed so the NB dispersion is interior, not at the Poisson limit
        mu = exp.(1.5 .+ 0.3 .* randn(p, n))
        Ynb = [rand(NegativeBinomial(1.5, 1.5 / (1.5 + mu[i, j]))) for i in 1:p, j in 1:n]
        Yga = [rand(Gamma(3.0, mu[i, j] / 3.0)) for i in 1:p, j in 1:n]
        for (fam, Y) in ((NegativeBinomial(), Ynb), (Gamma(), Yga))
            K = 1; rr = GLLVModels.rr_theta_len(p, K)
            for m in _sce_models(Y, fam, q, r; K = K)
                @testset "$(typeof(fam).name.name) $(m.name)" begin
                    ci = m.ci()
                    nlin = length(m.expected) - 1
                    @test length(ci.term) == length(m.expected)
                    @test ci.term[end] == "dispersion"
                    @test count(==("dispersion"), ci.term) == 1
                    # reported on the natural (positive) scale, equal to the fit's value
                    @test ci.estimate[end] > 0
                    @test ci.estimate[end] ≈ m.fit.dispersion rtol = 1e-8
                    @test ci.lower[end] > 0
                    if isfinite(ci.lower[end]) && isfinite(ci.upper[end])
                        @test ci.lower[end] ≤ ci.estimate[end] ≤ ci.upper[end]
                    end
                    @test ci.estimate ≈ m.expected rtol = 1e-8
                    d = m.ci(parm = "dispersion")
                    @test d.term == ["dispersion"]
                    @test d.estimate[1] ≈ ci.estimate[end]
                end
            end
        end
    end

    @testset "Binomial with and without N" begin
        K = 1
        Nt = fill(5, p, n)
        Ybin = [rand(Binomial(5, 0.3)) for _ in 1:p, _ in 1:n]
        Ybern = [rand() < 0.4 ? 1 : 0 for _ in 1:p, _ in 1:n]
        for (label, Y, N) in (("with N", Ybin, Nt), ("without N", Ybern, nothing))
            for m in _sce_models(Y, Binomial(), q, r; K = K, N = N)
                @testset "Binomial $label $(m.name)" begin
                    ci = m.ci()
                    @test length(ci.term) == length(m.expected)   # no dispersion term
                    @test !("dispersion" in ci.term)
                    @test ci.estimate ≈ m.expected rtol = 1e-8
                    @test ci.method == :wald
                    for i in eachindex(ci.term)
                        if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                            @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
                        end
                    end
                end
            end
        end
    end

    Y = rand(0:4, p, n)
    models = _sce_models(Y, Poisson(), q, r; K = 1)

    @testset "parm selection" begin
        for m in models
            @testset "$(m.name)" begin
                full = m.ci()
                # prefix selector picks exactly the matching terms, in order
                sub = m.ci(parm = m.bprefix[1:1])
                @test !isempty(sub.term)
                @test all(t -> startswith(t, m.bprefix[1:1]), sub.term)
                @test sub.term == filter(t -> startswith(t, m.bprefix[1:1]), full.term)
                # exact name and group names
                @test m.ci(parm = "beta[2]").term == ["beta[2]"]
                @test m.ci(parm = "beta").term == filter(t -> startswith(t, "beta["), full.term)
                @test m.ci(parm = ["beta[1]", m.bprefix[1:1]]).term ==
                      vcat(["beta[1]"], sub.term)
                # selected values agree with the full table
                for (k, t) in enumerate(sub.term)
                    i = findfirst(==(t), full.term)
                    @test sub.estimate[k] == full.estimate[i]
                    @test sub.lower[k] ≈ full.lower[i]
                end
                # nothing matched
                @test_throws ArgumentError m.ci(parm = "no_such_term")
                @test_throws ArgumentError m.ci(parm = String[])
                err = try; m.ci(parm = String[]); nothing; catch e; e; end
                @test err isa ArgumentError
                @test err.msg == "parm selector matched no parameters"
            end
        end
    end

    @testset "level widens the interval" begin
        for m in models
            @testset "$(m.name)" begin
                c90 = m.ci(level = 0.90, parm = "beta")
                c99 = m.ci(level = 0.99, parm = "beta")
                @test c90.term == c99.term
                @test c90.estimate == c99.estimate
                @test c90.se == c99.se
                for i in eachindex(c90.term)
                    if isfinite(width(c90, i)) && isfinite(width(c99, i))
                        @test width(c99, i) > width(c90, i)
                        @test c99.lower[i] < c90.lower[i]
                        @test c99.upper[i] > c90.upper[i]
                    end
                end
            end
        end
    end

    @testset "estimate column equals the fit's coefficients" begin
        for m in models
            ci = m.ci()
            @test length(ci.term) == length(m.expected)
            @test ci.estimate ≈ m.expected rtol = 1e-8
        end
    end

    @testset "K = 2" begin
        p2, n2 = 4, 30
        K = 2; rr = GLLVModels.rr_theta_len(p2, K)
        Y2 = rand(0:4, p2, n2)
        for m in _sce_models(Y2, Poisson(), q, r; K = K)
            @testset "$(m.name)" begin
                ci = m.ci()
                @test length(ci.term) == length(m.expected)
                @test count(t -> startswith(t, "Lambda["), ci.term) == rr
                @test ci.estimate ≈ m.expected rtol = 1e-8
                if m.name in ("rrr", "constrained")
                    @test count(t -> startswith(t, "B["), ci.term) == q * K
                end
                for i in eachindex(ci.term)
                    if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                        @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
                    end
                end
            end
        end
    end
end
