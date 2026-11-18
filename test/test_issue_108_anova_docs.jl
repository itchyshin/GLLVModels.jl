using Test
using GLLVModels
using Random

# Issue #108 (part): the tutorial documents gllvm_anova, its status vocabulary and its
# claim boundary. The doc lock checks the wording is present; the behaviour block checks
# every claim the section makes against the live function, so the docs cannot drift.

@testset "issue 108 gllvm_anova docs and stated behaviour" begin
    path = joinpath(@__DIR__, "..", "docs", "src", "tutorial.md")
    @test isfile(path)
    doc = replace(read(path, String), r"\s+" => " ")

    @test occursin("### Comparing nested fits with `gllvm_anova`", doc)
    # status vocabulary
    for status in ("`chisq`", "`chibar (q=…)`", "`chisq (requested)`", "`none`", "`refused`")
        @test occursin("| " * status * " |", doc)
    end
    # claim boundary
    @test occursin("Same data and nesting are assumed, not checked.", doc)
    @test occursin("do not read a `chisq` row as a boundary test", doc)
    @test occursin("The rank-step p-value is a rough guide, not a valid test.", doc)
    @test occursin("Wilks' theorem fails", doc)

    # Behaviour the section states.
    rng = MersenneTwister(108)
    p, n, K = 6, 120, 2
    Y = (randn(rng, p, K) ./ 1.2) * randn(rng, K, n) .+ 0.5 .* randn(rng, p, n)
    fits = [fit_gaussian_gllvm(Y; K = k) for k in 1:3]
    tab = gllvm_anova(fits...)

    # rank steps: chi-bar-square with q = p - K (K the smaller rank)
    @test tab.test == ["", "chibar (q=5)", "chibar (q=4)"]
    @test tab.d == [1, 2, 3]
    @test all(!ismissing, tab.pvalue[2:3])
    @test tab.pvalue[2] < 0.05          # true rank 2: K = 1 is rejected
    @test tab.pvalue[3] > 0.01          # K = 3 adds nothing
    @test tab.test == gllvm_anova(fits...; test = :chibar).test

    # requested options
    @test gllvm_anova(fits...; test = :chisq).test == ["", "chisq (requested)", "chisq (requested)"]
    tnone = gllvm_anova(fits...; test = :none)
    @test tnone.test == ["", "none", "none"]
    @test all(ismissing, tnone.pvalue)

    # same rank, one more parameter: plain chi-square; rank + something else: refused
    X = reshape(repeat((1:n) ./ n, inner = p), p, n, 1)
    fx = fit_gaussian_gllvm(Y; X = X, K = 2)
    @test gllvm_anova(fits[2], fx).test == ["", "chisq"]
    refused = gllvm_anova(fits[1], fx)
    @test refused.test == ["", "refused"]
    @test ismissing(refused.pvalue[2])
    @test refused.note[2] isa String

    # whole-call refusals listed in the section
    @test_throws ArgumentError gllvm_anova(fits[1])
    @test_throws ArgumentError gllvm_anova(fits[1], fits[1])
    @test_throws ArgumentError gllvm_anova(fits...; test = :bogus)
    Y5 = Y[1:5, :]
    @test_throws ArgumentError gllvm_anova(fits[1], fit_gaussian_gllvm(Y5; K = 2))
end
