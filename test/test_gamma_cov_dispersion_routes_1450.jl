# refs #1450: document and lock Gamma + X dispersion route counts (shared φ vs per-trait α).

using Test
using Random
using Distributions
using StatsModels
using GLLVModels

_doc_text(sym::Symbol) = string(Base.Docs.doc(Base.Docs.Binding(GLLVModels, sym)))

@testset "Gamma + X dispersion routes (#1450)" begin
    @testset "shared vs per-trait dispersion parameter counts" begin
        rng = MersenneTwister(77)
        p, n = 3, 40
        x = randn(rng, n)
        X = zeros(p, n, 1)
        @inbounds for t in 1:p, s in 1:n
            X[t, s, 1] = x[s]
        end
        Y = abs.(randn(rng, p, n)) .+ 0.1

        fm = fit_gllvm_cov(Y; family = Gamma(), X = X, K = 1, iterations = 200)
        ff = gllvm(@formula(y ~ 1 + x), Y, (x = x,); family = Gamma(), K = 1, iterations = 200)
        fg = fit_gamma_gllvm_grouped_cov(Y; X = X, K = 1, iterations = 200)

        @test fm isa GllvmCovFit
        @test ff isa GammaGroupedCovFit
        @test fg isa GammaGroupedCovFit
        @test isfinite(fm.dispersion)
        @test length(fg.α) == p
        @test length(ff.α) == p
        @test isapprox(ff.loglik, fg.loglik; rtol = 1e-10, atol = 1e-8)
        @test fm.loglik != ff.loglik
    end

    @testset "docstrings name the dispersion routes" begin
        cov_doc = _doc_text(:fit_gllvm_cov)
        @test occursin("fit_gamma_gllvm_grouped_cov", cov_doc)
        @test occursin("shared", lowercase(cov_doc))
        @test occursin("Gamma", cov_doc)

        form_doc = _doc_text(:gllvm)
        @test occursin("fit_gllvm_cov", form_doc)
        @test occursin("per-trait", form_doc)
        @test occursin("Gamma()", form_doc)
    end
end