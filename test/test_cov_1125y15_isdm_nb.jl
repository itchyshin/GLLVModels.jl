# Lane cursor-cloud-1125y15 — families/isdm_formula.jl, isdm_table.jl, nb_grouped_postfit.jl
using GLLVModels, Test, StatsModels, Distributions, Random

const G = GLLVModels
const _CL = (Binomial(), G.CLogLogLink())

@testset "cov 1125y15 iSDM formula and NB grouped postfit" begin
    @testset "_isdm_parse_latent admitted shape" begin
        ex = :(latent(0 + trait | cell, d = 2, unique = false))
        K, uniq = G._isdm_parse_latent(ex, :trait, :cell)
        @test K == 2 && uniq == false
        ex2 = :(latent(0 + trait | cell, d = 1))
        K2, u2 = G._isdm_parse_latent(ex2, :trait, :cell)
        @test K2 == 1 && u2 == true
        @test_throws ArgumentError G._isdm_parse_latent(:(latent(trait | cell, d = 1)), :trait, :cell)
        @test_throws ArgumentError G._isdm_parse_latent(:(latent(0 + trait | site, d = 1)), :trait, :cell)
        @test_throws ArgumentError G._isdm_parse_latent(:(latent(0 + trait | cell, d = 0)), :trait, :cell)
    end

    @testset "_isdm_term and refused structured" begin
        t = G._isdm_term(:env)
        @test t isa StatsModels.Term
        t2 = G._isdm_term(:(trait & env))
        @test occursin("trait", string(t2))
        @test G._isdm_refused_structured(:(phylo(trait | tip))) == true
        @test G._isdm_refused_structured(:(trait + env)) == false
        @test_throws ArgumentError G._isdm_term(:(sin(env)))
    end

    @testset "_isdm_declared_core predicate" begin
        fam = isdm_sources(gbif = Poisson(), lit = _CL)
        sel = [:gbif, :lit, :gbif, :lit]
        fid = fam.fid[[1, 2, 1, 2]]
        lid = fam.lid[[1, 2, 1, 2]]
        traits = ["sp1", "sp1", "sp2", "sp2"]
        @test G._isdm_declared_core(fam, sel, fid, lid, traits, 4) == true
        sel_bad = [:gbif, :gbif, :gbif, :gbif]
        @test G._isdm_declared_core(fam, sel_bad, fid, lid, traits, 4) == false
        @test G._isdm_declared_core(fam, sel, fid, lid, nothing, 4) == false
    end

    @testset "_nb_grouped_check_type" begin
        @test G._nb_grouped_check_type(:mean) === nothing
        @test G._nb_grouped_check_type(:response) === nothing
        @test G._nb_grouped_check_type(:link) === nothing
        @test_throws ArgumentError G._nb_grouped_check_type(:variance)
        @test G._nb_grouped_check_residual_type(:pearson) === nothing
        @test G._nb_grouped_check_residual_type(:dunnsmyth) === nothing
        @test_throws ArgumentError G._nb_grouped_check_residual_type(:deviance)
    end

    @testset "NB grouped predict smoke" begin
        rng = MersenneTwister(11)
        Y = rand(rng, NegativeBinomial(2.0, 0.4), 2, 10)
        fit = fit_nb_gllvm_grouped(Y; group = ones(Int, 2), K = 1, iterations = 60, g_tol = 1e-3)
        μ = predict(fit, Y; type = :mean)
        @test size(μ) == size(Y)
        @test all(μ .> 0)
        res = residuals(fit, Y; type = :pearson)
        @test size(res) == size(Y)
    end
end