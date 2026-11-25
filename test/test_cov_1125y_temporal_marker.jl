# Coverage round 4: temporal.jl constructor guards and show surfaces.
using Test
using GLLVModels

const G = GLLVModels

@testset "1125y temporal marker validation" begin
    good_f = :(0 + trait | series)
    @testset "TemporalTerm show" begin
        t = G.temporal_indep(good_f, :time; structure = :ou, replicate = :rep)
        s = sprint(show, t)
        @test occursin("temporal_indep", s)
        @test occursin("time = time", s)
        @test occursin(":ou", s)
        @test occursin("replicate = rep", s)
        tl = G.temporal_latent(good_f, :t; d = 1, unique = true, structure = :ar1)
        sl = sprint(show, tl)
        @test occursin("temporal_latent", sl)
        @test occursin("unique = true", sl)
    end

    @testset "formula and time guards" begin
        @test_throws G.TemporalContractError G.temporal_indep(:(trait ~ 1), :time)
        @test_throws G.TemporalContractError G.temporal_dep(:(trait), :time)
        @test_throws G.TemporalContractError G.temporal_indep(good_f, "time")
        @test_throws G.TemporalContractError G.temporal_indep(good_f, :time; replicate = 1)
    end

    @testset "structure and rank guards" begin
        @test_throws G.TemporalContractError G.temporal_indep(good_f, :time; structure = :rw)
        @test_throws G.TemporalContractError G.temporal_latent(good_f, :time; d = 2)
        @test_throws G.TemporalContractError G.temporal_latent(good_f, :time; unique = :yes)
    end

    @testset "TemporalContractError showerror" begin
        err = G.TemporalContractError("bad formula", :gllvmTMB_temporal_formula)
        buf = sprint(showerror, err)
        @test occursin("TemporalContractError", buf)
        @test occursin("gllvmTMB_temporal_formula", buf)
        @test occursin("bad formula", buf)
    end

    @testset "temporal_dep and temporal_latent admit good markers" begin
        td = G.temporal_dep(good_f, :occ; structure = :ar1)
        @test td.mode === :dep
        @test td.unique === false
        tl = G.temporal_latent(good_f, :occ; d = 1, unique = false)
        @test tl.mode === :latent
        @test tl.d == 1
        ti = G.temporal_indep(good_f, :t; structure = "ar1", replicate = :block)
        @test ti.structure === :ar1
        @test ti.replicate === :block
        @test ti.time === :t
        @test ti.unique === true
    end

    @testset "string structure alias" begin
        t = G.temporal_indep(good_f, :time; structure = "ou")
        @test t.structure === :ou
    end
end