# refs #1291: refuse Bool for K; name a polytomy before duplicate tip names.
using Test, GLLVModels, Random, Distributions

@testset "#1277 nits (refs #1291)" begin
    @testset "Bool K is refused at fit_gllvm" begin
        rng = MersenneTwister(3)
        p, n = 5, 40
        Yc = rand(rng, 0:6, p, n)
        err = try
            fit_gllvm(Yc; family = Poisson(), K = true)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("K must be a non-negative integer", err.msg)
        @test occursin("Bool", err.msg)
    end

    @testset "not-binary precedes duplicate tip names" begin
        msg(f) = try
            f()
            ""
        catch e
            sprint(showerror, e)
        end
        s = "((t1:1,t1:1,t2:1):1,(t3:1,t4:1):1);"
        for f in (augmented_phy, edge_phy)
            m = msg(() -> f(s))
            @test occursin("not binary", m)
            @test !occursin("tip names must be unique", m)
        end
    end

    @testset "duplicate tip names after a binary tree" begin
        m = try
            augmented_phy("((t1:1,t1:1):1,(t3:1,t4:1):1);")
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("tip names must be unique", m)
    end
end