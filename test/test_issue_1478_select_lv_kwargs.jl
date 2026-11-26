# Issue #1478: R-style select_lv keywords must not forward into fit_gllvm.
using GLLVModels, Distributions, Random, Test

@testset "select_lv issue 1478 — select_lv kwargs are not forwarded" begin
    Random.seed!(1126)
    Y = randn(4, 25)

    @testset "K_max and method raise ArgumentError at select_lv" begin
        err = try
            select_lv(Y; family = Normal(), K_max = 2, method = :aic)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("select_lv", msg)
        @test occursin("K_max", msg)
        @test occursin("Kmax", msg)
        @test occursin("method", msg)
        @test occursin("criterion", msg)
    end

    @testset "documented keywords still work (issue repro control)" begin
        sel = select_lv(Y; family = Normal(), Kmax = 2, criterion = :aic)
        @test sel isa LVSelection
        @test !isempty(sel.K)
        @test maximum(sel.K) <= 2
    end

    @testset "d_max alone names Kmax" begin
        err = try
            select_lv(Y; family = Normal(), d_max = 2)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("d_max", sprint(showerror, err))
        @test occursin("Kmax", sprint(showerror, err))
    end
end