# Lane cursor-cloud-1125y11 — model_selection.jl sweep branches
using GLLVModels, Test, Random

@testset "cov 1125y11 select_lv mini sweep" begin
    rng = MersenneTwister(1125112)
    Y = randn(rng, 6, 15)
    @testset "Gaussian K=1:2" begin
        sel = select_lv(Y; Kmax = 2, iterations = 40, g_tol = 1e-3)
        @test sel isa LVSelection
        @test !isempty(sel.attempts)
        @test length(sel.attempts) >= 2
        @test all(hasproperty(a, :status) for a in sel.attempts)
        @test sel.best_k ∈ sel.K
        @test sel.criterion === :bic_sites
        @test length(sel.aic) == length(sel.K)
    end

    @testset "criterion bic" begin
        sel = select_lv(Y; Kmax = 2, criterion = :bic, iterations = 30, g_tol = 1e-3)
        @test sel.criterion === :bic
    end

    @testset "show table" begin
        sel = select_lv(Y; Kmax = 1, iterations = 20, g_tol = 1e-3)
        txt = sprint(show, MIME"text/plain"(), sel)
        @test occursin("logLik", txt)
        @test occursin("npar", txt)
        @test occursin("conv", txt)
    end

    @testset "refusals" begin
        @test_throws ArgumentError select_lv(Y; Kmax = 0)
        @test_throws ArgumentError select_lv(Y; criterion = :unknown)
    end
end