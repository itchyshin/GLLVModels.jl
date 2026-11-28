# Maintainer decision 2026-11-28 (option B): refuse ambiguous weighted / grouped-N nobs
# with a clear ArgumentError until the grouped-nobs policy is set (refs #1538, #1553).
using Test, Random, Distributions
using GLLVModels
import GLLVModels: nobs
using GLLVModels: StatsAPI

function _nobs_grouped_err(test)
    err = try
        test()
        nothing
    catch e
        e
    end
    err isa ArgumentError || return false
    msg = sprint(showerror, err)
    occursin("grouped-nobs policy is not set yet", msg)
end

@testset "nobs weighted / grouped N — policy refusal (#1538, #1553)" begin
    Random.seed!(1)

    @testset "weighted Poisson: nobs(fit, Y; weights=...) (#1538)" begin
        p, n = 3, 10
        Y = rand(Poisson(2), p, n)
        w = ones(p, n)
        w[:, 1] .= 0
        fit = fit_gllvm(Y; family = Poisson(), K = 1, weights = w, iterations = 200)
        @test_throws ArgumentError nobs(fit, Y; weights = w)
        errmsg = try
            nobs(fit, Y; weights = w)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("grouped-nobs policy is not set yet", errmsg)
        @test occursin("weights", errmsg)
    end

    @testset "weighted PoissonFit: nobs(fit, Y) refuses (#1538)" begin
        p, n = 4, 20
        Y = [rand(Poisson(3)) for _ in 1:p, _ in 1:n]
        W = 0.5 .+ rand(p, n)
        fw = fit_gllvm(Y; family = Poisson(), K = 1, weights = W, iterations = 150)
        @test fw.weights !== nothing
        @test_throws ArgumentError nobs(fw, Y)
        @test _nobs_grouped_err(() -> nobs(fw, Y))
        # unweighted control unchanged
        fu = fit_gllvm(Y; family = Poisson(), K = 1, iterations = 150)
        @test fu.weights === nothing
        @test StatsAPI.nobs(fu, Y) == p * n
    end

    @testset "binomial trials: nobs(fit, Y; N=...) (#1553)" begin
        Random.seed!(1127)
        p, n = 6, 10
        N = fill(10, p, n)
        Y = rand(0:10, p, n)
        fit = fit_gllvm(Y; family = Binomial(), K = 1, N = N, iterations = 100)
        @test_throws ArgumentError nobs(fit, Y; N = N)
        errmsg = try
            nobs(fit, Y; N = N)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("grouped-nobs policy is not set yet", errmsg)
        # unit-trials binomial nobs unchanged
        Y0 = rand(Bool, p, n)
        fit0 = fit_binomial_gllvm(Y0; K = 1, iterations = 80)
        @test StatsAPI.nobs(fit0, Y0) == p * n
    end
end