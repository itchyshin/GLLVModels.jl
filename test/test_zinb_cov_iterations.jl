# #781: fit_zinb_gllvm_cov(iterations = 0) must not report converged = true,
# and θ_init must be rejected when it is non-finite. On origin/main
# e6cfcd4ee the zero-iteration path hard-coded converged=true, including
# for a NaN start that produced the 1e12 sentinel loglik.

using GLLVModels, Test, Random, Distributions

@testset "fit_zinb_gllvm_cov iterations <= 0 (#781)" begin
    Random.seed!(11)
    Y = Float64.(rand(Poisson(2), 4, 60))
    X = randn(4, 60, 1)

    @testset "zero iterations is not a converged fit" begin
        f = fit_zinb_gllvm_cov(Y; X = X, K = 1, iterations = 0)
        @test f isa ZINBCovFit
        @test !f.converged
        @test f.iterations == 0
        @test isfinite(f.loglik)
    end

    @testset "absurd finite start is still not converged" begin
        θ = vcat(fill(30.0, 4), 0.0, fill(30.0, 4), 0.0, zeros(4), 0.0)
        f = fit_zinb_gllvm_cov(Y; X = X, K = 1, iterations = 0, θ_init = θ)
        @test !f.converged
        @test f.iterations == 0
        @test isfinite(f.loglik)
    end

    @testset "non-finite θ_init is rejected" begin
        θ = vcat(fill(NaN, 4), 0.0, fill(30.0, 4), 0.0, zeros(4), 0.0)
        err = try
            fit_zinb_gllvm_cov(Y; X = X, K = 1, iterations = 0, θ_init = θ)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("θ_init", msg)
        @test occursin("non-finite", msg)
    end
end
