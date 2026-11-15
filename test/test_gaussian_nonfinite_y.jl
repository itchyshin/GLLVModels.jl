# A NaN or Inf in `Y` at an observed cell used to give `loglik = NaN, converged = false` from
# `fit_gaussian_gllvm(Y; K = p)` (and from `fit_dep_gllvm`, which forces K = p), and an unnamed
# LinearAlgebra error at K < p. It is now an ArgumentError naming the fitter and the cell.
# A non-finite value under `mask = false` stays legal.
using Test, GLLVModels, Random, Distributions

@testset "Gaussian fitters refuse NaN and Inf in Y" begin
    rng = MersenneTwister(3)
    p, n = 4, 30
    Y = randn(rng, p, n)
    thrown(f) = try; f(); nothing; catch e; e; end
    named(e, caller) = e isa ArgumentError && occursin(caller, e.msg) && occursin("Y[1, 2]", e.msg) &&
                       occursin("finite", e.msg)
    with(v) = (Z = copy(Y); Z[1, 2] = v; Z)

    @testset "fit_gaussian_gllvm, every K in 1:p" begin
        for K in 1:p, v in (NaN, Inf, -Inf)
            @test named(thrown(() -> fit_gaussian_gllvm(with(v); K = K, iterations = 20)), "fit_gaussian_gllvm")
        end
    end
    @testset "front doors that reach it" begin
        for v in (NaN, Inf)
            @test named(thrown(() -> fit_gllvm(with(v); family = Normal(), K = p, iterations = 20)), "fit_gaussian_gllvm")
            @test named(thrown(() -> fit_dep_gllvm(with(v); iterations = 20)), "fit_gaussian_gllvm")
        end
    end
    @testset "hidden by the mask stays legal; clean data fit" begin
        mask = trues(p, n); mask[1, 2] = false
        Yn = with(NaN); Y0 = with(0.0)
        a = fit_gaussian_gllvm(Yn; K = p, mask = mask, iterations = 60)
        b = fit_gaussian_gllvm(Y0; K = p, mask = mask, iterations = 60)
        @test isfinite(a.logLik)
        @test a.logLik ≈ b.logLik rtol = 1e-10
        @test isfinite(fit_gaussian_gllvm(Y; K = p, iterations = 60).logLik)
        @test isfinite(fit_gaussian_gllvm(Y; K = 1, iterations = 60).logLik)
    end
end
