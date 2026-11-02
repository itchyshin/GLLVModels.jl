# Issue #915: `n_init` best-of-starts for the Poisson Laplace fit.
using Test, GLLVModels, Distributions, Random

@testset "Poisson n_init (#915)" begin
    # ade4::mafragh flo table (97 sites x 56 species), species present at >= 3 sites kept.
    rows = readlines(joinpath(@__DIR__, "fixtures", "mafragh_flo.csv"))[2:end]
    M = reduce(vcat, [parse.(Float64, split(r, ',')[2:end])' for r in rows])
    M = M[:, vec(sum(M .> 0; dims = 1)) .>= 3]
    Y = round.(Int, permutedims(M))
    @test size(Y) == (47, 97)

    f1 = fit_gllvm(Y; family = Poisson(), K = 2)
    f1b = fit_gllvm(Y; family = Poisson(), K = 2, n_init = 1)
    # Default is unchanged: one start, and n_init = 1 is the same fit.
    @test f1.converged
    @test f1.loglik ≈ -1840.795461 atol = 1e-4
    @test f1b.loglik == f1.loglik && f1b.β == f1.β && f1b.Λ == f1.Λ

    f5 = fit_gllvm(Y; family = Poisson(), K = 2, n_init = 5)
    @test f5.converged
    @test f5.loglik > f1.loglik + 8
    @test f5.loglik ≈ -1832.348456 atol = 1e-3
    # Same call, same answer (fixed seeds).
    @test fit_gllvm(Y; family = Poisson(), K = 2, n_init = 5).loglik == f5.loglik
    # Direct entry point takes it too.
    @test fit_poisson_gllvm(Y; K = 2, n_init = 5).loglik == f5.loglik

    # Small problem: the global random stream is not consumed, a user Λ_init is start 1.
    rng = Random.Xoshiro(1)
    Ys = [rand(rng, Poisson(exp(0.5 + 0.4 * sin(i + 2j)))) for i in 1:6, j in 1:30]
    Random.seed!(7); expected = rand()
    Random.seed!(7)
    fs = fit_gllvm(Ys; family = Poisson(), K = 1, n_init = 3)
    @test rand() == expected
    @test fs.loglik >= fit_gllvm(Ys; family = Poisson(), K = 1).loglik - 1e-9
    fu = fit_poisson_gllvm(Ys; K = 1, n_init = 2, Λ_init = fill(0.1, 6, 1))
    @test fu.loglik >= fit_poisson_gllvm(Ys; K = 1, Λ_init = fill(0.1, 6, 1)).loglik - 1e-9

    # Bad requests.
    @test_throws ArgumentError fit_gllvm(Ys; family = Poisson(), K = 1, n_init = 0)
    @test_throws ArgumentError fit_poisson_gllvm(Ys; K = 1, n_init = -2)
    @test_throws ArgumentError fit_poisson_gllvm(Ys; K = 1, n_init = 2, aghq = 5)
    # Other families do not take the keyword.
    @test_throws MethodError fit_gllvm(Float64.(Ys); family = Normal(), K = 1, n_init = 2)
end
