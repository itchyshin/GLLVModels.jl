# Issue #1022: `n_init` best-of-starts for the NB2 Laplace fitters (shared and grouped r),
# mirroring `n_init` on `fit_poisson_gllvm` (#915 / #1012; see test_poisson_n_init.jl).
using Test, GLLVModels, Distributions, Random

@testset "NB n_init (#1022)" begin
    rng = Random.Xoshiro(11)
    p, n, K = 8, 50, 2
    Λt = 0.8 .* randn(rng, p, K); z = randn(rng, K, n)
    Y = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(1.0 + (Λt * z)[i, j])))) for i in 1:p, j in 1:n]
    grp = ones(Int, p)
    same(a, b) = a.loglik == b.loglik && a.β == b.β && a.Λ == b.Λ &&
                 a.converged == b.converged && a.iterations == b.iterations

    @testset "shared r: fit_nb_gllvm / fit_gllvm" begin
        f1 = fit_nb_gllvm(Y; K = K)
        # Default is unchanged: n_init = 1 is the same fit, bit for bit.
        @test f1.converged
        fa = fit_nb_gllvm(Y; K = K, n_init = 1)
        @test same(fa, f1) && fa.r == f1.r
        @test same(fit_gllvm(Y; family = NegativeBinomial(), K = K, n_init = 1), f1)

        f4 = fit_nb_gllvm(Y; K = K, n_init = 4)
        @test f4.converged
        @test f4.loglik >= f1.loglik - 1e-9           # never worse than start 1
        # Same call, same answer (fixed seeds); routes agree.
        @test fit_nb_gllvm(Y; K = K, n_init = 4).loglik == f4.loglik
        @test fit_gllvm(Y; family = NegativeBinomial(), K = K, n_init = 4).loglik == f4.loglik

        # A user Λ_init is start 1; the global random stream is not consumed.
        Li = fill(0.1, p, K); Li[1, 2] = 0
        Random.seed!(7); expected = rand()
        Random.seed!(7)
        fu = fit_nb_gllvm(Y; K = K, n_init = 3, Λ_init = Li)
        @test rand() == expected
        @test fu.loglik >= fit_nb_gllvm(Y; K = K, Λ_init = Li).loglik - 1e-9

        # Masked fits and other keywords pass through every start.
        M = trues(p, n); M[1, 1:5] .= false
        fm = fit_nb_gllvm(Y; K = K, mask = M, n_init = 2)
        @test fm.loglik >= fit_nb_gllvm(Y; K = K, mask = M).loglik - 1e-9
    end

    @testset "grouped r: fit_nb_gllvm_grouped / fit_gllvm(disp_group)" begin
        g1 = fit_nb_gllvm_grouped(Y; K = K, group = grp)
        @test g1.converged
        ga = fit_nb_gllvm_grouped(Y; K = K, group = grp, n_init = 1)
        @test same(ga, g1) && ga.r_group == g1.r_group
        @test same(fit_gllvm(Y; family = NegativeBinomial(), K = K, disp_group = grp, n_init = 1), g1)

        g4 = fit_nb_gllvm_grouped(Y; K = K, group = grp, n_init = 4)
        @test g4.converged
        @test g4.loglik >= g1.loglik - 1e-9
        @test fit_nb_gllvm_grouped(Y; K = K, group = grp, n_init = 4).loglik == g4.loglik
        @test fit_gllvm(Y; family = NegativeBinomial(), K = K, disp_group = grp, n_init = 4).loglik == g4.loglik

        Li = fill(0.1, p, K); Li[1, 2] = 0
        Random.seed!(7); expected = rand()
        Random.seed!(7)
        gu = fit_nb_gllvm_grouped(Y; K = K, group = grp, n_init = 3, Λ_init = Li)
        @test rand() == expected
        @test gu.loglik >= fit_nb_gllvm_grouped(Y; K = K, group = grp, Λ_init = Li).loglik - 1e-9
    end

    @testset "bad requests" begin
        @test_throws ArgumentError fit_nb_gllvm(Y; K = K, n_init = 0)
        @test_throws ArgumentError fit_nb_gllvm_grouped(Y; K = K, group = grp, n_init = -2)
        @test_throws ArgumentError fit_gllvm(Y; family = NegativeBinomial(), K = K, n_init = 0)
        # n_init does not combine with AGHQ (it has its own multistart).
        @test_throws ArgumentError fit_gllvm(Y; family = NegativeBinomial(), K = K,
                                             disp_group = grp, aghq = 5, n_init = 2)
        # Families without the keyword still reject it.
        @test_throws MethodError fit_gllvm(Float64.(Y); family = Normal(), K = K, n_init = 2)
    end
end

# mvabund::spider$abund (mvabund 4.2.8), species x sites (12 x 28); same matrix as
# test_convergence_553_554.jl.
const _SPIDER1022 = [
    25 0 15 2 1 0 2 0 1 3 15 16 3 0 0 0 0 0 0 0 0 7 17 11 9 3 29 15;
    10 2 20 6 20 6 7 11 1 0 1 13 43 2 0 3 0 1 1 2 1 0 0 0 1 0 0 0;
    0 0 2 0 0 0 0 0 0 1 2 0 1 0 0 0 0 0 0 0 0 16 15 20 9 6 11 14;
    0 0 2 1 2 6 12 0 0 0 0 0 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 7 5 0 18 4 1;
    4 30 9 24 9 6 16 7 0 0 1 0 18 4 0 0 0 0 0 0 0 0 0 0 2 0 0 0;
    0 1 1 1 1 0 1 55 0 0 0 0 1 3 6 6 2 5 12 13 16 0 2 0 1 0 0 0;
    60 1 29 7 2 11 30 2 26 22 95 96 24 14 0 0 0 0 0 0 1 2 6 3 11 0 1 6;
    12 15 18 29 135 27 89 2 1 0 0 1 53 15 0 2 0 0 1 0 0 0 0 0 6 0 0 0;
    45 37 45 94 76 24 105 1 1 0 1 8 72 72 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
    57 65 66 86 91 63 118 30 2 1 4 13 97 94 25 28 23 25 22 22 18 1 1 0 16 1 0 2;
    4 9 1 25 17 34 16 3 0 0 0 0 22 32 3 4 2 0 3 2 2 0 0 0 6 0 0 0]

@testset "NB n_init: spider grouped K = 2 (#1022)" begin
    grp = collect(1:12)
    g1 = fit_nb_gllvm_grouped(_SPIDER1022; K = 2, group = grp)
    g5 = fit_nb_gllvm_grouped(_SPIDER1022; K = 2, group = grp, n_init = 5)
    # The default start reports converged = true at a local optimum 0.64 below the best
    # of 5 starts, which is gllvmTMB's optimum (-705.786785, nbinom2 with per-species dispersion).
    @test g1.converged && g5.converged
    @test g5.loglik > g1.loglik + 0.3
    @test g5.loglik ≈ -705.786785 atol = 1e-3
    @test fit_gllvm(_SPIDER1022; family = NegativeBinomial(), K = 2, disp_group = grp,
                    n_init = 5).loglik == g5.loglik
end
