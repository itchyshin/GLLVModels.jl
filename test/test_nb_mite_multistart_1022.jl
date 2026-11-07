# Regression/documentation test for #1022: NegativeBinomial K = 2 on real data
# (vegan::mite, 70 sites x 35 oribatid mite taxa; fixture test/data/mite.csv,
# exported from the public `vegan` package, columns = taxa, rows = sites).
#
# Different starts report `converged = true` at DIFFERENT local optima.
# Observed on commit aed938e (Julia 1.10.10 and 1.12.7, Linux x86_64):
#
#   start    converged   logLik
#   default  true        -3759.0941
#   seed 1   true        -3792.3957     (33.30 below the default's optimum)
#   seed 2   true        -3792.3957
#   seeds 3-5 true       -3759.0941
#
# `n_init` (#1012) is Poisson-only, so NB users get no best-of-starts help.
# This file does NOT assert the pinned logLik values (optimiser-path
# sensitive across platforms); it asserts only the contract that SHOULD hold
# -- converged starts agree on the optimum -- as @test_broken. When #1022 is
# resolved that line reports "unexpected pass": flip it to @test.
#
# Not wired into runtests.jl: ~1 min of NB fitting. Run directly:
#   julia --project=. test/test_nb_mite_multistart_1022.jl
using GLLVModels, Test, Random, DelimitedFiles
using Distributions: NegativeBinomial

@testset "NB K=2 mite multistart spread (#1022)" begin
    d = readdlm(joinpath(@__DIR__, "data", "mite.csv"), ',', Int; skipstart = 1)
    Y = permutedims(d)                       # taxa x sites = 35 x 70
    @test size(Y) == (35, 70)
    p, K = size(Y)[1], 2

    f_default = fit_gllvm(Y; family = NegativeBinomial(), K = K, iterations = 3000)

    Random.seed!(1)
    Li = 0.3 .* randn(p, K); Li[1, 2] = 0
    f_seed1 = fit_gllvm(Y; family = NegativeBinomial(), K = K,
                        Λ_init = Li, iterations = 3000)

    @test isfinite(f_default.loglik)
    @test isfinite(f_seed1.loglik)
    # Both starts report convergence -- that is the problem: it is not
    # evidence of a shared optimum.
    @test f_default.converged
    @test f_seed1.converged

    spread = abs(f_default.loglik - f_seed1.loglik)
    @info "mite NB K=2 start spread" f_default.loglik f_seed1.loglik spread
    @test_broken spread < 1e-3
end
