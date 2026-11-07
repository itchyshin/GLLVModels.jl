# Regression for #1022 on real data: vegan::mite (70 sites x 35 oribatid mite taxa; fixture
# test/data/mite.csv, columns = taxa, rows = sites), NegativeBinomial, K = 2.
#
# A random start (`Λ_init`, seed 1 as in the issue) reports `converged = true` at a local
# optimum about 33.3 log-likelihood units below the one the default start reaches
# (-3792.40 against -3759.09 on aed938e, Julia 1.12.7). With `n_init = 3` that same bad
# first start is followed by seeded starts, and the best converged fit is returned.
#
# Not wired into runtests.jl: about 80 s of NB fitting. Run directly:
#   julia --project=. test/test_nb_n_init_mite_1022.jl
# Values are not pinned (optimiser-path sensitive across platforms); the test asserts the
# relation that n_init is meant to deliver.
using GLLVModels, Test, Random, DelimitedFiles
using Distributions: NegativeBinomial

@testset "NB n_init: mite K = 2 (#1022)" begin
    d = readdlm(joinpath(@__DIR__, "data", "mite.csv"), ',', Int; skipstart = 1)
    Y = permutedims(d)                       # taxa x sites = 35 x 70
    @test size(Y) == (35, 70)
    p, K = size(Y)[1], 2

    f_default = fit_gllvm(Y; family = NegativeBinomial(), K = K, iterations = 3000)

    Random.seed!(1)
    Li = 0.3 .* randn(p, K); Li[1, 2] = 0
    f_bad = fit_gllvm(Y; family = NegativeBinomial(), K = K, Λ_init = Li, iterations = 3000)
    f_multi = fit_gllvm(Y; family = NegativeBinomial(), K = K, Λ_init = Li, n_init = 3,
                        iterations = 3000)
    @info "mite NB K=2" f_default.loglik f_bad.loglik f_multi.loglik

    @test f_default.converged && f_bad.converged && f_multi.converged
    @test f_multi.loglik >= f_bad.loglik - 1e-9      # never worse than start 1
    @test f_multi.loglik > f_bad.loglik + 1.0        # and here it escapes the bad optimum
    @test abs(f_multi.loglik - f_default.loglik) < 1e-3
end
