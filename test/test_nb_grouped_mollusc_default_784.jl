using GLLVModels, Test, SHA, Random

# Red proof for #784 / #914: the default NB2 per-species route on ade4::mollusc
# (15 most frequent taxa, 163 sites, intercept-only, K = 2), with no `routes`
# keyword. On main (033d4e1) finite-difference L-BFGS from the default start
# ends at logLik -3129.531424 with converged = true — 8.37 below gllvmTMB
# (nbinom2, latent(d = 2), n_init = 10: -3121.164397). #914's default
# (`routes = :both`) reaches -3121.164397 from the same start. This file
# asserts logLik >= -3121.17 so it fails on base for the stall itself, not a
# missing-keyword MethodError.
#
# Data: test/fixtures/mollusc15.csv, built as in the issue:
#   f <- ade4::mollusc$fau; Y <- f[, order(-colSums(f > 0))[1:15]]
# md5 2cafc37f9e2614d388d5bfd05a88aede.

function _mollusc15()
    path = joinpath(@__DIR__, "fixtures", "mollusc15.csv")
    @test bytes2hex(sha256(read(path))) ==
          "bc2e15aace45f648fa4a2e965e1be1657fb411952853c9ab77c0c383b6fdfa03"
    rows = [parse.(Int, split(l, ",")[2:end]) for l in readlines(path)[2:end]]
    return reduce(hcat, rows)   # one column per site: p x n = 15 x 163
end

@testset "NB2 per-species dispersion, mollusc 15 x 163, K = 2: default route (#784)" begin
    Y = _mollusc15()
    @test size(Y) == (15, 163)
    Random.seed!(1)
    ft = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = 2, disp_group = :species)
    @test ft.loglik >= -3121.17
end
