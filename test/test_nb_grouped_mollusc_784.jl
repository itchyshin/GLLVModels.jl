using GLLVModels, Test, SHA, Random

# #784: the default NB2 per-species route on ade4::mollusc (15 most frequent taxa,
# 163 sites, intercept-only, K = 2). The initial L-BFGS stage stopped at its
# iteration cap with three taxa crawling toward the Poisson limit (r in the
# thousands, below the 1e6 boundary restart), so the ridge polish fixed them at
# r = 1e10 and reported converged = true at logLik -3129.531424, 8.37 below
# gllvmTMB (nbinom2, latent(d = 2), n_init = 10: -3121.164397, pdHess TRUE).
# Dense BFGS from the same start reaches gllvmTMB's optimum.
#
# Data: test/fixtures/mollusc15.csv, built as in the issue:
#   f <- ade4::mollusc$fau; Y <- f[, order(-colSums(f > 0))[1:15]]
# md5 2cafc37f9e2614d388d5bfd05a88aede.

function _mollusc15_784()
    path = joinpath(@__DIR__, "fixtures", "mollusc15.csv")
    @test bytes2hex(sha256(read(path))) ==
          "bc2e15aace45f648fa4a2e965e1be1657fb411952853c9ab77c0c383b6fdfa03"
    rows = [parse.(Int, split(l, ",")[2:end]) for l in readlines(path)[2:end]]
    return reduce(hcat, rows)   # one column per site: p x n = 15 x 163
end

@testset "NB2 per-species dispersion, mollusc 15 x 163, K = 2 (#784)" begin
    Y = _mollusc15_784()
    @test size(Y) == (15, 163)
    Random.seed!(1)
    ft = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = 2, disp_group = :species)
    @test ft.converged
    @test ft.loglik ≈ -3121.164397 atol = 1e-4
    # gllvmTMB's optimum keeps taxa 14 and 15 overdispersed (r about 0.62 and 0.10);
    # the stalled fit had put them at the Poisson limit.
    @test ft.r_group[14] < 10 && ft.r_group[15] < 10
end
