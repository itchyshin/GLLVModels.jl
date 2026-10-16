using GLLVModels, Test, SHA, TOML, LinearAlgebra

# #784: fit_nb_gllvm_grouped (the default NB2 route, per-species dispersion) on a real count
# table, ade4::mollusc (15 most frequent taxa, 163 sites, intercept-only, K = 2). Finite-
# difference L-BFGS from the default start stopped at logLik -3129.5314 with converged = true,
# 8.37 below gllvmTMB (nbinom2, latent(d = 2), n_init = 10: -3121.164397, convergence 0, the
# same optimum from 4 starts, ΛΛᵀ eigenvalues 84.43 / 24.01). The exact-gradient dense-BFGS
# route of fit_nb_gllvm_grouped_cov reaches that optimum from the same start, so the fitter
# now runs both routes (`routes = :both`, the default) and keeps the better. Neither route
# dominates (a 15 x 81 subsample of this table ends 1.99 higher on the L-BFGS route), hence
# "keep the better", not "replace".
#
# Data: test/fixtures/mollusc15.csv, built as in the issue:
#   f <- ade4::mollusc$fau; Y <- f[, order(-colSums(f > 0))[1:15]]
#
# The full-size default route (`:both`) includes the finite-difference L-BFGS fit, which takes
# tens of minutes on one core (2113 s on a loaded 4-core box), so this file does not run it:
# the mollusc case uses `routes = :exact` (about 30 s), and the "keep the better" rule is
# checked on small data, where all three routes are cheap. The `:both` result on the mollusc
# table (-3121.164397, converged) is recorded in the PR for #784.

function _mollusc15()
    path = joinpath(@__DIR__, "fixtures", "mollusc15.csv")
    @test bytes2hex(sha256(read(path))) ==
          "bc2e15aace45f648fa4a2e965e1be1657fb411952853c9ab77c0c383b6fdfa03"
    rows = [parse.(Int, split(l, ",")[2:end]) for l in readlines(path)[2:end]]
    return reduce(hcat, rows)   # one column per site: p x n = 15 x 163 (species x sites)
end

@testset "NB2 per-species dispersion, mollusc 15 x 163, K = 2: exact route (#784)" begin
    Y = _mollusc15()
    @test size(Y) == (15, 163)
    ft = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = 2, disp_group = :species,
                   routes = :exact)
    @test ft isa NBGroupedFit
    # gllvmTMB (0.8.0.9000, 8010cfd4): -3121.164397. The L-BFGS route alone: -3129.531424.
    @test ft.loglik >= -3121.164397 - 1e-3
    @test ft.converged
    # Same optimum, not a rotation of a different one: the spectrum of ΛΛᵀ matches gllvmTMB's
    # (84.43, 24.01; the stalled L-BFGS fit had 91.01, 36.02).
    ev = eigvals(Symmetric(Matrix(ft.Λ) * Matrix(ft.Λ)'))
    @test isapprox(ev[end], 84.43; atol = 0.05)
    @test isapprox(ev[end - 1], 24.01; atol = 0.05)
end

@testset "NB2 per-species dispersion: the default keeps the better of the two routes (#784)" begin
    # NATIVE-06-design data (p = 5, n = 80; see test_nb_boundary_restart.jl).
    d = TOML.parsefile(joinpath(@__DIR__, "fixtures", "nb2_restart_seed52.toml"))
    Y = reshape(Int.(d["Y_column_major"]), d["p"], d["n"])
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y))))) == d["data_sha256"]
    kw = (; family = GLLVModels.NegativeBinomial(), K = 2, disp_group = :species,
          g_tol = 1e-7, iterations = 800)
    fl = fit_gllvm(Y; kw..., routes = :lbfgs)
    fe = fit_gllvm(Y; kw..., routes = :exact)
    fb = fit_gllvm(Y; kw...)
    @test fb.loglik >= max(fl.loglik, fe.loglik) - 1e-9
    # On this fixture the L-BFGS route is the higher one (-833.1814); the default must not
    # trade it for the exact route's lower point.
    @test fb.loglik >= -833.181377 - 1e-5
end

@testset "NB2 per-species dispersion: routes argument checks (#784)" begin
    Y = reshape(Int.(TOML.parsefile(joinpath(@__DIR__, "fixtures", "nb2_restart_seed52.toml"))["Y_column_major"]), 5, 80)
    nb = GLLVModels.NegativeBinomial()
    @test_throws ArgumentError fit_gllvm(Y; family = nb, K = 1, disp_group = :species, routes = :bogus)
    # The exact gradient has no offset support: asking for it alone is an error, while the
    # default route silently stays on L-BFGS there.
    off = zeros(5, 80)
    @test_throws ArgumentError fit_gllvm(Y; family = nb, K = 1, disp_group = :species,
                                         routes = :exact, offset = off)
end
