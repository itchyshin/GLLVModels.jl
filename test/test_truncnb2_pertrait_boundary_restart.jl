using GLLVModels, Test, TOML

# The per-trait zero-truncated NB2 fitter can stall with the wrong trait's log r out at
# the Poisson limit, where the likelihood is nearly flat, below a better optimum. On two
# ordinary draws it stopped 0.35 and 2.0 log-likelihood units below gllvmTMB's fit of the
# same model. Restarting the boundary trait(s) from r = 1, as the NB2 grouped fitters do
# (`_nb_boundary_restart`, #477), reaches gllvmTMB's optimum.

const _TNB2_RS = TOML.parsefile(joinpath(@__DIR__, "fixtures",
                                         "truncnb2_pertrait_boundary_restart.toml"))

@testset "truncated NB2 per-trait: boundary restart reaches gllvmTMB's optimum ($seed)" for seed in
        ("seed93", "seed96")
    d = _TNB2_RS[seed]
    Y = reshape(Int.(d["Y_column_major"]), _TNB2_RS["p"], _TNB2_RS["n"])
    f = fit_truncated_nbinom2_gllvm_pertrait(Y; K = 1)
    # Before the restart: 0.35 (seed 93) and 2.0 (seed 96) below gllvmTMB.
    @test f.loglik >= d["gllvmtmb_loglik"] - 1e-3
end
