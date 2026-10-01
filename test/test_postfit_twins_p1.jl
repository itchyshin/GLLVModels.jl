# gllvm-parity-tag: P1
#
# Numeric twins of three gllvmTMB postfit rows against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9): tidy.gllvmTMB_multi (fixed effects),
# POST-COEF-NAMED (coef) and POST-DEVIANCE (deviance). No R at test time: R's recorded values
# are read from test/fixtures/postfit_twins_p1.toml (generated once by
# test/fixtures/gen_postfit_twins_p1.R against a lane-local gllvmTMB install at the pin; it
# records R version, commit and the data sha256).
#
# Fixture: the rank-2 Gaussian model of test/fixtures/ns_numeric_p1.toml [main], p = 6 traits,
# n = 200 units, UNCENTRED data (trait means 0.5, -0.3, 0.2, 0.8, -0.6, 0.1).
# R: value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE)
# <-> fit_gllvm(Y; family = Normal(), K = 2). The batch fixture behind the postfit rows was
# row-centred, so its R coefficients were ~1e-14 and a zero implementation passed; here the
# R values are far from zero. The R fit converged with a positive-definite Hessian (asserted).
# With K = 2 and no unique term the trait-mean MLE is the sample mean, so the engines agree to
# ~1e-13 (observed); the tolerances below follow the existing postfit rows (coef 1e-6,
# tidy 1e-4) and 2 x the log-likelihood guard for the deviance.
using Test
using GLLVModels
using Distributions: Normal
using TOML
using SHA

const _PF_DIR = joinpath(@__DIR__, "fixtures")

@testset "postfit twins: gllvmTMB P1 (9539352f6)" begin
    pf = joinpath(_PF_DIR, "postfit_twins_p1.toml")
    nsf = joinpath(_PF_DIR, "ns_numeric_p1.toml")
    if !(isfile(pf) && isfile(nsf))
        @warn "postfit twin P1 fixture absent; twin gate NOT RUN" pf
        @test_skip false
    else
        fx = TOML.parsefile(pf)
        ns = TOML.parsefile(nsf)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        m = fx["main"]
        dp = joinpath(_PF_DIR, m["data_file"])
        @test bytes2hex(sha256(read(dp))) == m["data_sha256"]
        @test m["converged"] && m["pd_hessian"]
        tn = String.(ns["trait_names"])
        p, n = Int(ns["p"]), Int(ns["n_unit"])
        Y = zeros(Float64, p, n)
        open(dp) do io
            readline(io)
            for line in eachline(io)
                isempty(line) && continue
                a = split(line, ",")
                Y[findfirst(==(strip(a[2], '"')), tn), parse(Int, strip(a[1], '"'))] = parse(Float64, a[3])
            end
        end
        fit = fit_gllvm(Y; family = Normal(), K = 2)
        @test fit.converged
        @test isapprox(fit.logLik, Float64(m["loglik"]); atol = 1e-6, rtol = 0)   # observed 1e-8
        # the R values are not degenerate: far from zero, not one constant
        @test minimum(abs, Float64.(m["coef"])) > 0.05
        @test length(unique(round.(Float64.(m["coef"]); digits = 3))) == p
        @test String.(m["coef_names"]) == ["trait" * t for t in tn]
        @test isapprox(coef(fit), Float64.(m["coef"]); atol = 1e-6, rtol = 0)   # POST-COEF-NAMED (observed 3e-13)
        rows = tidy(fit, Y)
        @test length(rows) == p
        @test isapprox([r.estimate for r in rows], Float64.(m["tidy_estimate"]); atol = 1e-4, rtol = 0)   # tidy fixed (observed 3e-13)
        @test isapprox(deviance(fit), Float64(m["deviance"]); atol = 2e-6, rtol = 0)   # POST-DEVIANCE (observed 2e-8)
    end
end
