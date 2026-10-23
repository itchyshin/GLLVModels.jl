# gllvm-parity-tag: P1
#
# Numeric twin of gllvmTMB's predict_missing() at P1 (9539352f66f2db2cc26b1c393e67212a359b60c9),
# row postfit/POSTFIT-SURFACE-predict_missing. No R at test time: R's recorded values are read from
# test/fixtures/predict_missing_p1.toml (generated once by test/fixtures/gen_predict_missing_p1.R
# against a lane-local gllvmTMB install at the pin; it records R version, commit and the data
# sha256).
#
# Fixture: the data of test/fixtures/ns_numeric_p1.toml [main] (p = 6 traits, n = 200 units,
# UNCENTRED), with 69 cells masked: traits 2 and 5 in every unit u with u % 10 == 3, trait 1 in
# every unit with u % 7 == 4. The mask is recorded in the fixture and read back here.
# R: value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE) with
# miss_control(response = "include"), so the masked cells stay in the likelihood as unobserved
# cells; predict_missing(fit) returns the fitted linear predictor (trait mean plus
# Lambda * posterior-mode score given the unit's observed cells) at each masked cell.
# <-> fit_gaussian_gllvm(Y; K = 2, X = one-hot trait design, mask = mask) and
#     predict_missing(fit, Y; mask = mask). The Julia fit takes the trait means through an
# explicit one-hot X (with X = nothing the masked Gaussian route fits a zero-mean model).
# Both fits converge to the same optimum (log-likelihoods compared). The R fit converged with a
# positive-definite Hessian (asserted).
#
# Limits, disclosed: the mask pattern was chosen when many masks made
# fit_gaussian_gllvm(...; mask) throw PosDefException (Cholesky in _gaussian_data_nll); that
# robustness bug is fixed (#716, test/test_masked_gaussian_posdef.jl). The mask here is benign (light missingness, at most three masked cells per unit), so this twin does
# not cover heavy or irregular missingness. Under the Gaussian identity link the response-scale
# values equal the link-scale values, so the :response check exercises that path but adds no
# independent numeric evidence.
using Test
using GLLVModels
using TOML
using SHA

const _PM_DIR = joinpath(@__DIR__, "fixtures")

@testset "predict_missing twin: gllvmTMB P1 (9539352f6)" begin
    pmf = joinpath(_PM_DIR, "predict_missing_p1.toml")
    nsf = joinpath(_PM_DIR, "ns_numeric_p1.toml")
    if !(isfile(pmf) && isfile(nsf))
        @warn "predict_missing P1 fixture absent; twin gate NOT RUN" pmf
        @test_skip false
    else
        fx = TOML.parsefile(pmf)
        ns = TOML.parsefile(nsf)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        m = fx["main"]
        dp = joinpath(_PM_DIR, m["data_file"])
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
        mu = Int.(m["masked_unit"]); mt = Int.(m["masked_trait"])
        @test length(mu) == length(mt) == Int(m["n_masked"])
        mask = trues(p, n)
        for i in eachindex(mu)
            mask[mt[i], mu[i]] = false
        end
        @test count(!, mask) == Int(m["n_masked"])
        X = zeros(Float64, p, n, p)
        for t in 1:p
            X[t, :, t] .= 1.0
        end
        Ym = copy(Y)
        Ym[.!mask] .= 0.0                  # the masked values are never read
        fit = fit_gaussian_gllvm(Ym; K = 2, X = X, mask = mask)
        @test fit.converged
        @test isapprox(fit.logLik, Float64(m["loglik"]); atol = 1e-6, rtol = 0)   # same optimum as R's masked fit

        out = predict_missing(fit, Ym; mask = mask, type = :link)
        # same cells, same order as R's rows (R: model_row ascending = unit, then trait)
        @test out.col == mu
        @test out.row == mt
        r_link = Float64.(m["est_link"])
        # the R values are not degenerate: not one constant, not zero, and they vary across traits
        @test length(unique(round.(r_link; digits = 3))) > length(r_link) ÷ 2
        @test maximum(abs, r_link) > 0.5
        @test isapprox(out.est, r_link; atol = 1e-5, rtol = 0)   # predict_missing, link scale
        # identity link: R's response-scale values are the link-scale values
        outr = predict_missing(fit, Ym; mask = mask, type = :response)
        @test isapprox(outr.est, Float64.(m["est_response"]); atol = 1e-5, rtol = 0)   # predict_missing, response scale
        # a masked cell's prediction does not read its own value
        Y2 = copy(Y)
        Y2[.!mask] .= 123.0
        out2 = predict_missing(fit, Y2; mask = mask, type = :link)
        @test isapprox(out2.est, out.est; atol = 1e-8, rtol = 0)
        # mask = nothing means every cell observed, so no rows (R's zero-argument call instead reads
        # the mask stored in the fit; Julia's fit does not store one, see the predict_missing docstring)
        @test isempty(predict_missing(fit, Ym).est)
    end
end
