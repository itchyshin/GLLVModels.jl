# gllvm-parity-tag: P1
#
# Numeric twin of gllvmTMB's confint_inspect() at P1 (9539352f66f2db2cc26b1c393e67212a359b60c9):
# namespace row export/confint_inspect. No R at test time: R's recorded values are read from
# test/fixtures/confint_inspect_p1.toml (generated once by test/fixtures/gen_confint_inspect_p1.R
# against a lane-local gllvmTMB install at the pin; it records R version, commit and the data
# sha256). The R fit converged (nlminb code 0) with a positive-definite Hessian (asserted).
#
# Model: the [main] fit of the namespace numeric twins (same CSV, ns_gauss_p1_data.csv):
#   value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE), Gaussian, p = 6, n = 200
#   <-> fit_gllvm(Y; family = Normal(), K = 2).
# R: confint_inspect(fit, parm, level = 0.95, ystep = 0.02)$bounds; Julia: confint_inspect(fit, Y;
# level = 0.95, parm) for the profile and Wald bounds and confint(fit, Y; parm).estimate for the
# point estimate. Four direct targets, one per kind of transformation:
#   R sigma_eps           (exp of log_sigma_eps)        <-> Julia "sigma_eps"     (log-SD Wald, exp back)
#   R b_fix[1]            (linear predictor)            <-> Julia "beta[1]"
#   R Lambda_B_packed[2]  (theta_rr_B[2], identity)     <-> Julia's 2nd Lambda_B term, "Lambda_B[2,2]"
#   R Lambda_B_packed[3]  (theta_rr_B[3], identity)     <-> Julia's 3rd Lambda_B term, "Lambda_B[2,1]"
# Both engines pack Lambda the same way (the d diagonal entries, then the strict lower triangle
# column by column): asserted in the R generator against extract_loadings(), and here by taking the
# k-th Lambda_B term of Julia's own term list. Compared per target: the estimate, the Wald bounds
# and the profile bounds, on the natural scale.
#
# Tolerances: sigma_eps and b_fix[1] at 1e-6 (observed at most 3.4e-7: the Wald bounds follow the
# two optimisers' estimates; the profile bounds agree to 2e-8); the two loadings at 5e-5, the same
# bound the namespace numeric twin uses for raw loadings on this fit (observed at most 1.0e-5,
# convergence noise in the loading estimates; profile bounds at most 4.4e-6).
#
# Limits, disclosed: R's $curve, $plot and $diagnostics are not compared (Julia returns no curve or
# plot; its single `disagree` flag uses the upper half-width, R's flags the full Wald half-width).
# R's bounds are read at ystep = 0.02 because the default grid (0.5) adds a linear-interpolation
# error of up to 2.2e-5 on these targets (recorded in the fixture, not compared). Gaussian only.
using Test
using GLLVModels
using Distributions: Normal
using TOML
using SHA

const _CI_DIR = joinpath(@__DIR__, "fixtures")

# R's write.csv long data (unit, trait, value) -> p x n response matrix.
function _ci_load_csv(path::AbstractString, trait_names::Vector{String}, n::Integer)
    p = length(trait_names)
    Y = fill(NaN, p, n)
    open(path) do io
        readline(io) == "\"unit\",\"trait\",\"value\"" || error("unexpected header in $path")
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, parse(Int, strip(parts[1], '"'))] = parse(Float64, parts[3])
        end
    end
    return Y
end

@testset "confint_inspect twin: gllvmTMB P1 (9539352f6)" begin
    cif = joinpath(_CI_DIR, "confint_inspect_p1.toml")
    if !isfile(cif)
        @warn "confint_inspect P1 fixture absent; twin gate NOT RUN" cif
        @test_skip false
    else
        fx = TOML.parsefile(cif)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        m = fx["main"]
        dp = joinpath(_CI_DIR, m["data_file"])
        @test bytes2hex(sha256(read(dp))) == m["data_sha256"]
        @test m["converged"] && m["pd_hessian"]
        @test m["level"] == 0.95
        Y = _ci_load_csv(dp, String.(m["trait_names"]), Int(m["n_unit"]))
        @test all(isfinite, Y)

        fit = fit_gllvm(Y; family = Normal(), K = 2)
        @test fit isa GllvmFit && fit.converged
        @test isapprox(fit.logLik, Float64(m["loglik"]); atol = 1e-6, rtol = 0)   # same optimum (observed 1e-8)

        lam = filter(startswith("Lambda_B["), confint(fit, Y).term)
        @test lam[2] == "Lambda_B[2,2]" && lam[3] == "Lambda_B[2,1]"   # packed order, as R's theta_rr_B
        jterm = Dict("sigma_eps" => "sigma_eps", "b_fix[1]" => "beta[1]",
                     "Lambda_B_packed[2]" => lam[2], "Lambda_B_packed[3]" => lam[3])
        ins = Dict(k => fx["inspect"][k] for k in ("sigma_eps", "b_fix_1", "lambda_packed_2", "lambda_packed_3"))
        J = Dict{String,Any}()
        for (k, r) in ins
            jp = jterm[r["parm"]]
            ci = confint_inspect(fit, Y; level = 0.95, parm = jp)
            @test ci.term == [jp]
            J[k] = (estimate = only(confint(fit, Y; level = 0.95, parm = jp).estimate),
                    wald = [only(ci.wald_lower), only(ci.wald_upper)],
                    profile = [only(ci.profile_lower), only(ci.profile_upper)])
            @test all(isfinite, J[k].wald) && all(isfinite, J[k].profile)
        end

        s, b, l2, l3 = ins["sigma_eps"], ins["b_fix_1"], ins["lambda_packed_2"], ins["lambda_packed_3"]
        @test isapprox(J["sigma_eps"].estimate, Float64(s["estimate"]); atol = 1e-6, rtol = 0)           # sigma_eps estimate (observed 3.0e-7)
        @test isapprox(J["sigma_eps"].wald, Float64.(s["wald"]); atol = 1e-6, rtol = 0)                  # sigma_eps Wald (observed 3.4e-7)
        @test isapprox(J["sigma_eps"].profile, Float64.(s["profile"]); atol = 1e-6, rtol = 0)            # sigma_eps profile (observed 3.5e-9)
        @test isapprox(J["b_fix_1"].estimate, Float64(b["estimate"]); atol = 1e-6, rtol = 0)             # b_fix[1] estimate (observed 4.3e-14)
        @test isapprox(J["b_fix_1"].wald, Float64.(b["wald"]); atol = 1e-6, rtol = 0)                    # b_fix[1] Wald (observed 8.9e-8)
        @test isapprox(J["b_fix_1"].profile, Float64.(b["profile"]); atol = 1e-6, rtol = 0)              # b_fix[1] profile (observed 1.9e-8)
        @test isapprox(J["lambda_packed_2"].estimate, Float64(l2["estimate"]); atol = 5e-5, rtol = 0)    # Lambda[2,2] estimate (observed 1.9e-6)
        @test isapprox(J["lambda_packed_2"].wald, Float64.(l2["wald"]); atol = 5e-5, rtol = 0)           # Lambda[2,2] Wald (observed 2.1e-6)
        @test isapprox(J["lambda_packed_2"].profile, Float64.(l2["profile"]); atol = 5e-5, rtol = 0)     # Lambda[2,2] profile (observed 4.4e-6)
        @test isapprox(J["lambda_packed_3"].estimate, Float64(l3["estimate"]); atol = 5e-5, rtol = 0)    # Lambda[2,1] estimate (observed 9.8e-6)
        @test isapprox(J["lambda_packed_3"].wald, Float64.(l3["wald"]); atol = 5e-5, rtol = 0)           # Lambda[2,1] Wald (observed 1.0e-5)
        @test isapprox(J["lambda_packed_3"].profile, Float64.(l3["profile"]); atol = 5e-5, rtol = 0)     # Lambda[2,1] profile (observed 1.0e-8)
    end
end
