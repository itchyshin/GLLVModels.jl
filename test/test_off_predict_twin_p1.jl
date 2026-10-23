# gllvm-parity-tag: P1
#
# Twin for the core070 row data/DATA-OFF-PREDICT against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9), under the maintainer ruling of 2026-10-05 (vault
# D-319, option (b)): a prediction on the training units with a NEW offset, keeping the training
# units' latent modes. No R at test time: R's values are read from
# test/fixtures/off_predict_twin_p1.toml (generated once by test/fixtures/gen_off_predict_twin_p1.R
# against a lane-local gllvmTMB install at the pin; the file records R version, commit and the data
# sha256s).
#
# R side: the Poisson exposure fit of the data twins (offset(log(e)), K = 1), then
# predict(fit, newdata = training rows with e = e_new, type = "link")$est. R re-evaluates the stored
# offset expression on newdata and keeps the training modes (asserted in the generator: the new
# prediction minus log(e_new) equals the training prediction minus log(e)). Julia side: the same fit
# through fit_gllvm(...; offset = log.(E)), then
# predict(fit, Y; type = :link, offset = log.(E_new), modes = :training).
#
# Tolerances are those of the sibling twin test_predict_offset_twin_p1.jl (same fit, same data),
# fixed before the Julia values were compared and not loosened to pass: logLik 1e-6, link predictor
# 1e-3 (absolute; the two optima agree to 1e-4 in the intercepts and 1e-3 in Lambda Lambda'). The
# link predictor is rotation- and sign-invariant.
using Test
using GLLVModels
using Distributions: Poisson
using TOML
using SHA

const _OP_DIR = joinpath(@__DIR__, "fixtures")
const _OP_TOML = joinpath(_OP_DIR, "off_predict_twin_p1.toml")

# R's write.csv long data (unit, trait "t#", then named columns) -> p x n matrix.
function _op_load(path::AbstractString, col::AbstractString, p::Integer, n::Integer)
    hdr = split(readline(path), ",")
    ci = findfirst(==("\"" * col * "\""), hdr)
    ci === nothing && error("column $col not found in $path")
    M = Matrix{Float64}(undef, p, n)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            a = split(line, ",")
            M[parse(Int, strip(a[2], ['"', 't'])), parse(Int, strip(a[1], '"'))] = parse(Float64, a[ci])
        end
    end
    return M
end
# A long vector in R's row order (unit-major, trait within unit) -> p x n matrix.
_op_wide(v, p, n) = reshape(Float64.(v), p, n)

@testset "training units, new offset, training modes kept: gllvmTMB P1 (9539352f6)" begin
    if !isfile(_OP_TOML)
        @warn "off-predict twin P1 fixture absent; twin gate NOT RUN" _OP_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_OP_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        p, n = Int(fx["p"]), Int(fx["n_unit"])
        d = fx["pois_exposure_new_offset"]
        csv = joinpath(_OP_DIR, d["data_file"])
        ncsv = joinpath(_OP_DIR, d["new_offset_file"])
        @test bytes2hex(sha256(read(csv))) == d["data_sha256"]
        @test bytes2hex(sha256(read(ncsv))) == d["new_offset_sha256"]
        @test d["converged"] && d["pd_hessian"]
        Y = Int.(_op_load(csv, "value", p, n))
        E = _op_load(csv, "e", p, n)
        Enew = _op_load(ncsv, d["new_offset_column"], p, n)
        fe = fit_gllvm(Y; family = Poisson(), K = 1, offset = log.(E))
        @test fe.converged
        @test isapprox(fe.loglik, Float64(d["loglik"]); atol = 1e-6, rtol = 0)
        # R: predict(fit, newdata = nd, type = "link")$est, nd = training rows with e = e_new
        R_new = _op_wide(d["eta_link_new_offset"], p, n)
        eta_new = predict(fe, Y; type = :link, offset = log.(Enew), modes = :training)
        @test isapprox(eta_new, R_new; atol = 1e-3, rtol = 0)
        # R's numbers are not degenerate, and the comparison discriminates: the new offset moves R's
        # predictor far from the training one, and the default Julia route (modes re-solved at the
        # new offset, a different quantity) misses R by far more than the tolerance
        R_train = _op_wide(d["eta_link_train"], p, n)
        @test maximum(abs, R_new .- R_train) > 0.5
        @test maximum(abs, predict(fe, Y; type = :link, offset = log.(Enew)) .- R_new) > 100 * 1e-3
    end
end
