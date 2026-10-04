# gllvm-parity-tag: P1
#
# Twin for the core070 row data/DATA-OFF-TRAIN-STORED against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9): a fit keeps the offset it was trained with, and the
# training-row prediction uses it. No R at test time: R's values are read from
# test/fixtures/predict_offset_twin_p1.toml (generated once by
# test/fixtures/gen_predict_offset_twin_p1.R against a lane-local gllvmTMB install at the pin).
# R side: the Poisson exposure fit of the data twins (offset(log(e)), K = 1), its stored training
# offset `.gllvmTMB_offset_vec(fit)` and `predict(fit, type = "link")$est`. Julia side: the same fit
# through `fit_gllvm(...; offset = log.(E))`, its stored `fit.offset` and `predict(fit, Y; type = :link)`.
#
# Tolerances were fixed BEFORE the Julia values were first compared with R's numbers and are not
# loosened to pass: the stored offset 1e-14 (absolute; both sides hold log(e) of the same CSV
# values, so only the last-bit rounding of `log` may differ), logLik 1e-6, the training-row link
# predictor 1e-3 (absolute; the two optima agree to 1e-4 in the intercepts and 1e-3 in
# Lambda Lambda', test_data_twins_p1.jl). The link predictor is rotation- and sign-invariant.
using Test
using GLLVModels
using Distributions: Poisson
using TOML
using SHA

const _PT_DIR = joinpath(@__DIR__, "fixtures")
const _PT_TOML = joinpath(_PT_DIR, "predict_offset_twin_p1.toml")

# R's write.csv long data (unit, trait "t#", then named columns) -> p x n matrix.
function _pt_load(path::AbstractString, col::AbstractString, p::Integer, n::Integer)
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
_pt_wide(v, p, n) = reshape(Float64.(v), p, n)

@testset "stored training offset and training-row predict: gllvmTMB P1 (9539352f6)" begin
    if !isfile(_PT_TOML)
        @warn "predict-offset twin P1 fixture absent; twin gate NOT RUN" _PT_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_PT_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        p, n = Int(fx["p"]), Int(fx["n_unit"])
        d = fx["pois_exposure_stored"]
        csv = joinpath(_PT_DIR, d["data_file"])
        @test bytes2hex(sha256(read(csv))) == d["data_sha256"]
        @test d["converged"] && d["pd_hessian"]
        Y = Int.(_pt_load(csv, "value", p, n))
        E = _pt_load(csv, "e", p, n)
        fe = fit_gllvm(Y; family = Poisson(), K = 1, offset = log.(E))
        @test fe.converged
        @test isapprox(fe.loglik, Float64(d["loglik"]); atol = 1e-6, rtol = 0)
        # the stored training offset, read back
        R_off = _pt_wide(d["offset_vec"], p, n)
        @test isapprox(fe.offset, R_off; atol = 1e-14, rtol = 0)
        # the training-row link predictor uses it (R: predict(fit, type = "link")$est)
        R_eta = _pt_wide(d["eta_link"], p, n)
        @test isapprox(predict(fe, Y; type = :link), R_eta; atol = 1e-3, rtol = 0)
        # R's numbers are not degenerate, and the comparison discriminates: the offset-free
        # prediction (what predict returned before the fit kept its offset) misses R by far more
        # than the tolerance
        @test maximum(abs, R_off) > 0.5
        @test maximum(abs, predict(fe, Y; type = :link, offset = zeros(p, n)) .- R_eta) > 100 * 1e-3
    end
end
