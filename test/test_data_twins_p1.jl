# gllvm-parity-tag: P1
#
# Fit-level twins for the core070 `data` rows that are about offsets and missing responses, against
# gllvmTMB at P1 (9539352f66f2db2cc26b1c393e67212a359b60c9). The data batch that paid these rows
# replayed R helper functions and produced no fit number; here a small simulated fit exercises each
# capability in R and in Julia and compares the maximised logLik and the parameters. No R at test
# time: R's recorded values are read from test/fixtures/data_twins_p1.toml (generated once by
# test/fixtures/gen_data_twins_p1.R against a lane-local gllvmTMB install at the pin; the file
# records R version, commit and the data sha256s). Each R fit converged with a positive-definite
# Hessian (asserted). Both sides are compared at their own optimum.
#
# Tolerances were fixed BEFORE the Julia fits were first compared with R's numbers and are not
# loosened to pass: logLik 1e-6 (absolute), intercepts 1e-4, Lambda Lambda' 1e-3, dispersion 1e-3.
# The sign of a K = 1 loading axis is not identified, so loadings are compared through
# Lambda Lambda' (rotation- and sign-stable).
#
#   pois_none / pois_scalar / pois_exposure : Poisson, K = 1, offset absent / log(2) / log(e)
#       (R: offset(), Julia: offset = p x n matrix; Julia's beta is the offset-free intercept, as in R).
#   pois_na_drop / pois_na_include : 12 NA response cells, R default miss_control() and
#       miss_control(response = "include") against Julia `missing` cells in Y and mask = .
#   nb2_exposure / nb1_exposure : count families other than Poisson with an exposure offset.
#   gauss_zero : rank-2 Gaussian with offset(0) (the one offset R allows on a non-count trait).
using Test
using GLLVModels
using Distributions: Normal, Poisson, NegativeBinomial
using TOML
using LinearAlgebra
using SHA

const _DT_DIR = joinpath(@__DIR__, "fixtures")
const _DT_TOML = joinpath(_DT_DIR, "data_twins_p1.toml")

# R's write.csv long data (unit, trait "t#", then named columns; NA for a missing cell) -> p x n matrix.
function _dt_load(path::AbstractString, col::AbstractString, p::Integer, n::Integer)
    hdr = split(readline(path), ",")
    ci = findfirst(==("\"" * col * "\""), hdr)
    ci === nothing && error("column $col not found in $path")
    M = Matrix{Union{Missing,Float64}}(missing, p, n)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            a = split(line, ",")
            v = a[ci] == "NA" ? missing : parse(Float64, a[ci])
            M[parse(Int, strip(a[2], ['"', 't'])), parse(Int, strip(a[1], '"'))] = v
        end
    end
    return M
end
_dt_counts(M) = any(ismissing, M) ? Matrix{Union{Missing,Int}}(M) : Int.(M)
_dt_mat(v, nrow, ncol) = permutedims(reshape(Float64.(v), ncol, nrow))

function _dt_check(fx, sec)
    d = fx[sec]
    @test bytes2hex(sha256(read(joinpath(_DT_DIR, d["data_file"])))) == d["data_sha256"]
    @test d["converged"] && d["pd_hessian"]
    return d
end

@testset "data twins (offset, missing response): gllvmTMB P1 (9539352f6)" begin
    if !isfile(_DT_TOML)
        @warn "data twins P1 fixture absent; twin gate NOT RUN" _DT_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_DT_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        p, n = Int(fx["p"]), Int(fx["n_unit"])
        pois = joinpath(_DT_DIR, "data_twins_pois_p1_data.csv")
        Yc = _dt_counts(_dt_load(pois, "value", p, n))
        Ona = _dt_load(pois, "value_na", p, n)
        E = Float64.(_dt_load(pois, "e", p, n))

        function compare(sec, fit; extra = nothing)
            d = fx[sec]
            @test fit.converged
            @test isapprox(fit.loglik, Float64(d["loglik"]); atol = 1e-6, rtol = 0)
            @test isapprox(fit.β, Float64.(d["beta"]); atol = 1e-4, rtol = 0)
            @test isapprox(fit.Λ * fit.Λ', _dt_mat(d["lambda_lambdat"], p, p); atol = 1e-3, rtol = 0)
            extra === nothing || extra(d)
            return d
        end

        @testset "offset: none / scalar / exposure (Poisson)" begin
            _dt_check(fx, "pois_none"); _dt_check(fx, "pois_scalar"); _dt_check(fx, "pois_exposure")
            f0 = fit_gllvm(Yc; family = Poisson(), K = 1)
            compare("pois_none", f0)
            fs = fit_gllvm(Yc; family = Poisson(), K = 1, offset = fill(log(2), p, n))
            compare("pois_scalar", fs)
            # a constant offset is absorbed by the intercepts: same logLik, beta shifted by log(2) (both engines)
            @test isapprox(fs.loglik, f0.loglik; atol = 1e-6, rtol = 0)
            @test isapprox(fs.β, f0.β .- log(2); atol = 1e-4, rtol = 0)
            fe = fit_gllvm(Yc; family = Poisson(), K = 1, offset = log.(E))
            compare("pois_exposure", fe)
            # the exposure offset is not absorbed: it moves the optimum by far more than any tolerance
            @test abs(fe.loglik - f0.loglik) > 1
            # R values are not degenerate: pairwise distinct intercepts
            @test length(unique(round.(Float64.(fx["pois_exposure"]["beta"]); digits = 3))) == p
        end

        @testset "missing response cells: drop (default) and include" begin
            dd = _dt_check(fx, "pois_na_drop"); di = _dt_check(fx, "pois_na_include")
            @test dd["n_na"] == 12 == di["n_na"] == count(ismissing, Ona)
            Ym = _dt_counts(Ona)
            fm = fit_gllvm(Ym; family = Poisson(), K = 1)
            compare("pois_na_drop", fm)
            # R documents that include reaches the drop optimum; both are compared with Julia
            @test isapprox(Float64(dd["loglik"]), Float64(di["loglik"]); atol = 1e-6, rtol = 0)
            fi = fit_gllvm(Yc; family = Poisson(), K = 1, mask = .!ismissing.(Ona))
            compare("pois_na_include", fi)
            @test isapprox(fi.loglik, fm.loglik; atol = 1e-8, rtol = 0)   # Julia mask and `missing` agree
            # NA cells change the optimum against the complete data
            @test abs(fm.loglik - fx["pois_none"]["loglik"]) > 1
        end

        @testset "offset with other count families" begin
            nb2 = joinpath(_DT_DIR, "data_twins_nb2_p1_data.csv")
            nb1 = joinpath(_DT_DIR, "data_twins_nb1_p1_data.csv")
            _dt_check(fx, "nb2_exposure"); _dt_check(fx, "nb1_exposure")
            f2 = fit_gllvm(_dt_counts(_dt_load(nb2, "value", p, n)); family = NegativeBinomial(1.0, 0.5),
                           disp_group = :species, K = 1, offset = log.(Float64.(_dt_load(nb2, "e", p, n))))
            @test f2.group == collect(1:p)
            compare("nb2_exposure", f2; extra = d -> @test isapprox(f2.r_group, Float64.(d["phi"]); atol = 1e-3, rtol = 0))
            f1 = fit_gllvm(_dt_counts(_dt_load(nb1, "value", p, n)); family = NB1(), K = 1,
                           offset = log.(Float64.(_dt_load(nb1, "e", p, n))))
            compare("nb1_exposure", f1; extra = d -> @test isapprox(f1.φ, Float64.(d["phi"]); atol = 1e-3, rtol = 0))
        end

        @testset "Gaussian with a zero offset" begin
            g = _dt_check(fx, "gauss_zero")
            ng = Int(fx["n_unit_gauss"])
            Y = Float64.(_dt_load(joinpath(_DT_DIR, g["data_file"]), "value", p, ng))
            fg = fit_gllvm(Y; family = Normal(), K = 2, offset = zeros(p, ng))
            @test fg.converged
            @test isapprox(fg.logLik, Float64(g["loglik"]); atol = 1e-6, rtol = 0)
            @test isapprox(fg.logLik, Float64(g["loglik_plain"]); atol = 1e-6, rtol = 0)
            @test isapprox(coef(fg), Float64.(g["beta"]); atol = 1e-4, rtol = 0)
            @test startswith(g["nonzero_offset_refusal"], "offsets are supported for count families")
            # a difference, not a match: Julia accepts a nonzero offset on a Gaussian fit where R refuses it
            fg5 = fit_gllvm(Y; family = Normal(), K = 2, offset = fill(0.5, p, ng))
            @test fg5.converged
        end
    end
end
