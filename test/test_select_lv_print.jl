# gllvm-parity-tag: P1
#
# select_lv() result: the nine fields gllvmTMB's print.gllvmTMB_select_lv()
# shows per rank (R/select-lv.R at gllvmTMB P1 = 9539352f66f2db2cc26b1c393e67212a359b60c9):
#   <marker> d npar logLik AIC BIC AICc conv pdHess
#
# What R does (read at the pin):
#   * AICc = AIC + 2*npar*(npar+1) / (nobs - npar - 1); NA when nobs - npar - 1 <= 0.
#     nobs is attr(logLik(fit), "nobs"), the likelihood-contributing cell count,
#     the SAME n that goes into BIC (here p*n observed cells).
#   * conv   = isTRUE(fit$opt$convergence == 0).
#   * pdHess = isTRUE(fit$sd_report$pdHess); NA when sdreport() was skipped.
#   * Printed with round(., 3), the marker "*" on the selected row, TRUE/FALSE/NA.
# R EXCLUDES a rank that did not converge or has a confirmed non-PD Hessian from
# the selection; Julia's select_lv keeps such ranks selectable (the known twin
# difference, GATES.md "select_lv"). This file pins that Julia behaviour so a
# change of it is deliberate.
#
# The expected printed lines below were produced by running R's own
# print.data.frame on the same numbers (R 4.6.0), not typed from memory.
using Test
using GLLVModels
using Distributions: Normal, Poisson
using TOML
using SHA
using Random

const _SELPR_DIR = joinpath(@__DIR__, "fixtures")
const _SELPR_TOML = joinpath(_SELPR_DIR, "select_lv_p1.toml")

function _selpr_load_csv(path::AbstractString, trait_names::Vector{String}, n_unit::Integer)
    Y = zeros(Float64, length(trait_names), n_unit)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            unit = parse(Int, strip(parts[1], '"'))
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, unit] = parse(Float64, parts[3])
        end
    end
    return Y
end

# Stand-in fit: exact numbers, no optimiser. `pd === nothing` makes confint throw
# (a fit type whose Hessian cannot be evaluated).
struct _SelPrFake
    ll::Float64
    np::Int
    converged::Bool
    pd::Union{Bool,Nothing}
end
GLLVModels._loglik(f::_SelPrFake) = f.ll
GLLVModels._nparams(f::_SelPrFake) = f.np
GLLVModels.StatsAPI.aic(f::_SelPrFake) = 2f.np - 2f.ll
GLLVModels.StatsAPI.bic(f::_SelPrFake, Y::AbstractMatrix; mask = nothing) =
    f.np * log(mask === nothing ? length(Y) : count(mask)) - 2f.ll
GLLVModels.StatsAPI.bic(f::_SelPrFake, n::Integer) = f.np * log(n) - 2f.ll
GLLVModels.confint(f::_SelPrFake, Y::AbstractMatrix; kwargs...) =
    f.pd === nothing ? throw(ArgumentError("no Hessian for this stand-in")) : (pd_hessian = f.pd,)

_selpr_fitter(lls, nps; conv = fill(true, length(lls)), pd = fill(true, length(lls)), fail_at = 0) =
    (Y; family, K, kwargs...) -> begin
        K == fail_at && error("singular at K=$K")
        _SelPrFake(lls[K], nps[K], conv[K], pd[K])
    end

_selpr_show(sel) = sprint(show, MIME("text/plain"), sel)

@testset "select_lv() result carries and prints R's nine fields" begin

    @testset "AICc: R's formula, n = the BIC n, NaN when the correction is undefined" begin
        Y = ones(4, 5)                                   # 20 observed cells
        sel = select_lv(Y; Kmax = 3, criterion = :aic,
                        _fitter = _selpr_fitter([-100.0, -90.0, -85.0], [10, 19, 25]))
        @test sel.K == [1, 2, 3]
        # n = 20: denominators 9, 0, -6.  Only K = 1 has a defined correction.
        @test sel.aicc[1] ≈ sel.aic[1] + 2 * 10 * 11 / (20 - 10 - 1)
        @test isnan(sel.aicc[2])                         # n - npar - 1 == 0
        @test isnan(sel.aicc[3])                         # n - npar - 1 < 0
        # n follows a mask exactly as BIC does (count of observed cells).
        mask = trues(4, 5); mask[1, 1] = false; mask[2, 3] = false
        selm = select_lv(Y; Kmax = 2, criterion = :aic, mask = mask,
                         _fitter = _selpr_fitter([-100.0, -90.0], [10, 12]))
        @test selm.aicc[1] ≈ selm.aic[1] + 2 * 10 * 11 / (18 - 10 - 1)
        @test selm.aicc[2] ≈ selm.aic[2] + 2 * 12 * 13 / (18 - 12 - 1)
    end

    @testset "conv: per-rank convergence flag; an unconverged rank stays selectable" begin
        Y = ones(6, 20)
        # K = 2 did not report convergence and has the lowest AIC.
        fit = _selpr_fitter([-300.0, -200.0, -199.5], [8, 14, 19]; conv = [true, false, true])
        sel = select_lv(Y; Kmax = 3, criterion = :aic, _fitter = fit)
        @test sel.converged == [true, false, true]
        @test sel.best_k == 2                            # selection rule unchanged
        @test sel.best_k == sel.K[argmin(sel.aic)]
        # the printed row for K = 2 says FALSE
        row2 = filter(l -> occursin(r"^ \* 2 ", l), split(_selpr_show(sel), '\n'))
        @test length(row2) == 1 && occursin("FALSE", only(row2))
    end

    @testset "pdHess: opt-in; NA when not computed; a non-PD rank is kept (twin fence)" begin
        Y = ones(6, 20)
        mk() = _selpr_fitter([-300.0, -200.0, -199.5], [8, 14, 19]; pd = [true, false, true])
        off = select_lv(Y; Kmax = 3, criterion = :aic, _fitter = mk())
        @test all(ismissing, off.pd_hessian)             # default: not computed
        on = select_lv(Y; Kmax = 3, criterion = :aic, pd_hessian = true, _fitter = mk())
        @test on.pd_hessian == [true, false, true]
        # Julia keeps the confirmed-non-PD rank selectable (R would exclude it).
        @test on.best_k == off.best_k == 2
        # a fit whose Hessian cannot be evaluated is NA, never FALSE
        nohess = _selpr_fitter([-300.0, -200.0], [8, 14]; pd = [true, nothing])
        sn = select_lv(Y; Kmax = 2, criterion = :aic, pd_hessian = true, _fitter = nohess)
        @test sn.pd_hessian[1] === true
        @test ismissing(sn.pd_hessian[2])
    end

    @testset "Gaussian P1 fixture: AICc, conv and pdHess equal R's recorded values" begin
        if !isfile(_SELPR_TOML)
            @warn "select_lv P1 fixture absent; print twin NOT RUN" _SELPR_TOML
            @test_skip false
        else
            fx = TOML.parsefile(_SELPR_TOML)
            @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
            trait_names = String.(fx["trait_names"])
            p, n = Int(fx["p"]), Int(fx["n_unit"])
            data_path = joinpath(_SELPR_DIR, fx["data_file"])
            @test bytes2hex(sha256(read(data_path))) == fx["data_sha256"]
            Y = _selpr_load_csv(data_path, trait_names, n)
            r = fx["r_reference"]
            @test Int(r["nobs"]) == p * n

            sel = select_lv(Y; family = Normal(), Kmax = 3, criterion = :bic, pd_hessian = true)

            @test isapprox(sel.aicc, Float64.(r["aicc"]); atol = 2e-6, rtol = 0)
            @test sel.converged == Bool.(r["converged"])
            @test sel.pd_hessian == Bool.(r["pd_hessian"])
            # n in AICc is the BIC n: nobs(fit, Y) = p*n
            @test GLLVModels.StatsAPI.nobs(sel.best, Y) == p * n
            for i in eachindex(sel.K)
                k = sel.nparams[i]
                @test sel.aicc[i] ≈ sel.aic[i] + 2k * (k + 1) / (p * n - k - 1)
            end
            # selection and the default criterion are untouched
            @test sel.best_k == Int(r["selected_d"])

            # Printed table: R's own print.data.frame lines for the same numbers.
            out = split(_selpr_show(sel), '\n')
            @test out[1] == "GLLVModels latent-dimension selection (criterion = bic, best K = 2)"
            @test out[2] == "   d npar    logLik      AIC      BIC     AICc conv pdHess"
            @test out[3] == "   1   13 -1183.645 2393.289 2455.720 2393.700 TRUE   TRUE"
            @test out[4] == " * 2   18 -1032.949 2101.898 2188.341 2102.674 TRUE   TRUE"
            @test out[5] == "   3   22 -1030.958 2105.915 2211.568 2107.069 TRUE   TRUE"
            @test !occursin("pd_hessian = true", join(out, '\n'))   # all pdHess known: no hint
        end
    end

    @testset "Poisson (Laplace) route: pd_hessian is the Wald flag; AICc uses nobs = p*n" begin
        Random.seed!(11)
        p, n = 4, 60
        Y = [rand(Poisson(exp(1.2 + 0.4 * randn()))) for _ in 1:p, _ in 1:n]
        sel = select_lv(Y; family = Poisson(), Kmax = 2, criterion = :bic, pd_hessian = true)
        @test all(x -> x isa Bool, sel.pd_hessian)
        ib = findfirst(==(sel.best_k), sel.K)
        @test sel.pd_hessian[ib] == confint(sel.best, Y; method = :wald).pd_hessian
        for i in eachindex(sel.K)
            k = sel.nparams[i]
            @test sel.aicc[i] ≈ sel.aic[i] + 2k * (k + 1) / (p * n - k - 1)
        end
        @test sel.converged isa Vector{Bool}
    end

    @testset "show: R's labels and layout, NA/FALSE/large-magnitude values" begin
        # Direct construction of the printed object with the cases R's printer
        # handles: a column of large magnitudes (R drops to 2 decimals), an NA
        # AICc, an unconverged rank and an undetermined pdHess.
        sel = GLLVModels.LVSelection(
            [1, 2, 3], [13, 18, 22],
            [-12345.6789, -11032.9494, -11030.9576],
            [24717.3578, 22101.8988, 22105.9152],
            [24779.7203, 22188.3411, 22211.5679],
            [24779.7203, 22188.3411, 22211.5679],
            [24717.7001, 22102.6744, NaN],
            [true, true, false],
            Union{Missing,Bool}[missing, missing, missing],
            2, nothing,
            NamedTuple{(:K, :status, :loglik, :message),Tuple{Int,Symbol,Float64,String}}[], :bic)
        out = split(_selpr_show(sel), '\n')
        @test out[1] == "GLLVModels latent-dimension selection (criterion = bic, best K = 2)"
        @test out[2] == "   d npar    logLik      AIC      BIC     AICc  conv pdHess"
        @test out[3] == "   1   13 -12345.68 24717.36 24779.72 24717.70  TRUE     NA"
        @test out[4] == " * 2   18 -11032.95 22101.90 22188.34 22102.67  TRUE     NA"
        @test out[5] == "   3   22 -11030.96 22105.92 22211.57       NA FALSE     NA"
        @test occursin("pd_hessian = true", join(out, '\n'))   # hint: pdHess not computed
    end

    @testset "show: ranks that were not used are still listed" begin
        Y = ones(6, 20)
        sel = select_lv(Y; Kmax = 3, criterion = :aic,
                        _fitter = _selpr_fitter([-300.0, -200.0, -199.5], [8, 14, 19]; fail_at = 3))
        @test sel.K == [1, 2]
        @test occursin("K = 3 not used: failed", _selpr_show(sel))
    end
end
