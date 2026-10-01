# gllvm-parity-tag: P1
#
# Numeric twin of select_lv() against gllvmTMB's select_lv() at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9, R/select-lv.R). No R at test time:
# R's recorded sweep is read from test/fixtures/select_lv_p1.toml (generated once
# by test/fixtures/gen_select_lv_p1.R against a lane-local gllvmTMB install at the
# pin; the file records R version, commit, seed and the data sha256) and the same
# dataset is swept in Julia.
#
# Design. Gaussian responses, p = 6 traits, n = 150 units, true latent rank 2,
# ranks 1:3, criterion = BIC with R's penalty log(p*n) (R's default). R side:
#   value ~ 0 + trait + latent(0 + trait | unit, d = k, unique = FALSE)
# i.e. Sigma = Lambda Lambda' + sigma_eps^2 I, the structure of Julia's
# fit_gllvm(Y; family = Normal(), K = k). Both engines converge at every rank;
# R reports a positive-definite Hessian at every rank (asserted below), so the
# comparison stays clear of the known fence: Julia's select_lv does not reject
# fits with a non-positive-definite Hessian as R does (signed on
# itchyshin/GLLVModels.jl#533). A fixture where R's Hessian is not PD at some
# rank would not be a valid twin and the test refuses to run on one.
#
# Tolerances. logLik: atol 1e-6 at each side's own optimum. AIC/BIC: atol 2e-6
# (they are -2 logLik plus an identical exact penalty). npar and the selected
# rank: exact. Observed differences are ~1e-8.
using Test
using GLLVModels
using Distributions: Normal
using TOML
using SHA

const _SELLV_DIR = joinpath(@__DIR__, "fixtures")
const _SELLV_TOML = joinpath(_SELLV_DIR, "select_lv_p1.toml")

# (unit, trait, value) CSV from R's write.csv -> p x n Float64 matrix.
function _load_select_lv_csv(path::AbstractString, trait_names::Vector{String}, n_unit::Integer)
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

@testset "select_lv() twin: gllvmTMB P1 (9539352f6) Gaussian rank sweep" begin
    if !isfile(_SELLV_TOML)
        @warn "select_lv P1 fixture absent; twin gate NOT RUN" _SELLV_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_SELLV_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        trait_names = String.(fx["trait_names"])
        p, n = Int(fx["p"]), Int(fx["n_unit"])
        data_path = joinpath(_SELLV_DIR, fx["data_file"])
        @test bytes2hex(sha256(read(data_path))) == fx["data_sha256"]
        Y = _load_select_lv_csv(data_path, trait_names, n)
        @test size(Y) == (p, n)

        r = fx["r_reference"]
        Ks = Int.(r["d"])
        # Valid-twin guard (the fence): R converged with a PD Hessian at every rank.
        @test all(r["converged"]) && all(r["pd_hessian"])
        @test Int(r["nobs"]) == p * n          # R's BIC penalty is log(p*n)
        @test r["criterion"] == "bic"

        sel = select_lv(Y; family = Normal(), Kmax = maximum(Ks), criterion = :bic)

        @test sel.K == Ks
        @test all(a.status === :ok for a in sel.attempts)
        @test sel.nparams == Int.(r["npar"])
        @test isapprox(sel.loglik, Float64.(r["loglik"]); atol = 1e-6, rtol = 0)
        @test isapprox(sel.aic,    Float64.(r["aic"]);    atol = 2e-6, rtol = 0)
        @test isapprox(sel.bic,    Float64.(r["bic"]);    atol = 2e-6, rtol = 0)
        @test sel.best_k == Int(r["selected_d"]) == Int(fx["true_rank"])
    end
end
