# Frozen-R parity: ZIP, Tweedie, Beta, Gamma GLLVM fits vs gllvmTMB (Refs #739).
#
# Tests only. No RCall and no R needed. Every R value below is a number a real
# gllvmTMB fit produced and the repo already stores:
#   ZIP     test/fixtures/zi_p1.toml          (gllvmTMB 0.7.1 @ 9539352f, zi_poisson())
#   Tweedie docs/dev-log/core070/true-parity-latest/receipts/family/runparity-tweedie/
#           values-NATIVE-07-TWEEDIE.toml     (replays test/parity/test_tweedie_parity.jl, seed 82)
#   Beta    .../runparity-beta/values-NATIVE-08-BETA.toml
#                                             (replays test/parity/test_beta_parity.jl, seed 45)
#   Gamma   .../runparity-dispersion/values-NATIVE-05-GAMMA.toml
#                                             (replays test/parity/test_nox_dispersion_parity.jl, seed 54)
# The three simulated cells (Tweedie seed 82, Beta seed 45, Gamma seed 54) read frozen,
# sha256-guarded Y matrices from test/fixtures/frozen_parity_739.toml instead of replaying
# the simulators: Julia's seeded RNG stream is not stable across versions (seed-82 Tweedie
# differs in 1 of 750 cells between 1.10.12 and 1.13.2). The data were generated on
# Julia 1.10.12 by test/fixtures/gen_frozen_parity_739.jl, the version behind the R receipts.
# A row that matches R is a plain @test. A row that does not is @test_broken with the
# issue it belongs to, so a later fix turns it into an "unexpected pass" to be promoted.
using Test
using GLLVModels
using TOML
using SHA

const _TP_FIXTURES = joinpath(@__DIR__, "fixtures")

# Same format reader as test_zi_twin.jl: (site, trait, y) CSV -> p x n.
function _tp_load_zi_csv(path::AbstractString, p::Integer, n::Integer)
    Y = zeros(Float64, p, n)
    for line in readlines(path)[2:end]
        isempty(line) && continue
        q = split(line, ",")
        Y[parse(Int, q[2]), parse(Int, q[1])] = parse(Int, q[3])
    end
    return Y
end

# Frozen simulated Y (sha256-checked) for one section of frozen_parity_739.toml.
function _tp_load_frozen(section::AbstractString)
    t = TOML.parsefile(joinpath(_TP_FIXTURES, "frozen_parity_739.toml"))[section]
    path = joinpath(_TP_FIXTURES, t["data_file"])
    @test bytes2hex(sha256(read(path))) == t["data_sha256"]
    p, n = Int(t["n_trait"]), Int(t["n_unit"])
    Y = fill(NaN, p, n)
    for line in readlines(path)[2:end]
        isempty(line) && continue
        q = split(line, ",")
        Y[parse(Int, q[2]), parse(Int, q[1])] = parse(Float64, q[3])
    end
    @test !any(isnan, Y)
    return Y
end

@testset "frozen gllvmTMB parity: ZIP / Tweedie / Beta / Gamma (Refs #739)" begin

    @testset "ZIP: fit_zip_gllvm vs gllvmTMB zi_poisson() (zi_p1.toml)" begin
        t = TOML.parsefile(joinpath(_TP_FIXTURES, "zi_p1.toml"))["zi_poisson"]
        path = joinpath(_TP_FIXTURES, t["data_file"])
        @test bytes2hex(sha256(read(path))) == t["data_sha256"]
        Y = _tp_load_zi_csv(path, 3, Int(t["n_unit"]))
        llR = Float64(t["r_reference"]["loglik"])
        @test t["r_reference"]["convergence"] == 0

        fit = GLLVModels.fit_zip_gllvm(Y; K = 1, iterations = 2000)       # default :observed
        @test fit.converged
        @test isapprox(fit.loglik, llR; atol = 1e-6)
        @test maximum(abs.(fit.βz .- Float64.(t["r_reference"]["logit_zi"]))) < 1e-3

        # Opt-in Fisher log-det weight is a different objective from TMB's (#739 mechanism);
        # it stays below R's value. Recorded as broken so the gap is visible, not asserted away.
        fit_f = GLLVModels.fit_zip_gllvm(Y; K = 1, hessian = :fisher, iterations = 2000)
        @test fit_f.converged
        @test_broken isapprox(fit_f.loglik, llR; atol = 1e-6)    # #739
    end

    @testset "Tweedie (frozen seed-82 data, p=5 K=1 n=150): per-species and fixed power vs R" begin
        Y = _tp_load_frozen("tweedie")
        K = 1
        rv = TOML.parsefile(joinpath(@__DIR__, "..", "docs", "dev-log", "core070", "true-parity-latest",
            "receipts", "family", "runparity-tweedie", "values-NATIVE-07-TWEEDIE.toml"))["values"]
        llR = Dict(v["label"] => Float64(v["r"]) for v in rv)

        fit = fit_tweedie_gllvm_grouped(Y; K = K, power = 1.5, iterations = 400)
        @test fit.converged
        @test isapprox(fit.loglik, llR["logLik, fixed common power"]; atol = 1e-6)
        fit = fit_tweedie_gllvm_grouped(Y; K = K, power_group = :shared, iterations = 400)
        @test fit.converged
        @test isapprox(fit.loglik, llR["logLik, estimated shared power"]; atol = 1e-6)
        fit = fit_tweedie_gllvm_grouped(Y; K = K, power_group = :species, iterations = 400)
        @test fit.converged
        @test isapprox(fit.loglik, llR["logLik, estimated per-species power"]; atol = 1e-6)
    end

    @testset "Beta (frozen seed-45 data, p=5 K=1 n=60): per-trait phi vs R" begin
        Y = _tp_load_frozen("beta")
        K = 1
        rv = TOML.parsefile(joinpath(@__DIR__, "..", "docs", "dev-log", "core070", "true-parity-latest",
            "receipts", "family", "runparity-beta", "values-NATIVE-08-BETA.toml"))["values"]
        llR = Float64(only(v for v in rv if v["label"] == "logLik")["r"])
        fit = fit_gllvm(Y; family = GLLVModels.Beta(), K = K, g_tol = 1e-7, iterations = 800)
        @test fit.converged
        @test isapprox(fit.loglik, llR; rtol = 1e-6)
    end

    @testset "Gamma (frozen seed-54 data, p=5 K=1 n=120): per-trait shape vs R" begin
        Y = _tp_load_frozen("gamma")
        K = 1
        rv = TOML.parsefile(joinpath(@__DIR__, "..", "docs", "dev-log", "core070", "true-parity-latest",
            "receipts", "family", "runparity-dispersion", "values-NATIVE-05-GAMMA.toml"))["values"]
        llR = Float64(only(rv)["r"])
        fit = fit_gamma_gllvm_grouped(Y; K = K, group = collect(1:size(Y, 1)))
        @test fit.converged
        @test isapprox(fit.loglik, llR; rtol = 1e-6)
    end
end
