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
# The three seeded cells replay the same Julia-side simulators as the opt-in parity
# files, so they hold only while Julia's seeded RNG stream is unchanged; a replay
# drift shows up here as a mismatch, not as a silent pass.
# A row that matches R is a plain @test. A row that does not is @test_broken with the
# issue it belongs to, so a later fix turns it into an "unexpected pass" to be promoted.
using Test
using GLLVModels
using TOML
using Random
using SHA
import Distributions

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

_tp_loadings() = [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]   # parity_loadings_p5k2()

# Johnk's algorithm, copied from test/parity/test_beta_parity.jl so the DGP is identical.
function _tp_rand_beta(a::Float64, b::Float64)
    a = max(a, 1e-12); b = max(b, 1e-12)
    while true
        u = rand(); v = rand()
        x = u^(1 / a); y = v^(1 / b); s = x + y
        s <= 1.0 && s > 0.0 && return x / s
    end
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

    @testset "Tweedie (seed 82, p=5 K=1 n=150): per-species and fixed power vs R" begin
        Random.seed!(82)
        p, K, n = 5, 1, 150
        β = [0.5, -0.2, 0.3, -0.4, 0.1]
        Λ = 0.5 .* _tp_loadings()[:, 1:K]
        φ = [0.8, 1.0, 1.2, 0.9, 1.1]
        Z = randn(K, n)
        μ = exp.(β .+ Λ * Z)
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            Y[t, s] = GLLVModels._tweedie_sample(μ[t, s], φ[t], 1.5, Random.default_rng())
        end
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

    @testset "Beta (seed 45, p=5 K=1 n=60): per-trait phi vs R" begin
        Random.seed!(45)
        p, K, n = 5, 1, 60
        β = [0.30, -0.20, 0.25, -0.15, 0.05]
        Λ = 0.15 .* _tp_loadings()[:, 1:K]
        Z = randn(K, n)
        η = β .+ Λ * Z
        Y = [begin
                 μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
                 _tp_rand_beta(μ * 12.0, (1 - μ) * 12.0)
             end for t in 1:p, s in 1:n]
        rv = TOML.parsefile(joinpath(@__DIR__, "..", "docs", "dev-log", "core070", "true-parity-latest",
            "receipts", "family", "runparity-beta", "values-NATIVE-08-BETA.toml"))["values"]
        llR = Float64(only(v for v in rv if v["label"] == "logLik")["r"])
        fit = fit_gllvm(Y; family = GLLVModels.Beta(), K = K, g_tol = 1e-7, iterations = 800)
        @test fit.converged
        @test isapprox(fit.loglik, llR; rtol = 1e-6)
    end

    @testset "Gamma (seed 54, p=5 K=1 n=120): per-trait shape vs R" begin
        Random.seed!(54)
        p, K, n = 5, 1, 120
        β = log.([2.0, 2.5, 1.8, 2.2, 2.1])
        Λ = 0.2 .* _tp_loadings()[:, 1:K]
        Z = randn(K, n)
        η = β .+ Λ * Z
        Y = [begin
                 μ = exp(clamp(η[t, s], -4.0, 4.0))
                 rand(Distributions.Gamma(2.5, μ / 2.5)) + 1e-6
             end for t in 1:p, s in 1:n]
        rv = TOML.parsefile(joinpath(@__DIR__, "..", "docs", "dev-log", "core070", "true-parity-latest",
            "receipts", "family", "runparity-dispersion", "values-NATIVE-05-GAMMA.toml"))["values"]
        llR = Float64(only(rv)["r"])
        fit = fit_gamma_gllvm_grouped(Y; K = K, group = collect(1:p))
        @test fit.converged
        @test isapprox(fit.loglik, llR; rtol = 1e-6)
    end
end
