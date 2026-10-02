# gllvm-parity-tag: P1
#
# Numeric twins of the aghq policy rows against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9; R/aghq-control.R, R/gllvmTMB.R). No R at test
# time: R's fits are read from test/fixtures/aghq_p1/aghq_p1.toml (generated once by
# test/fixtures/aghq_p1/gen_aghq_p1.R against a lane-local gllvmTMB install at the pin; it
# records R version, commit, seeds and the data sha256s) and the same data are fitted here.
#
# Rows (10): AGHQ-AUTO-K-{POISSON,BINOMIAL,GAUSSIAN}, AGHQ-DEFAULT-OFF, AGHQ-POLICY-{OFF,
# EXPLICIT,EXPLICIT-BYPASS-CUTOFF,AUTO-ENFORCE-CUTOFF,TRAITS19,TRAITS20}. The NB2, delta,
# ordinal and Tweedie AUTO-K rows have no `aghq=` surface in Julia and are not twinned.
#
# Model on both sides: eta_tj = beta_j + lambda_j z_t, z_t ~ N(0, 1) (one latent axis, no
# loading ridge; R: aghq_ridge = Inf), Gaussian with one residual SD, binomial with 10
# trials per cell. The data carry a real latent factor; the policy bind's toy data
# (independent eta, no factor) make both engines' binomial adaptation stall
# (itchyshin/GLLVModels.jl#586), so they are not used. Seeds at which either engine failed
# to converge were rejected when the fixture was generated and are listed in it.
#
# What is asserted per row, at each side's own optimum:
#   * both engines converged (R: aghq$converged for AGHQ fits, optimiser code 0 for Laplace;
#     recorded in the fixture; Julia: fit.converged and, for AGHQ, reason :converged);
#   * the integration actually used and its node count (used flag and k; Laplace = 0 nodes);
#     integers, so |difference| <= 0.5 is exact equality;
#   * logLik atol 1e-6; intercepts and loadings (sign-aligned) atol 1e-3, Gaussian residual
#     SD atol 1e-3 (parameters are determined to roughly the square root of the likelihood
#     tolerance; the ordinal twin uses the same 1e-3).
# The R fit never ran at test time and Julia's fits are not re-tuned to match it.
using Test
using GLLVModels
using Distributions: Normal, NegativeBinomial
using TOML
using SHA

include(joinpath(@__DIR__, "fixtures", "aghq_p1", "aghq_p1_helpers.jl"))

const _AGHQ_P1_ROWS = [
    "AGHQ-AUTO-K-POISSON", "AGHQ-AUTO-K-NB2", "AGHQ-AUTO-K-BINOMIAL", "AGHQ-AUTO-K-GAUSSIAN", "AGHQ-DEFAULT-OFF",
    "AGHQ-POLICY-OFF", "AGHQ-POLICY-EXPLICIT", "AGHQ-POLICY-EXPLICIT-BYPASS-CUTOFF",
    "AGHQ-POLICY-AUTO-ENFORCE-CUTOFF", "AGHQ-POLICY-TRAITS19", "AGHQ-POLICY-TRAITS20"]

@testset "aghq policy twins: gllvmTMB P1 (9539352f6)" begin
    if !isfile(AGHQ_P1_TOML)
        @warn "aghq P1 fixture absent; twin gate NOT RUN" AGHQ_P1_TOML
        @test_skip false
    else
        fx = TOML.parsefile(AGHQ_P1_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        for (nm, ds) in fx["dataset"]
            @test bytes2hex(sha256(read(joinpath(AGHQ_P1_DIR, ds["file"])))) == ds["sha256"]
        end
        cache = Dict{String,Any}()
        for row in _AGHQ_P1_ROWS
            @testset "$row" begin
                id = aghq_p1_fit_id(row)
                r = fx["case"][id]["r"]
                @test r["converged"] === true                       # valid-twin guard: R converged
                j = get!(cache, id) do
                    if fx["case"][id]["aghq_request"] == "auto" && fx["dataset"][fx["case"][id]["dataset"]]["p"] >= 20
                        @test_logs (:warn, r"AGHQ request retained Laplace") aghq_p1_fit(fx, id)
                    else
                        aghq_p1_fit(fx, id)
                    end
                end
                @test j.converged                                  # Julia converged too
                j.used && @test j.reason === :converged
                if row in ("AGHQ-POLICY-AUTO-ENFORCE-CUTOFF", "AGHQ-POLICY-TRAITS20")
                    @test j.actual === :laplace && j.reason === :auto_trait_cutoff
                    @test occursin("cutoff", r["reason"])
                end
                dec_j = Float64[j.used, j.nodes]
                dec_r = Float64[r["used"], r["k"]]
                @test maximum(abs.(dec_j .- dec_r)) <= 0.5
                @test isapprox(j.loglik, r["loglik"]; atol = 1e-6, rtol = 0)
                @test isapprox(j.beta, Float64.(r["beta"]); atol = 1e-3, rtol = 0)
                @test isapprox(j.lambda, Float64.(r["lambda"]); atol = 1e-3, rtol = 0)
                # NB2 dispersions are compared on the log scale (the optimised scale; the AGHQ
                # surface is flat in phi, so phi itself differs by up to 9e-4 absolute at values of 2-6).
                haskey(r, "phi") && @test isapprox(log.(j.phi), log.(Float64.(r["phi"])); atol = 1e-3, rtol = 0)
                haskey(r, "sigma_eps") && @test isapprox(j.sigma_eps, r["sigma_eps"]; atol = 1e-3, rtol = 0)
            end
        end
    end
end
