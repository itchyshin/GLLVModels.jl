#!/usr/bin/env julia
# runparity.jl — developer-opt-in / CI-required runner for GLLVModels.jl ↔ R gllvmTMB parity.
#
# Usage:
#   GLLVM_PARITY_TESTS=1 julia --project=test/parity test/parity/runparity.jl
#
# This file is the ONLY entry-point. It is not included by runtests.jl. CI
# invokes it in required mode; a local developer run without its opt-in variable
# is intentionally harmless and cannot satisfy the required evidence gate.

println("=" ^ 72)
println("GLLVModels.jl ↔ R gllvmTMB parity suite (Phase 1.0 scaffold)")
println("=" ^ 72)

# ── Gate: optional developer mode vs fail-closed required evidence ──────────
requested = get(ENV, "GLLVM_PARITY_TESTS", "0") == "1"
required = get(ENV, "CORE070_PARITY_REQUIRED", "0") == "1"
if required && !requested
    error("CORE070_PARITY_REQUIRED=1 requires GLLVM_PARITY_TESTS=1; an optional skip cannot satisfy required evidence")
end
if !requested
    println()
    println("SKIPPED — optional developer parity was not requested.")
    println()
    println("  To run: set GLLVM_PARITY_TESTS=1 and ensure R + gllvmTMB")
    println("  are installed, then:")
    println()
    println("    GLLVM_PARITY_TESTS=1 \\")
    println("      julia --project=test/parity test/parity/runparity.jl")
    println()
    exit(0)
end

# ── Wire the local GLLVModels package ─────────────────────────────────────────────
# In CI or isolated runner, develop GLLVModels if not already pointing to root
using Pkg
using GLLVModels
using Test

# ── Try to load RCall — optional skip only; required mode fails ──────────────
try
    @info "Loading RCall..."
    @info "R_HOME in Julia: " * get(ENV, "R_HOME", "<unset>")
    @eval using RCall
    @info "RCall loaded successfully."
catch err
    required && error("CORE070_REQUIRED_DEPENDENCY_ERROR: RCall/R is unavailable\n" *
                      sprint(showerror, err, catch_backtrace()))
    println()
    println("SKIPPED — RCall or R not available.")
    println("  Error: ", sprint(showerror, err, catch_backtrace()))
    println()
    println("  Install R and the gllvmTMB package, then rebuild RCall:")
    println("    julia --project=test/parity -e 'using Pkg; Pkg.build(\"RCall\")'")
    println()
    exit(0)
end

# ── Run the parity tests ──────────────────────────────────────────────────────
# Order locked by catch-up arc: Gaussian → Binomial → Poisson → NB2 → Beta →
# Ordinal-probit, then the 2026-08-24 no-X catch-up cells (lognormal fid 3,
# truncated_poisson fid 10), then shared site-X cohort (G/Bin/Pois + NB2/Beta/Gamma +
# Ordinal-probit). Light logLik oracles: NB2/Beta/Gamma via grouped-cov +
# per-trait dispersion; Ordinal+X via fit_ordinal_gllvm_pertrait_cov +
# ProbitLink. X cells use shared γ only.
# lognormal is the one no-X family pairing with a SHARED-scalar σ fitter (twin ties
# sigma_eps), and its logLik carries the −Σ log y Jacobian on both sides.
# Do not narrate as “full family parity” (named shared-φ / logit ordinal differ).
include(joinpath(@__DIR__, "parity_helpers.jl"))

function run_required_family_cell!(id::AbstractString, fixture::AbstractString)
    core070_execute_case!(id, fixture, () -> include(joinpath(@__DIR__, basename(fixture))))
end

function run_required_family_group!(ids::AbstractVector{<:AbstractString}, fixture::AbstractString)
    core070_execute_group!(ids, fixture, () -> include(joinpath(@__DIR__, basename(fixture))))
end

function run_required_family_smoke!()
    # This runner is deliberately only the 17 required family-smoke cells.  The
    # broader developer cohort below is not programme evidence and its optional
    # skips cannot appear in a CORE-070 receipt.
    @testset "CORE-070 required family smoke" begin
        run_required_family_cell!("NATIVE-01-GAUSSIAN", "test/parity/test_gaussian_parity.jl")
        run_required_family_cell!("NATIVE-02-BINOMIAL", "test/parity/test_binomial_parity.jl")
        run_required_family_cell!("NATIVE-03-POISSON", "test/parity/test_poisson_required.jl")
        run_required_family_cell!("NATIVE-06-NB2", "test/parity/test_negbin_parity.jl")
        run_required_family_cell!("NATIVE-08-BETA", "test/parity/test_beta_required.jl")
        run_required_family_cell!("NATIVE-15-ORDINAL-PROBIT", "test/parity/test_ordinal_probit_parity.jl")
        run_required_family_cell!("NATIVE-04-LOGNORMAL", "test/parity/test_lognormal_parity.jl")
        run_required_family_cell!("NATIVE-11-TRUNCATED-POISSON", "test/parity/test_truncated_poisson_parity.jl")
        run_required_family_group!(["NATIVE-05-GAMMA", "NATIVE-16-NB1", "NATIVE-09-BETABINOMIAL"],
                                   "test/parity/test_nox_dispersion_parity.jl")
        run_required_family_cell!("NATIVE-17-MULTINOMIAL-FIXED", "test/parity/test_multinomial_parity.jl")
        run_required_family_cell!("NATIVE-12-TRUNCATED-NB2", "test/parity/test_truncated_nbinom2_parity.jl")
        run_required_family_cell!("NATIVE-13-DELTA-LOGNORMAL", "test/parity/test_delta_lognormal_required.jl")
        run_required_family_cell!("NATIVE-14-DELTA-GAMMA", "test/parity/test_delta_gamma_required.jl")
        run_required_family_cell!("NATIVE-10-STUDENT", "test/parity/test_studentt_parity.jl")
        run_required_family_cell!("NATIVE-07-TWEEDIE", "test/parity/test_tweedie_parity.jl")
    end
end

function run_optional_developer_parity!()
    for file in (
        "test_gaussian_parity.jl", "test_binomial_parity.jl", "test_poisson_parity.jl",
        "test_negbin_parity.jl", "test_nb2_finite_dispersion_parity.jl",
        "test_beta_parity.jl", "test_ordinal_probit_parity.jl",
        "test_lognormal_parity.jl", "test_truncated_poisson_parity.jl",
        "test_nox_dispersion_parity.jl", "test_multinomial_parity.jl",
        "test_truncated_nbinom2_parity.jl", "test_x_covariate_parity.jl",
        "test_species_x_parity.jl", "test_delta_lognormal_parity.jl",
        "test_delta_gamma_parity.jl", "test_studentt_parity.jl", "test_tweedie_parity.jl",
    )
        include(joinpath(@__DIR__, file))
    end
end

if required
    core070_start_run!()
    try
        run_required_family_smoke!()
        run_required_family_group!(Core070CaseRegistry.GAUSSIAN_IDS,
                                   "test/parity/test_gaussian_original_required.jl")
        @testset "CORE-070 required interfaces" begin
            for id in setdiff(Core070CaseRegistry.INTERFACE_IDS, Core070CaseRegistry.GAUSSIAN_IDS)
                core070_execute_case!(id, _CORE070_FIXTURES[id],
                    () -> include(joinpath(@__DIR__, basename(_CORE070_FIXTURES[id]))))
            end
        end
        run_required_family_group!(Core070CaseRegistry.COVARIANCE_FIXED_IDS,
                                   "test/parity/test_covariance_fixed_required.jl")
        run_required_family_group!(Core070CaseRegistry.COVARIANCE_MODE_IDS,
                                   "test/parity/test_covariance_modes_required.jl")
        run_required_family_group!(Core070CaseRegistry.COVARIANCE_FIXED_FORMULA_IDS,
                                   "test/parity/test_covariance_fixed_formula_required.jl")
        run_required_family_group!(Core070CaseRegistry.COVARIANCE_MODE_FORMULA_IDS,
                                   "test/parity/test_covariance_modes_formula_required.jl")
        core070_finish_run!()
    catch err
        core070_abort_run!(err)
        rethrow()
    end
else
    run_optional_developer_parity!()
end
