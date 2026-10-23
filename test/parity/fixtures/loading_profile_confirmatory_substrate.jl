# Frozen Core070 pin matrices for D3 confirmatory `loading_profile` Stage 0.
# Source: test/parity/fixtures/core070_masks_known.R (sha256 in
# docs/dev-log/core070/loading-profile-confirmatory-substrate.json).
# Not exported; consumed by test/parity/loading_profile_confirmatory_substrate.jl
# and test/test_loading_profile_stage0.jl only.

const LOADING_PROFILE_CONFIRMATORY_P = 3
const LOADING_PROFILE_CONFIRMATORY_K = 2

# The L11 sign differs from R on purpose. R's MASK-B-PINS case pins L11 = +0.8
# (`core070_masks_known.R`, and `pins` in
# docs/dev-log/core070/masks-known-points-01/attempt1/out/maps.tsv). This Stage 0
# fixture was written separately and every test that uses it checks against
# -0.8. Tests that target the frozen R number use +0.8 themselves.
# Kept as is by decision: docs/dev-log/decisions/2026-09-24-parity-reference-julia-and-fixture-pins.md.
"""`MASK-B-PINS`: two user pins (L11 = -0.8, L32 = 0); three free packed θ coordinates."""
function loading_profile_fixture_mask_b_pins()
    M = fill(NaN, LOADING_PROFILE_CONFIRMATORY_P, LOADING_PROFILE_CONFIRMATORY_K)
    M[1, 1] = -0.8
    M[3, 2] = 0.0
    return M
end

"""`MASK-B-UPPER`: same as pins plus a bogus above-diagonal entry R must ignore."""
function loading_profile_fixture_mask_b_upper()
    M = loading_profile_fixture_mask_b_pins()
    M[1, 2] = 99.0
    return M
end

"""`MASK-B-ALLFIXED`: all five raw loading coordinates pinned (column-major R matrix)."""
function loading_profile_fixture_mask_b_allfixed()
    return [0.8 NaN; 0.1 0.7; 0.2 -0.15]
end

# Oracle: free (trait, axis) index pairs for R `loading_profile()` default `entries = NULL`
# on p = 3, K = 2 with engine strict-upper pins (see gllvmTMB/R/loading-profile.R).
const LOADING_PROFILE_ORACLE_FREE_MASK_B_PINS = [(2, 1), (2, 2), (3, 1)]
const LOADING_PROFILE_ORACLE_FREE_COUNT_MASK_B_PINS = 3
const LOADING_PROFILE_ORACLE_FREE_MASK_B_ALLFIXED = Tuple{Int,Int}[]

# Frozen likelihood oracles for MASK-B-PINS live under masks-known TSV artifacts;
# see docs/dev-log/core070/masks-known-contract.json case
# CORE070-MASKS-KNOWN-MASK-B-PINS-PAIRED-CONTROL (points P1/P2).
