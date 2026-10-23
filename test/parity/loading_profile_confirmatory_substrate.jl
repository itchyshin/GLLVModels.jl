# Stage-0 substrate helpers for confirmatory Λ profiling (D3). Test-only —
# not part of the public GLLVModels.jl API until Stage 1 + maintainer G0.
# Delegates to internal `src/loading_profile_confirmatory_internal.jl` so Stage 1
# fitter wiring shares one implementation with tests.

using GLLVModels

const lambda_constraint_is_pinned = GLLVModels._lambda_constraint_is_pinned
const normalize_lambda_constraint_pin_matrix = GLLVModels._normalize_lambda_constraint_pin_matrix
const enumerate_free_lambda_entries = GLLVModels._enumerate_free_lambda_entries
const profile_refit_lambda_constraint = GLLVModels._profile_refit_lambda_constraint
