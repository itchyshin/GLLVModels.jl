<!-- slop-ok: after-task field labels (Date / Lane / Base / PR) match repo protocol -->
# After-task: DRAFT Option A scaffold (Delta species CI + SO cells)

**Date:** 2026-09-16  
**Lane:** PLATFORM: claude | ON BRANCH: feat/delta-dispersion-a-scaffold-20260916 | LANE: delta-dispersion-a-scaffold  
**Worktree:** `~/local-scratch/gllvm-delta-disp-a-scaffold-20260916`  
**Base:** `origin/main` @ `b1c048f2f`  
**PR:** DRAFT; waits for paste `accept delta dispersion A` (do not mark ready; do not merge)

## Rose fence

- Not an ACCEPTED disposition for Delta dispersion A/B/C.
- Not a D1 pass at §4 tolerances (not remeasured as a promotion).
- Not a public fitter default flip to `:species` (postfit still assumes scalar σ/α).
- Not Stage 1 / S4 / Totoro / #357 / `Project.toml` bump / programme §7.
- Does wire `_family_ci` packs for `:species` (shared-η and separate predictors) and points SO twin cells at `:species`.

## What landed

1. `src/confint_family.jl`: DeltaLogNormalFit / DeltaGammaFit `_family_ci` for `:shared` and `:species`.
2. `tools/core070_second_order/cells.jl`: `cell_delta_{lognormal,gamma}` pass `disp_group=:species`; `parameterisation_gap=false`.
3. `src/postfit.jl`: `_nparams` uses `ndisp` from `disp_group`.
4. `test/test_second_order_delta_followup.jl`: species packing + Wald smoke; shared path kept.
5. Decision fence note on `docs/dev-log/decisions/2026-09-15-delta-dispersion-alignment-pending.md` (still PENDING).

## Still needs paste `accept delta dispersion A`

1. Append ACCEPTED (A) block to the decision doc.
2. Flip public fitter / `fit_gllvm` default to `:species` and fix postfit vector-σ/α.
3. Remeasure D1 on SO cells (no rtol widen).
4. Mark PR ready-for-review and merge on green.

## Checks run

```text
julia --project=. test/test_second_order_delta_followup.jl
# 23 pass / 1 broken (R live Δ skip without GLLVM_PARITY_TESTS)
```

## Goal

True-parity programme remains IN PROGRESS / incomplete.
