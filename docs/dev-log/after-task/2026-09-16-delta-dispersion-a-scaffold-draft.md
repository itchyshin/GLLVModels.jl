<!-- slop-ok: after-task field labels (Date / Lane / Base / PR) match repo protocol -->
# After-task: DRAFT Option A scaffold (Delta species CI + SO cells)

**Date:** 2026-09-16  
**Lane:** PLATFORM: claude | ON BRANCH: feat/delta-dispersion-a-scaffold-20260916 | LANE: delta-dispersion-a-scaffold  
**Worktree:** `~/local-scratch/gllvm-delta-disp-a-scaffold-20260916`  
**Base:** `origin/main` @ **`83f2e5224`** (rebased 2026-09-17; post-#420 paste packet tip)  
**PR:** DRAFT [#399](https://github.com/itchyshin/GLLVM.jl/pull/399); waits for paste `accept delta dispersion A` (do not mark ready; do not merge)

## Rose fence

- Not an ACCEPTED disposition for Delta dispersion A/B/C.
- Not a D1 pass at §4 tolerances (not remeasured as a promotion).
- Not a public fitter default flip to `:species` (`fit_gllvm` coerce still NB/Beta-only; explicit `disp_group=` wired for Delta).
- Not Stage 1 / S4 / Totoro / #357 / `Project.toml` bump / programme §7.
- Does wire `_family_ci` packs for `:species` (shared-η and separate predictors) and points SO twin cells at `:species`.

## What landed

1. `src/confint_family.jl`: DeltaLogNormalFit / DeltaGammaFit `_family_ci` for `:shared` and `:species`.
2. `tools/core070_second_order/cells.jl`: `cell_delta_{lognormal,gamma}` pass `disp_group=:species`; `parameterisation_gap=false`.
3. `src/postfit.jl`: vector σ/α in `getLV` / `predict` / `residuals` / `show`; `_nparams` uses `ndisp`.
4. `src/families/fit_gllvm.jl`: Delta `disp_group` routing + paste TODO on default coerce (not flipped).
5. `test/test_second_order_delta_followup.jl`: species postfit smoke + `fit_gllvm` explicit `:species`.
6. Paste-gate ACCEPTED draft + D1 remeasure runbook (`docs/dev-log/plans/2026-09-16-delta-dispersion-a-d1-remeasure-runbook.md`).
7. Decision doc still **PENDING_ACCEPTANCE** (HTML comment block only).

## Still needs paste `accept delta dispersion A`

1. Append ACCEPTED (A) block to the decision doc.
2. Flip public fitter / `fit_gllvm` default coerce to `:species` (one block in `fit_gllvm.jl`).
3. Remeasure D1 per runbook (no rtol widen).
4. Mark PR ready-for-review and merge on green.

## Checks run

```text
julia --project=. test/test_second_order_delta_followup.jl
# 2026-09-16: 34 pass / 1 broken (R live Δ skip without GLLVM_PARITY_TESTS)
# 2026-09-17 post-rebase @ 83f2e5224: 34 pass / 1 broken (same)
```

## Goal

True-parity programme remains IN PROGRESS / incomplete.
