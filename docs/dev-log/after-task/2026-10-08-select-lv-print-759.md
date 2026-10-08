# After-task: #855 CI, select_lv print pin vs #759

## 1. Goal

Diagnose the four failed checks on PR #855 (`cursor/oct4-759`, head `c7642e930`) and take only the action the logs support. Do not merge. Do not open the next draft.

## 2. Implemented

Real assertion failure, not a checkout flake. All four failed jobs (CI 37759725503 shard 3/4 on Julia 1.10 and Julia 1; P1 twin 37759725431 on both versions) failed the same three tests in `test/test_select_lv_print.jl`:

```
conv: per-rank convergence flag; an unconverged rank stays selectable
  Expression: sel.best_k == 2
   Evaluated: 3 == 2
  Expression: sel.best_k == sel.K[argmin(sel.aic)]
   Evaluated: 3 == 2
  Expression: length(row2) == 1 && occursin("FALSE", only(row2))
```

`c7642e930` already chose `best_k` among converged finite-criterion ranks (#759). The print test still pinned the old twin fence (unconverged K=2 stays selectable, and the `*` marker on that row). The third assertion failed because K=2 is no longer marked `*`.

Updated the print test to the #759 rule: K=2 stays in the table as FALSE; `best_k` is 3; the `*` is on K=3. Updated the matching `LVSelection` / `select_lv` docstring sentences so they no longer say an unconverged rank can win. Left the pdHess twin fence alone (a confirmed non-PD rank is still selectable). Did not change likelihoods, tolerances, or Documenter.

Did not edit `docs/dev-log/check-log.md` (Codex lease). Did not merge #855. Did not open 765/766.

## 3. Files Touched

- `test/test_select_lv_print.jl`
- `src/model_selection.jl`
- `docs/dev-log/after-task/2026-10-08-select-lv-print-759.md` (this report)

## 4. Checks Run

- `julia --project=. test/test_select_lv_print.jl`: 48 passed / 0 failed
- `julia --project=. test/test_select_lv_759.jl`: 7 passed / 0 failed
- Full suite and Documenter not rerun (Documenter already passed on `c7642e930`)

## 5. Rose

OK for this slice: the print pin now matches the #759 selection rule already on the branch. Remaining twin difference (non-PD Hessian still selectable) is unchanged. #855 not merged.
