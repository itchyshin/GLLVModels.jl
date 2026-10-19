# After-task: Optimizer selection examples in GLLVModels README

## 1. Goal

Make explicit outer-optimizer selection easy to discover from the package README and point readers to the existing detailed tutorial comparison.

## 2. Implemented

Added concise Poisson examples using `optimizer = :bfgs` and `optimizer = :lbfgs`, plus `optimizer_info()` calls. The text identifies the `responses × sites` count-matrix layout and states that omitting `optimizer` retains the existing default policy. Linked the example to the tutorial's matched-start comparison.

## 3. Decisions and rejected alternatives

Kept the README's existing detailed same-start code block and the tutorial's explanation intact. Added no implementation, API, CI, or test changes.

## 4. Files touched

- `README.md`
- `docs/dev-log/check-log.d/2026-10-19-optimizer-examples.md`
- `docs/dev-log/after-task/2026-10-19-optimizer-examples.md`

## 5. Checks run

`python3 tools/check_reader_surface.py` passed (`source_files=19`). Parsed the four Julia expressions in the README example with Julia 1.10.0. Confirmed the linked tutorial file exists and contains the existing optimizer-selection explanation and matched-start comparison. No fit or full Documenter build was run.

## 6. Tests of the tests

No optimizer fit was run: this change only adds a discovery example for an existing API, and the README snippet is not an executable Documenter block. The syntax and reader-facing link were checked directly.

## 7. Issue ledger

The snippet covers only the existing ordinary Poisson route. It makes no claim about AGHQ, grouped covariance, profile/bootstrap refits, or broader optimizer support.

## 8. Consistency audit

The example uses the `fit_poisson_gllvm` and `optimizer_info` names shown by the existing tutorial and documents the matrix orientation used by the package.

Rose verdict: independent audit pending parent review. The local review found no mismatch between the README scope statement, the example calls, and the linked tutorial.

## 9. What did not go smoothly

The first lease attempt overlapped active parity work and was refused. The original disposable worktree also could not update its shared Git metadata under the sandbox. Work resumed after the parent established a durable, scoped lease and a fresh isolated branch.

## 10. Known residuals

No new model fit or full Documenter build was requested or run. Existing same-start comparison and package validation evidence remain the source for numerical behavior.

## 11. Team learning

For a public API that already has a detailed tutorial, a short README example plus a direct tutorial link improves discovery without duplicating the full validation discussion.

## 12. Cross-product coverage

This is a GLLVModels documentation-only slice. It does not establish or alter optimizer behavior in other packages.
