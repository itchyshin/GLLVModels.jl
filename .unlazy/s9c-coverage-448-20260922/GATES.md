# Gates: S9c coverage PR #448

OWNS: docs/dev-log/plans/2026-09-22-s9c-coverage-448-next-slice.md
OWNS: docs/dev-log/plans/2026-09-22-ultra-plan-s9c-coverage-merge.md
OWNS: docs/dev-log/plans/2026-09-22-s9c-compute-routing.md
OWNS: .unlazy/s9c-coverage-448-20260922/**
OWNS: test/test_grouped_analytic_grad.jl
OWNS: docs/dev-log/check-log.d/**
OWNS: docs/dev-log/after-task/**

Scope: Execute the G0-approved S9c coverage merge for PR #448 without carrying unrelated old-base changes, widening `RTOL_FD`, or starting the latte-gap arc.

## Contract

- Interfaces: `test/test_grouped_analytic_grad.jl` gate CLI (`--gate coverage`) and in-suite `@testset`.
- Ownership: code merge owns only `test/test_grouped_analytic_grad.jl`; verification owns this `.unlazy` scope; PR hygiene owns check-log and after-task docs.
- Dependencies: leaf 2 and leaf 3 wait for leaf 1.
- Toolchain: Julia project at repository root; `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`; GitHub CLI only for PR #448 inspection/update after G0.
- Manual review: Rose checks no unsupported claim, no latte-gap scope creep, and no `RTOL_FD` widening.

## Tree

- 1 S9c coverage #448 .............. GATES.md .......................... State: OPEN
  - 1.1 main-tip certification merge gates/leaf-1-main-tip-merge.md .... Needs: - .... State: READY
  - 1.2 focused verification ........ gates/leaf-2-focused-verify.md .... Needs: 1.1 .. State: WAITING
  - 1.3 PR hygiene and closeout ...... gates/leaf-3-pr-hygiene.md ....... Needs: 1.1 .. State: WAITING
  - 1.4 owed reverify (DEFER) ...... gates/leaf-4-owed-reverify-DEFER.md . Needs: 1.2 .. State: DEFERRED

## Status log

Append events with:

```sh
node ~/shinichi-brain/skills/unlazy/scripts/gate-check.mjs --scope s9c-coverage-448-20260922 --log "<event>"
```
