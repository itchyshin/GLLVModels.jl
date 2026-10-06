# check_auto_residual public behaviour phase report

## 1. Goal

Prepare the check_auto_residual public behaviour local receipt phase for review. Main and full P1 remain open.

## 2. Implemented

Observed coherent and flagged public residual-check outcomes in R and Julia, with a crossed fixture, retained raw six-case capture and a source-selective writer. Only the residual sidecar is derived in this stage; the five iSDM bindings belong to the following stage.

## 3a. Decisions and Rejected Alternatives

Preserved P1R9539352f66f2db2cc26b1c393e67212a359b60c9, existing signed scope, case IDs, source engine, version0.3.0 and all tolerances. No signatures, Rengine modifications or public full-parity claim.

## 4. Files Touched

- `docs/dev-log/after-task/2026-10-06-p1-residual-public-receipt.md`
- `docs/dev-log/check-log.md`
- `docs/dev-log/core070/true-parity-latest/behaviour-equivalence.json`
- `docs/dev-log/core070/true-parity-latest/case-map-assembled.json`
- `docs/dev-log/core070/true-parity-latest/case-map-postfit.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/POSTFIT-SURFACE-check_auto_residual.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/behaviour-equivalence.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-06-v4/julia-public.tsv`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-06-v4/r-public.tsv`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-06-v4/run.json`
- `docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-WAVE7-CHECK-AUTO-RESIDUAL.json`
- `docs/dev-log/core070/true-parity-latest/scoreboard.md`
- `test/runtests.jl`
- `test/test_first_seven_behaviour_p1.jl`
- `tools/core070_behaviour_receipts.py`
- `tools/core070_postfit_p1_receipts.py`
- `tools/first_seven_behaviour_J.jl`
- `tools/first_seven_behaviour_R.R`
- `tools/first_seven_behaviour_derive.py`
- `tools/test_core070_behaviour_receipts.py`

## 5. Checks Run

Five owning checks, behaviour Python33/33 and assembler --check passed at this stage. The coverage assertion keeps prior57entries and this exact residual row (58 total). Registered enabled Julia wrappers passed2/2 in the integration lane; stage tools were checked directly. X2 299/317,C2 285/297 follows from this canonically assembled stage. Documenter of the unchanged site/API passed at aee0daf9b. Full/core sourceed11661f0 is pending and does not certify current candidate tests.

## 6. Tests of the Tests

Controls reject wrong pin/build/source, wrong refusal/call, missing or changed raw data and invalid comparison/context. Existing controls are preserved; no tolerance was widened.

## 7a. Issue Ledger

This phase binds only `postfit/POSTFIT-SURFACE-check_auto_residual` locally. The complete first checkpoint requires seven rows, independent review and consent for every merge. All13 later scoreboard rows and37 C6 names remain open.

## 8. Consistency Audit

Shared outputs were regenerated canonically. Numerical engine and public API/docs site remain unchanged. The phase adds no convergence or coverage evidence. Checkpoint tooling supports validating the later combined stage.

## 9. What Did Not Go Smoothly

The original early completion panel found missing test registration and nested raw-hash handling in the integrated candidate. Both were repaired. The original verdicts remain historical failed reviews; final signoff is pending. Sandbox limitations and failed measurements were preserved rather than waived.

## 10. Known Residuals

No main landing or full P1 completion is claimed. Draft PR and full acceptance remain consent-gated. Owned suite has a19:23:24UTC cap; current candidate exact-suite proof is pending.

## 11. Team Learning

Memory receipt: GLLVM.jl LOAD-FIRST manifest, signed repository scope and the approved compute/ownership contract shaped this phase. Golden Set: relevant model-tiering, completion, branch and refusal mistakes were consulted; detector self-tests discriminate good/bad fixtures. This is not a universal agent-compliance certificate.

## 12. Cross-Product Coverage

Covers: check_auto_residual public behaviour at the admitted P1 scope and its receipt regeneration. Does NOT cover: full-family numerical public-bridge success, realistic-size/data convergence, calibrated simulation/coverage, C6 decisions, animal random slopes, P2, releases, or landing on main.
