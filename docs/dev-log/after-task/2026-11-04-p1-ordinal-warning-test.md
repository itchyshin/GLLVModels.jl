# Ordinal warning test: local proposal

## 1. Goal

Correct a test's inference from a Laplace warm-start warning to a final AGHQ loading, without changing the model or numerical warning threshold.

## 2. Implemented

Removed only the final AGHQ maximum-loading >8 assertion and explained the stage distinction. Retained the positive AGHQ log assertion, both preceding Laplace/covariate loading checks and the helper diagnostic tests. Base f34d5c175773729cb5a8f228184b56f877234f1a.

## 3a. Decisions and Rejected Alternatives

Did not lower the warning threshold, alter the data or seed, change the engine, or force the final optimizer to remain at its warm start. Source inspection shows separate warm and final stages. Assumption without asking: this local test-only proposal is within the approved plan's diagnosis and bounded correction work; publication and merge remain separately gated.

## 4. Files Touched

Test file test/test_convergence_sweep.jl; dated check-log.d note; this report. Diagnostic scripts, runtime and test copies remain in the owned Totoro evidence directory. No engine, API or receipt file changed.

## 5. Checks Run

Hosted CI at main f34d5c175, Julia 1.13.3, reported a failed final-loading assertion with value 5.208452513947484 against >8. A bounded local-equivalent Julia 1.13.3 run of the original test file passed 21/21; the proposed file passed 20/20. Both exited 0. The original failure was not reproduced under the retained environment, whose dependency/BLAS setup is not certified identical to CI. Original run including first compilation took about 116 seconds; proposed run about 31 seconds. Each had a 180-second cap.

Static diff and fixture-preservation checks passed. The positive AGHQ warning assertion and Laplace checks are identical. No full suite or hosted rerun was performed. A warning-suppression mutation failed the retained AGHQ log assertion: 19 passed, 1 failed, 20 total, exit 1, about 31 seconds under a 90-second cap.

## 6. Tests of the Tests

The original CI failure is primary evidence of the invalid final-state requirement, but the local original run passed; do not call this a reproduced red-to-green case. The mutation wraps only the AGHQ call in a NullLogger; the intended warning assertion then fails while the other 19 assertions pass. It was run in an external test copy, not the committed source. This shows the patch preserves detection of a missing warning.

## 7a. Issue Ledger

No GitHub issue or PR change. Current-main CI failure remains unresolved as hosted evidence; this is a local candidate. No parity row was changed.

## 8. Consistency Audit

Read AGHQ's warm-start, fallback, selected-result and final-warning paths at src/families/aghq_ordinal.jl:182-226. File preflight found an old temporary merge-check ref that removes later ordinal tests; inspected it and left it untouched. Independent source and actual-diff review found no blocker. No unrelated source changes were taken from another lane.

## 9. What Did Not Go Smoothly

Julia 1.13.2 retained the warm start after adaptation failure and did not reproduce CI. An owned Julia 1.13.3 installation still did not reproduce the hosted failure. The source contract issue is clear, but the lower-level reason for the different final fit remains unknown.

## 10. Known Residuals

Hosted CI, full/core suites and quality tools are unverified. A passing focused file does not close ordinal parity or P1. The removed assertion changes test count by one, explicitly documented. Global report and shared phase-state updates remain integration work; no such clearance is claimed here.

## 11. Team Learning

Memory receipt: current handover, fresh exact-main gate measurement and systematic-debugging workflow informed this slice. Golden Set: not run; no numerical engine or policy changed. Warning evidence must be attributed to the stage that emitted it, rather than assuming later optimization preserves the same condition.

## 12. Cross-Product Coverage

Covers: one source-backed warning-test contract correction and focused runtime checks on Julia 1.13.3.

This does NOT cover a full-suite pass, exact hosted-environment reproduction, an engine correction, R parity, calibrated inference, row promotion or P1 completion.

## Review verdict

PASS WITH NOTES for the local test proposal: independent actual-diff review found no blocker, focused patched tests pass and the missing-warning mutation is rejected. Exact hosted failure reproduction, full-suite checks, shared status integration and publication remain outstanding. No parity promotion or whole-task completion is claimed.
