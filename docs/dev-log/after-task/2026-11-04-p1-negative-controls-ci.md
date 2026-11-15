# P1 negative-controls CI correction: local checkpoint

## 1. Goal

Make Julia available to the negative-controls job so its two existing fixture controls can execute. This checkpoint covers local preparation and verification only.

## 2. Implemented

Added julia-actions/setup-julia@v2 with version 1.10 after setup-r and before the control commands. Existing tests and thresholds are unchanged. Base source: f34d5c175773729cb5a8f228184b56f877234f1a.

## 3a. Decisions and Rejected Alternatives

The maintainer explicitly approved the plan and local CI correction. Used standard setup-julia rather than skipping controls or requiring a juliaup installation path. No package installation was added because the fixture constructors require only Printf. Assumptions without asking: validation on Linux Julia 1.10.12 approximates the action's requested 1.10 channel; it does not exercise GitHub provisioning itself.

## 4. Files Touched

.github/workflows/true-parity-check.yml; docs/dev-log/check-log.d/2026-11-04-p1-negative-controls-ci.md; this report. Scratch logs and the owned Totoro checkout history were also updated; no package source changed.

## 5. Checks Run

Unchanged tools/test_core070_behaviour_receipts.py without Julia: 35/37 controls, two FileNotFoundError failures. Totoro with Julia 1.10.12 on PATH and full Git history: 37/37 controls, exit 0. Intermediate shallow-history attempt: 36/37, with an older source commit unavailable. Full history is already required by the workflow, so no checkout setting changed.

Ruby YAML parse and step-order assertions passed. git diff --check passed. PyYAML and actionlint were unavailable. No fits, Pkg.test, documentation build or hosted Actions run was performed for this environment-only patch.

## 6. Tests of the Tests

The missing-Julia baseline reproduced the two target errors. Both controls passed once Julia was on PATH; the remaining intermediate failure tracked missing Git history and cleared after restoring it. Existing negative controls were retained, not relaxed. Environment comparisons span Mac and Linux and are not a same-host randomized experiment.

## 7a. Issue Ledger

No issue or PR changes. Receipt refreshes, signed FAIL holds and the separate ordinal-warning CI failure remain open. Push and merge remain gated.

## 8. Consistency Audit

Checked the subprocess resolution at tools/test_core070_behaviour_receipts.py:531 and :608, pure constructor imports, the workflow's fetch-depth 0, and Julia setup order. All existing control commands remain present. A bounded independent CI review found no blocking issue: no control was skipped and only Base/Printf fixture code needs Julia. The reviewer noted that other workflows use setup-julia@v3; this patch retains the explicitly proposed @v2, with no functional blocker identified. The final 37/37 result was collected separately after restoring full history.

## 9. What Did Not Go Smoothly

The remote execution checkout was initially shallow, causing a history-dependent receipt check to fail. Restored full history and reran the same controls. Used Ruby's standard YAML parser after finding PyYAML unavailable.

## 10. Known Residuals

Hosted setup-julia execution, R 4.5.3 provisioning and the complete negative-controls job are not verified. The 37/37 control-file result does not prove those other checks or a parity row. No full-suite pass or P1 completion is claimed. The global report checker has a previously observed foreign-ledger problem; report structure will be checked separately without claiming global clearance. Shared AGENTS phase-state edits are deferred to an owned integration slice.

## 11. Team Learning

Memory receipt: the current P1 handover, approved execution contract, exact-main measurement and LOAD-FIRST routing informed this slice. Golden Set: not run; no numerical code or new rule changed. Preserve full Git history when controls resolve receipt source commits. The applicable runtime is Julia on PATH; a juliaup-specific path is only the fallback.

## 12. Cross-Product Coverage

Covers: Julia availability for two pure fixture controls, all 37 controls under the reproduced runtime, YAML structure and step order.

This does NOT cover hosted Actions success, all receipt checks, package fits, numerical parity, F-capability coverage, signed FAIL promotion, N1c, release readiness or P1 closure.

## Review verdict

PASS WITH NOTES for the local CI correction: YAML/order and the affected control file pass; independent CI review found no blocker. Hosted execution, full job checks and global report clearance are not certified. This verdict does not close P1.
