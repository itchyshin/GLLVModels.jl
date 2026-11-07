# After Task: Signed-stop protection for receipt regeneration

## 1. Goal

Keep the three rows covered by the signed 2026-10-21 stop rule at `FAIL` during campaign receipt regeneration.

## 2. Implemented

The receipt writer checks that the ordinal, fungi, and iSDM headline rows are present, nonbinding, and still `FAIL`. Under `--apply`, candidate raw files are held until this guard passes. The receipt `--check` path applies the same rule and reports a violation as a failed check.

## 3a. Decisions and Rejected Alternatives

The guard preserves the existing scientific decisions and thresholds. It does not turn any measured receipt into a passing row, broaden tolerance, or infer a judgement from a polished fit. Only these three signed rows are protected by this change.

## 4. Files Touched

- `tools/true_parity/campaign/write_receipts.py`: defer raw writes until the signed-row guard passes; run the guard under `--check` too.
- `tools/test_true_parity_assemble.py`: add promotion, omission, write-order, current-status, and tolerance controls.
- `docs/dev-log/check-log.d/2026-10-24-signed-stop-receipt-guard.md`: record this checkpoint. The shared `docs/dev-log/check-log.md` remains untouched because another active lane holds a non-overlapping lease on it.
- `docs/dev-log/after-task/2026-10-24-signed-stop-receipt-guard.md`: this audit.

## 5. Checks Run

- `PATH=/Users/z3437171/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH PYTHONDONTWRITEBYTECODE=1 python3 tools/test_true_parity_assemble.py`: 535 passed, 0 failed, including Node-backed controls; output ended with `ALL PASS`.
- `PYTHONDONTWRITEBYTECODE=1 python3 tools/true_parity/campaign/write_receipts.py --check`: exit 0, `CAMPAIGN_RECEIPTS_OK`. Urbanisation is summary-only because its raw outputs are not in the repository.
- `PYTHONDONTWRITEBYTECODE=1 python3 tools/true_parity_assemble.py --check`: exit 0, `ASSEMBLE_OK 317 rows current`.
- `python3 -m py_compile tools/true_parity/campaign/write_receipts.py tools/test_true_parity_assemble.py`: exit 0.
- `git diff --check`: exit 0.
- `PATH=/Users/z3437171/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH Rscript ~/shinichi-brain/tools/check-after-task.R docs/dev-log/after-task/2026-10-24-signed-stop-receipt-guard.md`: report structure passed; repository-wide ledger validation failed to parse the existing `.unlazy/totoro-t4-p6-grid/GATES.md` because a gate lacks an explicit ID and colon.

## 6. Tests of the Tests

Ten new assertions exercise the guard, rejection of all three promotions, rejection of a missing held row, refusal to persist raw data before validation, and retained `FAIL` status and tolerances for each row. These are invalid-evidence and boundary controls. The regression test fails if the guard is absent or if it accepts an attempted promotion.

## 7a. Issue Ledger

No issue action was needed. This change implements the signed stop rule and does not alter the separate receipt mismatch diagnosis recorded in issue 991.

## 8. Consistency Audit

No user-facing documentation, example, API, or package status claim changed. The actual receipts retain `FAIL`; `LLt` remains `1e-4`, and iSDM prediction tolerances remain `1e-6`. The assembler still reports 317 rows current. The full P1 gates were not remeasured by this tooling slice.

The scans `rg "Gaussian only|not yet implemented|planned" README.md docs CLAUDE.md` and `rg "340.?x|machine precision|closed.?form" README.md docs/src docs/PERF-plus-design.md` returned existing capability, roadmap, and historical text. No matching text in the touched files was introduced or changed.

## 9. What Did Not Go Smoothly

The lane preflight found 27 active branches or PR lanes and four refs with receipt-writer changes. I inspected those diffs and ported only the signed-row guard. The shared lease helper did not return from its full registry scan. A read-only lease scan found only the separately leased joint-extractor paths, which do not overlap these files. The old candidate worktree and its dirty files remain untouched.

## 10. Known Residuals

- No automated lane lease was acquired because the shared registry scan did not return. Recheck and claim these scoped paths before further edits.
- Repository-wide acceptance-ledger validation is still blocked by the malformed `totoro-t4-p6-grid` ledger. It was not edited as part of this receipt-tool repair.
- Julia tests, JET, Allocs, Aqua, and R fits were not run; this change only touches Python receipt tooling and tests.
- Independent Rose review and the full P1 completion panel remain pending.
- The guard is local to this branch and is not on `origin/main`. It does not establish parity or complete the programme.

## 11. Team Learning

Run lane preflight with the explicit Codex runtime environment and inspect path ownership before porting a file that has several active historical variants.

## 12. Cross-Product Coverage

This tooling change covers only campaign receipt regeneration and its existing case-map/scoreboard derivation check. It does NOT cover engine fits, R bridge changes, row admission, other receipts, the wider parity gates, or public parity claims.

## Draft Assessment

Style: 2/10, medium confidence. Genre/coverage: this short internal audit report, read in full. Evidence/repair: the exact test counts and row tolerances make the result easy to check; the short sentences in sections 2 and 10 limit repetition, so no change was needed. Gates: science = pass for the narrow claim that no model code changed; facts = pass against the diff and command outputs; references = not applicable because no external reference is cited. Provenance: self-review of this report, 2026-10-24; the described controls were run and seen.

## Rose Verdict

Rose verdict: FAIL: the local signed-stop guard and its regression controls pass, but after-task closure is blocked by the unreadable existing acceptance ledger; lease acquisition, independent review, and full P1 closure also remain open.
