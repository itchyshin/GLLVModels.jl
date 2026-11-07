## 1. Goal

Make receipt writing, assembly, and the parity checker enforce the signed P1 convergence rule consistently without changing comparison tolerances, row statuses, or admissions.

## 2. Implemented

- The receipt writer now records R and Julia Hessian flags and maximum absolute gradients for each required row, using JSON `null` when gradient evidence is missing. It adds an explicit failed pass-rule leg when any required evidence is absent or fails the strict rule.
- The writer, assembler, and JS checker now scope the required convergence block to the four campaign comparisons named in the signed note: ordinal, iSDM headline, spider, and fungi.
- Both validators require each Hessian flag to be exactly `true`, each gradient to be finite and nonnegative, and each gradient to be strictly less than `1e-5`. A gradient equal to `1e-5` fails.
- The real-tree `--check` permits only the expected N9 block, false pass-rule leg, and matching reason text on those four pinned FAIL snapshots. It verifies the new block from unchanged engine fields, still compares their other derived receipt fields, comparison values, and tolerances, and refuses any promotion or edit. No signed receipt, case-map row, raw evidence, scoreboard value, or tolerance was changed.
- Other rows, including urbanisation and beetle, are not added to that four-row rule.
- Mathematical contract: N/A for the model likelihood. This is an evidence gate only; it leaves model parameterisation, fitted quantities, and every comparison tolerance unchanged.

## 3a. Decisions and Rejected Alternatives

- Kept the signed `1e-5` threshold and the existing comparison tolerances. Rejected accepting the exact boundary because the signed rule says strictly below.
- Required Hessian evidence to be explicit and positive definite for both engines. Missing, false, or non-boolean evidence refuses to bind.
- Preserved signed FAIL rows. This checker change creates no fit evidence and admits no row.
- Added a compatibility exception only for the four existing signed FAIL snapshots: it accepts the exact N9 block derived from the unchanged engine fields, the single false N9 leg, and its corresponding reason text. It rejects other checked field changes and does not rewrite those snapshots.
- The shared `docs/dev-log/check-log.md` is conductor-owned and was left untouched. This task uses the established per-task `check-log.d` record.
- Assumptions without asking: none.

## 4. Files Touched

- `tools/true_parity/campaign/write_receipts.py`
- `tools/true_parity_assemble.py`
- `tools/true_parity_check.mjs`
- `tools/test_true_parity_assemble.py`
- `tools/test_true_parity_check.mjs`
- `docs/dev-log/check-log.d/2026-10-27-n9-signed-convergence-gate.md`
- `docs/dev-log/after-task/2026-10-27-n9-signed-convergence-gate.md`

## 5. Checks Run

- `python3 tools/test_true_parity_assemble.py`: **549 passed, 0 failed** (`ALL PASS`), including null-gradient serialization, exact signed-snapshot compatibility, and tolerance/status/evidence/type mutation refusals.
- `node tools/test_true_parity_check.mjs`: **261 negative controls passed** (`All true-parity negative controls passed.`), including the N9 boundary, Hessian, and row-scope controls.
- `python3 tools/true_parity_assemble.py --check`: `ASSEMBLE_OK 317 rows current`; signed FAIL snapshots rederive under the narrow compatibility rule.
- `node --check tools/true_parity_check.mjs`: passed.
- `python3 -m py_compile tools/true_parity/campaign/write_receipts.py tools/true_parity_assemble.py tools/test_true_parity_assemble.py`: passed.
- `git diff --check`: passed.
- `python3 ~/shinichi-brain/tools/slop_check.py /Users/z3437171/.codex/worktrees/true-parity-p1-current-main-20261024/GLLVM.jl/docs/dev-log/after-task/2026-10-27-n9-signed-convergence-gate.md /Users/z3437171/.codex/worktrees/true-parity-p1-current-main-20261024/GLLVM.jl/docs/dev-log/check-log.d/2026-10-27-n9-signed-convergence-gate.md`: 0 findings; report 1.9 hits per 1,000 words (two faux-reveal matches), check-log 0, and no em dashes.
- `Rscript -e 'source("~/shinichi-brain/tools/check-after-task.R"); check_after_task("docs/dev-log/after-task/2026-10-27-n9-signed-convergence-gate.md")'`: after-task structure check passed.
- `Rscript ~/shinichi-brain/tools/check-after-task.R docs/dev-log/after-task/2026-10-27-n9-signed-convergence-gate.md`: structure passed, then the full closeout check exited 1 on unrelated `.unlazy/totoro-t4-p6-grid/GATES.md`, which lacks an explicit gate ID. That ledger was not edited, so repository-wide acceptance clearance is not claimed.
- No Julia fits, parity capture, full Julia suite, or documentation build was part of this tools-only repair.
- JET: not run; no Julia implementation or hot path changed. Allocs: not run; no hot path changed. Aqua: not run; no dependency, export, or project metadata changed.
- Benchmark: N/A; no hot path changed. R parity: N/A; this repair changes validation tooling only.

## 6. Tests of the Tests

Negative controls reject a gradient at exactly `1e-5`, a gradient above the threshold, a missing gradient, a missing R Hessian, a non-positive-definite Julia Hessian, a loose declared bound, an unknown compared point, and a malformed convergence block. They verify that the writer serializes unavailable gradients as `null`, refuses to bind that block, and adds a failed pass-rule leg. The four named rows reject a missing signed block, held and unlisted rows do not expand the rule, and unmarked legacy receipts preserve their existing behavior. Snapshot controls accept only the block derived from unchanged engine fields, then reject tolerance, altered-evidence, FAIL-to-PASS, and boolean-to-number mutations in receipts, pass-rule legs, and case-map flags. Both validator implementations and the Python receipt writer are exercised.

## 7a. Issue Ledger

- Fixed: assembler and checker previously allowed the exact `1e-5` boundary and did not require positive-definite Hessian evidence.
- Deferred: existing signed FAIL rows remain FAIL until separately authorized measurements satisfy the scientific rule. No capture or fit was run here.
- No issue admission or scoreboard change was made.

## 8. Consistency Audit

Compared the Python assembler, JS checker, receipt writer, the signed 2026-10-21 note, and both test suites. Their field names, strict threshold, Hessian semantics, four required row IDs, and held-row boundaries agree. `ASSEMBLE_OK 317 rows current`; the four committed case-map rows, receipt files, scoreboard, raw evidence, and tolerances remain untouched.

Stale-wording inventory commands and results: `rg "Gaussian only|not yet implemented|planned next|TODO|FIXME" README.md docs CLAUDE.md` returned 135 existing matches; `rg "340.?x|machine precision|closed.?form" README.md docs/src docs/PERF-plus-design.md` returned 62. These broad matches were not treated as evidence of defects because this tooling-only change did not alter user-facing claims or docs pages.

## 9. What Did Not Go Smoothly

The first JS test attempt exposed a test fixture that omitted both the Julia gradient and Hessian, so it failed on the Hessian check before testing the missing-gradient case. The fixture now supplies a positive-definite Hessian and omits only the gradient. The first writer pass-rule update then made the four existing signed FAIL snapshots appear stale. The first snapshot-mutation control also set a Hessian flag to its existing value; the suite caught this ineffective mutation and it now flips the recorded value. Rose then found Python's `True == 1` comparison could hide a type substitution; exact JSON comparison and boolean-to-number controls now reject it. The `--check` compatibility rule accepts only the N9 block derived from unchanged engine fields and its false pass-rule annotation. The closeout helper also first resolved a relative path into the brain repository; that blank accidental template was removed, and the report was recreated at its explicit worktree path.

## 10. Known Residuals

This is a local checker and receipt-contract repair only. It does not establish numerical parity, qualify any signed FAIL row, verify the full P1 goal, or place this code on `origin/main`. The full goal remains unmet. The repository-wide closeout scan is blocked by the unrelated acceptance-ledger syntax issue. The shared conductor-owned `check-log.md` was not edited; the task record is in `check-log.d`.

## 11. Team Learning

Memory receipt: used the routed GLLVM parity context and the signed stop-rule boundary; the strict inequality and Hessian requirement shaped the checker and mutation tests.

Golden Set: not run. This slice changes receipt validation and test controls, not numerical model code or a known numerical failure class.

## 12. Cross-Product Coverage

Covers: R and Julia campaign receipt serialization; Python assembly; JavaScript checkpoint/parity validation; the four signed P1 campaign comparisons.

Does NOT cover: R or Julia optimization behavior, model fitting, full C0-C8/X2 closure, F-capability coverage, numerical row admission, public parity claims, or the required landing on `origin/main`.

## 13. Independent Rose Audit

Rose verdict: PASS WITH NOTES. Rose accepted all four intended snapshot compatibility cases and rejected 14 boolean/integer type mutations. The full P1 goal remains unmet, and the repository-wide closeout compiler is blocked by the unrelated malformed `.unlazy/totoro-t4-p6-grid/GATES.md`.
