# After Task: First-Seven Behaviour Receipt Provenance

## 1. Goal

Harden first-seven public-behaviour derivation against fixture substitutions and runner-provenance mismatches, then refresh the six canonical leaf receipt behaviour blocks from retained v4 raw evidence.

## 2. Implemented

The derivation now pins the expected fixture recipe digest independently of agreement between R and Julia, requires `ISDM-EXTRA-SOURCE` to map to the signed `guard:family-length` category, validates captured R and Julia runner hashes, preserves the capture-time derivation hash in immutable `run.json`, and records the current derivation hash in refreshed derived receipts. The six canonical leaf receipt behaviour blocks now cite those refreshed derivations. No raw captures, case maps, scoreboard rows, classifications, admissions, likelihoods, or tolerances changed.

## 3a. Decisions and Rejected Alternatives

Kept `run.json` and raw TSVs immutable. Retained the old `runner_sha256.derive` as capture provenance instead of rewriting provenance to match the revised validator; bound the new derivation version in the derived receipt. Refreshed only the authorized six leaf receipt behaviour blocks and their derivation citations. No assumptions without asking affected the evidence contract.

## 4. Files Touched

- `tools/first_seven_behaviour_derive.py`
- `tools/test_core070_behaviour_receipts.py`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-COUNT.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-EXTRA-SOURCE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-MISSING-IN-TRAIT.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-MISSING-SOURCE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-WRAPPER-LAW.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/POSTFIT-SURFACE-check_auto_residual.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/behaviour-equivalence.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-COUNT-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-EXTRA-SOURCE-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-MISSING-IN-TRAIT-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-MISSING-SOURCE-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-WRAPPER-LAW-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-WAVE7-CHECK-AUTO-RESIDUAL.json`
- `docs/dev-log/check-log.md`
- `docs/dev-log/after-task/2026-10-10-p1-behaviour-receipt-provenance.md`

## 5. Checks Run

- `python3 tools/test_core070_behaviour_receipts.py`: 35/35 controls passed.
- `python3 tools/first_seven_behaviour_derive.py --self-test`: `CORE070_FIRST7_DERIVATION_SELFTEST_OK`.
- `python3 tools/true_parity_assemble.py --check`: `ASSEMBLE_OK 317 rows current`.
- `git diff --check`: clean. The raw v4 capture directory had no diff.
- The parent independently verified the exact seven-row checkpoint and canonical negative controls on this PR branch. No Julia fit or R/Julia capture was run for this change.
- The report structure check passed. Full after-task validation stopped because the inherited `.unlazy/totoro-t4-p6-grid/GATES.md` cannot be parsed by the current gate checker. That cursor-owned ledger remains unchanged.

Mathematical contract: N/A; no likelihood, parameterization, estimator, or model output changed.

Benchmark numbers: N/A; no hot path changed.

R-parity: N/A; this receipt-validator change makes no numerical agreement claim and does not modify a numerical parity surface.

JET: N/A; no Julia code changed.

Allocs: N/A; no Julia hot path changed.

Aqua: N/A; no package metadata, exports, or Julia code changed.

## 6. Tests of the Tests

The focused controls reject a changed expected fixture digest, tampered captured R or Julia runner hashes, and an EXTRA-SOURCE label that does not match the approved expected category. The regression control also proves the unchanged capture metadata and unchanged raw files can be re-derived after a validator revision while preserving the capture-time derivation hash and recording the current derivation hash. The existing public-door controls continue to reject route substitutions and mismatches.

## 7a. Issue Ledger

This follow-up hardens evidence in the existing P1 PR lane. It does not close additional receipt gaps or change admissions.

## 8. Consistency Audit

Re-derived the six requested cases from retained `raw-2026-10-06-v4` files. The canonical leaf diff changes only each `behaviour` block’s `read_from` hashes. The derivation and aggregate receipts record the current derivation SHA. Raw TSVs and `run.json` remain byte-for-byte unchanged; no map, scoreboard, other receipt, or admission file is in the change. The assembly check confirms all 317 rows remain current. User-facing docs and mathematical claims were not affected.

Consistency searches: broad public-claim scans were N/A because this change alters no public API, supported-family claim, mathematical statement, or performance claim. The exact targeted source/test search was `rg -n "first_seven_capture_runner|fixture_digest_is_anchored|EXTRA-SOURCE|receipt_derivation_sha256|runner_provenance_problem" tools/test_core070_behaviour_receipts.py tools/first_seven_behaviour_derive.py`; it located the signed fixture, expected category, runner-provenance guard, and regression controls.

## 9. What Did Not Go Smoothly

The lane has no local `tools/lane_preflight.sh` or after-task wrapper. The hub `closeout.py` resolves paths against the Shinichi vault rather than this worktree; its accidentally created empty template was removed after checking its contents. The hub R checker validates this report's structure, but its full validation also rechecks every repository ledger and fails on the inherited cursor-owned Totoro ledger syntax. I left that unrelated ledger unchanged.

## 10. Known Residuals

This evidence is limited to the approved first-seven public-door checkpoint. It does not establish numerical R/Julia parity, complete P1 parity, main-branch landing, or release readiness. No package test suite, fit, JET, allocation, Aqua, or new R/Julia capture was run as part of this validator/receipt-only change.

## 11. Team Learning

Memory receipt: the supplied repo instructions and the project after-task protocol were consulted; the missing lane-local wrapper was recorded above. No cross-project discovery or routed guard set was needed.

Golden Set: not in scope; this change modifies Python evidence derivation and receipt metadata, not Julia model behavior.

Prose self-review: style 2/10, moderate confidence; after editing em dashes and a faux-reveal sentence, the final style check found 0 hits in 872 words. Scientific gate: not applicable; factual gate: pass against the cited commands, current diff, and parent’s independent checkpoint result; reference gate: not applicable, no external references cited.

## 12. Cross-Product Coverage

Covers: first-seven public-behaviour receipt derivation, the five iSDM behaviours, `check_auto_residual`, immutable capture provenance, fixture digest pinning, expected EXTRA-SOURCE category, and the six corresponding canonical leaf receipt blocks.

Does NOT cover: likelihoods, fitting, numerical R/Julia parity, any other receipt cases, case-map or scoreboard content, the remaining P1 rows, admissions, main-branch integration, or release claims.

## Rose Verdict

Rose verdict: PASS WITH NOTES for the seven-row receipt checkpoint at `637100d9b0244942063616f9c1bbab7f47746ab4`. The delta review compared this tree with the reviewed receipt commit and current PR head, then re-ran the checkpoint, behaviour controls and receipt derivation checks. It found no P0-P2 defect in this bounded change. Full P1 and main landing remain UNMET. The full-repository after-task validator remains blocked by the unreadable cursor-owned Totoro ledger recorded above.
