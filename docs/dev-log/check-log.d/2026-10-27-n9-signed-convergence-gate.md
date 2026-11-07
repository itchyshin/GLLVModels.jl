## 2026-10-27: Strict signed P1 convergence evidence gate

- Aligned campaign receipt writing, Python assembly, and JS parity checking with the signed P1 stop rule: both engine Hessians must be recorded positive definite and both maximum absolute gradients must be strictly below `1e-5`. The writer records Hessian evidence and gradient values, using JSON `null` when a gradient is unavailable, and records a failing pass-rule leg. The real-tree `--check` accepts only the N9 block derived from unchanged engine fields plus the false leg/reason on the four existing signed FAIL snapshots.
- Exactly the four campaign comparisons named in the signed 2026-10-21 note require the gate: ordinal, iSDM headline, spider, and fungi. Urbanisation, beetle, and other rows are not added to that scope.
- Python assembler/receipt tests: 549 passed, 0 failed. JS negative controls: 261 passed. Campaign receipt check: `CAMPAIGN_RECEIPTS_OK`; assembler check: `ASSEMBLE_OK 317 rows current`. JSON type-mutation controls reject booleans replaced with numerically equal integers. `node --check`, Python bytecode compilation, and `git diff --check` passed.
- Absolute-path slop scan found 0 issues; after-task structure check passed. Full closeout validation stopped on the unrelated `.unlazy/totoro-t4-p6-grid/GATES.md` missing an explicit gate ID.
- Independent Rose review: PASS; accepted all four intended snapshot compatibility cases and rejected 14 boolean/integer type mutations.
- No raw capture, fit, row admission, status change, tolerance change, push, or merge occurred. Full P1 remains unmet.
- After-task report: `docs/dev-log/after-task/2026-10-27-n9-signed-convergence-gate.md`.
