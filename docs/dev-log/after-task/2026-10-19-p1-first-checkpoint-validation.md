# P1 first checkpoint provenance and regression validation

Status: candidate checkpoint verified at `7d6806a59d840aafb3e557fb2dddbc529d345136`; the receipt-only provenance correction is committed in `fc908cf4ec37092a644d3f8a6c4395734d775dad`. The exact committed-source checkpoint passed after that commit. Main remains separate. This report does not close full P1.

## 1. Goal

Verify the seven approved receipt rows and the exact candidate regression evidence, repair the provenance mismatch without rerunning observations, and state the remaining P1 blockers against current `origin/main`.

## 2. Implemented

Finalized the retained raw capture metadata so `derivation_commit` names `7d6806a59`; the derive snapshot hash matches the committed deriver. Regenerated six derived sidecars and six paired case receipts. The R and Julia TSVs, both runner snapshots, and the empty launch-source patch are byte-identical before and after. The recorded Julia capture remains pinned to `7a2b8dcbe` and its captured source tree. No new R or Julia observations were run.

The source files changed between capture and candidate are limited to `src/phylo_binomial_xlv.jl`, `src/phylo_gamma_xlv.jl`, and `src/phylo_nb_xlv.jl`. Review found these changes do not touch the seven receipt routes. Therefore the raw cases are validated as route-local carry-forward evidence; they were not rerun against candidate source `7d6806a59`.

## 3a. Decisions and Rejected Alternatives

The deterministic finalizer was used to bind the current deriver to existing captured outputs. Re-running R or Julia would add observations without addressing the metadata-only discrepancy. No scoreboard admission beyond the seven already scoped rows, C6 classification change, tolerance change, API change, or merge is authorized here.

## 4. Files Touched

The first-seven `run.json`, six derived sidecars, and six paired case receipts are in provenance-correction commit `fc908cf4e`. This report records the post-commit validation in its follow-up. The candidate's environment configs `.claude/settings.json`, `.cursor/hooks.json`, and `.mcp.json` remain foreign dirty edits and are excluded.

## 5. Checks Run

On candidate `7d6806a59`, raw derivation validation passed; behavior receipts are current at 32 receipts and 24 classes; postfit receipts are current at seven twin rows and 52 rows; postfit recount passed; assembly reports 317 rows current; checkpoint mutation controls passed; behavior negative controls passed 37/37; and the full true-parity negative-control suite exited 0.

The canonical current-origin/main measurements (locally fetched `882292ac6539d00bc0a08f5d0d06dc9f471251b3`) are C0, C1, C5, C7, and C8 MET; C2 is 284/297, C3 6/8, C4 4/8, C6 not met, and X2 298/317. Candidate checkpoint remains X2 304/317 and C2 290/297.

All four full and four core Julia partitions on Totoro exited 0 on exact ref `7d6806a59`; each result records a matching ref, successful source smoke, empty tracked diff, and its log SHA-256. Full partitions include the quality check in shard 1. The exact-ref local Documenter build completed its checks and VitePress render; deployment was skipped. Its log is `.unlazy/true-parity-p1-next/regressions/7d6806a/docs-build.log` (SHA-256 `d9a75d7f5eda60309a910376d6dadcccca9a0da0c8014975927180eee68191a6`). No deployment or visual audit is claimed.

Post-commit validation at `fc908cf4e`: the exact checkpoint verifier passed with seven rows, X2 304/317 and C2 290/297. The behavior suite passed 37/37; checkpoint mutation controls passed; all true-parity checker negative controls passed; receipt checks reported 32 behavior receipts/24 classes, seven postfit twins/52 rows and 20 ISDM case receipts/20 rows; the assembler reported 317 rows current. C6 measured MET on this candidate with the approved delta (17 extras, 11 scoped twins, nine held names). These results establish the first checkpoint on the candidate only.

## 6. Tests of the Tests

Checkpoint controls reject a wrong-row substitute, missing P1 provenance, unexecuted bridge context, and mismatched behavior. The broader checker controls reject invalid receipts and unsigned dispositions. The behavior suite reports 37/37 passing controls. Retained earlier failed setup attempts remain evidence of missing isolated manifests; tests did not start on those attempts. The retry used the validated manifest and passed independent source smokes before all eight partitions ran.

## 7a. Issue Ledger

No issue or pull request was changed. No new row admission, classification signature, GitHub merge, push, release, or public parity claim occurred.

## 8. Consistency Audit

Receipt derivation, sidecars, behavior table, assembled map, and exact seven-row candidate checkpoint agree. Independent Emmy and Pat reviews passed with notes on the corrected metadata; Rose's independent Sol review passed with notes and requested refreshed completion wording. The latest fetched main head is `882292ac6`; this candidate contains the already-approved `7cd56b3d0` main integration but not the later main commits.

## 9. What Did Not Go Smoothly

The initial provenance record named a deriver that did not match the captured snapshot. The deterministic finalizer repaired the metadata and receipt hashes without changing observations. Pat initially inspected a different stale checkout; the exact corrected candidate was subsequently independently verified by Emmy and Rose. The main checkout advanced during the work. A per-change entry was added under `docs/dev-log/check-log.d/` to avoid the F-capability lane’s concurrent central `check-log.md` rewrite. The AGENTS phase snapshot remains held because that lane also changes AGENTS.md.

## 10. Known Residuals

Thirteen scoreboard rows remain open, including four C3/C4 diagnostic rows outside this checkpoint; five F-capability gates remain open. C6 is met only on the approved candidate scope and is not met on current main. The latest locally available main ref remains at X2 298/317. Current main integration, focused PR landing, full completion-panel closeout, updated shared check-log/phase snapshot, validated handover, and full P1 closure remain open. GitHub and the remote were unreachable during this turn, so `882292ac6` is not confirmed as the current remote tip. A fresh integration preview against that ref is clean but integrating it requires separate approval.

## 11. Team Learning

Bind derived receipts to the exact deriver snapshot and launch tree, preserve raw observations across a metadata repair, and state explicitly when route isolation supports carry-forward rather than a rerun. Refresh every completion record when evidence changes; old reports remain historical until their owner reconciles them.

Memory receipt: the approved plan and exact scope decisions remain the authority; this report records measurements and does not create signatures.

Golden Set: the exact seven-row checkpoint, raw hashes, eight partition result receipts, and independent review records are retained under `.unlazy/true-parity-p1-next/`.

## 12. Cross-Product Coverage

This checkpoint covers seven public-door receipt rows, receipt provenance, repository regressions, and local documentation generation. This does NOT cover new numerical R/Julia fit comparisons, remaining scoreboard rows, the five F-capability gates, main landing, a release, or full P1 parity. R parity fit reruns and benchmarks: N/A for this metadata-only correction. Aqua/JET ran in the full Pkg.test partition; Allocs was not run. Rose verdict: PASS WITH NOTES for this bounded checkpoint, not full P1.
