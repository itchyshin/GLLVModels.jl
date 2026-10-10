# P1 first-seven candidate checkpoint

## 1. Goal

Close the seven exact P1 receipt gaps locally and provide the remaining 13-row/37-name decision packet. This is a candidate checkpoint; full P1 and landing on main remain open.

## 2. Implemented

FAMILY-11 now cites two observed public R bridge refusals with frozen provenance. check_auto_residual and five iSDM rows cite paired public-door observations with real mixed-source positive controls. Raw TSVs and finalized process/source/build metadata are retained. Canonical writers re-derive raw evidence, then maps and scoreboards. The internal verifier fixes baseline f220379d0937d0afffc6a030023c63c0f715e168 and checks exact IDs, unchanged admission/case contracts, raw hashes and baseline predicates. Three stacked PR branches separate family, residual and five iSDM bindings. A consolidated unsigned decision map and six-axis reconciliation are saved.

## 3a. Decisions and Rejected Alternatives

Preserved P1 R9539352f66f2db2cc26b1c393e67212a359b60c9, engine source, version 0.3.0 and tolerances. Rejected unexecuted bridge context, wrong-row substitutions, internal-predicate surrogates, hand-authored sidecar claims and deleted existing controls. COUNT is the ordinary all-count public admission with a separate mixed positive control. Extra/missing source cases record the actual first public family-length guard. Wrapper-law evidence uses standalone logit construction, collector refusal and valid cloglog controls. No C6 exclusion, identity, Monte Carlo or animal-slope contract is signed.

## 4. Files Touched

The later two-file CI repair also changes `.github/workflows/true-parity-check.yml`; its control-tool and check-log files are listed below. The delta adds its audit and dispatch manifest.

- `AGENTS.md`
- `docs/dev-log/audits/2026-10-06-p1-final-panel.md`
- `docs/dev-log/audits/2026-10-06-p1-verification-state.json`
- `docs/dev-log/audits/2026-10-06-p1-draft-assessments.json`
- `docs/dev-log/handover/2026-10-06-p1-next-continuation.md`
- `LOOP/lanes/true-parity-p1-next/dispatch/final-emmy.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/final-pat.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/final-rose.events.dispatch.txt`

- `LOOP/lanes/true-parity-p1-next/GOAL.md`
- `LOOP/lanes/true-parity-p1-next/arcs.md`
- `LOOP/lanes/true-parity-p1-next/checkpoint.md`
- `LOOP/lanes/true-parity-p1-next/checks/GATES.md`
- `LOOP/lanes/true-parity-p1-next/checks/exact-seven.json`
- `LOOP/lanes/true-parity-p1-next/dispatch/completion-emmy.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/completion-pat-fresh.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/completion-rose.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/ebbinghaus-continue.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/ebbinghaus.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/gauss.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/hopper-continue.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/hopper.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/melissa.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/pat-continue.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/pat-repair.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/pat.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/shannon-app.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/shannon-reverify.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/dispatch/shannon.events.dispatch.txt`
- `LOOP/lanes/true-parity-p1-next/gates.lock.json`
- `LOOP/lanes/true-parity-p1-next/recon/shannon-g0.md`
- `LOOP/lanes/true-parity-p1-next/ultra-plan.md`
- `docs/dev-log/after-task/2026-10-06-p1-first-seven-candidate.md`
- `docs/dev-log/audits/2026-10-06-p1-initial-panel.md`
- `docs/dev-log/check-log.md`
- `docs/dev-log/core070/true-parity-latest/behaviour-equivalence.json`
- `docs/dev-log/core070/true-parity-latest/case-map-assembled.json`
- `docs/dev-log/core070/true-parity-latest/case-map-family.json`
- `docs/dev-log/core070/true-parity-latest/case-map-isdm.json`
- `docs/dev-log/core070/true-parity-latest/case-map-postfit.json`
- `docs/dev-log/core070/true-parity-latest/receipts/family/cases/CORE070-FAMILY-02-LOG-PUBLIC-R-BRIDGE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/family/cases/CORE070-FAMILY-05-LOG-PUBLIC-R-BRIDGE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/family/cases/CORE070-FAMILY-07-LOGIT-PUBLIC-R-BRIDGE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/family/cases/CORE070-FAMILY-BETA-ALIAS-COMPATIBILITY-ADAPTER.json`
- `docs/dev-log/core070/true-parity-latest/receipts/family/first-seven-boundary/CORE070-FAMILY-11-LOG-PUBLIC-R-BRIDGE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/family/first-seven-boundary/r-public-bridge.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-COUNT.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-EXTRA-SOURCE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-MISSING-IN-TRAIT.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-MISSING-SOURCE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-WRAPPER-LAW.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/POSTFIT-SURFACE-check_auto_residual.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/behaviour-equivalence.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-06-v3/julia-public.tsv`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-06-v3/r-public.tsv`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-06-v4/julia-public.tsv`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-06-v4/r-public.tsv`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-06-v4/run.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-COUNT-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-EXTRA-SOURCE-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-MISSING-IN-TRAIT-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-MISSING-SOURCE-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-WRAPPER-LAW-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-WAVE7-CHECK-AUTO-RESIDUAL.json`
- `docs/dev-log/core070/true-parity-latest/scoreboard.md`
- `docs/dev-log/plan-actual/2026-10-06-p1-next.md`
- `docs/dev-log/plans/2026-10-06-p1-c6-evidence.md`
- `docs/dev-log/plans/2026-10-06-p1-decision-map.md`
- `docs/dev-log/plans/2026-10-06-p1-method-evidence.md`
- `test/runtests.jl`
- `test/test_family11_p1_boundary.jl`
- `test/test_first_seven_behaviour_p1.jl`
- `tools/core070_behaviour_receipts.py`
- `tools/core070_family_bridge_p1.R`
- `tools/core070_family_p1_receipts.py`
- `tools/core070_isdm_p1_receipts.py`
- `tools/core070_postfit_p1_receipts.py`
- `tools/first_seven_behaviour_J.jl`
- `tools/first_seven_behaviour_R.R`
- `tools/first_seven_behaviour_derive.py`
- `tools/test_core070_behaviour_receipts.py`
- `tools/test_true_parity_checkpoint_check.mjs`
- `tools/true_parity_checkpoint_check.mjs`

## 5. Checks Run

The later measured results appear in the verification delta below. Earlier RUNNING statements in this section describe the19:31UTC checkpoint.

Canonical exact checkpoint passed at ce3432b95, independently at aee0daf9b, at the pushed integration head e9cb474ad and at the reviewed candidate d879a9e21: X2 304/317,C2 290/297, exact sevenchanged; C3 6/8,C4 4/8,C6 37 undecided unchanged. Five owning --check commands pass. Behaviour Python 33/33 and enabled Julia wrappers2/2 pass. Family boundary has1positive/6rejected mutations. Checker, assembler and checkpoint controls pass. Unlazy root5 runnable gates reverified with a fixed Python/Node PATH; main gate fails as expected. G0 lock verification passes. Documenter at the exact d879a9e21 candidate passed in 98.617 seconds; the earlier aee0daf9b build passed in 109.2 seconds. Independent Emmy, Pat and Rose reviewers each returned receipt-candidate OK at d879a9e21, conditional on package verification and merge consent. The older ed11661f0 unsharded run stopped at its two-hour cap, exit 124; its core run was NOT_EXECUTED and all four owned process IDs were verified stopped. An initial four-shard archive run at d879a9e21 was stopped after missing Git metadata and the default R library's absent jsonlite were identified; exit 143 after 1153 seconds, no suite pass claimed. Both existing tests that exposed those prerequisites passed in 17.319 seconds in real Git clones with existing R jsonlite 2.0.0. Corrected exact-candidate full/core verification is running on Totoro using the existing four-shard partition, launched 19:29:03 UTC, estimate 100-120 minutes, cap 21:29:03 UTC. Julia four threads per worker, BLAS/OMP one, maximum sixteen cores. No source tests or tolerances changed to repair the environment. Fresh main remained f220379d0. Full after-task acceptance validation remains open while landing and closure gates are unmet.

## 6. Tests of the Tests

Retained all 26 existing behavioural controls and added seven. Table coverage is prior 57entries plus the exact sixtargetbehaviours. Negative controls reject swapped labels, wrong/absent scoped entries, wrong calls, incomplete raw cases, wrong fixture, wrong installed library/source/build, wrong bridge refusal class, NOT_EXECUTED context, missing/stale byte hashes, nested-hash tampering, changed contracts and unrelated-row substitutions. Raw v3 failed observations remain non-binding. Source selection rejects unapproved IDs and validates all six raw cases before a staged write.

## 7a. Issue Ledger

Fixed local receipt-tool and integration defects described below. First checkpoint exact seven rows are candidate-bound. Remaining 13 scoreboard rows and 37 C6 names remain open. Draft PRs #842, #843 and #844 are open and attached. Full and repaired core suites pass at `d879a9e21`; fresh Pat, Rose and Emmy reviews pass the bounded candidate scope. Hosted Julia-only jobs, consent for every merge, and main verification remain pending. P2/releases/version changes are deferred.

## 8. Consistency Audit

Rose independently verified exact sevenchanges and13/37 coverage, and found missing test registration, stale C6 FAMILY11 prose and missing consolidated decision map. Parent repaired all three. Pat identified the earlier checkpoint-hash storage mismatch; the new checker validates top-level and nested behavioural hash blocks without rewriting legacy batch provenance. Both wrappers are registered and accept an explicit compatible Python interpreter. All shared maps are canonically regenerated, with unchanged admission/classifications/signatures. README, CLAUDE and user API/engine are unchanged. Existing covariance/grouping check limitations remain: overlay/runner freshness is not a whole-engine certificate.

## 9. What Did Not Go Smoothly

Old CLI rejected Luna before work; switched to the app CLI. Initial SSH/process-list sandbox blocks did not establish network/auth failure. Pat initially removed455 existing test lines; they were restored. Parent repaired incomplete/confounded panel construction, wrong wrapper route, unsupported R arguments, unit keys, R factors and quoted multiline TSVs. Failed v1-v3 attempts were retained; only v4 binds. Canonical overlays initially skipped iSDM and a loader name was wrong; corrected both. An exact table-count assertion was strengthened to 63. Initial completion review started before all candidate tooling was ready; its failures are retained. One additional fresh three-reviewer repair round returned receipt-candidate OK, still conditional on suites and main consent. Consolidation miscounted aliases; corrected to 12 scoped assessments and 5 mismatches directly from17rows. Five retained parent compaction boundaries and eleven successful production invocations for six distinct roles exceeded the planned invocation discipline. The suite archive environment also lacked Git metadata and the default R jsonlite dependency; both were corrected and smoke-tested before restarting. Automatic approval review rejected one report commit helper that selected every dirty path; its replacement staged only named task paths. Output path was corrected to Melissa’s owned plan-actual path. No failures were waived.

## 10. Known Residuals

Full P1 is NOT complete. Main remains baseline until explicit merge consent. C2 has 7 open capabilities; C3 has 2 open realistic-size rows; C4 has 4 open data rows; C6 has 37 undecided names. Method recommendations are unsigned; private urbanisation raw has no authorised verified location; reported 7/40 seed audit raw was not located. The full suite passed all four shards at d879a9e21, and the repaired core suite also passed all four shards (see the 21:00 UTC verification delta). The full after-task validator exits 1 because an inherited unrelated .unlazy/totoro-t4-p6-grid/GATES.md lacks explicit gate IDs. That historical ledger is protected and unchanged. The isolated PR worktree also contains tracked historical ledgers. A complete twenty-gate task-only validation container reached only current G7/G8: structure passed, acceptance exit1. No required task gate was omitted or abandoned; no closure pass is claimed. Every later checkpoint requires its own contracts and acceptance ledger.

## 11. Team Learning

Memory receipt: queried shinichi-brain vault first; repository signed D-319 rules remain technical authority. Re-ran route.py with registered key GLLVM.jl and loaded its LOAD-FIRST manifest. Absolute-path and renamed-key lookups supplied no manifest and are recorded as routing limitations. Applied compute caps, scope identity, source provenance, signed boundaries and separate method/coverage claims. Golden Set: listed relevant completion-overclaim, worktree/branch drift, model-tiering, partial-arc, pdHess and refusal-contract cases; deterministic self-tests passed for all detectors. This proves detector discrimination, not universal agent compliance. Retain a complete crossed fixture and a successful public admission control alongside refusal cases. Keep failed raw and original review verdicts; repair the cause and reverify an immutable candidate.

## 12. Cross-Product Coverage

Covers: the exact seven P1 rows and their public refusal/behaviour receipts, two language runners, raw metadata, canonical derivation/assembly, controls and the 13/37 decision packet. Does NOT cover: full-family numerical parity, realistic-size/data convergence, calibrated Monte Carlo/coverage, animal random slopes, joint phylo/grouped signal, C6 signed dispositions, R engine changes, P2, release or main landing. No numerical convergence or coverage claim follows from a matching behaviour label.

## Verification delta, 21:00 UTC

Repaired core session31075 finished successfully: driver_exit0/core_exit0 and all four shards exited0. Shard summaries were 7607 pass/11 Broken/7618 total, 10017/4/10021, 6717/5/6722 and 6556/70/6626. README checks passed8/8 per shard. Elapsed time was2331.498seconds. The local retained aggregate is `repaired-core-shards.log`, SHA256 `8831a86d71a365458aa32d69cdd0e2027b3392fc54b6362e676d987bcd239eba`; Totoro's retained status reads `driver_exit=0`, and the owned process group399887 is absent. The full suite at d879a9e21 remains four successful Pkg.test shards. The core scratch environment contains test-only dependencies only; no package dependency, source test, or tolerance changed.

Hosted repaired negative-controls, guard, Documenter and deploy checks passed on #843 and #844. The two Julia-only twin jobs remain in progress on each PR as of21:00UTC. The three fresh completion reviewers are examining the candidate; no final panel verdict is claimed yet. Candidate exact-checkpoint evidence remains X2 304/317 and C2 290/297, with C3 6/8, C4 4/8, C6 37 undecided unchanged. Main remains f220379d0 pending explicit merge consent. The after-task structure check is not the acceptance closure: task G7 and G8 remain unmet, total18met/2unmet/0abandoned.

## Verification delta, 20:22 UTC

All four Pkg.test() shards passed at d879a9e21: shard passes7619/10021/6719/6556, zero failures or errors, with9/4/5/70 recorded Broken assertions. The separate README checks passed8/8 per shard. The full/core driver finished after2765.447seconds, exit1 because every direct core invocation stopped on missing test-only dependencies: StableRNGs on shards1/2/4 and JSON3 on shard3. These failed core attempts are retained. No package regression follows from missing test dependencies. Owned original driver descendants were verified absent.

The scratch core environment copies test/Project.toml without Aqua/JET and resolves in offline mode against already cached packages. Repository source and dependency files stay unchanged. Its import smoke passed in4.215seconds, including exact package path and absent Aqua/JET checks. Core rerun session31075 started20:20:01UTC, stops21:20:01UTC, estimate45-60minutes. Driver399888, verified process group399887; existing four isolated Git checkouts, JULIA_LOAD_PATH points to the scratch environment, Julia4/BLAS1/OMP1 per worker, maximum sixteen cores. Remote repaired-core-driver/status/source/started/finished files and repaired-core-k.log/status files retain outcomes. Do not claim core passed until all four exits are zero; the continuing lane owns cleanup.

FAMILY-11 source8a74f62c5 CI37515650751 completed successfully with all eight Julia jobs green. Current PR844 repair f5b90fa19 provisions R4.5.3/Julia1.10 and selects Julia from PATH before the local fallback. Previous hosted negative controls failed32/33 because Rscript was absent; the Juliaup-only path was also unsuitable for hosted setup-julia. All33/33 controls pass locally after repair. One additional fresh enforced Emmy gpt-6-luna/medium follow-up returned OK for this two-file change. Its closing generic reference to passing full/core evidence does not certify core; core remains pending. Hosted repaired controls are queued. Receipt runners, raw evidence, numerical engine, Julia tests, package dependencies, versions, signatures and tolerances are unchanged from the full-suite source. The prior receipt panel remains scoped to d879a9e21.

The isolated PR worktree still has twenty tracked historical .unlazy files, so copying this task's ledgers there did not isolate unrelated malformed ledgers. Instead, /private/tmp/p1-current-task-validation used unchanged candidate ff46e14c9 source aliases and copied every one of this task's seven ledger files, all twenty gates, omitting zero required gates. All six leaves reverified with exit0; root exit1 only for open G7/G8. The canonical full report validator passed structure and then exited1 for this task's UNMET gates, without a historical-ledger parse failure. It is a validated open contract, not a completion pass. Historical ledgers were neither modified nor abandoned. The900second adapter changes only timeout from the canonical120second default.

Status remains18met/2unmet/0abandoned. No merge consent or C6/model/Monte Carlo signature has arrived. Main remains f220379d0; full P1 is open. The parent has crossed a fifth compaction boundary. Production remains eleven successful invocations for six roles; the fresh completion panel is in progress. A whole-file slop check hit historical check-log prose; the exact new entry separately passed with zero hits, and historical prose was preserved. No new P1 arc was started.

## Verification delta, 21:08 UTC

Fresh Pat, Rose and Emmy reviews pass the scoped first-seven checkpoint at candidate `2451c176d7c098d91f4f29a5b1aba1b1117dc9a7`. Pat found no P0-P2 issue in the seven public-door rows. Rose found no scientific or numerical blocker within scope and explicitly leaves full P1 and main unmet. Emmy verified exact rows, tooling, provenance and regression evidence; the source and test trees are unchanged from tested suite source `d879a9e21`. Melissa's dated reconciliation records the same distinction and open gates.

Hosted negative-controls, guard, Documenter and deploy checks pass on #843 and #844. At21:08UTC, #844's Julia1.10 twin job passed in38m7s; #844 Julia1 and both #843 Julia jobs were still in progress. No main merge or maintainer signature occurred. G7/G8 remain unmet, and full P1 remains open.


## Verification delta, 22:17 UTC

The exact current PR #844 candidate `54f0c2e00e13b5daa7bebb7e3419cf6561206ad9` reverified with the pinned baseline. It binds exactly the seven named rows and reports X2 304/317 and C2 290/297; C3 remains 6/8, C4 4/8, and C6 has 37 undecided names. The four owning derivation checks pass: family 30 receipts/21 rows, behaviour 32 receipts/24 classes, postfit 7 twin rows/52 rows, and iSDM 20 receipts/20 rows. Assembly reports 317 rows current. The canonical checker, assembler and checkpoint negative controls pass. The same exact candidate's hosted P1 twin workflow run 37533654264 passed on Julia 1 and 1.10; hosted negative-controls run 37533678454 passed. The staged #842 and #843 P1 twin runs also pass on both versions.

Full `Pkg.test()` at `d879a9e21fe2203ae52d5ceb3a4d2ed5c3e3e886` passed all four shards: 30,915 passes, 88 existing Broken assertions, zero failures or errors. Repaired core verification at that source passed all four shards: 30,897 passes, 90 existing Broken assertions, zero failures or errors; README checks passed 8/8 per shard. The retained aggregate SHA256 is `8831a86d71a365458aa32d69cdd0e2027b3392fc54b6362e676d987bcd239eba`. Local Documenter and rendered reader-surface checks passed on the candidate; no numerical engine, test, dependency, tolerance, version, admission or classification changes were made.

The independent completion panel (Emmy, Luna/medium; Pat, Luna/medium; Rose, Sol/high) passed within the seven-row scope. Melissa's plan/actual reconciliation is retained in `docs/dev-log/plan-actual/2026-10-06-p1-next.md`. The task-scoped ledger now has 19 met, 1 unmet and 0 abandoned: G8 is met with the completed panel, this updated report and Melissa reconciliation; G7 remains open for explicit merge consent and verification on main. PRs #842, #843 and #844 remain drafts. Main is still `f220379d0937d0afffc6a030023c63c0f715e168`; no merge consent has been given. Full P1 remains open: 13 scoreboard rows and 37 C6 names require the reviewed, unsigned decision packet and maintainer contracts. P2 and release work remain deferred.

### Required audit fields

- Mathematical contract: no likelihood, parameterization or numerical estimator changed; this checkpoint records public-door receipts and exact behavioural twins.
- Tests of tests: retained and extended negative controls reject wrong-row substitutions, malformed or stale evidence, invalid bridge context, and behaviour mismatches.
- Benchmarks: N/A; no hot path changed.
- R parity: N/A as a numerical parity verdict; this checkpoint makes no numerical agreement claim.
- JET: the package-quality gate passed within the full `Pkg.test()` run; no separate report was run.
- Allocs: N/A; no hot path changed.
- Aqua: passed within the full `Pkg.test()` run.
- Documentation consistency: local Documenter and 34-page rendered reader-surface checks passed; no user API changed.
- GitHub issue maintenance: no issue action needed; this is the existing P1 checkpoint tracked by drafts #842-#844.
- Remaining risks: G7 and full P1 remain open as described above.
- Next command after approval: merge #842 to main, then remeasure the main checkpoint before requesting the next merge.

Prose assessment: exact-draft self-review is recorded in `docs/dev-log/audits/2026-10-06-p1-after-task-assessment.json` after final text validation.

Rose verdict: PASS WITH NOTES; the seven-row receipt checkpoint is verified, while main landing and full P1 remain open.


## Final verification delta, 22:58 UTC

The exact current candidate `54f0c2e00e13b5daa7bebb7e3419cf6561206ad9` passed a fresh lane-root unlazy `--reverify`: 19/20 gates met, 0 abandoned, and only G7 remains open. All 12 runnable checks passed, including the exact seven-row checkpoint, receipt derivations, negative controls, and C0/C1 measurements. The G7 main-ref check returned the expected `P1_CHECKPOINT_NOT_MET` at FAMILY-11 because no merge consent was given and main remains at the pinned baseline.

A fresh local Documenter build completed successfully, including doctests and VitePress rendering. The full repository after-task validator still stops on the unrelated historical Totoro ledger syntax error; the report structure itself passes. The task-scoped ledger has no other unmet gate. PRs #842, #843 and #844 remain drafts and unmerged. This verifies the first candidate checkpoint only; full P1 remains open for 13 scoreboard rows and 37 C6 names.
