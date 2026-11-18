# Session Handoff: true-parity-latest P1 closure

Meta: 2026-10-19 · from Codex · to Cursor · context handoff after focused test-harness repair

You are Cursor, taking over the `true-parity-latest` P1 lane of GLLVModels.jl. Continue the approved P1 goal from this candidate worktree. The first checkpoint is partial; full parity still requires every finish-line gate listed below.

## Critical Context

The approved finish line is **C0–C8 and X2 MET on `origin/main`, F-capability coverage verified, negative controls passing, independent completion review, reconciliation, after-task report, and final handover on main**. Candidate reports and agent summaries do not count as main evidence. The first seven receipt rows previously formed a met candidate checkpoint; that is below full P1 closure.

The handover gate is currently non-green. It found uncommitted and unpushed candidate work, 710 unpushed commits on other local branches, and six unmet acceptance ledgers belonging to unrelated lanes. These are declared below. Do not alter, stage, push, merge, abandon, or “fix” those unrelated items. The shared coordination board and AGENTS snapshot were deliberately left unchanged because multiple lanes are live and their current pointers are stale.

## Mission-control snapshot

| Repository | Candidate / main | CI | What is covered | Plan by priority |
|---|---|---|---|---|
| GLLVModels.jl | candidate `codex/p1-isdm-current-main-20261019` at `0961bf4f5`; local `origin/main` at `882292ac6` | No fresh candidate CI result recorded here | First-seven receipt checkpoint was met on a prior candidate. Full P1 is open; current fresh-main measurement in the ledger is from `7cd56b3d0`, before current local `origin/main` `882292ac6` | First reconcile current refs and active jobs; finish the exact candidate regression verdict; then take the remaining unowned P1 rows and F gates through separately approved checkpoint contracts; land only after fresh main checks and explicit consent |
| gllvmTMB | local `origin/main` at `2eb8caeb4` | GitHub API was unreachable during the history check | `select_lv()` exists on R main; `latent(d = "auto")` is in the separate auto-d branch, not in this local main ancestry | Separate question from P1; do not fold it into this lane without a new scoped contract |

## What Was Accomplished

- Rehydrated the P1 candidate at `0961bf4f5` and inspected its ledger, working tree, branch history, and sibling worktrees.
- Reproduced and repaired three candidate-only test-harness issues, with the maintainer’s explicit authorization:
  - `test/test_gp1_verdict.jl`: exact recorded Julia versions only; does not select a patch release’s fixture by minor version.
  - `test/test_nb_grouped_upper_warn.jl`: recorded boundary-specific checks require exact Julia `1.10.12` plus macOS aarch64.
  - `test/test_weights_byte_identity.jl` and its fixture/generator: cross-process bit checks run only when host fingerprint and case/project/manifest/test-project/input hashes all match.
- Focused results on this worktree, Julia 1.10.0:
  - GP-1: 291/291 pass.
  - Grouped NB upper/lower warning: 14/14 pass (expected boundary warnings printed).
  - Weights identity: 127 pass, 63 `@test_skip`; there were no failures. The cross-process comparison was skipped because the capture and candidate differ in `Manifest.toml` and `test/Project.toml` hashes. The in-process omitted-vs-`weights=nothing` checks passed. Thus cross-process bit identity remains unverified for this environment.
  - `git diff --check` passed.
- Checked the `d = "auto"` history because it was raised in this chat: Julia’s merged feature is `fit_gllvm(Y; K omitted)`, which routes through `select_lv`; it does not expose the R spelling `latent(d = "auto")`. On local gllvmTMB `origin/main` `2eb8caeb4`, `select_lv()` exists but `latent(d = "auto")` commits `36cf5d704` and `e7e5eaaf0` are not ancestors. Their branch refs exist. The GitHub API was unreachable, so current PR status was not verified.

## Current Working State

- Working tree: `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`
- Branch / HEAD: `codex/p1-isdm-current-main-20261019` / `0961bf4f5ba3bb67716f0b761295c87e95359c9e`
- Local `origin/main`: `882292ac6539d00bc0a08f5d0d06dc9f471251b3`
- Eleven commits on this candidate branch are not present on any remote ref. Nothing was pushed in this handoff. A push requires explicit maintainer instruction under this repository’s AGENTS.md.
- Six authorized test/fixture files are modified and remain uncommitted. Three other dirty config files and two untracked proposal files are protected foreign/unrelated state; do not stage them.
- The repository-wide handoff gate also found six unrelated unmet ledgers: `.unlazy/grouped-analytic-20260920/gates/leaf-S8.md`, `leaf-S9.md`, `leaf-S9c.md`; and `.unlazy/s9c-coverage-448-20260922/gates/leaf-1-main-tip-merge.md`, `leaf-2-focused-verify.md`, `leaf-3-pr-hygiene.md`. These are not this lane’s worktree, so they remain unchanged and unmet.
- A read-only Totoro check found concurrent Pkg.test jobs for `main` and PR suites `pr952`–`pr957` under `/home/snakagaw/lanes/fix1019/suites/GLLVM.jl`; nested Julia test processes were already running for about 62 minutes. Do not launch duplicate full suites, kill these jobs, or infer their result. One separate CMR evidence fit was also running and is unrelated.
- No `Pkg.test()` or core-suite pass is established for this exact candidate. A prior full-suite attempt was interrupted after overrun and did not produce an aggregate pass. The latest ledger still has REGRESSIONS, COMPLETION-PANEL, LANDING, CLOSURE, and FULL-P1 open.

## Key Decisions & Rationale

- P1 is not complete until the measured finish line in Critical Context is met on committed main. No P2, release, version change, public parity claim, or unsigned scope expansion is included.
- Approved C6 scope decisions and C3/C4 diagnostic contract are recorded in `docs/dev-log/decisions/2026-10-17-p1-c6-and-c3-c4-contract-approval.md`, `.unlazy/true-parity-p1-next/GATES.md`, and `docs/dev-log/plans/2026-10-19-p1-remaining-decision-packet.md`. The signed C6 choices do not admit scoreboard rows or authorize merges. Nine named C6 items remain held. Do not invent name-only exclusions.
- The approved `extract_phylo_signal` contract is implementation-only for `JointPhyloGroupedGaussianFit`: an explicit species selector, point estimates only (`ci=false`), refusal for `ci=true`, and shared residual mode to match the R model. It does not authorize row admission or merging.
- Every main/GitHub merge needs explicit consent. Unsigned admissions/classifications, shared Laplace-mode changes, DRAC jobs, release actions, and public claims remain gated.
- Do not refresh a single AGENTS snapshot pointer while the multi-lane split is unresolved. Use the coordination board as a routing map, but revalidate every branch/worktree because its current entries are stale.

## Landing State

The pre-write `handoff_gate.sh` result was **FAIL**, with each finding declared here. The handover is intentionally local and unpushed so Cursor must resume from the exact shared worktree path below. Do not mistake the gate failure for completion.

| Artifact / branch | Committed | Pushed | PR | State |
|---|---:|---:|---|---|
| `codex/p1-isdm-current-main-20261019` at the local handover commit (candidate code parent `0961bf4f5`) | yes | no; 11 local-only commits | none verified | CARRIED-OVER; resume in this exact worktree. Push only after explicit instruction. |
| `test/test_gp1_verdict.jl` | no | no | none | CARRIED-OVER; approved edit, focused test passes. |
| `test/test_nb_grouped_upper_warn.jl` | no | no | none | CARRIED-OVER; approved edit, focused test passes. |
| `test/test_weights_byte_identity.jl`, `test/fixtures/weights_byte_identity/cases.jl`, `gen_main.jl`, `main_capture.toml` | no | no | none | CARRIED-OVER; approved provenance guard; 63 cross-process checks skipped due environment mismatch. |
| `.claude/settings.json`, `.cursor/hooks.json`, `.mcp.json` | no | no | none | PROTECTED foreign modifications; never stage or revert. |
| `docs/dev-log/check-log.d/2026-10-19-p1-post-coef-empty-contract.md`, `docs/dev-log/plans/2026-10-19-p1-post-coef-empty-discriminating-contract.md` | no | no | none | PROTECTED unrelated untracked proposals; preserve. |
| Six unmet ledgers listed under Current Working State | no change | n/a | other lanes | PROTECTED; declare only, do not edit. |
| Other local branches reported by the gate (710 unpushed commits) | mixed | no | multiple / unknown | PROTECTED; not inventoried or changed by this lane. |
| This handover document | yes, local handover commit | no | none | CARRIED-OVER with the branch; do not push without explicit instruction. |

FINDINGS-OF-RECORD: none

## Files Created / Modified

Branch paths from `git diff --name-only origin/main...HEAD` at the handover candidate:

- `.github/workflows/true-parity-check.yml`
- `AGENTS.md`
- `docs/dev-log/after-task/2026-10-19-p1-c6-approved.md`
- `docs/dev-log/after-task/2026-10-19-p1-check-auto-receipt.md`
- `docs/dev-log/after-task/2026-10-19-p1-first-checkpoint-validation.md`
- `docs/dev-log/after-task/2026-10-19-p1-isdm-receipts.md`
- `docs/dev-log/audits/2026-10-19-p1-decision-packet-review.json`
- `docs/dev-log/audits/2026-10-19-p1-thirteen-row-source-audit.md`
- `docs/dev-log/check-log.d/2026-10-19-p1-first-checkpoint-validation.md`
- `docs/dev-log/check-log.md`
- `docs/dev-log/core070/true-parity-latest/GATES.md`
- `docs/dev-log/core070/true-parity-latest/behaviour-equivalence.json`
- `docs/dev-log/core070/true-parity-latest/c6-approved-decisions-2026-10-17.toml`
- `docs/dev-log/core070/true-parity-latest/case-map-assembled.json`
- `docs/dev-log/core070/true-parity-latest/case-map-isdm.json`
- `docs/dev-log/core070/true-parity-latest/case-map-postfit.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-COUNT.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-EXTRA-SOURCE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-MISSING-IN-TRAIT.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-MISSING-SOURCE.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/ISDM-WRAPPER-LAW.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/POSTFIT-SURFACE-check_auto_residual.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/derived/behaviour-equivalence.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/environment-snapshot/Manifest.toml.txt`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/environment-snapshot/Project.toml.txt`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/julia-environment.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/julia-public.err`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/julia-public.log`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/julia-public.tsv`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/julia-run-status.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/julia-src.patch`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/r-env.txt`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/r-public.tsv`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/r.stderr.log`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/r.stdout.log`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/r.time.log`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/retained-r-origin.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/retained-r-run-status.txt`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/run.json`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/runner-snapshots/first_seven_behaviour_J.jl`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/runner-snapshots/first_seven_behaviour_R.R`
- `docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour/raw-2026-10-19-main/runner-snapshots/first_seven_behaviour_derive.py`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-COUNT-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-EXTRA-SOURCE-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-MISSING-IN-TRAIT-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-MISSING-SOURCE-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/isdm/cases/CORE070-ISDM-WRAPPER-LAW-PAIRED-CONTROL.json`
- `docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-WAVE7-CHECK-AUTO-RESIDUAL.json`
- `docs/dev-log/core070/true-parity-latest/reverse-gap-decisions.json`
- `docs/dev-log/core070/true-parity-latest/reverse-gap-inputs.json`
- `docs/dev-log/core070/true-parity-latest/reverse-gap.json`
- `docs/dev-log/core070/true-parity-latest/scoreboard.md`
- `docs/dev-log/decisions/2026-10-17-p1-c6-and-c3-c4-contract-approval.md`
- `docs/dev-log/plans/2026-10-19-p1-approved-execution-contract.md`
- `docs/dev-log/plans/2026-10-19-p1-remaining-decision-packet.md`
- `tools/core070_behaviour_receipts.py`
- `tools/core070_isdm_p1_receipts.py`
- `tools/core070_postfit_p1_receipts.py`
- `tools/first_seven_behaviour_J.jl`
- `tools/first_seven_behaviour_R.R`
- `tools/first_seven_behaviour_derive.py`
- `tools/test_core070_behaviour_receipts.py`
- `tools/test_core070_isdm_p1_receipts.py`
- `tools/test_core070_postfit_recount.py`
- `tools/test_true_parity_assemble.py`
- `tools/test_true_parity_c6_classify.jl`
- `tools/test_true_parity_check.mjs`
- `tools/test_true_parity_checkpoint_check.mjs`
- `tools/true_parity_assemble.py`
- `tools/true_parity_c6_classify.jl`
- `tools/true_parity_check.mjs`
- `tools/true_parity_checkpoint_check.mjs`

Candidate working-tree modifications:

- `test/fixtures/weights_byte_identity/cases.jl`
- `test/fixtures/weights_byte_identity/gen_main.jl`
- `test/fixtures/weights_byte_identity/main_capture.toml`
- `test/test_gp1_verdict.jl`
- `test/test_nb_grouped_upper_warn.jl`
- `test/test_weights_byte_identity.jl`
- Protected unrelated paths: `.claude/settings.json`, `.cursor/hooks.json`, `.mcp.json`, `docs/dev-log/check-log.d/2026-10-19-p1-post-coef-empty-contract.md`, `docs/dev-log/plans/2026-10-19-p1-post-coef-empty-discriminating-contract.md`.
- Handover file committed in this branch: `docs/dev-log/handover/2026-10-19-cursor-handover.md`.
- No `AGENTS.md` snapshot or coordination-board pointer was changed.

## Next Immediate Steps

1. Start in the exact candidate worktree, run `git status --short`, confirm branch and HEAD, and classify every row in this document `OWED`, `DONE`, `RETRACTED`, or `PROTECTED`. Do not switch the shared root checkout or other worktrees.
2. Read `AGENTS.md`, `docs/dev-log/coordination-board.md`, `docs/dev-log/handover/2026-10-06-codex-handover.md`, `.unlazy/true-parity-p1-next/GATES.md`, the approved execution contract, the remaining decision packet, and the C6/C3/C4 approval record. Treat the board as stale for current tips; inspect each active lane’s own state before assigning shared files.
3. Reverify the three focused test results and inspect the weights provenance mismatch. Do not regenerate the capture or mark cross-process identity met until the capture is made from the exact candidate environment or a valid matching capture is available.
4. Read the Totoro matrix completion files for `main` and `pr952`–`pr957` before any new suite. Do not launch another full suite while those are active. After they finish, identify which exact candidate ref each result covers and whether any full/core partition remains unpaid. Re-run failed or incomplete gates only under their exact accepted contracts and compute estimates.
5. Continue the remaining P1 rows from the decision packet. Keep row admission, unsigned C6 changes, public claims, merges, releases, and material scope changes behind their stated approval gates. Each new checkpoint needs its own scope contract, ownership claim, model manifest, and executable acceptance ledger.
6. At a verified checkpoint, run `--reverify` on the leaf gates, run applicable full/core/docs checks, obtain the fresh independent reviewer panel, complete Melissa’s reconciliation, and update after-task/check-log evidence. Full P1 remains open until fresh `origin/main` measures C0–C8/X2 and F coverage meet the approved finish line.
7. Do not push this branch without explicit instruction. The uncommitted candidate fixes and local branch commits are visible to Cursor only if it opens this same local worktree; keep this handover and the candidate worktree intact until the user explicitly chooses a landing route.

## Blockers / Open Questions

- The latest main measurements recorded in the P1 ledger are bound to `7cd56b3d0`: C0/C1/C5/C7 met; C2 284/297, C3 6/8, C4 4/8, C6 37 unresolved, C8 38/38, X2 298/317. Candidate `origin/main` has since advanced locally to `882292ac6`; remeasure before making any current-main claim.
- Thirteen P1 scoreboard rows and five F gates remain open according to the latest ledger. Owner-held contracts and C6 work still constrain the route; see the approved remaining decision packet.
- Full/core suite completion and final Rose verdict are not established for the candidate. The Totoro matrix was live during handover.
- The handoff gate remains red for the declared uncommitted/unpushed state and six unrelated unmet ledgers. Do not “clear” other lanes to make the gate green.
- Local GitHub API access failed during the separate R `d = "auto"` check. Its live PR status remains unknown.

## Gotchas & Failed Approaches

- `lane_preflight.sh` reports ownership and does not reserve it. A lease is separate. The repo route script returned “NO manifest” for this worktree, so there is no LOAD-FIRST manifest to rely on.
- The current checkout has many coexisting worktrees and local branches. A path visible in one worktree is not proof it is on `main`; compare refs and file versions explicitly.
- The GP-1 capture is exact-version-specific; Julia `1.10.0` must not inherit `1.10.12`'s optimum.
- Weights capture fingerprints alone were insufficient. Even on matching Julia/macOS/BLAS/bounds settings, changed Manifest and test-project hashes require skipping cross-process bit comparisons.
- A `Pkg.test()` process that reports interruption and no aggregate `Test Summary` is not a pass, even if an outer shell returned exit code zero.
- Never run two Julia test processes in one GLLVM.jl worktree; Totoro is shared and currently has the listed matrix jobs. Pin `OPENBLAS_NUM_THREADS=1`, `JULIA_NUM_THREADS=1` for local focused runs; obey the ≤150-core Totoro cap and estimate before heavier work.
- Do not `git add -A` or `git add .`. The protected config files and untracked proposals listed above are not part of this P1 repair.

## How to Resume

Open Cursor in `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`. Use Julia from the installed toolchain (`julia --version`; the author observed Julia 1.10.0). For the focused checks, set `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1`; weights provenance reproduction additionally used `--check-bounds=yes`. Use the existing Totoro control socket only for read-only status unless a concrete approved compute contract authorizes a run.

The author could not obtain a green global handoff gate because it includes other lanes’ ledgers. This handoff declares every reported blocker and preserves those lanes unchanged. Do not push or merge. First reconcile this document against live refs and classify the ledger.

Paste this in the fresh Cursor session:

```text
Read AGENTS.md and docs/dev-log/handover/2026-10-19-cursor-handover.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
```
