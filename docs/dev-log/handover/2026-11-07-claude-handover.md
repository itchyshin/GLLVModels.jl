# Session Handoff: R–Julia P1 parity

2026-11-07. From Codex to Claude. The user asked to pause the goal and hand over. The Codex goal is paused, not complete. This handover is committed locally on the review branch and is not on a remote.

## Critical context and goal

You are Claude, taking over the entire R–Julia parity goal. Rehydrate before acting: read `AGENTS.md` and its canonical contract, inspect the coordination board and all current lane pointers, run lane preflight if available, inspect live Git state and classify items OWED, DONE, RETRACTED or PROTECTED. Preflight is not an ownership lease. Do not orphan other lanes. Preserve other owners' work and keep R `gllvmTMB` read-only at reference commit `9539352f66f2db2cc26b1c393e67212a359b60c9`.

The goal is true P1 parity on exact origin/main: C0 through C8 and X2 all MET, required F-capability coverage and negative controls verified, independent completion review, reconciliation, validated after-task report and final handover on main. API shape, ledger counts, fixture replays, diagnostic tests and CI alone do not prove parity. Retain signed failures, tolerances and optimizer stopping rules. Full bootstrap-Sigma suite attempts at `b11624712` were incomplete, not passes. The user approved scoped report validation while leaving historical ledgers untouched. Keep the 90-minute per-run cap and obtain authorization for capture, workflow changes, row promotion, push and merge. The current review branches must stay unpushed and unmerged unless the user changes that instruction.

| Repo and lane | Current state | Next work by leverage |
|---|---|---|
| POST-COEF review | Local commit `e062c016c1b2cc9ae860333c5d4ce760b1c2d958`; unpushed, unmerged | Resolve numerical admission and verify the full authenticated transaction before considering new evidence. |
| F-ZI control | Separate local worktree, uncommitted test and log | Reconcile the candidate with the F contract and its scoped evidence. |
| P1 on main | Incomplete; recheck live main and all gates | Continue the remaining C, X2 and F obligations without assuming this work promotes a row. |

## What was accomplished

PR #1095 was merged by user instruction. The separate POST-COEF branch registers the current-format oracle and prepares evidence verification and health collection. Capture-only gate `b53f8221a6fa78a3efaf57667b51b5acb32aff0d` deliberately has an unbound launcher; `17f74e36803cb568835970b10592440698b49e78` binds its caller and `e062c016c1b2cc9ae860333c5d4ce760b1c2d958` adds the reviewed verifier hardening. At `e062c016`, `python3 -B -m unittest discover -s tools -p 'test_core070_postcoef_live_ref.py'` passed 43 tests in 56 seconds. It is not a full suite or complete live transaction.

A fixed-point F-ZI shared-dispersion negative control was independently reviewed and tested in a private source-matched environment at `147dd3be22eb91befdc17ab2b497d3bae88d16df`, Julia 1.10.12, one Julia and BLAS thread. Four testsets passed 22, 5, 40 and 16 assertions. This used Julia fits and retained R fixtures, not a new R capture; it tests one selected shared dispersion, not a refit or the full F gate. The same patch is now uncommitted in the separate F-ZI worktree and has not been rerun there.

## Working state and risks

- POST-COEF checkout: `/tmp/p1-postcoef-fresh-committed-20261106`, branch `codex/p1-postcoef-health-unbound-main-20261106`, HEAD `e062c016c1b2cc9ae860333c5d4ce760b1c2d958`. The handover is committed as a child of that source commit. This `/tmp` checkout and all local-only files are fragile.
- F-ZI checkout: `/Users/z3437171/.codex/worktrees/p1-zi-shared-dispersion/p1-postcoef-fresh-committed-20261106`, branch `codex/p1-zi-shared-dispersion-control-20261107`, HEAD `147dd3be22eb91befdc17ab2b497d3bae88d16df`. Intended uncommitted changes: `test/test_zi_twin.jl` and `docs/dev-log/check-log.md`. The modified `.claude/settings.json` and `.cursor/hooks.json` are unrelated protected changes: do not stage, revert or overwrite them.
- Original checkout `/Users/z3437171/.codex/worktrees/p1-postcoef-source-current/GLLVM.jl` also has protected dirty settings and hooks. Other lane branches and PRs have independent owners; consult `docs/dev-log/coordination-board.md` and current state before touching them.
- The signed Julia gradient limit is `< 1e-5` in optimizer coordinates; record full-coordinate gradient as diagnostic. The user later applied the fit-health rule to POST-COEF, which the original 2026-10-21 ruling excluded. The Hessian coordinate and reliable numerical admission require review.
- A synthetic exact binary64 matrix exposes a false positive: the pinned floating Cholesky check reports positive definite although its exact determinant is negative. The live comparison function accepts a fabricated record with this classification. This does not prove an actual fit or receipt is affected. A private fail-closed prototype holds preliminary PASS as INCOMPLETE, passes five controls and preserves 18 P0 verifier controls, but it is not installed and is not a numerical solution. See `exact-matrix-prototype/` and `fail-closed-health-prototype/` under the private evidence root.
- Full authenticated launcher and evidence positive controls, objective error bounds, curvature accuracy, near-singular classification and same-user/object-store assumptions remain unresolved. The private numerical design is `optimizer-coordinate-ruling/numerical-admission-design-20261107.md`. No row was promoted, no fresh capture is authorized, and fixed-effect priors as well as ridge priors exist; do not generalize the no-prior diagnostic to prior-enabled models.

## Landing state and recovery

The handoff gate ran and exited 1: six historical `.unlazy` acceptance ledgers have unmet gates, alongside unpushed branch work and the then-untracked handover. The user specifically authorized leaving historical ledgers untouched. Do not clear or mark them abandoned merely to pass a global check. Full and core Julia suites and after-task closure remain incomplete. This handover is committed locally and not pushed. Do not claim the gate passed.

| Artifact | State | Why and resume |
|---|---|---|
| POST-COEF branch at `e062c016` | CARRIED-OVER, committed local only, no PR | User requires unpushed and unmerged; `cd /tmp/p1-postcoef-fresh-committed-20261106 && git status --short && git log -1` |
| F-ZI branch at `147dd3be` | CARRIED-OVER, two intended uncommitted files | Local candidate pending integration review; `cd /Users/z3437171/.codex/worktrees/p1-zi-shared-dispersion/p1-postcoef-fresh-committed-20261106 && git status --short && git diff -- test/test_zi_twin.jl docs/dev-log/check-log.md` |
| Handover | CARRIED-OVER, committed local only | User requires no push or merge; inspect this branch and the local recovery copy. |
| Six historical ledgers | PROTECTED, existing unmet state | Scoped validation approved; preserve ownership and do not edit them. |
| Private evidence, prototypes, prior branches | CARRIED-OVER, local only | Inspect their exact source and limitations before use; do not publish raw evidence. |

Evidence root: `/Users/z3437171/.codex/worktrees/p1-postcoef-receipt-integration/evidence-cd358290/postcoef-fit-health-followup-20261106`. Full-history local bundle `handover-p1-e062c016-20261107.bundle` was verified and includes the POST-COEF branch and two earlier review branches; SHA-256 `858a5d408f30e18b6273394b95989ad114d45d134b825e585634c928fef9bcf4`. The F-ZI intended two-file patch is `zi-control-wip-20261107.patch`, SHA-256 `ab7f6d92e43fa14f9f1b2c9b301f6fcd2445a3e6305429cf30a95480a79e2ba7`; it does not include the protected settings changes. Verify the hashes and base before recovery. For a new unused directory, `git clone --branch codex/p1-postcoef-health-unbound-main-20261106 <absolute-bundle-path> <new-directory>` recovers the committed POST-COEF branch; the uncommitted handover and F-ZI patch must be copied separately. Do not overwrite an existing worktree.

FINDING-OF-RECORD: the pinned health classifier admits a synthetic indefinite matrix and the mutable checker pre-execution risk was addressed by the local verifier candidate. These are recorded in the private evidence root and local branch. No vault distillation note was created; do not claim the handoff distillation gate passed or that these findings are indexed in the vault.

## Files created or modified

The POST-COEF branch differs from base `147dd3be22eb91befdc17ab2b497d3bae88d16df` in these tracked paths:

- `docs/dev-log/after-task/2026-11-06-postcoef-preload-evidence-path.md`
- `docs/dev-log/check-log.md`
- `docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json`
- `docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build-totoro-20261104.json`
- `test/parity/core070_pin.jl`
- `test/parity/fixtures/postcoef_old_target_row.json`
- `test/parity/test_core070_pin.jl`
- `test/parity/test_current_oracle_registry.py`
- `tools/core070_family_p1_receipts.py`
- `tools/core070_json_semantics.py`
- `tools/core070_postcoef_depot_check.py`
- `tools/core070_postcoef_fit_health.R`
- `tools/core070_postcoef_fit_health.jl`
- `tools/core070_postcoef_fit_health.py`
- `tools/core070_postcoef_gate_launcher.py`
- `tools/core070_postcoef_live_ref.py`
- `tools/core070_postcoef_offline_check.py`
- `tools/core070_postcoef_ref_check.mjs`
- `tools/core070_postcoef_ref_compatible.py`
- `tools/core070_postcoef_transition.py`
- `tools/core070_postfit_p1_receipts.py`
- `tools/core070_postfit_policy_batch.R`
- `tools/core070_postfit_policy_batch.jl`
- `tools/core070_ref_bytes.mjs`
- `tools/core070_verify_postfit_policy_batch.py`
- `tools/test_core070_postcoef_gate_launcher.py`
- `tools/test_core070_postcoef_live_ref.py`
- `tools/test_core070_postcoef_ref_check.mjs`
- `tools/test_core070_postcoef_transition.py`
- `tools/test_core070_ref_bytes.mjs`
- `tools/test_postcoef_contract_consistency.py`
- `tools/test_postcoef_fit_health.py`
- `tools/test_postcoef_fit_health_jl.jl`
- `tools/test_postcoef_fit_health_r.R`
- `tools/test_postcoef_health_contract.py`
- `tools/test_postcoef_historical_preservation.py`
- `tools/test_postcoef_preload.py`
- `tools/test_postcoef_receipt_integration.py`
- `tools/test_postcoef_unheld_contract.py`
- `tools/test_true_parity_check.mjs`
- `tools/true_parity_assemble.py`
- `tools/true_parity_check.mjs`

Additional local paths: this handover; the F-ZI worktree's `test/test_zi_twin.jl` and `docs/dev-log/check-log.md`; the evidence bundle and WIP patch identified above. The original draft is preserved privately as `claude-handover-stale-draft-20261107.md`. Do not rely on its stale status and commands.

## Next immediate steps

1. Rehydrate and reconcile all lanes and live main, then classify each listed item. Keep the paused goal and explicit no-push/no-merge instruction visible; follow the user's direction on resumption.
2. Review the numerical admission design and the synthetic counterexample. Preserve all failed and incomplete evidence. Do not use the temporary hold prototype as proof that numerical health is fixed.
3. Review the F-ZI candidate against its actual signed F contract and retain its narrow fixed-point interpretation. Run further checks only when justified, estimate compute first, and do not duplicate an active process.
4. Address the remaining authenticated capture and P1 gates on exact source, subject to their explicit approvals. Obtain independent review, required full-suite evidence, after-task audit and Rose verdict before any completion claim.

## How to resume

From the POST-COEF checkout in the user's authenticated terminal, start a fresh Claude session:

```sh
claude "Read AGENTS.md and docs/dev-log/handover/2026-11-07-claude-handover.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps."
```

Claude may need access to the local Julia and R toolchains for future live checks; verify availability and credentials rather than assuming them. The current private evidence and unpushed commits are not available from GitHub. No automatic session launch is implied.

Read AGENTS.md and docs/dev-log/handover/2026-11-07-claude-handover.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
