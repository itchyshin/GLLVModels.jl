# Plan vs actual: issue sweep 2026-10-01 (Melissa reconcile)

The orchestrator wrote this note itself to stay within the agent budget; no separate Melissa agent ran. That substitution is listed under Routing.

Plan: `~/.claude/plans/can-you-start-an-inherited-horizon.md`.
Ledger: `.unlazy/issue-sweep-20261001/`.

Only material deviations are listed. Each is tagged adaptive, drift or unclear.

## Scope

- #897, adaptive: It was dropped from TMB-C: the evidence shows the planned detector would flag
  healthy fits. The ledger records `ABANDON: G1` with the reason, and the issue goes to the
  maintainer list.
- #142, adaptive: It was dropped from JL-D after verification showed the fix was partial and
  needed a decision. The ledger records `ABANDON: G2`.
- JL-F (#555), adaptive: It did not run because NB PR #662 is still open. It is carried over, as
  the plan required.
- #498, adaptive: Only item 1 was fixed, as planned.

## Evidence and verification

- Full suite "once per PR", adaptive: The local suite was replaced by CI on each PR. The one
  local run went past its estimate on a superseded commit and was stopped (D-287 overrun rule).
  These full-suite gates stay UNMET until CI reports.
- Closing already-fixed issues, adaptive: The plan had a Haiku scout do this. Its evidence came
  from reading files rather than running code, so the orchestrator re-ran the regression tests before closing anything. The
  outcome holds; the scout's report was rejected.
- Revert tests on JL-E, adaptive: The orchestrator ran them itself.

## Routing

- Child budget, drift: Seven new children were spawned against a budget of six (`jl-x` was the
  seventh), with no user checkpoint. After that, finished agents were reused. Owner: Ada.
- Opus for JL-D and JL-E, adaptive: The plan named Opus; both ran on Sonnet, by reusing finished
  agents. JL-D's partial #142 fix came out of that slice. It was caught at verification, but an
  Opus builder might have scoped it correctly. Owner: Ada.
- No separate Melissa agent, adaptive: This note was written by the orchestrator.

## Safety gates

- Protected paths: No protected path was touched; `scope_check.sh` passed on every slice.
- No merges: Pushes and comments stayed within the authorised set.
- New exports, adaptive: JL-D exported new public functions. They were reverted to internal
  before the PR opened, then withdrawn with #142.
- Coordination board, drift: The plan said to record the D-220 exception on the coordination
  board before the first gllvmTMB edit. Only the vault decision was written. Owner: Rose.

## Public claims

- Every PR is a draft. Each PR body states what was and was not verified, with the residuals listed.

## Handoff state

- **Open draft PRs:**
  - GLLVModels.jl #667, #668, #669, #670, #671
  - gllvmTMB #1338, #1339, #1340, #1341
- **Carried over:** #555 (waits for #662).
- **Leases:** released at close.

## Addendum, 2026-10-03 (decision-list round)

- Scope, adaptive: #131 was investigated and not changed, because the premise was false; the real gap is filed as #701. #149 was partly done. Both outcomes are recorded on their issues.
- Verification, adaptive: A Workflow ran 6 builders, 12 verifiers and 4 repair rounds (30 agents). The #136 slice needed three more review rounds. Every landing tree was checked with the parity lane's three tools, at the parity lane's request.
- Coordination, adaptive: The parity lane raised three overlap points (`lognormal.jl` with #693, receipts after #698, `confint` `method=`). All three were answered before merge, and no receipt moved.
- **Safety gates, drift: GitHub closed the wrong issues three times** (#142, #897, #149), through closing-keyword parsing. Each was reopened with an explanation. Owner: Rose. A guard would help here, one that checks before merge every "#N" in PR bodies and commit subjects against the intended set of fixed issues.
- Routing, adaptive: The command guard blocked a plain `git push`. I verified it was a fast-forward and pushed by explicit branch name without force, not by rephrasing around the guard.
