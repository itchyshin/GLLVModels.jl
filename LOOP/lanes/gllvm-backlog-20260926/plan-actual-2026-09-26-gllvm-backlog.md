# Plan vs actual: lane gllvm-backlog-20260926 (2026-09-26)

Reconciled by Melissa. Receipts: plan `hidden-meandering-newell.md`; lane kit on branch
`claude/lane-gllvm-backlog-20260926` (GOAL.md, arcs.md, checkpoint.md, backlog-GATES.md,
draft-after-task.md, reviews/); `git log --oneline` on the worktree; `gh pr`/`gh issue` on
itchyshin/GLLVModels.jl. Six axes: scope, evidence/verification, model routing, safety gates,
public claims, handoff state.

## Planned vs actual, per slice

| # | Slice (plan) | Model/effort (plan) | Planned artifact/check | Actual |
|---|---|---|---|---|
| 0 | Merge #494, confirm CI starts | Opus (orchestrator) | merged | Matches. `d4da31544`, advisory R 277/9, within main's own range. |
| 1 | Recon #487-#491,#493,#495 | Haiku, low | `scratchpad/recon.md` | Matches. Feeds S2/S3. |
| 2 | Adversarial review of #493, #491, #494 knock-ons | Opus, high | verdict per PR | Scope grew. The same reviewer also produced `pr-500-verify.md`, `pr-502-correctness.md`, `pr-502-rereview.md`, and `pr-502-ci-crosscheck.md` for the two carried fixes (S4/S5), which this slice did not name. |
| 3 | Bounded review of #490, #488, #489, #495, #487 | Sonnet, medium | verdict per PR | Matches. All five reviewed; `pr-487.md` through `pr-497.md` on file; all five merged. |
| 4 | Finish #484 (two-part mode search) | Sonnet, high, 1-1.5 h | PR | PR #500 opened, then ran through at least two further fix/verify rounds (agents `finish-500`, `fix-500-r1`, `verify-500` in this session's roster). A third fix round is still in flight at reconciliation time. Elapsed time already runs to several times the 1-1.5 h estimate. |
| 5 | Finish #485 (zero-step verdict) | Sonnet, high, 1-1.5 h | PR | Became PR #502. The first design, a global `_fit_verdict` gradient rule, was returned DO_NOT_MERGE (broke 5 tests, false flips on genuine optima). Shinichi chose option (d), per-family verdicts. The reworked head has been re-reviewed; one blocking item is still open (a seed-101 hard assertion) and CI was still pending at last check. In flight (agents `rework-502`, `rework-502b`). Also blocked for about an hour early on by S4's overly broad lease on `test/` and `CHANGELOG.md`, under the old per-session `LANE_ID` scheme. |
| 6 | Reverse-gap classes + packet update | Haiku (tool), then Sonnet (docs) | PR(s) | Matches, with one extra fix round: #496 first went out with a bad citation (#1192), which the Rose audit caught; it was fixed and then merged. Ordinary review iteration. |
| 7 | Merge train | Ada + returning builder, 3-5 h wall | merges | #487, #488, #489, #490, #495, #496, #497 merged. #491 and #493 were correctly withheld under the plan's own must-stop gate (science-changing). #491 now carries a written `RETURNED:` maintainer hand-off, confirmed live via `prstate.sh 491` returning `PR_491_OPEN_WITH_REASON`; the lane's own `backlog-GATES.md` still shows P491 unticked, because the run ledger has not been re-verified since the return comment posted. #493 is still `PR_493_UNRESOLVED` (CI running, no return comment yet), which is a genuinely open item, not a stale one. |
| 8 | D-43 panel (2 Sonnet + 1 Opus) | not stated | verdict | Not started. `D43: pending` in `backlog-GATES.md`. In flight, not judged. |
| 9 | Close: ledger reverify, after-task, handover, vault log, Mission Control status, Melissa | Sonnet, low | handover, plan-actual | `draft-after-task.md` exists but is explicitly a draft with pending fields. Vault log is done: the D-220 amendment is recorded at `memory/DECISIONS.md:8118`, dated 2026-09-26. Handover is not yet on main (`H1: pending`). Melissa's output is this file. In flight. |

## Deviations

### Drift, route to Ada (scope/routing)

1. **Live-agent ceiling breached.** GOAL.md and the plan both cap live agents at 5 (plan wording: "≤5 live," 6 new children per checkpoint, 1 ceiling child). `checkpoint.md` records "USAGE LIMIT HIT ~17:00Z ... about 7 agents live," with four killed mid-task. The fan-out budget was over the stated cap before the usage limit forced a correction; it was not a judgment call within budget. The session self-corrected afterward (resumed at 3 or fewer), but the breach itself needs recording as drift.
2. **Acceptance ledger written after dispatch.** Ultra-plan Phase 2.5 calls for the ledger before dispatch. The plan document itself concedes the ledger "was written after dispatch rather than before." The git log confirms it: `dfdda8d32 true-parity acceptance ledger` and the backlog `GATES.md` both postdate the wave-1 dispatch commits. This is a process gap against a named phase.
3. **Lane-lease granularity caused a real block.** The #484 builder claimed a lease over the whole `test/` directory and `CHANGELOG.md`, which blocked the #485 builder for about an hour (checkpoint.md, S5; after-task §9). The fix, per-agent `LANE_ID` and exact-file leases, was applied mid-session, but the hour was already lost. This belongs to Ada as a lane-mechanics gap, and it recurs (see recurring classes, below).

### Adaptive, no owner action needed

4. **#502 redesign after DO_NOT_MERGE.** The global `_fit_verdict` gradient rule broke five existing tests and flagged genuine optima. The maintainer decision (option d, per-family verdicts) is recorded, reasoned, and re-reviewed. This is what the plan's own G5 verify-then-fix discipline is for.
5. **Sibling screen and mode-search triage beyond the slice table.** The plan flagged the sibling screen as optional, to run only if time remained. It ran, found ordered-beta convergence problems severe enough for two independent verifiers to confirm, and produced issue #501 plus a correction comment about the Linux/macOS optimizer-path finding. The mode-search triage (issues #503, #504) is a direct, in-scope follow-on from the same #479/#480/#486/#484/#500 bug family the plan was already fixing one instance at a time. Both extend the plan's own headline rather than add unrelated scope.
6. **Beta realistic-size cell deferred, not dropped.** The plan listed it as optional. The after-task gives an explicit reason for deferring it: it would duplicate #491's pattern while #491 was still under review. A documented deferral, not a silent drop.
7. **D-220 amendment.** Recorded in the vault with direct quotes from Shinichi's own chat turn (`memory/DECISIONS.md:8118`), matching the plan's context section. Correctly kept out of the GLLVModels.jl repo and logged where it belongs.

### Unclear, route to the domain reviewer (method evidence)

8. **Opus review load may exceed "one child."** The plan states that Opus stays with the orchestrator plus one child. In practice, review work covering #491, #493, #494, #500, and #502 spans at least five separate review documents at the plan's stated "Opus, high" effort tier. The review files carry no model or effort header, so it cannot be settled from the lane kit alone whether this is the orchestrator itself doing the extra reviews (plan-compliant) or additional Opus children beyond the one budgeted (a routing deviation). Needs a read from whoever tracked which agent name ran which review, then a decision on whether it goes to Ada instead.

## In flight (receipts insufficient, not judged)

- PR #500: third fix round in progress (Newton small-step / R1 rule; commit `ba0722c00` still open).
- PR #502: rework re-review flags one blocking item outstanding (the seed-101 assertion), and CI was not yet resolved on the reworked head at last check.
- PR #493: CI running, no return comment yet.
- D-43 panel: not dispatched.
- Handover: not yet landed on main (`H1` gate unmet).

## Recurring drift classes for the monthly ledger

1. **Fan-out ceilings are stated but not mechanically enforced.** The same "≤5 live" language appears in this plan and in GOAL.md, yet the session still ran to about 7 before a usage-limit event forced a correction. A checker, of the kind that already exists for compute targets and agent mentions, would catch this before the limit does.
2. **Lease granularity defaults too broad.** A builder claiming a whole directory, `test/`, or a top-level file, `CHANGELOG.md`, blocks unrelated sibling work. The fix applied here, exact-file leases with per-agent `LANE_ID`, should be the default, not a mid-session patch.
3. **Acceptance ledgers get written after dispatch.** Per the plan's own admission, this is the second time Phase 2.5 has been skipped under time pressure. A lighter-weight ledger template, fillable in a few minutes before dispatch, would remove the excuse for skipping it.

## Verification

`python3 ~/shinichi-brain/tools/slop_check.py` run on this file: clean, 0 findings.

## Resolution of the unclear item (orchestrator, from the dispatch record)

The Opus review load was extra Opus children, not the orchestrator. Opus children dispatched: the #491/#493 science review (wave 1), the #500 first review (wave 2), the wave-3 verification (killed by the usage limit before it produced a result), the #502 correctness review, the #491/#493 verification, and the #500 verification. That is six ceiling children against the plan's "orchestrator plus one child". Tag: drift, owner Ada (routing). Each of these reviews found a defect the builders had not (the #502 regressions, the platform-fragile tests in #493 and #502, the broken step rule in #500), which is the case for the spend, not an exemption from recording it. For the next plan: budget the verification tier explicitly per milestone instead of per checkpoint.
