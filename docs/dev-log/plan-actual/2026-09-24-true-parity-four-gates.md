# Plan vs actual: true-parity-20260924 four paste gates

Reconciler: Melissa (ultra-plan Phase 4.5). Scope: diff the approved plan
(`LOOP/lanes/true-parity-20260924/ultra-plan.md`, `GOAL.md`, `arcs.md`,
`checkpoint.md`) against what actually happened, using the acceptance ledger
(`.unlazy/true-parity-20260924/`), the after-task report, the closeout
handover, and live GitHub state (`gh pr list/view` against
`itchyshin/GLLVModels.jl`, run 2026-09-24). I record deviations; I do not
review implementation correctness and I do not escalate each item myself —
routing is listed for Ada / Rose / the domain reviewer to act on.

Read-only pass: no input file was modified. `gate-check.mjs --status` was run
(not `--reverify`) as instructed, so leaf files below still show whatever
evidence state they were in when I read them; where `--status` reported a
gate as unmet or unchecked I cross-checked the underlying fact independently
(via `gh pr view`) rather than trusting the ledger file alone.

## Summary (counts by tag)

- **Adaptive:** 9
- **Drift:** 2
- **Unclear / needs Rose to close:** 2

Everything tagged drift or unclear is repeated in "Open items for Rose" at
the end, one line each, per the closing-reply format.

## Deviations table

| # | Axis | Planned | Actual | Tag | Owner | Evidence |
|---|---|---|---|---|---|---|
| 1 | Verification | X409 G2: a written pre-run time estimate for the S4 probe, *before* the run (D-139 discipline; plan text: "State an estimate first"). | The `S4_ESTIMATE` line in the receipt was written at 17:06:00Z; the probe ran at 16:59:43Z — estimate written *after* the run. Only a 2–6 min orchestrator brief and a 1.58s define-only smoke preceded it, neither logged as the formal estimate. | **Drift** | Ada (process/routing — D-139 discipline) | `.unlazy/true-parity-20260924/gates/leaf-X409.md` ABANDON note for G2; `after-task §9` |
| 2 | Evidence/verification | Slice table's dependency chain: `V2 → D43 → Z → M` — the D-43 completion panel gates the after-task/handover write-up. | The after-task report and closeout handover were drafted (uncommitted, no PR opened) with D-43 not yet run at all (no panel doc exists anywhere in the lane or closeout tree), and while two of the six leaves (X411 Stage 1 slice #471, X409 S4 receipt #472) are still open with CI in progress. `arcs.md` itself still marks V1, V2, D43, Z, M all `TODO`, contradicting the fact that Z's outputs already exist. | **Drift** | Rose (closeout/claims — order of operations) | `arcs.md` rows V1/V2/D43/Z/M; no `D43`/`completion-panel` file found under either tree; `gh pr view 471/472` both OPEN |
| 3 | Safety gates | Ultra-plan core-gates prose for leaf-X399: "G4 the D1 receipt is written with its verdict. **FAIL → PAUSE, no merge.**" — D1 remeasurement was meant to gate the #399 merge decision. | The operationalized ledger (`leaf-X399.md`) reorders this: G4 = "#399 merged", G5 = "both D1 remeasure receipts on main". #399 was merged (94a7b56f9, 16:47:28Z) *before* D1 was remeasured; checkpoint.md records this explicitly: "a D1 FAIL pauses the D1 claim, not the merge." D1 passed both cells, so no harm resulted, but the gate no longer had the power to block a bad merge. | **Adaptive** (recorded, cites the D1 runbook as authoritative for this sequencing; consequence was benign because D1 passed) — **flagged**, since it silently weakens the "FAIL → PAUSE, no merge" safety property the plan's own prose promised | Rose (confirm no closeout wording implies D1 gated the merge) | `checkpoint.md` DEVIATIONS line; `leaf-X399.md` G4/G5; ultra-plan.md "Acceptance ledger" §leaf-X399 |
| 4 | Evidence/verification | B* leaf G3 ("includes" gate) was meant to prove the merged `runtests.jl` include list is correct. | The check initially measured a stale branch reference instead of the merge result — a false-target risk — caught by a self-run "test of the tests," then corrected to evaluate the actual pushed/merge-bound head before being relied on for any merge decision. Disclosed in after-task §6, §9, and Team Learning ("a gate must measure what lands, not what a stale branch holds"). | **Adaptive** (self-caught before reliance, documented) — worth a spot re-check | domain reviewer (confirm all 4 merged heads still satisfy INCLUDES_OK against `origin/main`, not just the pre-fix branch state) | after-task §6/§9; `leaf-B411.md` G3 (`INCLUDES_OK main=329 merged=331 new=2`) |
| 5 | Safety gates | X410 G2 (plan text): a Totoro pre-run environment check. | Redefined *before any check ran against it*: "no prior oracle build exists, so a gradient_max needs the 45–90 min build, which belongs to the full run" — G2 became an environment-only receipt (dry-run OK, source fetched, archive prepared, toolchain versions), not a gradient number. | **Adaptive** (corrected an incorrect assumption before evaluation, recorded plainly) | Ada (gate design) | `leaf-X410.md` line 12 and ABANDON note; `checkpoint.md` "leaf-X410 G2 redefined..." |
| 6 | Evidence | X410 G4: finite `gradient_max` for all three #323 holdout cells (NATIVE-06/10/12). | Only NATIVE-12 produced a refreshed gradient (5.90e-4). NATIVE-06 never reached its R check (seeded-data hash guard failed on Julia 1.10.12). NATIVE-10 records no `r_gradient_max` (its parity cell passes; only a diagnostic fails to converge). Ledger reports all three honestly with `not_reached`/`not_recorded` rather than fabricating values. | **Adaptive** (genuine campaign finding, honestly classified, matches the plan's own "met or ABANDONED with a reason" rule) | domain reviewer (NATIVE-06 Julia-version pin, NATIVE-10 missing gradient — both are open follow-ups, not closure blockers) | `leaf-X410.md` ABANDON note; `docs/dev-log/after-task/2026-09-24-totoro-323-track-a-receipt.md` |
| 7 | Scope | Plan text (Members plan-review, BLOCKING #1): `runtests.jl`/`check-log.md` conflicts across the merge chain need a documented procedure; review recommended keeping both dated sections on conflict. | Execution went further than the review's fix: check-log sections were dropped from each individual PR entirely and centralized into `LOOP/lanes/true-parity-20260924/check-log-carry.md`, to land verbatim in one closing PR (X411 leaf G5 explicitly records "check-log clause removed 2026-09-24"). | **Adaptive** (extends the review's own finding, recorded in the leaf gate text and `check-log-carry.md`) | Ada (scope) | `leaf-X411.md` G5; `check-log-carry.md`; ultra-plan.md plan-review item 1 |
| 8 | Model routing | Totoro Track A runbook named Codex (or the maintainer) as executor. | Shinichi explicitly chose "Claude runs it here" (quoted, timestamped) — plan-review flagged this as BLOCKING and resolved it before dispatch; the deviation is recorded in the receipt, after-task, and handover. | **Adaptive** (explicit maintainer decision, fully disclosed in 3 places) | none — closed | `docs/dev-log/after-task/2026-09-24-totoro-323-track-a-receipt.md` "Executor" line; ultra-plan plan-review item 2 |
| 9 | Model routing | B411 slice: "Sonnet · high (promote to Opus if the substrate merge needs a design call)" — an in-place promotion of the same builder. | When B411 found the σ_eps pin-scaling bug, a *separate* independent Opus verifier ("Gauss") was spawned to reproduce and confirm the finding, rather than promoting the B411 Sonnet builder itself. Recorded as "ceiling slot 1/1" in checkpoint.md, matching the plan's "ceiling 0 (promotion only on named triggers)" allowance. | **Adaptive** (arguably stronger than the literal plan — an independent second opinion rather than self-promotion; consistent with the ceiling-slot budget) | none — closed | `checkpoint.md`; `logs/gauss-pin-scaling-verdict.md` |
| 10 | Scope | Slice table named fresh 2026-09-24-dated worktree paths per builder (e.g. `…/GLLVM.jl-tp-399-20260924`). | Builders reused existing clean 2026-09-16 worktrees already checked out on the right branches (`~/local-scratch/gllvm-{delta-disp-a-scaffold,s4-probe-rebase,totoro323-rebase,d3-stage1-rebase}-20260916`). | **Adaptive** (disclosed efficiency reuse, no scope bleed found) | Ada (routing) | `checkpoint.md` line 6 |
| 11 | Merge order | Plan: `#409 → #410 → #399 → #411-harness → Stage 1 slice PR`. | Matches exactly by merge timestamp: #409 15:20:46Z, #410 16:37:27Z, #399 16:47:28Z, #411 16:54:17Z. D1 remeasure PR #470 merged after, at 18:05:19Z. Stage 1 slice (#471) and S4 receipt (#472) remain open. | **Adaptive** (no deviation on the harness-PR order; the two trailing PRs are legitimately still running, not out of order) | none | `gh pr view 409/410/399/411/470` `mergedAt` fields |
| 12 | Public claims | GOAL.md "Definition of done": `#409 #410 #399 #411 + Stage 1 slice MERGED`. | Four harness PRs are merged; the Stage 1 slice (#471) is open with 10/10 CI checks still `IN_PROGRESS` at time of this reconciliation, and the S4 receipt (#472) is open with an expected advisory Frozen-R failure. The after-task and handover both state this plainly (no claim of Stage-1-slice or S4-receipt being merged; "Files Touched" and "Landing State" list #471/#472 by number without a MERGED tag). | **Unclear** — the underlying facts are honestly disclosed (no overclaim), but the closeout was written before its own stated definition-of-done was met | Rose (decide: hold the after-task/handover until #471/#472 land, or explicitly re-scope "done" to "four harness gates merged; two trailing PRs OWED") | GOAL.md "Definition of done"; `gh pr view 471/472`; after-task §4/§10, handover "Landing State" table |
| 13 | Evidence/verification | Acceptance ledger meant to be kept current; after-task §5 states "re-verified with `gate-check --reverify` after every builder return." | `gate-check --status` on the current ledger shows several X-leaf gates still `[ ]`/`EVIDENCE: pending` even though the underlying facts are already true — X409 G1 (#409 is in fact MERGED per `gh pr view`), X410 G1 (#410 is in fact MERGED), X410 G2 (the receipt already carries the `TRACKA_PRERUN` line the check greps for), X410 G5 (no independent check run here to confirm no lane process remains on Totoro). These were never re-run against the ledger file to convert `pending` to recorded evidence. The claim in after-task §5 is literally true for the B-leaves (all 6/6 met, confirmed independently) but does not extend to these X-leaf entries as currently persisted on disk. | **Unclear** — most likely a bookkeeping gap (checkpoint boundary before the reverify pass ran), not a false claim, since the underlying facts check out independently | Rose (run `gate-check --reverify` on X409/X410/X411 leaves before treating the ledger as closed; also independently confirm X410 G5's Totoro-process check, which this reconciliation did not execute) | `.unlazy/true-parity-20260924/gates/leaf-X409.md`, `leaf-X410.md` raw `[ ]`/`pending` markers vs `gh pr view 409/410` `MERGED` |

## Budget and routing

Per the task brief's own count: 4 builders (b399, b409, b410, b411, reused
across both the B-phase and their X-phase follow-on work rather than
re-spawned) + 1 Haiku recon/scout (V1) + 1 Sonnet plan-review pass (dual
persona Rose+Noether) + 1 Opus independent verifier (Gauss, the named
"ceiling slot 1/1") + Melissa (this task) = 7 distinct production children
across the lane, plus the separately-tracked vault workflow for the
`lane_launch.sh` fix (explicitly noted in checkpoint.md as running outside
this lane's fan-out budget, on a separate vault task).

This reconciles against the plan's own accounting: "new production children
5/6 (4 builders + 1 Haiku scout) · ceiling 0 (promotion only on the named
triggers) · plus the plan-review child." 4 builders + 1 scout + 1 plan-review
= 6, at the base cap; Gauss used the single named "ceiling" slot the plan
allowed for a design-call trigger, bringing the total to 7 — within what the
plan itself authorized, not an overrun.

**Live-agent cap (5) — not directly instrumented.** I found no per-agent
start/stop timestamp log, so I cannot certify from artifacts alone that more
than 5 agents were never simultaneously live. Inferred from the arc
ordering in `arcs.md`/`checkpoint.md`: the plan-review pass ran at G0,
before any builder was dispatched (so only the orchestrator + 1 review child
were live then); the 4 builders ran in parallel during the B-phase (4 + the
orchestrator = 5, at the cap, not over); V1 (Haiku) ran after the builders
returned; Gauss was spawned later, during X411/B411 work, by which point
B399/B409/B410 had already merged and presumably exited (the live set at
that point looks like orchestrator + b411 + Gauss = 3). On this evidence, the
cap of 5 does not appear to have been exceeded, but this is an inference
from sequencing, not a measurement. The system reminder for this task lists
the currently-live agents as `main, b399, b409, b410, b411` — 5, exactly at
the cap, consistent with the builders being kept alive/reused rather than
torn down between B and X phases.

## Gates changed after planning

Four items were named explicitly in the task brief; all four are addressed
in the Deviations table above with fuller evidence:

- **X410 G2 redefined before the run** (row 5) — adaptive, corrected a wrong
  assumption (no prior Totoro oracle build existed) before any check ran
  against it.
- **B\* G3 changed to evaluate the merge result** (row 4) — adaptive but
  flagged: this was a real verification gap (checking a stale branch instead
  of the merge target) that existed until a self-run "test of the tests"
  caught it. Recommend Rose spot-check the four merged heads against
  `origin/main` directly rather than trusting the historical pass alone.
- **X411 G5 check-log clause removed** (row 7) — adaptive, extends the
  plan-review's own BLOCKING finding about `runtests.jl`/`check-log.md`
  merge-chain conflicts.
- **X409 G2 and X410 G4 ABANDONED** — two different kinds of deviation
  under one heading:
  - X409 G2 (row 1) is **drift**: the D-139 "estimate before you run" rule
    was violated in sequence (estimate written after the probe ran), even
    though it was then honestly disclosed rather than backdated or hidden.
  - X410 G4 (row 6) is **adaptive**: a genuine, honestly-classified campaign
    result (2 of 3 holdout cells did not produce the target number, for
    named, distinct reasons), not a process violation.

## Open items for Rose

1. **Drift — X409 G2, the S4 probe's D-139 pre-run estimate was written
   after the run, not before** (16:59:43Z run vs 17:06:00Z estimate line).
   Disclosed honestly in the ledger's ABANDON note and after-task §9, but
   the underlying D-139 discipline was not followed. Evidence:
   `.unlazy/true-parity-20260924/gates/leaf-X409.md`.
2. **Drift — the after-task and handover were drafted before the D-43
   completion panel ran at all**, and before two of the six PRs in scope
   (#471 Stage 1 slice, #472 S4 receipt) merged; `arcs.md` still marks
   V1/V2/D43/Z/M as TODO. This inverts the plan's own dependency order
   (`V2 → D43 → Z → M`). Recommend either running D43 before these documents
   are finalized/committed, or explicitly re-scoping what "closeout" means
   at this checkpoint and saying so in the documents themselves. Evidence:
   `arcs.md`; absence of any D-43/completion-panel artifact; `gh pr view
   471/472` (both OPEN).
3. Unclear (row 12): the closeout's own definition-of-done (per GOAL.md)
   is not yet met — Stage 1 slice and S4 receipt are still open PRs. The
   documents are honest about this, but Rose should decide whether to hold
   publication of the after-task/handover until they land, or amend the
   claimed scope.
4. Unclear (row 13): several X-leaf ledger gates (X409 G1, X410 G1/G2/G5)
   show `pending`/unchecked on disk despite the underlying facts already
   being true (PRs independently confirmed MERGED via `gh pr view`). Run
   `gate-check --reverify` on the X409/X410/X411 leaves, and independently
   confirm X410 G5 (no lane process left on Totoro) before treating the
   ledger as closed.
5. Flagged (row 3, not drift but worth a second look): #399 merged before
   its D1 remeasurement, reversing the ultra-plan's own gate prose ("FAIL →
   PAUSE, no merge"). D1 passed, so no harm occurred, but confirm no
   closeout wording claims D1 gated the merge decision.
6. Flagged (row 4): spot-check that all four merged branch heads still pass
   `INCLUDES_OK` against `origin/main` directly, since the check that proves
   this was itself found measuring the wrong target earlier in the lane.
7. Carried from checkpoint.md (not independently verified here): B399/X399
   pinned one offset test (`delta_gamma`) to `:shared` because the new
   `:species` default reached a different optimum — checkpoint.md says
   "FLAG to Shinichi: the flip can change convergence." This is recorded in
   after-task §10 as a residual but should be surfaced to Shinichi
   explicitly, not left inside a residuals list, per this repo's own
   "surface review touchpoints" rule.
