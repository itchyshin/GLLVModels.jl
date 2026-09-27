# Handover to Claude: true-parity-latest lane (2026-09-27)

You are Claude, picking up the true-parity programme for GLLVModels.jl against the latest gllvmTMB. You inherit no chat context. This file, the lane kit and the repository are the record.

## Critical Context

- **Goal.** GLLVModels.jl at true parity with gllvmTMB main pinned at P1 = 9539352f6 (DESCRIPTION 0.7.1, untagged), inside a claim boundary Shinichi signs; then re-pin to the newest main and repeat. Recorded as vault decision D-294 (supersedes T2, frozen at 0.7.0).
- **His decisions (2026-09-27):** pin now and re-pin at milestones; integrated SDM first; port R's semantics where Julia has its own design, keep Julia's design as a documented extra.
- **Plan:** approved, Fable-reviewed: `LOOP/lanes/true-parity-latest/ultra-plan.md` on this branch. Estimate 18 to 22 weeks, or 10 to 12 if column grammar and spatial sit outside the P1 boundary.
- **No parity claim.** 0 of 32 gate-tier rows are done; the 0.7.0 ledger reads 1 of 10 (C7, via #513).

## What Was Accomplished (previous session, 2026-09-26 to 27)

- Merged: #507 NB1 grouped, #508 confint refits, #509 Student-t grouped, #510 beta-binomial, #511 ordered beta, #512 COM-Poisson, #513 parity page (C7), #516 Poisson bootstrap verdict, #517 overnight handover. #501 closed (checked on macOS and Linux). Issues filed: #515. #503 reopened after a title closed it.
- Surveys for the plan: capability inventory of gllvmTMB at P1 against Julia, the parity machinery, and recorded decisions. Their results are summarised in the plan's Context and Workstreams sections.

## Current Working State

| Item | State | Owner |
|---|---|---|
| #514 mixed-family bridge rework | build then review workflow running in the PREVIOUS session (opened in the glmmTMB folder); lease wb-mixed2 on `src/families/mixed.jl` | previous session |
| #515 beta-binomial loglik +7e54 | build then review workflow running in the previous session; lease wc-bb515 on `src/families/beta_binomial.jl`; cause is loggamma cancellation at large phi | previous session |
| Lane kit | `LOOP/lanes/true-parity-latest/` (GOAL, arcs, checkpoint, ultra-plan, HANDOVER) on branch `claude/lane-true-parity-latest`, pushed | you |
| Arcs | none started | you |

Mission control:

| Repo | Branch / main | What shipped | Next by leverage |
|---|---|---|---|
| GLLVModels.jl | main 1385b0490 | fixes above; C7 | A0 additive re-pin; D1 Packet 1 and boundary; iSDM spec |
| gllvmTMB | main 9539352f6 (P1) | nothing from this lane | #1236 bridge rebase (A4a); iSDM bridge route later |

## Key Decisions & Rationale

- Re-pin is additive: the 0.7.0 oracle stays as evidence; many tools refuse a wrong reference, and the pin appears in about 555 tracked files.
- A 0.7.0 receipt counts at P1 only if every file in its source pins is byte-identical at P1 (62 R/src files changed between the pins).
- Receipts must resolve on origin/main: 20 isdm rows currently cite a gitignored path that no longer exists.
- The ledger and its oracle get tracked (`docs/dev-log/core070/true-parity-latest/`, `tools/true_parity_check.mjs`); gitignored `.unlazy/` lost receipts before.
- iSDM twin target: R's public door, non-spatial, Laplace, first order plus predict (R claims no intervals for it). Needs Shinichi's yes in Packet 1.

## Files Created / Modified (this handover)

- `LOOP/lanes/true-parity-latest/GOAL.md`, `arcs.md`, `checkpoint.md`, `ultra-plan.md`, `HANDOVER.md`
- `docs/dev-log/handover/2026-09-27-claude-handover-true-parity-latest.md` (this file)
- Vault (local-only): `memory/DECISIONS.md` D-294.
- No `AGENTS.md` snapshot edit: four GLLVM lanes are live, so a single pointer would orphan the others.

## Landing State

`tools/handoff_gate.sh` (2026-09-27): GATE FAIL, for reasons outside this lane. The six unmet ledgers belong to earlier lanes (`.unlazy/grouped-analytic-20260920/` and `.unlazy/s9c-coverage-448-20260922/`), carried in the shared clone; this lane has none. Its branch `claude/lane-true-parity-latest` is pushed.

## Next Immediate Steps

1. Run `~/shinichi-brain/tools/lane_preflight.sh "/Users/z3437171/Dropbox/Github Local/GLLVM.jl"`, read `AGENTS.md`, and classify each item here as OWED, DONE, RETRACTED or PROTECTED against the current git state (for example, #514 and #515 may have landed by then).
2. Arc D1: draft Packet 1 (13 items, listed in the plan's "What needs Shinichi") and the P1 claim-boundary question, each with a recommendation and a drafted reply. Send it as a message; do not block on it.
3. Arc A0: additive re-pin with a Sonnet builder and a Fable reviewer, using `LOOP/lanes/gllvm-backlog-20260926/fix-build-review.workflow.js` (branch `claude/lane-gllvm-backlog-20260926`). First output: the stale-row count under the carry rule.
4. Ask Shinichi to reopen the Totoro socket before stale-row re-measurements.

## Blockers / Open Questions

- Packet 1 and the P1 claim boundary (Shinichi).
- iSDM build (A1b) waits for #514 to merge (shared `mixed.jl` area; build in new files).
- gllvmTMB #1236 and #1283 are open drafts in conflict.

## Other lanes (PROTECTED)

- NB per-species session: owns the NB2 kernel in `src/families/grouped_dispersion.jl` (Shinichi assigned); will ask for an independent review.
- Gaussian intercepts (#519): `gaussian_intercept.jl`, confint entry points, `cv.jl`, the Normal branch of `formula.jl`, `postfit.jl`.
- auto-d (#518, gllvmTMB #1324): `select_lv` and `src/model_selection.jl` until those merge or close. R also rejects fits with a non-positive-definite Hessian, a known twin difference. Shinichi decides who signs the select_lv row.

## Gotchas & Failed Approaches

- Entering plan mode pauses running builders; finish or park them first.
- Never ask a blocking question while agents run; the overnight loop stalled 15 hours that way.
- A squash merge uses the PR title as its subject; "Fixes #N" in a title closes N. `merge_train.sh` refuses such titles.
- CHANGELOG entries at the same line conflict; place them at distinct positions.
- One Julia process per agent; about 4 cores per lane on this shared Mac.
- Damping can move looked-healthy whole fits; Shinichi accepts better optima, not worse. Always compare whole fits.

## Environment

- Working directory: `/Users/z3437171/Dropbox/Github Local/GLLVM.jl` (main clone); edit only in `~/local-scratch/lanes/GLLVM.jl-true-parity-latest` or new worktrees under `~/local-scratch/`. That worktree's settings deny `git push`; push with `git -C "/Users/z3437171/Dropbox/Github Local/GLLVM.jl" push origin <branch>`.
- Julia: `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia +1.10 --project=. -e 'using Test, GLLVModels; include("test/<file>.jl")'`, and the same with `+1.13`. Never the full suite locally.
- R twin: `/Users/z3437171/Dropbox/Github Local/gllvmTMB` (read with `git show 9539352f6:<path>`).
- Never stage: `.unlazy/` scratch, other lanes' files.

## How to Resume

In a fresh Claude session opened in `/Users/z3437171/Dropbox/Github Local/GLLVM.jl`, paste:

```text
Read AGENTS.md and docs/dev-log/handover/2026-09-27-claude-handover-true-parity-latest.md (on branch claude/lane-true-parity-latest; worktree ~/local-scratch/lanes/GLLVM.jl-true-parity-latest). Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
```
