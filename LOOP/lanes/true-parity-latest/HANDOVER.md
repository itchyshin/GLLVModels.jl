# Handover into the true-parity-latest lane (2026-09-27)

## Critical context

Shinichi approved a plan (`ultra-plan.md` here) for true parity of GLLVModels.jl with the latest gllvmTMB. His decisions on 2026-09-27: pin to gllvmTMB main 9539352f6 now and re-pin at milestones; integrated SDM first; port R's semantics and keep Julia's own designs as documented extras. This replaces T2 (frozen at 0.7.0). A Fable review of the first draft found ten problems; all are folded into the plan (additive re-pin, receipt-carry rule, receipts must resolve, tracked ledger, iSDM scope, trimmed Packet 1, 18 to 22 week estimate). No parity claim: 0 of 32 gate-tier rows are done; the 0.7.0 ledger reads 1 of 10 (C7).

## Where to work

Open the new session in the GLLVM.jl folder (`/Users/z3437171/Dropbox/Github Local/GLLVM.jl`). Make code changes only in worktrees under `~/local-scratch/`; this lane's kit is in `~/local-scratch/lanes/GLLVM.jl-true-parity-latest` on branch `claude/lane-true-parity-latest`. That worktree's settings deny `git push`; push from the main clone or ask, as the plan allows pushing branches and draft PRs.

## Carried over from the previous session (now closed)

The previous session (glmmTMB folder) stopped its workflows and closed; the work below is yours. Details and exact resume steps are in docs/dev-log/handover/2026-09-27-claude-handover-true-parity-latest.md.
- #514 rework (mixed-family bridge; lease wb-mixed2 on `src/families/mixed.jl`). iSDM (A1b) must not start until #514 merges.
- #515 fix (beta-binomial loglik +7e54; lease wc-bb515 on `src/families/beta_binomial.jl`). Cause found: loggamma cancellation at large phi.
Shinichi already said "merge #514 and #515 when green" (after a clean fresh review). Merged today: #510, #511, #512, #513 (clause C7 met), #516, #517; #501 closed.

## Other lanes (agreed by message)

| Lane | Owns | Note |
|---|---|---|
| NB per-species (assigned by Shinichi) | `src/families/grouped_dispersion.jl` NB2 kernel | will ask for an independent review of its PR |
| Gaussian intercepts (#519) | `gaussian_intercept.jl`, confint entry points, `cv.jl`, `formula.jl` Normal branch, `postfit.jl` | shares only runtests.jl and CHANGELOG.md appends |
| auto-d (#518, gllvmTMB #1324) | `select_lv`, `src/model_selection.jl` until those PRs merge or close | R and Julia `select_lv` now share semantics; one known difference: R also rejects fits with a non-positive-definite Hessian. Who signs the select_lv row is Shinichi's call |

## First steps

1. Read `GOAL.md`, `checkpoint.md`, `ultra-plan.md` here, and the repo's `AGENTS.md`.
2. Draft Packet 1 and the P1 claim-boundary question (arc D1) with a recommendation and a drafted reply per item; ask in a message, never with a blocking question while agents run.
3. Start A0 (WS0 additive re-pin) with a builder and a Fable reviewer. The machinery to reuse: `LOOP/lanes/gllvm-backlog-20260926/fix-build-review.workflow.js` and `merge_train.sh` on branch `claude/lane-gllvm-backlog-20260926`; the 0.7.0 oracle `check.mjs` is in `~/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/true-parity/` (to be committed as `tools/true_parity_check.mjs`).
4. Ask Shinichi to reopen the Totoro socket (needed for the stale-row re-measurements).

## Gotchas

- Entering plan mode pauses running builders. Finish or park them first.
- A squash merge uses the PR title as its commit subject, so "Fixes #N" in a title closes issue N. `merge_train.sh` refuses such titles.
- Two PRs adding a CHANGELOG entry at the same line conflict; place entries at distinct positions.
- One Julia process per agent; the Mac is shared (about 4 cores per lane).
- Check whole fits main vs branch, not only sites: damping can move looked-healthy fits (Shinichi accepts better optima, not worse ones).
- Receipts under gitignored `.unlazy/` have been lost before; track them.
