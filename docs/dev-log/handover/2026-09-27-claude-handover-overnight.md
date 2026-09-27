# Handover: overnight silent-failure fixes (2026-09-26 to 27)

## Critical Context

The lane ran from 16:10 MDT on 2026-09-26 under the maintainer's overnight rule (D-290): fix fits that fail silently, one family per PR, and merge a fix after an independent review and green CI unless it changes fits that looked healthy. The run was due to end at 05:00 MDT with this handover; the handover is late because a blocking question stopped the loop from about 00:10Z to 15:10Z. No parity claim: the true-parity oracle reads 0 of 32 gate-tier rows done on main.

## What Was Accomplished

- Merged: #509 (Student-t grouped), #508 (profile and bootstrap refits), #507 (NB1 grouped), #511 (ordered beta), #510 (beta-binomial). #512 (COM-Poisson) is in the merge queue on head 74c555202.
- The maintainer's rule for mode-search fixes, given during the run: a fit that looked healthy may move to a better optimum, provided no looked-healthy fit gets worse and every site on the branch is stationary.
- Filed #515 (beta-binomial reports converged at loglik +7e54). Reopened #503 after #507's title closed it. The merge script now refuses titles with closing keywords.

## Current Working State

| PR | State | What it needs |
|---|---|---|
| #512 COM-Poisson kernel | queued, CI running on 74c555202 | nothing; merges when green (maintainer's word) |
| #516 Poisson bootstrap verdict (#504) | review MERGE, rebased to fa4966d56 | maintainer's word ("merge #516 when green") |
| #513 parity page, what parity does not mean | CI green, docs only | maintainer's word ("merge #513") |
| #514 mixed-family bridge kernel | review NEEDS_SHINICHI, own test fails on Linux | rework (Next Immediate Steps, item 3) |

Lane kit: branch `claude/lane-gllvm-backlog-20260926`, `LOOP/lanes/gllvm-backlog-20260926/`. `OVERNIGHT.md` holds the timed log; `merge_train.sh` is the head-pinned merge script; `fix-build-review.workflow.js` is the builder-then-reviewer workflow used for each fix.

## Key Decisions & Rationale

- Whole fits are now compared as well as sites. The damping in these fixes acts at intermediate outer-optimiser steps, so a fit can move even where every site is unchanged at the end (#511: +52 and +9; #510: +7.35 and +6.08; #512: +0.059).
- #514 did not merge because its review found a fit that was healthy on main and ends worse on the branch: with a Normal trait whose sigma is near zero, the absolute gradient test can never pass.

## Landing State

This report and the after-task report land through the closing PR. The lane's branches for #512, #513, #514 and #516 are pushed. CARRIED-OVER: #514 (branch `claude/mixed-bridge-mode-search-503`, head 9261fae9a; resume from its review, `reviews/pr-514.md` in the lane kit, and item 3 below). The shared clone holds a ledger belonging to another lane, `.unlazy/totoro-t4-p6-grid/GATES.md`, that `check-after-task.R` cannot parse; this lane did not touch it.

## Next Immediate Steps

1. Maintainer: #516 and #513 (suggested replies above).
2. #501: run the five macOS-flagged seeds #511 did not check (2002, 2004, 2006, 2007, 2008) and a Linux check; close #501 if they are clean.
3. #514 rework: replace the absolute gradient test with a scale-aware one (for example the Newton decrement), line-search every step after a rejection, correct the Normal Fisher-weight claim in the body, CHANGELOG and code comment, and assert relations in the test instead of main's exact garbage value. Then re-review, including whole fits.
4. #503 next family: NB2 grouped (`_nb_grouped_loglik_site`, default NB2 route, 25 of 187 sites non-stationary); the workflow arguments are in the lane kit as `args-nb2-grouped.json`, for `fix-build-review.workflow.js`. Then Tweedie grouped, then #505 starting with `fit_nb1_gllvm_grouped_cov` (same file, so one at a time).
5. #504: copy #516's pattern to Binomial, NB2 and Gaussian, one per PR.
6. One cleanup PR for the em dashes in the CHANGELOG entries added this week and in #516's test comments.

## Blockers / Open Questions

- GP1 and shared Student-t use the shared generic `_laplace_mode`; changing it needs the maintainer's sign-off.
- #515 needs a reproduction as a committed fixture before a fix; the seeds in the issue are macOS and Julia 1.10 only.

## Gotchas & Failed Approaches

- Never block an unattended loop on a question; post it and keep working.
- A squash merge uses the PR title as the commit subject, so "Fixes #N" in a title closes issue N.
- Two PRs that add a CHANGELOG entry at the same line conflict whichever merges second; an entry placed elsewhere in the list merges cleanly.
- One Julia process at a time per agent on the shared Mac.
- Never message a running agent; it forks a second copy that writes the same worktree.

## How to Resume

Read `LOOP/lanes/gllvm-backlog-20260926/OVERNIGHT.md` on branch `claude/lane-gllvm-backlog-20260926`, then this handover, then #503's latest comment for the family list.
