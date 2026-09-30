# GOAL — overnight run 2026-09-27/28 (IMMUTABLE for the run; re-read at the top of every arc)

Shinichi, 00:10Z: "I am going till 5 am - please keep going autonomously by then." Run until about
11:00Z (05:00 local).

## Mission
Use the night on reversible verification and preparation that makes tomorrow's decisions faster.
Nothing merges; nothing is submitted to DRAC; no new public claims. Finish line: every arc below is
DONE or explicitly BLOCKED with a reason, and MORNING-REPORT-2026-09-28.md says what happened.

## Invariants
- Draft PRs only. Never merge, never release, never submit a DRAC/Totoro job.
- One push per PR per batch (every push restarts CI). Never push a branch whose CI is mid-run unless
  the push fixes that run.
- Local compute only, ≤4 threads per process, ≤4 processes; estimate before any run over a few minutes.
- Other lanes' files (truncated NB2, `_grouped_laplace_mode` root fix) are not ours.
- A surprise that changes scope or a public claim: write it in the morning report and stop that arc.

## Arcs
- A0: #518 CI finishes → push the held handover commit `d40874ac5`.
- A1: Julia vs R binary-ridge gap at n = 120, p = 10 (8/10 vs 1/10 under `bic`): run gllvmTMB on
  the exact Julia datasets, so the two engines see the same data. Record in design/74 T7 (local commit).
- A2: Gaussian grid re-run prep for #519: pre-run test of what trait intercepts change for select_lv
  on uncentred Gaussian data; plan note in the lane kit (local commit).
- A3: CI watch on everything open; fix real failures on our branches.
- A4: Morning report; checkpoint; batch-push the lane kit to #518 once at the end.

## Pre-authorised
Scoped edits in our worktrees, local tests/fits, local commits, one batched push per PR, PR comments
reporting CI or evidence.

## Must stop for
Merge, DRAC submission, new cost beyond local compute, public claims, anything touching another lane.
