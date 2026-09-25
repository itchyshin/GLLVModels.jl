# GOAL: true-parity-finish-20260925 (IMMUTABLE; re-read at the top of EVERY arc)

Read this first, every cycle. After a compaction, re-read THIS, then checkpoint.md, then continue.

## Mission
Close every true-parity gap in GLLVModels.jl that engineering can close without a maintainer
decision, each as a reviewed PR (never merged by the lane). Leave a 32-row gate-tier scoreboard,
one decision packet with drafted replies, and a handover.

## Headline
The gate-tier scoreboard (A-13). It turns "true parity" into a checkable table; every later
sign-off runs from it.

## Invariants
- True parity is NOT reachable this session (C4 blocked on gllvmTMB #1236; phylo rows 5-25 days;
  ~20 maintainer decisions). Never claim it.
- Frozen oracle is gllvmTMB 0.7.0 at b4d5fee64. Never edit the frozen contract, a required-cell
  assertion, or a ledger signature.
- Nothing is written to gllvmTMB (code, PRs or issues).
- Each slice runs in its own worktree off origin/main under ~/local-scratch/. At most 5 agents live.
- Compute: Mac at most 4 Julia threads per lane; Totoro runs at most 30 min on at most 4 cores.

## Authoritative WHAT
The approved plan: ~/.claude/plans/precious-enchanting-frog.md, copied here as ultra-plan.md.

## Definition of done
Scoreboard PR, decision packet, and every engineering slice (A-03, A-04, A-05, A-08..A-12, X-06,
cov-Gamma, Class-A/B audit follow-ups) are either PR-open with tests and review, or recorded with the
reason they stopped. Handover and after-task landed.

## Pre-authorisation
Worktrees under ~/local-scratch/, scoped edits, local Julia/R tests, Totoro runs of 30 min or less,
local commits, branch pushes and PRs in GLLVModels.jl, GLLVModels.jl issues for confirmed defects,
closing GLLVModels.jl issues only with a receipt.

## Must stop for
Any merge; frozen contract/required-cell/ledger-signature edits; anything in gllvmTMB; Totoro over
30 min; Project.toml, tags, releases; public API change; any claim that true parity is reached.
