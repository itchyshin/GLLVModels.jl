# Handover: gllvm-backlog lane (2026-09-26)

DRAFT. Items marked PENDING are filled at close from GitHub and the ledgers.

## Critical Context

This lane landed the backlog the 2026-09-25 true-parity-finish lane left open. It does not move true parity: the true-parity acceptance ledger reads 0 of 10 gates met (191 of 497 required ledger rows are held by BLOCKED or PENDING dispositions, and all 32 gate-tier rows are PARTIAL or OPEN). Ownership: D-220 was amended on 2026-09-26 (Claude owns GLLVModels.jl true parity; Cursor keeps gllvmTMB twin work).

## What Was Accomplished

- Merged (each pinned to its reviewed head, CI green apart from the advisory frozen-R cell at main's own range): #494, #488, #495, #487, #489, #490, #497, #496, #502 (squash; Fixes #485 as a per-family NB1 verdict), and PENDING #500 (Fixes #484).
- Returned to the maintainer with a written note, both recommended to merge: #491 (realistic-size Binomial receipt), #493 (per-trait truncated-NB2 Wald CI; adds a public `confint` method; squash with a corrected message).
- Issues filed: #498, #499, #501, #503 (tracking issue for the remaining undamped mode searches), #504 (confint bootstrap and profile refit accept non-converged replicates). #501 carries a correction comment on platform-dependent optimiser paths.

## Current Working State

- Lane kit: branch `claude/lane-gllvm-backlog-20260926`, `LOOP/lanes/gllvm-backlog-20260926/` (GOAL, arcs, checkpoint, plan, both ledgers, merge_train.sh, prstate.sh, reviews/, plan-actual).
- Ledgers: `.unlazy/gllvm-backlog/GATES.md` (PENDING final count) and `.unlazy/true-parity/GATES.md` (0 of 10), both git-ignored in the worktree `~/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926`; committed copies are in the lane kit.

## Key Decisions & Rationale

- #502, maintainer 2026-09-26: option (d), per-family verdicts. A global gradient criterion in `_fit_verdict` broke five existing tests and flagged genuine optima where the objective has small jumps.
- `_phylo_verdict` is deliberately unchanged: its flips under the global rule were finite-difference steps hitting the 1e12 failure value. It gets its own verdict only after a family-specific reproduction.
- #491 and #493 change scientific results, so the lane returned them instead of merging.

## Landing State

PENDING: paste `tools/handoff_gate.sh` output at close.

## Next Immediate Steps

1. Maintainer: sign off #491 and #493 (drafted replies: "merge #491 when green"; "merge #493 when green, squash with a corrected message").
2. #504 first (small, affects every family's intervals), then #503 one family at a time (NB1 grouped and Student-t first), following the #479/#480 pattern.
3. #501 (ordered beta): diagnose the jump in the objective before fixing; any regression test must assert a relation over several seeds.
4. `_phylo_verdict`: build a family-specific reproduction before changing it.

## Blockers / Open Questions

- gllvmTMB #1283 (recorder fix) still blocks the true-parity S4 probe; gllvmTMB is read-only from this lane.
- The gate-tier scoreboard is dated 2026-09-25 and some rows cite PR states that have since changed; refresh before reading its "still open" language as current.

## Gotchas & Failed Approaches

- Never assert one seed's fitted outcome in a test: CI runs Julia 1.10 and 1.13 on Linux, the same seed draws different data on 1.10 and 1.13, and optimiser paths differ between Linux and macOS on the same data. Load committed, hash-verified fixtures and assert relations.
- Give each subagent its own `LANE_ID` and claim exact files, never directories.
- Filter unlazy ledger output for `APPROVAL REQUIRED`; a lapsed approval leaves a stale tick that `--status` trusts. `EXPECT:` is a substring unless written `/regex/`.
- A `pgrep -f` watcher whose command line contains its own pattern never exits.
- Keep live agents at three or fewer on a long run; the shared usage window runs out before context does.
- `merge_train.sh` checks mergeability only at start; after another PR merges, re-check before relying on it.

## How to Resume

Read `LOOP/lanes/gllvm-backlog-20260926/checkpoint.md` on branch `claude/lane-gllvm-backlog-20260926`, then this handover, then the open issues above.
