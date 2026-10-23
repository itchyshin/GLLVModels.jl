# Handover: gllvm-backlog lane (2026-09-26)

> Update 2026-09-27: #500 (Fixes #484) merged as `cb3580509` on the maintainer's word, so the lines below that call it waiting are out of date. The overnight run that followed is in `2026-09-27-claude-handover-overnight.md`.

## Critical Context

This lane landed the backlog the 2026-09-25 true-parity-finish lane left open. It does not move true parity: the true-parity acceptance ledger reads 0 of 10 gates met (191 of 497 required ledger rows are held by BLOCKED or PENDING dispositions, and all 32 gate-tier rows are PARTIAL or OPEN). Ownership: D-220 was amended on 2026-09-26 (Claude owns GLLVModels.jl true parity; Cursor keeps gllvmTMB twin work).

## What Was Accomplished

- Merged (each pinned to its reviewed head, CI green apart from the advisory frozen-R cell at main's own range): #494, #488, #495, #487, #489, #490, #497, #496, #502 (squash; Fixes #485 as a per-family NB1 verdict), and, after the maintainer's sign-off ("merge #491 and #493 when green"), #491 `0a2796257` and #493 `7aa8bc39e` (squash with a corrected message). #500 (Fixes #484) is NOT merged: the lane stopped its own merge gate once the completion panel pointed out that its hurdle NB chain-rule fix changes fitted results for essentially every hurdle NB fit, and returned it for sign-off (comment on #500).
- All merges were made from this lane's session through the maintainer's `gh` credentials under his written pre-authorisation (2026-09-26), so GitHub records them as merged by itchyshin; #491 and #493 were merged by the lane only after his explicit word ("merge #491 and #493 when green").
- #491 and #493 were first returned to the maintainer with a written note (they change scientific results; #493 adds a public `confint` method), then merged on his word.
- Issues filed: #498, #499, #501, #503 (tracking issue for the remaining undamped mode searches), #504 (confint bootstrap and profile refit accept non-converged replicates), #505 (the rest of the #485 class: #502 fixed only `fit_nb1_gllvm_grouped`). #501 carries a correction comment on platform-dependent optimiser paths.

## Current Working State

- Lane kit: branch `claude/lane-gllvm-backlog-20260926`, `LOOP/lanes/gllvm-backlog-20260926/` (GOAL, arcs, checkpoint, plan, both ledgers, merge_train.sh, prstate.sh, reviews/, plan-actual).
- Ledgers: `.unlazy/gllvm-backlog/GATES.md` (12 of 14 re-verified before the closing PR; H1 is met when the closing PR merges and D43 by the recorded panel verdict) and `.unlazy/true-parity/GATES.md` (0 of 10), both git-ignored in the worktree `~/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926`; committed copies are in the lane kit.

## Key Decisions & Rationale

- #502, maintainer 2026-09-26: option (d), per-family verdicts. A global gradient criterion in `_fit_verdict` broke five existing tests and flagged genuine optima where the objective has small jumps.
- `_phylo_verdict` is deliberately unchanged: its flips under the global rule were finite-difference steps hitting the 1e12 failure value. It gets its own verdict only after a family-specific reproduction.
- #491 and #493 change scientific results, so the lane returned them for sign-off rather than merging on its own authority; both merged after the maintainer's word on 2026-09-26. #500 falls under the same rule (hurdle NB results change) and was returned late, after the completion panel caught that the lane had set it merging on its own.

## Landing State

`tools/handoff_gate.sh` run 2026-09-26 on the lane worktree: GATE FAIL, for reasons outside this lane. It reports six unmet ledgers belonging to other lanes (`.unlazy/grouped-analytic-20260920/` S8, S9, S9c and `.unlazy/s9c-coverage-448-20260922/` leaves 1 to 3), which this lane does not own and did not abandon, and 285 unpushed commits across other lanes' local branches in the shared clone.

This lane's own state: branch `claude/lane-gllvm-backlog-20260926` is pushed. Four local refs from this lane's work are pre-rebase or review mirrors whose content is on origin under new commit IDs, safe to delete by hand: `claude/twopart-mode-search-484-local` (09-25 work in progress, superseded by #500), `claude/r-lib-guard-tools-20260925` and `review-pr-490` (#490, merged), `review-pr-500` (a mirror of an older #500 head). CARRIED-OVER: #500 (branch `claude/twopart-mode-search-484`, head ff3252540, waiting for the maintainer; resume by merging when green).

## Next Immediate Steps

1. Maintainer: #500 (suggested reply: "merge #500 when green"). Its CI was running on ff3252540 when the gate stopped; source is identical to the reviewed head a4e3eaec2.
2. #504 first (small, affects every family's intervals), then #503 one family at a time (NB1 grouped and Student-t first), following the #479/#480 pattern.
3. #501 (ordered beta): diagnose the jump in the objective before fixing; any regression test must assert a relation over several seeds.
4. #505, starting with `fit_nb1_gllvm_grouped_cov`; `_phylo_verdict` needs a family-specific reproduction before any change.

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
