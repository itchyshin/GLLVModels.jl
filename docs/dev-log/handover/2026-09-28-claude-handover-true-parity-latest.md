# Handover: true-parity-latest lane, 2026-09-28 (overnight run)

Goal: GLLVModels.jl at true parity with gllvmTMB main pinned at P1 = 9539352f6 (0.7.1); boundary D-295 (temporal and phylo latent inside; column grammar and spatial outside). Status: IN PROGRESS, paused at the maintainer's gates (merges, signatures, Packet 2 rulings). Not done: C0 to C8 are not MET on origin/main.

## Where truth lives

- Lane branch `claude/lane-true-parity-latest`, kit `LOOP/lanes/true-parity-latest/` (in the lane worktree `~/local-scratch/lanes/GLLVM.jl-true-parity-latest`): `GOAL.md`, `checkpoint.md` (resume pointer), `morning-report-2026-09-28.md` (full state, per-PR entries, CI table), `packet-2-draft.md` (all open decisions with replies to paste), `reviews/` (every independent review), `scripts/` (refresh and merge-train scripts).
- Ledger and checker: `docs/dev-log/core070/true-parity-latest/GATES.md`, `tools/true_parity_check.mjs` (hardened in #561), assembler `tools/true_parity_assemble.py` (#589).

## State on origin/main at hand-over

- Merged overnight: #548 (chibar2_pvalue / variance_lrt twin), #531 (extract_latent_scores twin).
- Gates on main: C0 NOT MET (default pin P0), C7 MET, the rest not measurable until #533 (case map, signature pending) and #589 (scoreboard, reverse gap) land.

## Open PRs, in merge order

1. Renewed word needed (changed after the landing word to fix CI failures present at the reviewed head): #543, then #563; #546, then #558; #556; #547.
2. Ready for word: #557, #561, #576, #581; the A3 re-measurement stack #567 -> #569 -> #571 -> #579 -> #584 -> #586 / #587 -> #589; specs #525, #535, #545.
3. Signature: #533 (case-map classes), text drafted in the morning report.

A3 result: 52 of 306 required 0.7.0 rows carry real R-vs-Julia numbers at P1 (none before). Most of the remainder waits on Packet 2 rulings or new comparison cases. C3, C4, C5 select no rows until the realistic-size, real-data and grouping-level rows proposed in Packet 2 Part A exist.

## Lessons

- A review verdict is not CI: check the current head's check runs exist and pass. A PR that conflicts with its base gets no pull_request CI at all. The P1 twin job enrols any .jl file whose text contains the tag string, comments included.
- Refresh after every merge (CHANGELOG and check-log conflict); one PR per train, pinned to a verified head.

## Resume

Read `GOAL.md`, `checkpoint.md`, `ultra-plan.md` in the kit; then the morning report; continue from the maintainer's replies.
