# Plan: land the true-parity-finish backlog (GLLVModels.jl, with gllvmTMB read-only)

```
🎯 GOAL
Solo platform: Claude Code (this session; Opus 5.5 orchestrating)
Deliverable: #494 merged; the 9 open drafts from 2026-09-25 (#487-#491, #493, #495) each reviewed
  and either merged or returned with a written reason; the two unfinished fixes (#484 two-part
  search, #485 zero-step verdict) as reviewed PRs; the decision packet updated and Rose-audited; a
  handover. NO parity claim (0 of 32 gate-tier rows are evidenced today).
HEADLINE: the merge train. Nine reviewed-but-unmerged PRs are the biggest pile of finished-but-unlanded
  work, and each one that sits collects conflicts in the append-only logs.
IN PARALLEL: #484 fix · #485 verdict fix · reverse-gap classes doc · packet update · Beta realistic cell
DEFER: gllvmTMB writes (#1283 recorder, #1236 bridge, #1323 is the Cursor lane's); multi-day builds
  (bridge spine, phylo latent A14/A15, real-data C4); every Totoro run over 30 min; Project.toml/tags.
DISCIPLINE: verify = test red then green + adversarial review before merge · compute = Mac ≤4 Julia
  threads, Totoro ≤30 min/≤4 cores · closure = ledger --reverify all met + handover landed
```

## Context

The previous session (2026-09-25, plan `precious-enchanting-frog.md`) opened PRs #487 to #495 and
filed issues #484 to #486, then paused at your request. Since then #481 (Gamma), #483 (Beta) and #492
merged. #494 is ready: all 8 Julia shards and Documenter pass; only the known advisory "Frozen R 0.7.0
family smoke" fails. Nothing in #488 to #495 has had an adversarial review yet, and the packet does not
yet list #495's 11 needs-decision issues or live divergences #129 and #131.

## Prior-work sweep receipt

- **Repo git state** → `git status -sb`, `git worktree list`, `git for-each-ref refs/remotes` →
  local checkout sits on a gone branch `docs/claude-handover-20260924`; carried-over branches
  `claude/twopart-mode-search-484` (test + unverified WIP fix), `claude/reverse-gap-classes-20260925`
  (WIP, doc unwritten), lane kit `claude/lane-true-parity-finish-20260925` (PAUSED) → **resume all three**.
- **Twin** → gllvmTMB open PRs listed; #1283 (recorder) blocks S4; #1323 belongs to a Cursor lane →
  **read-only, no writes**.
- **Brain** → `search_notes "GLLVModels.jl true-parity programme gate tier next step"` +
  `grep -in GLLVModels memory/AGENT_LOG.md` (09-24 entries: #474/#475/#478 merged, #479/#480 filed) +
  Mission Control gllvmTMB board (NOW: S4 waits on #1283) → **reuse the 09-25 plan's slice design**.
- **Verdict** → resume, not rebuild. The only new work is review, merge, the two carried fixes, and the packet update.

## Lane pre-flight

`lane_preflight.sh GLLVM.jl`: PLATFORM claude, **9 other live Claude lanes** in this repo. Each builder
claims a lease on its paths first (`tools/lane_lease.sh --claim GLLVM.jl --paths …`). A refused lease
means that slice waits.

## Slices

Model rule: Haiku for mechanical reads, Sonnet for building and bounded review, one Opus for the
load-bearing review. Usage bars not readable from here; the plan keeps Opus to the orchestrator plus one child.

| # | Slice | Member · model · effort | Time | Output | Dep |
|---|---|---|---|---|---|
| 0 | Merge #494 (head-pinned gate on `60105df5b`); confirm main CI starts | Ada · Opus (me) | 5 min + CI | merged | — |
| 1 | Recon: for #487-#491, #493, #495, report mergeability vs new main, conflict files, CI state, review notes in each body | Scout · **Haiku · low** | 15 min | `scratchpad/recon.md` | 0 |
| 2 | Adversarial review of the code-and-science PRs: #493 (adds `src/` Wald intervals), #491 (intercept changed after a separated run whose log was lost), #494's knock-on results | Gauss · **Opus · high** | 60 min | verdict per PR | 1 |
| 3 | Bounded review of tooling/docs PRs #490, #488, #489, #495, #487: citations resolve on `origin/main`, no row claims more than its receipt | Rose-lite · **Sonnet · medium** | 45 min | verdict per PR | 1 |
| 4 | Finish #484 (two-part mode search): re-run the failing test, verify the WIP kernel fix, fail-without-fix check, open PR "Fixes #484" | Julia engineer · **Sonnet · high** | 1-1.5 h | PR | — |
| 5 | #485 central verdict (converged on a zero-length step), TDD, from lane `arcs.md` row S8c | Julia engineer · **Sonnet · high** | 1-1.5 h | PR | lease must not overlap S4 |
| 6 | Reverse-gap classes: re-run `tools/parity_ledger.py`, confirm unclassified REVERSE = 0, write the doc + D8 text; then add #495's 11 decisions and #129/#131 to the packet | **Haiku · low** (tool run) → **Sonnet · medium** (doc, packet) | 1 h | PR(s) | 3 for #495 list |
| 7 | Apply review fixes, rebase, merge train: one PR at a time through `merge-when-green`, order docs-light first, append-only log conflicts hand-resolved | Ada · Opus + the returning builder | 3-5 h wall, mostly CI waits | merges | 2, 3 |
| 8 | D-43 panel on the milestone "backlog landed": 2 Sonnet + 1 Opus fresh reviewers | Sonnet ×2 medium, Opus high | 45 min | verdict | 7 |
| 9 | Close: ledger `--reverify`, after-task, handover, vault log, Mission Control status; Melissa plan-vs-actual | Rose/Melissa · **Sonnet · low** | 45 min | handover, plan-actual | 8 |

Parallel: {2, 3, 4, 5, 6} after 1. Sequential: 7 after 2 and 3; 8 after 7.
Fan-out budget, checkpoint 1: 6 new children (1 Haiku, 4 Sonnet, 1 Opus), ≤5 live. Checkpoint 2 (after
7): the 3 D-43 reviewers plus Melissa. SCOUT SUITABILITY: yes (slices 1 and 6's tool run).

Optional, only if time remains: sibling screen (lane row S7) and the Beta realistic-size cell on Totoro
(≤30 min, estimate stated before launch).

## Estimate

- Active agent work: about 6 to 8 hours across about 10 children.
- Wall clock: **about 8 to 10 hours**. CI dominates: each rebased PR needs a full run of about 30 to 75
  minutes, and the logs force merges one at a time.
- Fits one long session if run as an `/arc-loop`; otherwise it ends with a handover at the merge train.

## Acceptance ledger (`.unlazy/gllvm-backlog/`, git-ignored)

- G0 #494: `gh pr view 494 --json state` → `MERGED`.
- G2/G3 each draft PR: a review verdict file exists and every BLOCKING item is fixed or the PR is left open with the reason.
- G4 #484: `julia --project=test test/test_twopart_mode_search.jl` fails on `main` and passes on the branch; the two-part test files pass.
- G5 #485: the same red/green pair for the new verdict test, plus the family test files that call `_fit_verdict`.
- G6: `tools/parity_ledger.py` reports unclassified REVERSE = 0; the packet lists all 11 #495 decisions plus #129 and #131.
- G7 each merge: `gh pr view N --json state` → `MERGED`, and main CI on the merge commit is green except the advisory R cell.
- G9: `check-after-task.R`, `slop_check.py`, `agent_mention_check.py` pass on the after-task and every PR body.

## Pre-authorisation

PRE-AUTHORISED AFTER APPROVAL: merge #494; worktrees under `~/local-scratch/`; local Julia/R tests;
local commits; push branches and open PRs in GLLVModels.jl; Totoro runs ≤30 min on ≤4 cores.
MERGES (your answer, 2026-09-26): merge reviewed PRs through `merge-when-green` without asking, once
the review verdict has no open BLOCKING item and CI is green apart from the advisory R cell. #491 and
#493 change scientific results, so they come to you first with the review verdict and a drafted reply.
RUN MODE (your answer): `/arc-loop` to completion. Goal on disk at
`LOOP/lanes/gllvm-backlog-20260926/` (lane kit via `lane_launch.sh`), re-read every arc.
MUST STOP: frozen-contract, required-cell or ledger-signature edits; anything written to gllvmTMB;
Totoro over 30 min; `Project.toml`, tags, releases; public API changes; any parity claim.
