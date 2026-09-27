# GOAL: true-parity-latest (IMMUTABLE for the run; re-read at the top of every arc)

## Mission
GLLVModels.jl at true parity with gllvmTMB pinned at P1 = 9539352f6 (0.7.1 candidate), inside a claim boundary Shinichi signs. Every P1 capability inside the boundary gets a scoreboard row with a receipt made at P1, or a carried receipt whose source files are byte-identical at P1. Everything outside the boundary closes by a signed disposition. Then re-pin to the newest gllvmTMB main (P2) and repeat for what is new.

## Headline
Integrated SDM through R's public door (`gllvmTMB(..., family = isdm_sources(...))`): non-spatial, Laplace, first order plus predict, scoped exactly as R's own ledger claims it. Port R's semantics; Julia's SourceCovariance stays as a documented Julia-only extra.

## Invariants
- No parity claim until the tracked ledger (`docs/dev-log/core070/true-parity-latest/GATES.md`, oracle `tools/true_parity_check.mjs`) reverifies all met.
- The re-pin is additive: the P0 (0.7.0, b4d5fee64) oracle stays as historical evidence.
- A P0 receipt counts at P1 only if every file in its source pins is byte-identical at P1.
- Lanes: stay out of `grouped_dispersion.jl` (NB per-species lane), the #519 files (Gaussian lane), and `select_lv` / `model_selection.jl` (auto-d lane, until #518 and gllvmTMB #1324 merge or close).
- At most 3 agents live, at most 2 builds at once; one Julia process per agent; exact-file leases; no agent messages another.
- Tests assert relations on literal or hash-verified data, never one seed's outcome; whole-fit checks main vs branch.

## Authoritative WHAT
`ultra-plan.md` in this folder (the approved plan, 2026-09-27). Detail wins there; this file wins on what must never be lost.

## Definition of done
C0 to C8 and the F-gates reverify MET from tracked files on origin/main, the negative controls fail as expected, and a handover is on main.

## Pre-authorisation (Shinichi approved the plan, 2026-09-27)
Worktrees under ~/local-scratch/; local Julia and R tests; local commits; pushing branches and opening draft PRs in GLLVModels.jl and gllvmTMB; Mac at 4 cores per lane; Totoro <=30 min on <=4 cores once its socket is back; kohaku <=8 vCPU for short checks; new public API mirroring an R export at P1 under R's name (merged only on his word).

## Must stop for
Every merge (his word, batchable); releases, tags, Project.toml; public parity claims; the shared `_laplace_mode`; DRAC jobs (estimate first); runs over 3 hours; another lane's files; case-map classifications (he signs).
