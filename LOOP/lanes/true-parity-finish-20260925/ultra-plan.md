# Plan: finish the true-parity session (GLLVModels.jl ↔ gllvmTMB)

```
🎯 GOAL
Solo platform: Claude Code (this session; Opus 5.5 orchestrating)
Deliverable: every true-parity gap that engineering can close without your decision is closed as a
  reviewed PR; a 32-row gate-tier scoreboard shows, row by row, where parity stands; every open
  decision sits in one packet with a drafted reply; a handover lets the next session carry on.
HEADLINE: the gate-tier scoreboard (A-13), because it turns "true parity" from a feeling into a
  checkable table, and every later sign-off (B-14) runs from it.
IN PARALLEL: R-library guard (A-04) · docs truth (A-08/A-10/A-11) · truncated-NB2 second-order cell
  (A-03) · reverse-gap classes (A-09) · cross-audit issue triage (A-12) · remaining sibling screen (X-06)
DEFER: multi-day builds (bridge spine A-07, bridge-route first-order rows X-01, D3 item 3 A-06,
  phylo latent A14/A15, real-data workflows C4); every Totoro campaign over 30 min; anything in gllvmTMB.
DISCIPLINE: verify = tests red then green + independent review per PR · compute = Mac ≤4 Julia,
  Totoro ≤30 min per run (≤4 cores) · closure = scoreboard + packet + handover landed; NO parity claim.
```

## Context

You asked to finish true parity this session. A read-only inventory (5 agents, completeness critic,
saved at `/private/tmp/claude-503/gap_inventory.json`) found that the programme defines true parity
precisely, in `docs/dev-log/core070/true-parity-decision-map.md` and the 2026-09-05 programme map:
seven clauses C1 to C7 against frozen gllvmTMB 0.7.0 (`b4d5fee64`), all 32 gate-tier rows evidenced
or signed, and a signed joint note before `Project.toml` leaves 0.3.0.

Measured today: 0 of 32 gate-tier rows are promoted; real-data workflows (C4) are blocked on gllvmTMB
#1236 and need 18 to 28 agent-days; the phylo rows need 5 to 25 days; about 20 decisions are yours
alone. **So true parity cannot be completed in one session, by the programme's own definition.** This
plan does the most that honestly can be done, and says plainly what is left and who holds it.

Already done this session (do not redo): #474, #475, #478 merged; #481 (Gamma, #479) and #483 (Beta,
#480) open for your review; issues #476, #477, #482 filed; holdout diagnoses for NATIVE-06, NATIVE-10
and NATIVE-12 (none is a Julia defect: each is a genuine boundary plus R-side numerics at large
dispersion or ν).

## Slices (≤5 agents live at once; each in its own worktree off `main`; local tests; PR, never merge)

| # | Slice | Member · model | Time | Output | Dep |
|---|---|---|---|---|---|
| 0 | Lane kit: `lane_launch.sh` → `LOOP/lanes/true-parity-finish-20260925/` (GOAL, arcs, checkpoint) | Ada · Opus (me) | 10 min | committed kit | — |
| 1 | A-13 gate-tier scoreboard: one table for A1–A15, B1–B4, C1–C5, D1–D8 (receipt path, signed disposition, or blocking gap id); fixes the 42-vs-32 count | Sonnet · high | 1–2 h | draft PR | — |
| 2 | A-04 R-library guard for the 7 remaining tools + `parity_helpers.jl` fail-closed; list receipts as build-confirmed or provenance-unknown | Sonnet · medium | 1–2 h | PR + tests | — |
| 3 | Docs truth: A-08 capability-status (header, split rows, R-name rows, stale notes) + A-10 parity page + A-11 stale programme docs (status blocks) | Sonnet · medium | 1–2 h | one docs PR | — |
| 4 | A-03 truncated-NB2 second-order cell → per-trait route, finite-dispersion data, remeasure against the frozen oracle | Sonnet · high | 1–2 h | PR + receipt | — |
| 5 | A-09 reverse-gap classes for 91 Julia exports (classes proposed, you sign) + D8 hand-off text | Haiku recon → Sonnet | 1 h | PR (draft) | — |
| 6 | A-12 triage of ~31 cross-audit issues: reproduce against the oracle; comment and close only those verified fixed; list live divergences | Sonnet · medium | 2 h | triage table + comments | — |
| 7 | X-06 remaining sibling screen (truncated-NB2 per-trait, beta-binomial, ordinal, Delta :species, NB1 cov) for the #477/#480 classes; plus the running class-wide audit's findings | workflow, Sonnet | 1–1.5 h | issues / fix PRs | — |
| 8 | Land in-flight work: `fit_gllvm_cov` Gamma fix (agent running) → PR; audit findings → issues | Ada | 30 min | PR, issues | 7 |
| 9 | A-05 realistic-size cell for Binomial-logit (and Beta once #483 lands) on Totoro, D-139 estimate first | Sonnet · medium | 1 h | receipt PR | #483 for Beta |
| 10 | Decision packet (below) as a committed doc with drafted one-line replies | Ada · Opus | 45 min | `docs/dev-log/owed/2026-09-25-true-parity-decision-packet.md` | 1–9 |
| 11 | Review: an adversarial Opus reviewer per code PR; Rose (Opus) on claims across the scoreboard and packet | Opus · high | 1 h | verdicts | 1–10 |
| 12 | Close: refresh scoreboard, after-task, handover, vault log; Melissa plan-vs-actual | Sonnet · low | 45 min | handover + plan-actual | all |

SCOUT SUITABILITY: yes, slice 5's name inventory is mechanical (Haiku).
Estimate: about 6 to 10 hours wall, about 12 agents in batches of ≤5 (batch A: 1, 2, 3; batch B: 4,
5, 6, 7; batch C: 9, 11). May need a handover near the end; the lane kit makes that lossless.

## Decisions for you (the packet; the three that unlock the most first)

1. **The three #323 holdouts (frozen-contract revision).** Diagnosed today; none is a Julia defect.
   Drafted reply: *"approve contract v2 for NATIVE-06/10/12"*, meaning:
   - NATIVE-10: assert the boundary flag and `!converged` (your A6 #11 rule) and a one-sided logLik;
   - NATIVE-12: policy v2, Newton polish on R's own gradient, boundary dispersions held;
   - NATIVE-06: B-full, finite-dispersion data (the interior n = 200 draw from #478) plus the boundary-agreement check.
2. **Ledger sign-off (B-01, B-02, X-11).** One T9 draft PR with every disposition (47 open questions,
   the Ada defaults, the 3 animal rows, 14 AGHQ rows receipted on a later R). Drafted reply:
   *"I will review the T9 ledger PR"*.
3. **Merge #481 and #483** after review. Drafted reply: *"merge #481 and #483 when green"*.

Also in the packet, each with a recommendation and a drafted reply:
- B-03 five second-order holdouts (GP-1, Student free ν, raw Λ, joint Tweedie power, BB φ);
- B-04 re-scope C5 for grouping levels;
- B-05 phylo transport Q1–Q4 and the A14/A15 build;
- B-07 62 forward export gaps;
- B-08 DIFFER rows;
- B-09 a closure rule that actually tests parity;
- B-10 NB2 via AGHQ or restart only;
- B-11 the S4 fallback if gllvmTMB #1283 never lands;
- B-12 `latent()` Ψ through the bridge;
- B-13 R-lane renames;
- X-02 the frozen oracle calls `using GLLVM` (bridge rows need a decision);
- X-03 which oracle build is the authority;
- X-07/X-08/X-09/X-10 dispersion granularity, dispersion SEs, REML, and which families count as "paired";
- gllvmTMB issues to file (NB/truncated-NB2 precision at large dispersion, Student df CI, the NB2 stall on 6/16 datasets);
- Totoro campaigns (D-02 Tweedie screen, D-03 Stage 1 grid, D-04 grids, X-05 full re-campaign), each with a D-139 estimate.

## Out of this session (named owner)

- gllvmTMB lane (read-only from here): #1283 recorder, #1236 bridge, ledger-tool fixes, R-side defects (C-01 to C-09).
- Multi-day builds: bridge spine and first-order bridge rows, D3 item 3, phylo latent rows, real-data workflows.

## Pre-authorisation after approval

PRE-AUTHORISED: new worktrees under `~/local-scratch/` and scoped edits there; local Julia and R
tests; Totoro runs of ≤30 min each on ≤4 cores through the existing socket; local commits; push
branches and open PRs in GLLVModels.jl; file GLLVModels.jl issues for confirmed defects; comment on
and close GLLVModels.jl issues only when a receipt shows them fixed.
MUST STOP: any merge (#481, #483 and every new PR); any edit to the frozen contract, a required-cell
assertion or a ledger signature; anything written to gllvmTMB, including issues there; a Totoro run
over 30 min; `Project.toml`, tags, releases; a public API change (for example X in the confirmatory
fitter); any claim that true parity is reached.

## Verification

- Every code PR: a test that fails before and passes after, the existing files that reach the
  changed code, and an adversarial reviewer (Opus) before the PR opens.
- Parity-touching PRs: the relevant parity files against the frozen oracle `b4d5fee64`, and the
  Totoro reference run where the claim depends on it.
- Scoreboard and packet: Rose checks every row's citation resolves on `origin/main` and that no row
  claims more than its receipt.
- Close: `check-after-task.R`, `slop_check.py`, `agent_mention_check.py` on every PR body; Melissa's
  plan-vs-actual at `docs/dev-log/plan-actual/2026-09-25-true-parity-finish.md`.
