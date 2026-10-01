# Session Handoff: truncated-NB2 mode jump, r intervals and follow-ups

Meta: 2026-09-30, 17:45 UTC · from Claude Code (Opus 5.5) · lane closed, nothing in flight.
After-task report: `docs/dev-log/after-task/2026-09-30-truncnb2-mode-jump-closeout.md`.

## Critical Context

1. **Everything from this lane is merged.** #601, #613, #621, #627, #605 and #607 are on `main`. `main`'s CI and Documenter on the last merge (`7e25cbb51`) completed green. Nothing is owed from these PRs.
2. **The dispersion-boundary rule for every NB route (maintainer's decision, 2026-09-29):** r below 1e-6 means `converged = false`; r above 1e6 (the Poisson limit) only warns. For NB1 the ends are reversed (its Poisson limit is phi -> 0). This was reached after gllvmTMB 0.7.1 was shown to put one trait's phi above 1e6 on 4 of 5 ordinary per-trait draws. Do not flip truncated NB2 back to flagging the upper end without the maintainer; the reasoning is in #627's PR body and §3a of the report.

## What Was Accomplished

- #601: the grouped mode search no longer 2-cycles (truncated NB2, NB1, truncated and censored Poisson). The truncated-NB2 objective is smooth in log r (+2.1e-8 / -2.7e-8 for a 1e-5 step, was -5.5e-4 / -1.5e-8).
- With #601, the #581 fixture draw-104 r intervals are sane: Wald [0.034, 0.530] (was [0.131, 0.136]), profile upper 0.331 with an open lower end (was `:failed`).
- #605 and #607: an open lower profile end on a log-scale family parameter is reported as 0 (`:profile`), found by a floor check before the search (draw 104: 583 s to 20.6 s).
- #621: truncated-NB2 fits with r below 1e-6 report not converged; above 1e6 warn.
- #627: the per-trait truncated-NB2 fitter restarts traits stalled at the Poisson limit and now matches gllvmTMB on the two draws where it fell 0.35 and 2.0 log-likelihood units short.
- #613 (and #634 from another lane): R's end-of-fit gradient is recorded, not a gate, in the truncated-NB2 parity and formula cells.

## Current Working State

- Working: all of the above, on `main`.
- In progress: none.
- Not working / blocked: none from this lane.

## Key Decisions & Rationale

- Upper-end dispersion rule: see Critical Context 2.
- The grouped mode-search guard is limited to backtracking families, so NB2, Beta and Gamma grouped `getLV` are bit-identical; `laplace.jl` was not touched.
- #621's test checks the verdict helper directly and wires each fitter with r started at the boundary and zero iterations, because where a fit ends is machine-dependent (a 10^13 count gave r = 3.1e-46 on macOS and r = 0.98 on a Linux runner).
- Merges used a hand-rolled gate keyed on `status == COMPLETED` for the exact head SHA, with `--match-head-commit`; never `--auto` (auto-merge is off on this repo, so `--auto` merges at once).

## Landing State

`tools/handoff_gate.sh` on the shared checkout reports many other lanes' unpushed branches and 12 unrunnable `.unlazy` ledgers; none belong to this lane and none were touched. This lane's rows:

| Artifact / branch | Committed | Pushed | PR | State |
|---|---|---|---|---|
| `claude/fix-truncnb2-mode-jump` `b39f97611` | y | y | #601 merged `e794b4cb2` | LANDED |
| `claude/truncnb2-smoke-record-gradient` `30e1980f1` | y | y | #613 merged `d337a2302` | LANDED |
| `claude/truncnb2-dispersion-boundary` `0f981d216` | y | y | #621 merged `be25621c5` | LANDED |
| `claude/truncnb2-pertrait-boundary-restart` `203ccb0b0` | y | y | #627 merged `0f31f6baf` | LANDED |
| `claude/profile-open-lower-bound` `442c03f0d` | y | y | #605 merged `efdc8be31` | LANDED |
| `claude/profile-floor-first` `9b7f4804d` | y | y | #607 merged `7e25cbb51` | LANDED |
| `docs/claude-closeout-truncnb2-20260930` (this handover and the after-task report) | y | awaiting the maintainer's push approval (repo rule: no push without instruction) | none yet | CARRIED-OVER |

CARRIED-OVER, `docs/claude-closeout-truncnb2-20260930`: not pushed because the repo's `CLAUDE.md` requires an explicit instruction to push. Resume: `git -C .worktrees/closeout-truncnb2 push -u origin docs/claude-closeout-truncnb2-20260930`, then open a docs-only PR.

FINDINGS-OF-RECORD: none on unmerged branches; every code finding is in merged PR bodies and the after-task report.

## Next Immediate Steps

None owed. Optional follow-ups, each a separate decision for the maintainer:

1. Give the truncated-NB2 `_family_ci` adapters a `dispersion_boundary` flag, as the other NB routes have (currently `boundary = false` for r).
2. Probe the shared-r truncated-NB2 fitter for the wrong-optimum stall #627 fixed on the per-trait fitter.
3. Decide whether the Gaussian `profile_ci` and `phylo_beta_xlv.jl` should also report 0 for an open lower end (they share `_profile_bisect_side`, which #605 and #607 left unchanged).

## Blockers / Open Questions

- None blocking. Open: whether an exact-marginal profile would close the draw-104 r interval at a positive r (inference from the #581 review's Laplace-vs-exact table; not tested).
