# After-task: landing the 2026-09-25 true-parity backlog (lane gllvm-backlog-20260926)

DRAFT. Items marked PENDING are filled at close from the ledger and GitHub, not from memory.

## 1. Goal

Land the backlog the 2026-09-25 true-parity-finish lane left open: merge #494, review and then merge or return each of the seven drafts (#487 to #491, #493, #495), turn #484 and #485 into reviewed PRs, update and audit the decision packet, and hand over. No parity claim.

## 2. Implemented

- Merged after review, each pinned to its reviewed head and gated on CI (the advisory frozen-R cell allowed only at main's own 277/9 to 278/8 range): #494 `d4da31544`, #488 `8001b0523`, #495 `39886c705`, #487 `d89179d41`, #489 `5bc5818d5`, #490 `b71c0047f`, #497 `ef0488df8`, #496 `b90641c97`.
- New PRs: #496 (reverse-gap classes, proposed and unsigned), #497 (decision packet brought to main with #495's 11 decisions and live divergences #129 and #131), #500 (Fixes #484, two-part mode search), #502 (Fixes #485, reworked as a per-family NB1 verdict after review).
- Issues filed from independent reviews and a screen: #498 (Binomial quasi-separation flagged differently by the engines), #499 (per-trait truncated-NB2 stall, boundary flags, shared-r default), #501 (ordered beta reports converged at non-stationary points; restarts reach valid optima up to +2859 logLik; confirmed by two independent verifiers; a correction comment records that the optimiser path differs between Linux and macOS on the same data).
- A true-parity acceptance ledger (seven clauses plus two row gates plus one manual gate) with an oracle that reads origin/main, run with positive and negative controls. Baseline: 0 of 10 met.
- PENDING: #500 outcome; #502 rework outcome; #491 and #493 handed to the maintainer with verification.

## 3a. Decisions and Rejected Alternatives

- #502: a global gradient criterion in `_fit_verdict` was rejected after review (it failed five existing tests and flagged genuine optima where the objective has small jumps). Maintainer chose option (d), per-family verdicts (2026-09-26). Rejected: (a) a stall check needing the objective at about 100 call sites; (b) sentinel and tolerance exemptions on the global rule; (c) a curvature-scaled test, left as a later option.
- `_phylo_verdict`: not changed. Its flips under the global rule were finite-difference steps hitting the 1e12 failure value, so it waits for a family-specific reproduction under (d).
- #491 and #493 change scientific results (and #493 adds a public `confint` method), so they are returned for the maintainer's sign-off after their blocking items were fixed, not merged by the lane.
- D-220 amended in the vault (Claude owns GLLVModels.jl true parity from 2026-09-24; Cursor keeps gllvmTMB twin work).

## 4. Files Touched

- GLLVModels.jl, through the PRs above (see each PR's file list). Lane kit on branch `claude/lane-gllvm-backlog-20260926` under `LOOP/lanes/gllvm-backlog-20260926/` (GOAL, arcs, checkpoint, plan copy, both ledgers, merge_train.sh, prstate.sh, reviews/).
- Vault: `memory/DECISIONS.md` (D-220 amendment), `Shinichi/Dashboards/mission-control/live/status/gllvmTMB.json`.
- PENDING: final list at close.

## 5. Checks Run

- Every merge: latest check run per name on the pinned head; advisory count read from the job log.
- Run ledger `.unlazy/gllvm-backlog/GATES.md` re-verified after each merge (PENDING final count). True-parity ledger 0 of 10 (all gates measurable from main since #487).
- Test evidence per PR as recorded in the review reports under `reviews/`.

## 6. Tests of the Tests

- True-parity oracle: a synthetic all-evidenced fixture makes every mode print MET; reverting one row flips only that row's clauses. This caught a vacuous pass (the realistic-size gate passed with zero rows) before any baseline was recorded.
- Run ledger: two false passes caught and removed (a substring EXPECT that could never match; a fuzzy GitHub search that matched #494 for "Fixes #485"), and one stale tick from a lapsed approval.
- #500 and #502: red on main, green on branch (PENDING for the reworked heads).

## 7a. Issue Ledger

Filed #498, #499, #501. Corrected #501 by comment. PENDING: the mode-search tracking issue and the confint bootstrap/profile-refit issue from the triage.

## 8. Consistency Audit

- Sibling screen of eight unscreened families on Totoro (ordered beta severe, ordinal and mixed modest, COM-Poisson anomaly, the rest clean).
- Mode-search triage of the audit's sibling list against current main (PENDING).

## 9. What Did Not Go Smoothly

- The session hit its usage limit around 17:00Z with seven agents live, killing four mid-task. Resumed with at most three live.
- Lane leases keyed on the session PID made sibling subagents overwrite each other's claims; `LANE_ID` per agent fixed it. A broad lease (a whole `test/` directory) blocked the #485 builder for an hour.
- A Totoro completion watcher of the form `until ! pgrep -f PATTERN` matched its own command line and never exited, holding up a report for 90 minutes.
- zsh passed a space-joined argument list as one argument to the merge train; the train refused (fail-closed).

## 10. Known Residuals

PENDING at close. Known now: the gate-tier scoreboard is dated 2026-09-25 and some rows cite PR states that have since changed; #501's fix is not started; the triage's still-live families have no fixes.

## 11. Team Learning

- Filter ledger output for `APPROVAL REQUIRED` as well as PASS and FAIL; a lapsed approval leaves a stale tick that `--status` trusts.
- `EXPECT:` is a literal substring unless written `/regex/`.
- Use GitHub's `closingIssuesReferences`, not text search, to ask "which PR closes issue N".
- Give each subagent its own `LANE_ID`; claim exact files, never directories.
- Never write a `pgrep -f` watcher whose own command line contains its pattern.
- Keep live agents at three or fewer on a long run; the shared usage window, not context, runs out first.

## 12. Cross-Product Coverage

Not covered: gllvmTMB (read-only by rule); any true-parity row promotion (none attempted, none claimed); multi-day builds (bridge spine, phylo latent A14/A15, real-data C4); Totoro campaigns over 30 minutes; the Beta realistic-size cell (deferred because it would copy #491's pattern while #491 was under review).
