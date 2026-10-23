# After-task: landing the 2026-09-25 true-parity backlog (lane gllvm-backlog-20260926)

## 1. Goal

Land the backlog the 2026-09-25 true-parity-finish lane left open: merge #494, review and then merge or return each of the seven drafts (#487 to #491, #493, #495), turn #484 and #485 into reviewed PRs, update and audit the decision packet, and hand over. No parity claim.

## 2. Implemented

- Merged after review, each pinned to its reviewed head and gated on CI (the advisory frozen-R cell allowed only at main's own 277/9 to 278/8 range): #494 `d4da31544`, #488 `8001b0523`, #495 `39886c705`, #487 `d89179d41`, #489 `5bc5818d5`, #490 `b71c0047f`, #497 `ef0488df8`, #496 `b90641c97`, #502 `6cb0649e3` (squash), and after the maintainer's sign-off #491 `0a2796257` and #493 `7aa8bc39e` (squash, corrected message).
- New PRs: #496 (reverse-gap classes, proposed and unsigned), #497 (decision packet brought to main with #495's 11 decisions and live divergences #129 and #131), #500 (Fixes #484, two-part mode search), #502 (Fixes #485, reworked as a per-family NB1 verdict after review).
- Issues filed from independent reviews and a screen: #498 (Binomial quasi-separation flagged differently by the engines), #499 (per-trait truncated-NB2 stall, boundary flags, shared-r default), #501 (ordered beta reports converged at non-stationary points; restarts reach valid optima up to +2859 logLik; confirmed by two independent verifiers; a correction comment records that the optimiser path differs between Linux and macOS on the same data).
- A true-parity acceptance ledger (seven clauses plus two row gates plus one manual gate) with an oracle that reads origin/main, run with positive and negative controls. Baseline: 0 of 10 met.
- #500 (Fixes #484) reviewed three times and rebased (head ff3252540, source identical to the reviewed head a4e3eaec2), then returned for sign-off rather than merged, because its hurdle NB chain-rule fix changes fitted results.
- All merges were made from this lane's session through the maintainer's `gh` credentials under his written pre-authorisation (2026-09-26), so GitHub records them as merged by itchyshin; #491 and #493 were merged by the lane only after his explicit word ("merge #491 and #493 when green").
- #493's CHANGELOG entry on main corrected in the closing PR (it still said the method awaited sign-off, and described only `method = :wald` although `:profile` and `:bootstrap` are also reachable and untested for that type).

## 3a. Decisions and Rejected Alternatives

- #502: a global gradient criterion in `_fit_verdict` was rejected after review (it failed five existing tests and flagged genuine optima where the objective has small jumps). Maintainer chose option (d), per-family verdicts (2026-09-26). Rejected: (a) a stall check needing the objective at about 100 call sites; (b) sentinel and tolerance exemptions on the global rule; (c) a curvature-scaled test, left as a later option.
- `_phylo_verdict`: not changed. Its flips under the global rule were finite-difference steps hitting the 1e12 failure value, so it waits for a family-specific reproduction under (d).
- #491 and #493 change scientific results (and #493 adds a public `confint` method), so they were returned for the maintainer's sign-off after their blocking items were fixed; both merged on his word.
- D-220 amended in the vault (Claude owns GLLVModels.jl true parity from 2026-09-24; Cursor keeps gllvmTMB twin work).

## 4. Files Touched

- GLLVModels.jl, through the PRs above (see each PR's file list). Lane kit on branch `claude/lane-gllvm-backlog-20260926` under `LOOP/lanes/gllvm-backlog-20260926/` (GOAL, arcs, checkpoint, plan copy, both ledgers, merge_train.sh, prstate.sh, reviews/).
- Vault: `memory/DECISIONS.md` (D-220 amendment), `Shinichi/Dashboards/mission-control/live/status/gllvmTMB.json`.
- Closing PR: `CHANGELOG.md` (#493 entry), this report, and `docs/dev-log/handover/2026-09-26-claude-handover-gllvm-backlog.md`.

## 5. Checks Run

- Every merge: latest check run per name on the pinned head; advisory count read from the job log.
- Run ledger `.unlazy/gllvm-backlog/GATES.md` re-verified after each merge (12 of 14 before the closing PR; H1 is met by the closing PR, D43 by the panel verdict). True-parity ledger 0 of 10 (all gates measurable from main since #487).
- Test evidence per PR as recorded in the review reports under `reviews/`.

## 6. Tests of the Tests

- True-parity oracle: a synthetic all-evidenced fixture makes every mode print MET; reverting one row flips only that row's clauses. This caught a vacuous pass (the realistic-size gate passed with zero rows) before any baseline was recorded.
- Run ledger: two false passes caught and removed (a substring EXPECT that could never match; a fuzzy GitHub search that matched #494 for "Fixes #485"), and one stale tick from a lapsed approval.
- #500: the R1 relation test fails on the pre-fix head and passes on a4e3eaec2; an independent probe over 1800 matched sites found no regressions. #502: the verdict test is red on main and green on the branch, and its fixed-seed case was replaced by a platform-free unit contract that passes on Julia 1.10 and 1.13.

## 7a. Issue Ledger

Filed #498, #499, #501, #503 (remaining undamped mode searches, re-measured on main), #504 (confint bootstrap and profile refit accept non-converged replicates). Filed #505 (the rest of the #485 class). Corrected #501 (optimiser paths differ between Linux and macOS on the same data) and #503 (it wrongly said #502 covered the `_fit_verdict` class) by comment.

## 8. Consistency Audit

- Sibling screen of eight unscreened families on Totoro (ordered beta severe, ordinal and mixed modest, COM-Poisson anomaly, the rest clean).
- Mode-search triage of the audit's sibling list against current main: NB1 grouped and Student-t highest; filed as #503.

## 9. What Did Not Go Smoothly

- The session hit its usage limit around 17:00Z with seven agents live, killing four mid-task. Resumed with at most three live.
- Lane leases keyed on the session PID made sibling subagents overwrite each other's claims; `LANE_ID` per agent fixed it. A broad lease (a whole `test/` directory) blocked the #485 builder for an hour.
- A Totoro completion watcher of the form `until ! pgrep -f PATTERN` matched its own command line and never exited, holding up a report for 90 minutes.
- zsh passed a space-joined argument list as one argument to the merge train; the train refused (fail-closed).
- The lane set #500 merging on its own although its hurdle NB fix changes fitted results, the same reason it had returned #491 and #493; the completion panel caught it and the gate was stopped before any merge.
- About six Opus review children ran against a plan of one (recorded as routing drift in the plan-actual note); each found a defect the builders had missed.

## 10. Known Residuals

- #500 awaits the maintainer; no check against gllvmTMB was run on hurdle NB after its chain-rule fix, and the frozen-R smoke has no hurdle cell.
- #502 fixes only `fit_nb1_gllvm_grouped`; the rest of the class is #505.
- `confint(:profile)` and `confint(:bootstrap)` on `TruncatedNegBin2PerTraitFit` are reachable but untested; the bootstrap route inherits #504 and the fitter inherits #499.
- For #487, #489, #490 and #496 the merged head differs from the head the first review read; the panel confirmed each difference is only the requested fix, but no in-repo record of that check existed at merge time.
- The gate-tier scoreboard is dated 2026-09-25 and some rows cite PR states that have since changed. #501, #503 and #504 have no fixes yet.

## 11. Team Learning

- Filter ledger output for `APPROVAL REQUIRED` as well as PASS and FAIL; a lapsed approval leaves a stale tick that `--status` trusts.
- `EXPECT:` is a literal substring unless written `/regex/`.
- Use GitHub's `closingIssuesReferences`, not text search, to ask "which PR closes issue N".
- Give each subagent its own `LANE_ID`; claim exact files, never directories.
- Never write a `pgrep -f` watcher whose own command line contains its pattern.
- Keep live agents at three or fewer on a long run; the shared usage window, not context, runs out first.
- Apply a "returns for sign-off" rule by what a PR does to users' results, not by which PR was flagged first; check every code PR against it before starting its merge gate.
- When a merged head differs from the reviewed head, record the reviewed-to-merged diff check in the PR before merging.

## 12. Cross-Product Coverage

- **Convergence flag (`converged`).** Covers `fit_nb1_gllvm_grouped` (#502: converged only when the scale-aware gradient test holds). This arc does NOT cover `fit_nb1_gllvm_grouped_cov`, scalar `fit_nb1_gllvm`, the other roughly 100 `_fit_verdict` call sites, `_phylo_verdict`, or ordered beta; all are tracked in #505 and #501.
- **Per-site mode search.** Covers the two-part families through #500 once it merges (ZIP, ZINB, ZIB, hurdle Poisson, hurdle NB, delta Gamma, delta lognormal, beta hurdle), on top of Gamma and Beta (#481, #483) and the covariate kernel (#494). It does NOT cover NB1 grouped, Student-t (shared and grouped), GP1, beta-binomial, the mixed bridge, COM-Poisson, Tweedie grouped or NB2 grouped; all are tracked in #503.
- **`confint` methods.** Covers `method = :wald` for `TruncatedNegBin2PerTraitFit` (#493, tested on an interior and a boundary fixture). It does NOT cover `:profile` or `:bootstrap` for that type (reachable, untested), the shared-r adapter's boundary gap (#499), or the bootstrap and profile-refit acceptance of non-converged replicates for every family (#504).
- **Platforms and Julia versions.** Every merge was gated on CI for Julia 1.10 and 1.13 on Linux, and local runs used macOS on Julia 1.10. This arc does NOT cover Windows, and it does NOT cover the frozen-R smoke's hurdle families (it has no hurdle cell).
- **Twin repository.** This arc does NOT cover gllvmTMB (read-only by rule), including #1283 and #1236.
- **True parity.** This arc does NOT cover any true-parity row promotion: none was attempted or claimed, and the true-parity ledger stays at 0 of 10.
- **Deferred by plan.** It does NOT cover multi-day builds (bridge spine, phylo latent A14 and A15, real-data C4), Totoro campaigns over 30 minutes, or the Beta realistic-size cell.
