# After-task: overnight silent-failure fixes (lane gllvm-backlog-20260926, 2026-09-26 to 27)

## 1. Goal

Run unattended from 16:10 MDT on 2026-09-26 to 05:00 MDT on 2026-09-27 (D-290 in the maintainer's decision log): fix, one family per PR, the fits that fail silently, meaning code that returns a plausible value when the computation failed. Scope named by the maintainer: #503 (undamped per-site mode searches), #505 (convergence flags), #504 (bootstrap and profile refits), #501 (ordered beta), plus anything else in that class. The lane could merge a fix after an independent review and green CI, unless it changed results for fits that looked healthy, added public API, or touched a shared kernel. No parity claim.

## 2. Implemented

- Merged: #509 Student-t grouped kernel (`8f0bc97f6`), #508 profile and bootstrap refits reject non-converged replicates (`2847b5dbf`), #507 NB1 grouped kernel (`57a0266dd`), #511 ordered-beta kernel (`d55a8e3af`), #510 beta-binomial kernel (`46461f883`). #512 (COM-Poisson kernel) was rebased and is in the merge queue at the time of writing. All squash merges, each pinned to its reviewed head, CI green apart from the advisory frozen-R cell (277 to 278 passed, 8 to 9 failed, main's own range).
- #507, #508, #509, #510 and #511 merged, and #512 is queued, each on the maintainer's explicit word. #511, #510 and #512 were first returned to him because they change some fits that looked healthy (section 3a).
- Opened and not merged: #513 (parity page: one section on what parity does not mean), #514 (mixed-family bridge kernel), #516 (Poisson bootstrap refit reports its own convergence verdict, the first family for #504).
- Filed #515: `fit_beta_binomial_gllvm` on main reports `converged = true` at a log-likelihood of about +7e54, with the precision parameter near 3e65.
- A census of the 62 R exports with no Julia twin, kept in the lane kit (`reviews/p4-forward-census-2026-09-26.md`).

## 3a. Decisions and Rejected Alternatives

- Better optima at fits that looked healthy. The review of #511 found that at five times the usual loading scale, two fits that main reported converged, with every site stationary at the end, reached a better optimum on the branch (log-likelihood +52 and +9). The damping acts at intermediate steps of the outer optimiser, so its path changes. A whole-fit check then found the same for #510 (+7.35 and +6.08) and #512 (+0.059), with no fit worse. The maintainer accepted this for #511 ("merge #511 when green") and then for the class: a looked-healthy fit moving to a better optimum is acceptable for a mode-search fix, provided no looked-healthy fit gets worse and every site on the branch is stationary.
- Per-site "unchanged" is not enough. Reviews before #511 checked only sites where the old search converged. From #511 on, reviews also compare whole fits on main and branch.
- #514 was not merged: its review found a looked-healthy fit that gets worse (below), which the class rule still sends to the maintainer.
- Rejected: batching the remaining #504 adapters into one PR tonight. One family per PR stays the rule; #516 is the template.

## 4. Files Touched

- GLLVModels.jl through the PRs above (each PR lists its files).
- This report, `docs/dev-log/handover/2026-09-27-claude-handover-overnight.md`, and a dated correction at the top of `docs/dev-log/handover/2026-09-26-claude-handover-gllvm-backlog.md` (it still said #500 was waiting; #500 merged as `cb3580509`).
- Lane kit on branch `claude/lane-gllvm-backlog-20260926`, `LOOP/lanes/gllvm-backlog-20260926/`: `OVERNIGHT.md` (goal, queue, timed log), `merge_train.sh`, `fix-build-review.workflow.js`, and the review and check reports under `reviews/` (pr-510, pr-511, pr-512, pr-516, wholefit-510-512, p4-forward-census).

## 5. Checks Run

- Every merge: latest check run per name on the pinned head; the advisory count read from the job log.
- Every rebase before a merge: the PR's code and test patch compared with the reviewed head (#512: `git diff` of source and test files empty; #516: identical hash of the changed lines), then the PR's tests re-run locally (#512: 950 of 950; #516: 24 of 24).
- The backlog run ledger re-verified: 14 of 14 met. The true-parity oracle on main: 0 of 32 gate-tier rows done, clause C1 191 of 497 required rows unsigned, C7 not met (met on #513's branch).

## 6. Tests of the Tests

- Each fix has a test that fails on main and passes on its branch, on Julia 1.10 and 1.13, asserting relations (every returned value is either -Inf or a stationary point certified by an independent ForwardDiff check) rather than one seed's outcome.
- Each reviewer wrote its own probe with its own seeds and its own copy of main's kernel, and reproduced the failure rate on main (for example #510: 26 of 200 sites on main, 0 of 200 on the branch).
- #514's own new test failed on Linux CI while passing on macOS (`test/test_mixed_mode_search.jl:161`), which is the platform fragility its reviewer flagged. It is why #514 did not merge.

## 7a. Issue Ledger

- Filed #515 (beta-binomial reports converged at loglik +7e54).
- Reopened #503, which #507's merge closed by mistake: its squash title said "Fixes #503". The merge script now refuses a PR whose title would close an issue.
- #501 stays open: #511 checked two of the seven seeds flagged on macOS. Still to run: 2002, 2004, 2006, 2007 and 2008, plus a Linux check.

## 8. Consistency Audit

- The whole-fit finding on #511 was applied to its siblings: #510 and #512 were held and checked before merging, and #514's review ran the same check (it found the Heywood case below).
- #507, #509 and #500 merged before this check existed. They were merged on the maintainer's word, but their whole-fit behaviour was not measured.

## 9. What Did Not Go Smoothly

- A blocking question to the maintainer at about 00:10Z stopped the loop until he answered at about 15:10Z. Background agents finished in that time, but nothing new started, and the 05:00 MDT handover was not written on time. In an unattended run, a question belongs in a text message while work continues.
- #507's title contained "Fixes #503", and the merge closed the tracking issue. Neither the review nor the merge script checked titles.
- The #504 builder ran three Julia processes at once (about five cores, above the lane's four). Later briefs allow one Julia process at a time.
- Early timestamps in the lane's progress log were written ahead of the clock and later corrected.
- A duplicate copy of the ordered-beta builder wrote the same worktree after another agent messaged it; the branch was checked and found coherent.

## 10. Known Residuals

- #514 (mixed-family bridge) needs rework before it can merge. From its review: an absolute gradient test that cannot pass when the Normal trait's sigma is near zero (a Heywood case), so a fit that was healthy on main ends at a lower log-likelihood; a small-step shortcut that can oscillate (4 of 3000 sites return -Inf although a mode exists); an inaccurate justification in the body, CHANGELOG and code comment about the Normal Fisher weight; a test that pins main's garbage value across platforms (it fails on Linux).
- #503 remaining: NB2 grouped (live on the default NB2 route, 25 of 187 sites), Tweedie grouped, and GP1 and shared Student-t (both use the shared generic kernel, which needs the maintainer's sign-off).
- #504: about 40 bootstrap adapters still return a bare vector; #516 migrates Poisson only.
- #505: the remaining convergence-flag fitters, starting with `fit_nb1_gllvm_grouped_cov`.
- Em dashes remain in tonight's CHANGELOG entries and in #516's test comments; a single cleanup PR would clear them without a CI run per PR.

## 11. Team Learning

- Check whole fits, not only sites, when a fix changes an inner search that an outer optimiser calls.
- Never block an unattended loop on a question; post it and keep working.
- Check every PR title for closing keywords before a squash merge; the title becomes the commit subject.
- Two PRs that add a CHANGELOG entry at the same line conflict whichever merges second; placing an entry at a different position in the list lets both merge.
- One Julia process at a time per agent on the shared Mac.

## 12. Cross-Product Coverage

- **Per-site mode search (#503).** Covers Student-t grouped, NB1 grouped, ordered beta, beta-binomial, and COM-Poisson once #512 merges. It does NOT cover the mixed-family bridge (#514, rework), NB2 grouped, Tweedie grouped, GP1 or shared Student-t.
- **Whole-fit behaviour.** Measured for #510, #511, #512 and #514 on p = 6, n = 120, K = 2 at two loading scales. It does NOT cover #507, #509 or #500, larger problems, or other platforms (all checks ran on macOS, Julia 1.10).
- **Bootstrap and profile refits (#504).** #508 covers the shared helpers; #516 would cover Poisson's adapter. It does NOT cover the other families' adapters.
- **Platforms and Julia versions.** Merges were gated on Linux CI for Julia 1.10 and 1.13; local runs used macOS. This run does NOT cover Windows.
- **Twin repository.** This run does NOT cover gllvmTMB; nothing was written there.
- **True parity.** This run does NOT promote any true-parity row. The ledger reads 0 of 32 gate-tier rows done; #513 would move clause C7 once merged.
