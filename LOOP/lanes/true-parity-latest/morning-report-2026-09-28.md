# Morning report, true-parity-latest lane (for 2026-09-28)

Written first at 18:27 MDT on 2026-09-27 and updated through the night. The newest state is at the top of each section.

## What needs you (each has a reply to paste)

1. Sign the P1 case-map classes on draft PR #533 (one GitHub comment). Draft: "Signed: accept all 38 classes as proposed, except ordination_uncertainty moves to semantic_divergence and chibar2_pvalue / variance_lrt move to required_core (D-297, 2026-09-27)."
2. Landing words for PRs that became ready overnight: see "Ready for your word" below.
3. A15 (phylo latent): "promote A15 on the objective and estimate receipts; send the converged-flag rule to the convergence lane (#485)."
4. CI capacity: trains time out after 3 h waiting for runners. If you want faster merges, approve a CI change such as running the Julia 1.10 shards only on workflow_dispatch or when a PR is marked ready.

## Landed on main (2026-09-27)

#522 (beta-binomial verdict, closes #515), #530 (docs phrase), #523 (P1 ledger and check tool), #524 (P1 pin), #514 (mixed bridge mode search), #532 (Documenter per-branch), #528 (ordinal_logit twin), #539 (harness pin source and P1 oracle).

## Ready for your word

- #563 temporal slice 2 (unit/unit_obs composition; head acfd02a69). Lands after #543 (it contains #543's commits). Review NON-BLOCKING; all fixes applied: receipts regenerated from R match byte-for-byte; NLL/gradient vs R's TMB 3e-9; logLik vs R 9e-8. The optimizer change leaves every slice-1 logLik identical to 1e-12; the Newton polish now respects the iteration cap; all 25 cells are classified (10 interior, 7 flat, 8 on R's degenerate engine fixture) instead of 14 skipping silently. 1.10 and 1.13 both pass. DECISION: keep update() exported? Reply: "merge #563 after #543; keep update exported" (or "make update internal").
- #535 temporal spec: count typo fixed (13 blocks, 20 expectations; head 9ec406e07). Docs only. Reply: "merge #525, #535, #545 when green" (the three spec PRs).
- #561 namespace re-measurement at P1 (head 1598b2142, supersedes #559). Review was BLOCKING; fixed and re-checked by me: a row labelled "numeric" must now cite a receipt with a real R-vs-Julia comparison inside tolerance, or it fails C1 and C8 (the review's mutation now reads C1_NOT_MET / C8_NOT_MET). Signed dispositions accept only "Shinichi Nakagawa" or "itchyshin" with a real, non-future date. Disclosed: at P0, 14 aghq rows now read name-only in C8 (C8 was NOT_MET before and after). Known limit: the checker trusts the numbers written in a receipt; it does not re-run them. 50/50 negative controls pass. Reply: "merge #561 when green".

## In a merge train (word given)

19:10: all seven train PRs had gone into conflict with main in CHANGELOG / check-log only. Refreshed the chain heads (#531, #543, #548) by merging main in; checked each refresh adds only main's commits and log lines; trains restarted on the new heads. #546, #556, #547, #558 get the same refresh when the PR ahead of them merges.

#531 extract_latent_scores; #543 temporal slice 1, then #546 iSDM build; #548 chibar2/variance_lrt, then #556 anova, then #547 phylo latent; #558 ISDM-PSI (after #546).

## In progress overnight

- #557 zi_* twin: blocking fix DONE at d0a57e05d (sites with a Laplace precision eigenvalue below 0.1 are walled off; optima near the floor report converged = false; NB2 moment start). The reviewer's case now matches R (-3311.4547). Over 20 NB2 draws: silent breakdowns 3 -> 0; 2 draws honestly flagged not converged (R also fails there). Second review NON-BLOCKING (blocker fixed, red-then-green on the saved case; R refits reproduce all three literal fixtures; the guard leaves the twin objective unchanged at R's optima). Builder applying the smaller items: a retry for 1-in-15 Julia-only stalls at the guard, honest wording on recovery range, a zi_poisson/zi_binomial sweep for the floor, missing-value refusal, and moving the R-pinned cases into the P1-tagged test file.

## Findings worth knowing

- Temporal: on 3 cells built from R's own engine test data, R and Julia both stop at the same saddle point (Hessian eigenvalue about -2.5, logLik equal to 1e-12). Recorded in the test, not investigated; worth a look on the R side.

- Namespace at P1 (#561): 71 R exports give 0 numeric twins, 44 name-only matches, 6 mismatches (Julia counterpart is a type or unexported: extract_Sigma_B/W, extract_cutpoints, extract_loadings, extract_proportions, extract_residual_split), 2 need a Julia function (animal_slope, dep), 2 retired, 17 still need live R fits.

- zi_* Laplace can fail outright on some data: on 2 of 20 NB2 draws neither R nor Julia has a usable Laplace optimum. The real fix is adaptive quadrature; that is a decision for you, not overnight work.

- 0 of 306 required 0.7.0 rows carry to P1; 278 receipts had lived in the gitignored .unlazy/. Re-measurement (A3) is the main remaining work; compute is small, the harness is the work.
- A namespace "pass" was only a name-existence check; the ledger now separates registration-only rows from numeric twins, so a name match cannot count toward parity (D-295 row 5).
- zi_nbinom2: observed-curvature Laplace can break down at y = 0 (negative curvature), giving a spurious maximum with a higher value; R's model shares the surface. Worth reporting to the gllvmTMB side.
- ordination_uncertainty is a name match only (R: joint-precision conditional covariance; Julia: bootstrap).

## Not done / not covered

(updated overnight)
