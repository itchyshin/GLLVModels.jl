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

(updated overnight)

## In a merge train (word given)

#531 extract_latent_scores; #543 temporal slice 1, then #546 iSDM build; #548 chibar2/variance_lrt, then #556 anova, then #547 phylo latent; #558 ISDM-PSI (after #546).

## In progress overnight

- #557 zi_* twin: blocking fix (Laplace breakdown guard, recovery test, NB2 start).
- #559 namespace re-measurement: re-created on main with the evidence-tier fix.
- Temporal slice 2 (unit/unit_obs composition).

## Findings worth knowing

- 0 of 306 required 0.7.0 rows carry to P1; 278 receipts had lived in the gitignored .unlazy/. Re-measurement (A3) is the main remaining work; compute is small, the harness is the work.
- A namespace "pass" was only a name-existence check; the ledger now separates registration-only rows from numeric twins, so a name match cannot count toward parity (D-295 row 5).
- zi_nbinom2: observed-curvature Laplace can break down at y = 0 (negative curvature), giving a spurious maximum with a higher value; R's model shares the surface. Worth reporting to the gllvmTMB side.
- ordination_uncertainty is a name match only (R: joint-precision conditional covariance; Julia: bootstrap).

## Not done / not covered

(updated overnight)
