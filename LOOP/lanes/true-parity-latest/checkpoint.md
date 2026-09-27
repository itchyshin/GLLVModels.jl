# Checkpoint (overwritten every arc), late 2026-09-27

## Signed decisions (vault)
D-294 re-pin to P1; D-295 Packet 1 + boundary (temporal, phylo latent inside; column grammar, spatial outside); D-296 Packet 1b (iSDM scope); D-297 #526/#533 classes accepted in chat (signature still to be given on #533 by Shinichi himself); D-300 Packets 1c (temporal) and 1d (phylo latent).

## Landed on main today
#522 (52ed4281b, closes #515), #530 (d286ac4c5, docs phrase fix), #523 (0a94b1cfc, P1 ledger + check tool), #524 (824d22a4b, additive P1 pin), #514 (855542118, mixed bridge mode search), #532 (97e11be04, Documenter per-branch build, serialised deploy).

## Ready, landing word given, train running (scratchpad/merge_train_v2.sh ignores "Documenter deploy")
- #528 ordinal_logit twin (5c05e5b30), then #539 harness pin source + P1 oracle (e40179ceb): log scratchpad/train_528_539.log
- #531 extract_latent_scores twin (d0e454150): log scratchpad/train_531.log
CI is runner-starved (about 24 queued runs across lanes); trains wait, nothing failing.

## Ready, awaiting landing word
- #543 temporal slice 1 (146c6fdf5): NLL = R's TMB fn to 2.3e-13, gr 1.8e-12; getLV vs R 1.3e-15; review NON-BLOCKING, fixes applied.

## In review / in progress
- #546 iSDM build (e2b53b25b): cross-objective 1.3e-11 on 4 cases; cloglog grid bit-identical; 3 b_fix rows @test_broken vs R's stopped nlminb (pass vs polished optimum); admission 19/19; tests 158 + 41 + 213 on 1.10 and 1.13. DEVIATION for Shinichi: R latent() defaults unique = TRUE (theta_diag_B), omitted by the spec; the Julia door refuses it. No gllvm() door (formula.jl belongs to the grammar lane). Review r546 running.
- Phylo latent build (b9, branch claude/phylo-latent-build) from spec #545 (c0fd5f5be) and D-300.
- #533 case-map rows (9c1f55038, re-created #526) and #534 carry scan (21433f1bd, stacked): waiting for Shinichi's own signature comment on #533 (proposal: ordination_uncertainty moves to semantic_divergence).
- Specs #525 iSDM (032d90284), #535 temporal (af130f704), #545 phylo latent (c0fd5f5be): reviewed and revised; docs-only, land when convenient.

## Findings to remember
- Carry scan: 0 of 306 required P0 rows carry; 278 receipts dangling (gitignored .unlazy/), 21 stale. P1 evidence must be re-made (A3); compute is small (10-45 min on kohaku); the harness is the work (#539 is step 1).
- ordination_uncertainty is name-only (R: TMB joint-precision conditional covariance; Julia: bootstrap + Procrustes).
- Remaining hardcoded-P0 path that could stamp the wrong pin silently: tools/t4_p6_write_receipt.py:186 (off the CI path).
- Temporal slice 2 needs the grammar lane (#519) to add the gllvm() long-data hook (D-300, 1c-3).

## Next
1. When trains finish: confirm merges; re-stack anything that conflicts.
2. #546 review, then fixes; Shinichi decides unique = TRUE.
3. Phylo latent build report, then review.
4. A3 next harness PR: regenerate per-family contracts at P1 with tracked receipts (after #539 lands).
5. A2 remaining twins: zi_* with R semantics; chibar2_pvalue / variance_lrt semantic check; AIC/BIC/anova/update; select_lv (auto-d lane).

## Where truth lives
Branch claude/lane-true-parity-latest in ~/local-scratch/lanes/GLLVM.jl-true-parity-latest; kit LOOP/lanes/true-parity-latest/ (GOAL, arcs, packets, reviews/). Compute: Totoro down until 2026-09-28; kohaku (8 vCPU) or DRAC with an estimate first.

RESUME: read GOAL.md, then checkpoint.md, then ultra-plan.md in LOOP/lanes/true-parity-latest/, and continue from Next.
