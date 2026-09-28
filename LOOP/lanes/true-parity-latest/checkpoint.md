# Checkpoint (overwritten every arc), late 2026-09-27

## Signed decisions (vault)
D-294 re-pin to P1; D-295 Packet 1 + boundary (temporal, phylo latent inside; column grammar, spatial outside); D-296 Packet 1b (iSDM scope); D-297 #526/#533 classes accepted in chat (signature still to be given on #533 by Shinichi himself); D-300 Packets 1c (temporal) and 1d (phylo latent).

## Landed on main today
#528 (e9f7d949e, ordinal_logit twin), #539 (cb5688f7e, harness pin source + P1 oracle), #522 (52ed4281b, closes #515), #530 (d286ac4c5, docs phrase fix), #523 (0a94b1cfc, P1 ledger + check tool), #524 (824d22a4b, additive P1 pin), #514 (855542118, mixed bridge mode search), #532 (97e11be04, Documenter per-branch build, serialised deploy).

## Ready, landing word given, train running (scratchpad/merge_train_v2.sh ignores "Documenter deploy")
- #528 ordinal_logit twin (5c05e5b30), then #539 harness pin source + P1 oracle (e40179ceb): log scratchpad/train_528_539.log
- #531 extract_latent_scores twin (d0e454150): log scratchpad/train_531.log
CI is runner-starved (about 24 queued runs across lanes); trains wait, nothing failing.

## Landing word given 2026-09-27 (train scratchpad/train_543_546.log): #543 (refreshed dafe3b9d2), #546 (983c9979b); D-301 keeps the unique = TRUE refusal, ISDM-PSI follow-up.
- #543 temporal slice 1 (146c6fdf5): NLL = R's TMB fn to 2.3e-13, gr 1.8e-12; getLV vs R 1.3e-15; review NON-BLOCKING, fixes applied.

## In review / in progress
- #546 iSDM build (e2b53b25b): cross-objective 1.3e-11 on 4 cases; cloglog grid bit-identical; 3 b_fix rows @test_broken vs R's stopped nlminb (pass vs polished optimum); admission 19/19; tests 158 + 41 + 213 on 1.10 and 1.13. DEVIATION for Shinichi: R latent() defaults unique = TRUE (theta_diag_B), omitted by the spec; the Julia door refuses it. No gllvm() door (formula.jl belongs to the grammar lane). Review NON-BLOCKING (reviews/pr-546-review.md); fixes DONE at 983c9979b (polish convergence asserted; no @test_broken, door gap bound 1e-4; level-order doc; refusal prints the full corrected formula; 159 + 41 + 219 both versions). READY to land on Shinichi's word. unique = TRUE: reviewer recommends refuse now + follow-up port ISDM-PSI (1-2 days, p >= 3 fixture); Shinichi decides.
- A2 small twins running: chibar2_pvalue/variance_lrt = SAME function (R's file says it is a faithful port of Julia's), draft PR #548 (da1fd562c, 1e-12 twin, 71/54 both versions); NaN refusal added at 40d4578b6 (83/54 both versions, red-then-green shown); READY to land; the case-map row can move semantic_divergence -> twin (Shinichi signs); anova twin = draft PR #556 (045c26e85; AIC/BIC already match R's conventions, no new code; update STOPPED: Julia fits store no call; 56/56 both versions); r556 blocking FIXED at 1fc705699 (clean rank step only when dnpar == p - d_prev, else refused; type/family/trait-count refusals; AIC/BIC claim narrowed; 79/79 both versions; conductor checked the logic). READY to land. Side finding: base Gaussian fit without X is zero-mean (overlaps #519, Gaussian-intercepts lane); builder filed chip task_cd488039 on formula.jl's comment.
- Phylo latent build = draft PR #547 (3dd69810e): A14 logLik 4e-14, cross 1e-14, log-det 0; A15 logLik 2.8e-13, cross 3.7e-11, gradient gap (R 4e-3, Julia 2e-4, converged = false, within 2e-10 of optimum), 3 checks @test_broken; Ainv refused pending decision (R rewrites keyword Ainv to dense solve); 81/81 and 98 + 4 broken both versions. r547 NON-BLOCKING (reviews/pr-547-review.md); fixes DONE at e64388709 (A15 real bounds, 0 broken; Ainv twinned as R's dense keyword route incl. marginalised unobserved tip; g_tol docstring; R branch-length wording; 89/89 and 113/113 both versions). READY to land. For Shinichi: A14/A15 promotion block; route the A15 converged-flag to the convergence lane (#485).
- #533 case-map rows (9c1f55038, re-created #526) and #534 carry scan (21433f1bd, stacked): waiting for Shinichi's own signature comment on #533 (proposal: ordination_uncertainty moves to semantic_divergence).
- Specs #525 iSDM (032d90284), #535 temporal (af130f704), #545 phylo latent (c0fd5f5be): reviewed and revised; docs-only, land when convenient.

- zi_* = draft PR #557 (8bab78578): all three families verdict (b), new R-semantics route (observed-curvature Laplace, per-trait phi, per-row trials) alongside unchanged Julia routes; logLik 1e-8, cross 4e-9; r557 BLOCKING: zi_nbinom2 converges (converged = true) to a spurious Laplace maximum with HIGHER value (phi 42; y = 0 mixture observed curvature goes negative, A near singular); shared R surface, R's start avoids it; b2z adding recovery test + PD-floor guard (sentinel on breakdown) + better start. Finding to report to the gllvmTMB side. iSDM and mixed bridge not affected (cloglog log-concave; Poisson observed = Fisher). Running: ISDM-PSI = draft PR #558 (fc91c477f, base #546): unique = TRUE via Lambda_aug; 4-trait fixture cross 4.4e-10 / 1.1e-11, sd 4.2e-7; unique = FALSE bit-identical to #546; 183 / 41+219 / 68 both versions; r558 NON-BLOCKING (model equivalence exact; bit-identity verified); K = 0 on a unique table refused up front (R errors on d = 0) at 7ce161d92; 185 / 68 both versions. READY to land (stacked on #546). Follow-up noted: copied mode search can stop with a sqrt(eps) leftover step (log-det moves ~2e-8).

- Landing word given for #548, #556, #547 (train scratchpad/train_548_556_547.log, order 548 -> 556 -> 547).

- A3 PR2 = draft PR #559 (ac78084cb, base #539 branch: RETARGET to main after #539 merges): namespace 71 rows: 50 pass (registration parity only), 0 fail, 2 retired, 2 blocked, 17 need live R fits; tracked receipts; separate case-map-namespace.json (checker must read several files or fold after #533). CLOSED when #539 merged; b3n re-creating on main (v2) + evidence-tier fix (review: Tier 0 pass is isdefined only; checker must separate registration-only from numeric; reviews/pr-559-review.md). #544 (ordination X-forward fix, the user's side task) landed on main b3d86fe62.

- Temporal slice 2 started (b6b, branch claude/temporal-slice2): unit/unit_obs composition via fit_temporal_gllvm's structure argument; no formula.jl edit (the gllvm() hook still needs the grammar lane).
- #558 retargeted to main; its train waits for #546 to merge first (scratchpad/train_558.log). Trains #528/#539 and #531 timed out on CI capacity and were restarted (logs ...b.log).
- Remote Control turned on for this session and the two running sessions (#541 grouped beta-binomial; DRM.jl package completion).

## Overnight state (23:40 MDT; the morning report is the fuller view)
- Merged overnight (word given): #548 (4357e4652), #531 (5b9af3763).
- Trains running (tracked bash tasks; logs scratchpad/train_*): #556 (32d3f2097) then #547; #546 (512f90dc5) then #558. Refresh with scratchpad/refresh_pr.sh <N> (union-resolves only CHANGELOG/check-log, checks the PR touches no new files), verify, restart merge_train_v2.sh pinned to the new head. The auto-refresh-and-merge train (v3) was refused by the permission classifier: do not rebuild it.
- #543: twin job failed on CI Linux at the reviewed head (one fit just above g_tol); cherry-picked #563's reviewed optimizer commits (now 7921c55ab after refresh); NEEDS SHINICHI'S RENEWED WORD.
- Ready for word (all reviewed, fixes applied): #557 (1834fd515), #561 (92cf39571), #563 (acfd02a69, after #543), #567 (4176a9dce), #569 (073e90e78), #571 (1b99ae6b8), #576 (94fd6abb9), #579 (a857516df), #581 (1dc7c6dc6), #584 (0fb2b9411); specs #525/#535 (9ec406e07)/#545.
- A3 stack: #567 -> #569 -> #571 -> #579 -> #584 -> {aghq builder, isdm builder} running.
- Task chip filed: truncated NB2 mode-search jump and r intervals (grouped_dispersion.jl _grouped_laplace_mode oscillation).
- Waits for Shinichi: Woodbury fix in likelihood.jl (headline Gaussian path); shared post-fit gate in _fit_verdict; nobs ruling (unblocks 4 wave6 rows).

## Findings to remember
- Carry scan: 0 of 306 required P0 rows carry; 278 receipts dangling (gitignored .unlazy/), 21 stale. P1 evidence must be re-made (A3); compute is small (10-45 min on kohaku); the harness is the work (#539 is step 1).
- ordination_uncertainty is name-only (R: TMB joint-precision conditional covariance; Julia: bootstrap + Procrustes).
- Remaining hardcoded-P0 path that could stamp the wrong pin silently: tools/t4_p6_write_receipt.py:186 (off the CI path).
- Temporal slice 2 needs the grammar lane (#519) to add the gllvm() long-data hook (D-300, 1c-3).

## Overnight mandate (Shinichi, 2026-09-27: "I am going till 5 am - please keep going autonomously by then")
- Merge only PRs with a landing word already given (#531, #543, #546, #548, #556, #547, #558). Newly ready PRs queue for the morning report.
- No DRAC, no runs over 3 h, no outward actions beyond draft PRs and PR bodies; no CI config changes.
- Write the morning report (docs in the lane kit: morning-report-2026-09-28.md) by 04:30; keep it current as things land.

## Next
1. When trains finish: confirm merges; re-stack anything that conflicts.
2. #546 review, then fixes; Shinichi decides unique = TRUE.
3. Phylo latent build report, then review.
4. A3 next harness PR: regenerate per-family contracts at P1 with tracked receipts (after #539 lands).
5. A2 remaining twins: zi_* with R semantics; chibar2_pvalue / variance_lrt semantic check; AIC/BIC/anova/update; select_lv (auto-d lane).

## Where truth lives
Branch claude/lane-true-parity-latest in ~/local-scratch/lanes/GLLVM.jl-true-parity-latest; kit LOOP/lanes/true-parity-latest/ (GOAL, arcs, packets, reviews/). Compute: Totoro down until 2026-09-28; kohaku (8 vCPU) or DRAC with an estimate first.

RESUME: read GOAL.md, then checkpoint.md, then ultra-plan.md in LOOP/lanes/true-parity-latest/, and continue from Next.
