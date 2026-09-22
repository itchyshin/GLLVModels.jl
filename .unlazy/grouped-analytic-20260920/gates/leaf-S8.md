# Gates: leaf-S8 GLLVModels grouped route, analytic outer gradient (overnight arc 2026-09-20)

OWNS: src/grouped_laplace.jl, src/grouped_nongaussian_fit.jl, src/grouped_fit.jl, test/test_grouped_analytic_grad.jl, test/runtests.jl (one include line), bench/profile_grouped_glmm.jl, bench/results/grouped_sections_*.tsv, docs/design/grouped-analytic-gradient.md

BASELINE: origin/main 69a69b0a0. Latte 200x5 fixture, banked walls: 0.192 s (pre-S7c), 0.184721 s before / 0.150383 s after S7c (bench/results/grouped_warm_68c2f067c.tsv, git-ignored). Latte.jl 0.015 s. Per fit at that fixture: 118 objective calls, 711 summed inner Newton iterations, nθ = 2.

SCOPE: replace the finite-difference outer gradient with an analytic one derived through the implicit function theorem, reusing src/takahashi_selinv.jl for the log-det trace term; then unconfine the S7c warm start, demote Nelder-Mead to a fallback, and drop the final O(nθ²) FD Hessian. Every step an identity against 69a69b0a0. Never widen a tolerance.

## GATE A (the arc's own go/no-go, runs BEFORE any src/ change)

- [x] GA.1: `bench/profile_grouped_glmm.jl` gains a second fixture with nθ >= 6 and a section partition (Nelder-Mead evaluations, BFGS line search, FD gradient calls, final FD Hessian, inner Newton iterations, GLM state/score/curvature, CHOLMOD factorisation, log-det). Sections sum to within 10% of the measured wall on both fixtures.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. bench/profile_grouped_glmm.jl --gate sections
  EXPECT: GATE GA.1 PASS
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-s9a-hessian-20260921; path=01d9749a8aeb/36 entries; output=GA.2: FD-attributable share -- large fixture (glmm_5000x3_g500): 0.954  small fixture (glmm_200x5): 0.830 | GA.2 VERDICT: PROCEED

- [ ] GA.2: THE DECISION. On the LARGER fixture, the FD-attributable share (FD gradient calls + final FD Hessian + Nelder-Mead evaluations) is >= 25% of the fit wall.
  CHECK: read the TSV written by GA.1 and state the three section shares and their sum
  EXPECT: share >= 0.25 -> proceed to GATE B; share < 0.25 -> STOP the arc, the measurement is the deliverable, write the finding into the progress record and the after-task
  EVIDENCE: pending

## GATE B (only if GA.2 says proceed)

- [ ] GB.1: docs/design/grouped-analytic-gradient.md derives the gradient before any code, with the symbolic-alignment table: implicit db̂/dθ, the tr(A⁻¹ dA/dθ) log-det term, and the observed-curvature dependence on y. Every symbol maps to the function that will compute it.
  CHECK: manual read by the orchestrator
  EXPECT: PASS
  EVIDENCE: pending
  Section 6 is the symbolic-alignment table: 28 rows, each mapping one symbol to the function that computes it with a `file:line`, and each marked `exists` or `must be written`. It is honest about what did not exist -- it flags `Fo` as computed but discarded (hence the `JointGroupedLaplaceResult.factor` field the implementation added), `takahashi_selinv` as existing but never wired to the grouped route, and `_grouped_laplace_design` as Float64-hard-typed so no AD can be carried through it.
  ONE DOCUMENTED DEVIATION, recorded rather than smoothed: four table rows name functions the implementation chose to INLINE inside `_grouped_analytic_loglik_gradient` instead of writing as named functions -- `_grouped_eta_explicit` (e_k), `_grouped_mode_rhs` (v_k), `_grouped_mode_jacobian` (u_k), `_grouped_eta_total` (edot_k). The computations are all present and in the derived form; only their packaging differs from the table. This does not affect GB.1, which gates the derivation, but it means the table is now one revision ahead of the code's structure and should be reconciled before the PR body quotes it.

- [x] GB.2: the analytic gradient agrees with the existing central-difference gradient (`_grouped_fd_gradient`, src/grouped_fit.jl:213-224) at 20 random theta, rtol 1e-6, which is the FD reference's own accuracy and not a widened bound.
  **AMENDED 2026-09-21 by the orchestrator after reading B1's section 7.** The gate as first written was
  unable to catch four of the nine named failure modes, because both fixtures are Poisson and the
  comparison was not per-coordinate. It now REQUIRES all four of:
  (a) **per-coordinate** comparison, never a norm or a cosine similarity, because a sign slip confined to
      the log-det direction leaves the mean coordinates matching to two digits while the log-SD
      coordinates are wrong by about a factor of -1 (hazard 7.2);
  (b) at least one fixture with a **nonzero loading coordinate**, because losing the factor of 2 on the
      design-derivative trace is identically zero in every gamma coordinate (hazard 7.3);
  (c) at least one **non-Poisson, non-Binomial family** (Beta or NB2), because for those two families the
      Fisher and observed weights coincide pointwise, so picking up `Ff` where `A` belongs is undetectable
      in Poisson (hazard 7.5), as is dropping the dispersion terms (hazard 7.9);
  (d) the `inner_tol` in force **recorded in the evidence**, because the analytic and FD gradients degrade
      differently as it loosens, so a disagreement at loose tol is the fixture's fault, not the gradient's
      (hazard 7.7).
  A Poisson-only, norm-summarised version of this gate would pass while four real defects shipped.
  **AMENDED AGAIN 2026-09-21 ~05:00Z, after a defect all three fixtures were blind to.** A fifth
  requirement:
  (e) at least one fixture with **TWO OR MORE grouping terms, with UNEQUAL group counts**. With a
      single term, `_grouped_laplace_design_jacobian` never pushes a zero PLACEHOLDER block for the
      terms the coordinate does not belong to, so the placeholder's width is never exercised and
      can be wrong. It was wrong: it used the trait-factor `width`, where the real block is
      `kron(incidences[s], Lstar)`, i.e. `size(incidences[s], 2) * width` columns. Unequal group
      counts are required as well as two terms, because equal counts let a coincidental width
      match hide the same defect.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate fd_agreement
  EXPECT: GATE GB.2 PASS
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-s9a-hessian-20260921; path=01d9749a8aeb/36 entries; output=PASS poisson_percoord: worst per-coordinate rel 1.749e-07 <= 1.0e-06 | GATE GB.2 PASS
  - `poisson_latent` (Poisson, `mode=:latent, rank=1`, p=3, n=60, G=12, ntheta=6 -- requirement (b), nonzero loading coordinates): worst 7.281e-08 at coord 6 (analytic 9.4693509067e-02 vs fd 9.4693515962e-02). Per-coordinate worsts 6.70e-09, 1.67e-09, 1.67e-08, 2.84e-09, 1.05e-09, 7.28e-08.
  - `beta_shared` (Beta, shared log_phi, p=2, n=60, ntheta=4 -- requirement (c), Fisher != observed weight AND a dispersion block): worst 6.211e-09 at coord 4, the log_phi coordinate itself.
  - `nb2_shared` (NegativeBinomial, shared log_r, ntheta=4 -- requirement (c) again, and the only fixture exercising the hand-coded `_glm_obs_weight` override in src/families/negbin.jl, i.e. a different `_glm_obs_weight_deta` dispatch): worst 1.986e-07 at coord 2.
  No coordinate anywhere had its FD reference below 1e-6, so none of these is a small-denominator artefact; the check counts and prints that number rather than excusing it.
  **The gate was shown to have teeth by MUTATION, not assumed to.** Dropping the log-det implicit term (`- 0.5 * dot(wdot, t)` -> `- 0.0 * ...` at all three sites, B1 section 2's "cheap wrong answer"), re-running, and restoring src from a byte copy: GATE GB.2 FAIL on all three fixtures, worst rel 7.995e-01 / 1.057e+00 / 1.322e+00, the Beta and NB2 dispersion coordinates flipping SIGN (beta coord 4 analytic -4.83 vs fd +0.275). A gate that cannot fail is not evidence; this one fails on the exact defect it was amended to catch.
  **RE-RUN 2026-09-21 ~05:05Z with requirement (e)'s fixture added, and still PASS.** Fourth
  fixture `poisson_twoterm` (Poisson, two `:indep common=true` terms -- `unit` with 12 groups and
  `cluster` with 5 -- p=2, n=60, ntheta=4): worst per-coordinate rel **2.890e-08** against the 1e-6
  bound, 20 valid theta, zero skipped, zero FD fallbacks, no coordinate with an FD reference below
  1e-6. The other three fixtures' worsts are UNCHANGED to every printed digit (7.281e-08,
  6.211e-09, 1.986e-07), which is the evidence that the src fix touched nothing they exercise.
  **Mutation-tested again, and this is the part that matters.** Reverting the one-line fix (back to
  `spzeros(Float64, N, width)`), re-running, and restoring src from a byte copy: the three original
  fixtures ALL STILL PASS at their identical worsts, and only `poisson_twoterm` fails -- with a
  hard `DimensionMismatch`, not a wrong number. The gate as it stood at ~03:20Z could not have
  caught this defect, and the new fixture is the only thing now standing between it and a
  release.

- [ ] GB.3: fitted parameters and logLik equal origin/main 69a69b0a0 within rtol 1e-8 on both fixtures, and the existing grouped identity fixtures A, B and D still pass.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate identity && env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_laplace_identity.jl --gate identity
  EXPECT: GATE GB.3 PASS
  EVIDENCE: pending
  First half PASSES. `--gate identity` on the new file fits each fixture twice, once with `analytic_gradient=false` (which reaches the same `_grouped_fd_gradient(objective_cold, ...)` call origin/main always used) and once with it true: `poisson_latent` loglik -2.448399966295e+02 both ways, rel 0.000e+00, max per-coordinate beta rel 1.091e-10; `beta_shared` +4.357672061406e+01, rel 3.424e-15, beta 2.570e-10; `nb2_shared` -2.123724366875e+02, rel 2.677e-16, beta 3.442e-10. All well inside rtol 1e-8, `converged` identical both ways. Stated openly in the gate's own output: this is an IN-WORKTREE PROXY for the ledger's origin/main 69a69b0a0 comparison, which is still owed -- it proves the S8 branch changes no answer, it does not independently re-derive origin/main's numbers.
  **ORIGIN/MAIN HALF NOW DISCHARGED, 2026-09-21 ~04:15Z, scheduled session.** The proxy above is no
  longer the only evidence. A detached worktree was created at 69a69b0a0 itself
  (`~/local-scratch/lanes/GLLVM.jl-s8-baseline-69a69b0a0`), given a byte copy of the lane's
  `Manifest.toml` so both sides resolve the IDENTICAL dependency versions and the comparison isolates
  the source change (`git diff 69a69b0a0 HEAD -- Project.toml` is empty, so nothing was forced). One
  script (session scratchpad, not in the lane) builds the three fixtures from the same seeds and calls
  `fit_grouped_nongaussian` with `inner_maxiter=200, inner_tol=1e-10` and NO kwarg that exists on only
  one side, so 69a69b0a0 takes its only path (FD) and the S8 branch takes its default
  (`analytic_gradient=true`, and `warm_start_inner=true` from S7c). Result, rtol 1e-8:
  - `poisson_latent`: loglik main -2.44839996629515724e+02, S8 -2.44839996629515724e+02, rel 0.000e+00; max per-coordinate beta rel 1.091e-10.
  - `beta_shared`: main +4.35767206140607328e+01, S8 +4.35767206140605836e+01, rel 3.424e-15; beta 2.570e-10.
  - `nb2_shared`: main -2.12372436687485845e+02, S8 -2.12372436687485788e+02, rel 2.677e-16; beta 3.442e-10.
  `converged=true` on all six fits. Worst disagreement anywhere 3.442e-10 against a bound of 1e-8.
  **RE-RUN ~05:10Z after the multi-term fix recorded under GB.4, with a fourth fixture added to the
  comparison** (`poisson_twoterm`, the shape the defect lived in): main -1.49516291816394840e+02
  against S8 -1.49516291816345273e+02, loglik rel **3.315e-13**, max per-coordinate beta rel
  **3.753e-09**, `converged=true` both. The other three reproduce their earlier numbers exactly.
  Worst anywhere across all four fixtures is **3.753e-09** against 1e-8 -- inside the bound, and
  the two-term fixture is the closest to it, which is worth saying rather than rounding away.
  Two things this buys beyond ticking a box. First, it reproduces the in-worktree proxy's three beta
  numbers to every printed digit, so the proxy was a faithful stand-in rather than a convenient one --
  that is a check ON the earlier evidence, not a repeat of it. Second, it shows the S8 branch as a
  WHOLE, S7c's `warm_start_inner=true` default included, still lands on origin/main's answer; the
  earlier proxy only compared two paths inside one worktree and could not have seen a shared drift.
  **GATE NOW TICKED, 2026-09-21 ~13:10Z, scheduled session.** Both halves pass. The second half's
  STOP was Shinichi's item-1 decision, taken 2026-09-21 12:25-12:35Z as option (a) ("guard both
  paths") and applied in `3ca07a491`, so `test_grouped_laplace_identity.jl --gate identity` now
  reports **GATE G7b.1 PASS, 21/21** -- the FD path pinned at 118 calls with all its reuse
  invariants, the analytic path pinned at 94 with 0 fallbacks. Re-run by this session on the GB.4
  text below, not accepted from the earlier run: `test_grouped_analytic_grad.jl --gate identity`
  **GATE GB.3 PASS** on all five fixtures (`poisson_latent` rel 0.000e+00 / beta 1.091e-10,
  `beta_shared` 3.424e-15 / 2.570e-10, `nb2_shared` 2.677e-16 / 3.442e-10, `poisson_twoterm`
  3.315e-13 / 3.753e-09, `latent_plus_indep` 1.183e-16 / 7.155e-10, `converged` true both ways
  everywhere), then `test_grouped_laplace_identity.jl --gate identity` **GATE G7b.1 PASS**. The
  origin/main half remains discharged by the detached 69a69b0a0 worktree recorded above; nothing
  in the GB.4 change moved any number it measured.

  **SUPERSEDED 2026-09-21 ~13:10Z by the tick above.** Shinichi took the decision as option (a) and it was applied in `3ca07a491`, which pins fixture A's inner-fit count on BOTH gradient paths (118 FD, 94 analytic). The paragraph below is the pre-decision ~03:30Z record, kept verbatim rather than summarised away; it is not a live failure and `No action taken` below is no longer true.

  Second half FAILS, and the failure is a FINDING rather than a defect. `test/test_grouped_laplace_identity.jl --gate identity`: 16 passed, 1 failed -> `GATE G7b.1 FAIL fixture A: inner Laplace-fit call count changed (94 vs 118)`. Every NUMERIC identity in that file passed (loglik, logdet_precision, fitted parameters, all at rtol 1e-8); the single failing assertion is `stats.calls == BASELINE_A_OBJ_CALLS`, a call-COUNT invariant banked for slice S7b whose stated rationale is "the reuse must not change the optimiser's path". That rationale is correct for S7b, a CHOLMOD-reuse change that must be numerically and procedurally invisible. It is the opposite of what S8 is for: replacing a 2*ntheta-call FD gradient with one inner solve is SUPPOSED to cut the objective-call count, and 118 -> 94 on fixture A (-20.3%) is the first measured evidence that it does.
  **No action taken.** The tolerance was not widened, the assertion was not edited, and `test/test_grouped_laplace_identity.jl` is not in this leaf's OWNS list. The decision -- whether that S7b invariant should become conditional on `analytic_gradient`, or be rebanked at 94, or whether S8's default should be `analytic_gradient=false` until it is -- is Shinichi's, because it changes a gate another slice depends on.

- [x] GB.4: with the warm start UNCONFINED (the S7c restriction to Nelder-Mead removed), fixture D's regression test still passes and the converged answer is unchanged at rtol 1e-8. This is the gate that S7c could not pass with an FD gradient.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_laplace_identity.jl --gate warm_identity
  EXPECT: GATE G7c.1 PASS
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-s9a-hessian-20260921; path=01d9749a8aeb/36 entries; output=│   cold_iters = 6 | └   warm_iters = 6
  scheduled session ran this gate on 2026-09-21 ~04:35Z, before making any GB.4 change, purely to
  learn whether it was green. It was not: `GATE G7c.1 FAIL`, 15 passed and 1 ERRORED, an uncaught
  `DimensionMismatch` thrown from `_grouped_analytic_loglik_gradient` at `dW * bhat`
  (src/grouped_nongaussian_fit.jl:478) on fixture D, the 4-source `common=true` design, reached
  through the ordinary public `fit_gllvm`. Not a tolerance and not a count: the analytic gradient
  CRASHED on any model with two or more grouping terms, and it crashed rather than falling back to
  `_grouped_fd_gradient`, so it was a hard user-facing regression sitting on the branch.
  Root cause proved by direct probe BEFORE any edit (two terms, 12 and 5 groups, p=2: `size(W)` =
  (24,14) against `size(dW)` = (24,10) for k=1 and (24,8) for k=2, short by exactly the missing
  group factor). Fixed with one line plus its explanatory comment, at the placeholder push in
  `_grouped_laplace_design_jacobian`. Gate re-run after the fix: **GATE G7c.1 PASS, 21/21**;
  fixture A cold vs warm `rel_ll=0.0`, `rel_par=0.0`, iterations 3/3; fixture D cold and warm
  gradient norms bit-identical at 7.815970093361102e-8, iterations 6/6.
  **The gate is NOT ticked, and the distinction is the point.** GB.4 asks for the warm start to be
  UNCONFINED -- the S7c restriction to Nelder-Mead removed -- and only then for this check to pass.
  The restriction is still in place. What the PASS above establishes is the weaker, still useful
  fact that S8 does not break S7c's identity. Unconfining stays blocked behind Shinichi's item-1
  decision: option (c) of that decision (`analytic_gradient=false` by default) puts the FD gradient
  back, and the S7c confinement exists precisely to protect an FD gradient from a warm inner mode.
  Doing GB.4 before the decision would build on a default that may move.
  **GATE NOW MET, 2026-09-21 ~13:05Z, scheduled session, with the restriction actually removed.**
  Item 1 closed as option (a), so the default stays `analytic_gradient=true` and the FD-protection
  argument for the confinement no longer applies on the default path. One change in
  `src/grouped_nongaussian_fit.jl` (in this leaf's OWNS list), commit `5a9e37feb`: the BFGS
  refinement optimises `objective_refine = analytic_gradient ? objective_warm : objective_cold`
  instead of always `objective_cold`. The FD fallback inside `grad_fn`, the reported gradient and
  the final FD Hessian all still difference `objective_cold`, so nothing that is differenced sees a
  warm mode; the S7c comment block and the `warm_start_inner` docstring were corrected to say so.
  CHECK re-run on the committed text: **GATE G7c.1 PASS, 21/21** -- fixture A cold vs warm
  `rel_ll=0.0`, `rel_par=0.0`, iterations 3/3; fixture D (the 4-source `common=true` design) cold
  and warm gradient norms bit-identical at 7.815970093361102e-8, iterations 6/6, both converged.
  **The change was shown to be REACHED rather than assumed to be**, which matters because the gate
  output above is byte-identical to the pre-change run. A scratchpad probe (not in the lane) fitted
  fixture A with `analytic_gradient=true` under both builds and read the `_grouped_chol_stats`
  counters: with the confinement, warm run `calls=94 fresh=188 reused=744`; with it lifted,
  `calls=94 fresh=187 reused=685`. Summed inner CHOLMOD factorisations fall **744 -> 685 (-7.9%)**
  while the objective-call count, the loglik (-2.025469254255519e+03) and both parameters
  (0.9211786338480774, -0.3668330786521219) are bit-identical. The cold runs are identical under
  both builds (`reused=1040`), which is the control. So the gate's identity is a real identity
  across a path that genuinely changed, not a no-op passing itself.
  What this gate does NOT establish: the -7.9% is inner-solve work on one small fixture, not a wall
  measurement, and GB.5's 1.217x / 1.782x were measured BEFORE this change. The arc's headline
  numbers are unchanged and are not claimed to improve.

- [ ] GB.5: objective calls and summed inner Newton iterations are reported before and after (118 and 711 banked at fixture A); the wall on fixture A is recorded against 0.150383 s and Latte's 0.015 s; the larger fixture against its own GA.1 baseline. Numbers reported whatever they are, no claim beyond them.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. bench/profile_grouped_glmm.jl --gate sections_after
  EXPECT: GATE GB.5 PASS
  EVIDENCE: pending
  06:30-07:00Z), `pgrep -x julia` empty before every run, no other Julia process on the machine. The
  `--gate sections_after` mode did not exist when this gate was written; it was implemented this session
  in `bench/profile_grouped_glmm.jl` (in this leaf's OWNS list) and the TSV is
  `bench/results/grouped_sections_after_6f2a98f36.tsv` (git-ignored).
  Both settings are measured inside ONE process -- one untimed warm-up then 5 timed reps (small) or 3
  (large) per setting, median reported, min and max printed -- so the machine state that contaminates an
  absolute number is shared by both halves and the RATIO survives it.

  | fixture | wall BEFORE (FD) | wall AFTER (analytic) | speedup | objective calls | inner Laplace fits | inner Newton iters |
  |---|---|---|---|---|---|---|
  | `glmm_200x5` (nθ=2, 5 reps) | 0.149013 s | **0.122395 s** | **1.217x** | 116 -> 84 | 116 -> 92 | 551 -> 407 |
  | `glmm_5000x3_g500` (nθ=6, 3 reps) | 10.566280 s | **5.928687 s** | **1.782x** | 512 -> 284 | 512 -> 303 | 2735 -> 1437 |

  Integrity checks, all of which had to hold for the PASS and did: section sums land within 0.2% of the
  measured wall on all four measurements (bound 10%); after-vs-before loglik rel 4.154e-15 (small) and
  2.011e-15 (large), inside rtol 1e-8; the AFTER driver's loglik matches a real `fit_gllvm` call at rel
  **0.000e+00** on both fixtures, so the after path is the shipped code and not a shadow; `converged=true`
  on all four; **zero fallbacks to `_grouped_fd_gradient`** on the analytic path, so every analytic second
  is analytic; and call counts were identical across every rep, checked rather than assumed.
  Where the time went on the large fixture (the GA.2 partition, re-measured after): the FD gradient's
  20 invocations and 240 objective calls costing 5.7901 s collapse to 19 analytic invocations costing
  **0.6441 s**, a 9.0x cut in that one section, while Nelder-Mead (2.85 s) and the single final FD
  Hessian (1.69 s) are paid in full exactly as before. That is why the wall speedup is 1.78x and not 9x,
  and it is also why the number is a FLOOR: GB.4 (unconfine the warm start) and dropping the diagnostics
  Hessian are both still undone, and GA.2 put the Hessian alone at 16.0% of this fixture's wall.
  Fixture A against the banked numbers, as the gate asks: pre-S7c 0.184721 s, post-S7c **0.150383 s**,
  this AFTER **0.122395 s**; against Latte's 0.015 s the gap closes from **10.03x to 8.16x**. The banked
  118 objective calls / 711 summed inner Newton iterations are restated rather than differenced: they
  predate S7c's counter, and this run's own before-half measures 116 / 551 on the same fixture, so the
  honest comparison is 116 -> 84 and 551 -> 407 within this run.

  **A BANKED NUMBER IS CORRECTED, AND IT IS THE LESS FLATTERING DIRECTION.** The progress record's
  04:00Z interim entry reported the small fixture at **0.085829 s and 1.64x**. That does not reproduce.
  THREE independent measurements taken this session put the after-wall at 0.120141 s, 0.122395 s and
  0.117778 s, the last one from a separate cross-check script calling `fit_grouped_nongaussian` directly
  -- the same entry point the interim script used -- which gave **1.265x** on the small fixture and
  **1.772x** on the large. So the LARGE fixture's interim 1.78x is reproduced to within noise and stands;
  the SMALL fixture's 0.0858 s / 1.64x is an outlier and is **withdrawn**. The machine was LESS loaded
  for these runs than for the interim one (load average 7.7 against 28.9), so the direction cannot be
  explained by contention. The arc's stated goal -- materially faster than 0.150 s on the Latte 200x5
  fixture -- is still MET at 0.122 s, by about 19% rather than by the 43% the interim number implied.

  **A REGRESSION THIS SESSION INTRODUCED AND THEN CAUGHT, recorded because the catching is the lesson.**
  The first version of the patch gave `_S8_measure_driver` ONE gradient closure that branched on
  `analytic_gradient` inside its body. GB.5 passed on it. But re-running GA.1 -- a ticked gate nobody
  had asked to re-run -- returned `GATE GA.1 FAIL glmm_200x5: sections sum gap 0.127 exceeds 10%`.
  Cause: a single branching closure is inferred as a whole on its first call, which drags
  `_grouped_analytic_gradient` through compilation even on the FD path; that compilation lands inside
  `driver_wall` but in none of the section buckets, and on a fixture whose whole wall is 0.17 s it moved
  the gap from 8.0% to 12.7%. Fixed by selecting between TWO separate closures up front, so the FD path
  never references the analytic function. **The 10% bound was not touched.** GA.1 re-run after the fix:
  `GATE GA.1 PASS`, small-fixture sections gap **0.082** against the banked 0.080 and 0.084, FD-attributable
  share **0.829** against the banked 0.830 and 0.827, large fixture **0.954** unchanged -- so GA.1's ticked
  evidence is reproduced, not merely restored to green. Worth carrying: the small fixture's 10% bound has
  always been marginal (8.0%, 8.4%, 8.2% across three runs), and a fixed overhead of about 15 ms is enough
  to breach it. Anything added inside that driver has to be checked against GA.1, not just against GB.5.

  Scope note on the implementation: `_S8_measure_driver`'s new `analytic_gradient` kwarg DEFAULTS TO
  FALSE, which is not the package's default (src/grouped_nongaussian_fit.jl:579 defaults it true). That is
  deliberate -- `--gate sections` is GA.1's ticked CHECK and its banked EVIDENCE was measured on the
  all-FD path, so flipping this default would have made a ticked gate quietly stop reproducing its own
  numbers. `--gate sections_after` passes both settings explicitly.

- [ ] GB.6: full `Pkg.test()` green apart from the known pre-existing test_em_louis.jl:127 flake; test/test_grouped_laplace.jl unchanged from 69a69b0a0.
  CHECK: test -z "$(git diff --name-only 69a69b0a0 -- test/test_grouped_laplace.jl)" && env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Pkg; Pkg.test()'
  EXPECT: tests passed, or exactly 1 failed being test_em_louis.jl:127
  EVIDENCE: pending
  `git diff --name-only 69a69b0a0 -- test/test_grouped_laplace.jl` is EMPTY, so that file is
  unchanged from the baseline as the CHECK requires.

  **Run 3 (2026-09-21, source state of `7f175835e`, the run this gate passes on).** Launched 14:41Z by the
  scheduled session `6ff2e4cb`, wrapper PID 73451 under `script -q` (a pty, so the buffered output
  survives), julia worker PID 73633; exited 16:21Z. Read fresh by a later session, not inherited.
  **`GLLVModels.jl | 16328 pass, 1 fail, 0 error, 19 broken, 16348 total, 100m27.3s`**
  (log line 981; `Pkg.test()` exits nonzero on any failure, so the trailing `ERROR: Some tests did
  not pass` at line 1404 is that one failure being reported, not a second fault).
  **Exactly one `Test Failed` line in the whole log** (line 395): `test_em_louis.jl:127`,
  "SE PRIMARY GATE: EM-SEM SEs match dense-Hessian SEs (p=10)", 65 passed and 1 failed -- the
  pre-existing flake this gate explicitly allows. The EXPECT is therefore satisfied.
  **Run 2's SECOND failure is gone, and that is the point of the run.** `test_grouped_laplace_identity.jl`
  now reports "grouped Laplace CHOLMOD reuse identity (S7b) | 20 | 20" (line 1108), because item 1 was
  decided as D-273 (guard BOTH optimiser paths: the S7b pin stays on the FD path, a bound is asserted
  on the analytic path) and applied in `3ca07a491`. The count rose 16 -> 20 as the second path's
  assertions were added, and nothing was re-pinned or switched off to get there.
  `test/test_grouped_analytic_grad.jl` ran INSIDE the suite and passed: "grouped analytic outer
  gradient vs finite differences | 12 | 12 | 0.9s" (line 1110), up from 8 on run 2 -- the four added
  assertions are the compacted-column regression test from `7f175835e`.
  Precondition re-checked against this exact HEAD, not carried over: `git diff --name-only 69a69b0a0
  -- test/test_grouped_laplace.jl` EMPTY; `git status --porcelain` EMPTY, so the source state tested
  is the tree as committed.
  **CORRECTED 2026-09-21 ~00:50Z, and the correction matters more than the verdict.** An earlier
  version of this block said the suite ran on HEAD `58fdee27f`. It did not, and could not have:
  `7f175835e` is timestamped 14:40:30Z, the suite launched at 14:41Z, and `58fdee27f` was not created
  until 14:59:13Z, eighteen minutes into a hundred-minute run. The VERDICT is unaffected, because
  `58fdee27f` touches `docs/src/low-level-reference.md` alone and `Pkg.test()` never reads `docs/src/`,
  so its `src/` and `test/` are identical to `7f175835e`'s. What was wrong was naming a commit that did
  not yet exist as the thing under test.
  **The run also does not describe this branch's CURRENT test corpus.** `test/test_poisson_grad_perf.jl`
  was re-pinned twice after the run finished: `9db03e8e9` (version-keyed, 19:54:54Z) and `4b3832f76`
  (StableRNG, one baseline valid on every Julia, 2026-09-22 00:32:57Z). So the 16,348 total counts the
  pre-re-pin corpus. Neither re-pin adds or removes a test; both change one constant and its comment.
  Environment: Julia 1.10.0 (`juliaup` default, aarch64-apple-darwin), `JULIA_NUM_THREADS=4`,
  `OPENBLAS_NUM_THREADS=1`, alone on the Mac Studio. Log:
  `/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-Shinichi/6ff2e4cb-4cde-45b7-8d5e-63ce63ba3d8b/scratchpad/gb6_suite_run3.log`
  (158,796 bytes, retained).
  **Wall was 100m27s against the 95-minute estimate below** -- the estimate holds, and a future run
  should budget 95 to 105 minutes.
  **This gate's verdict is LOCAL only, and the CI story has since moved.** When this run finished,
  `Julia 1 (1.13.0) ubuntu shard 3/4` was failing `test/test_poisson_grad_perf.jl:70`, which no Julia
  1.10 run can reproduce. The cause was found: `MersenneTwister(20260901)` yields a different stream
  after Julia 1.10, so the fixture was DIFFERENT DATA on that runner, not a regression and not a
  better optimum. It was version-keyed in `9db03e8e9` (shard 3/4 then passed on both Julia legs) and
  properly fixed in `4b3832f76`, which moves the fixture to `StableRNGs` and one baseline valid on
  every Julia. Both landed on THIS leaf's branch AFTER this suite ran, so the new constant and the
  StableRNG code path have been executed by CI and by targeted gates, but by no full local suite.
  That is the honest residual: GB.6's 16,348 total does not cover them.

  **Runs 1 and 2 are kept below as history, not as the verdict.**
  Run 2 (2026-09-21, after the Printf fix below), the first genuinely COMPLETE suite this arc has
  had: **`GLLVModels.jl | 16320 pass, 2 fail, 0 error, 19 broken, 16341 total, 94m37.3s`**.
  The two failures are exactly the two already known, and there is no third:
  - `test_em_louis.jl:127`, "SE PRIMARY GATE: EM-SEM SEs match dense-Hessian SEs (p=10)", 65 passed
    and 1 failed -- the pre-existing flake this gate explicitly allows.
  - `test_grouped_laplace_identity.jl:49`, 16 passed and 1 failed -- the S7b call-count assertion,
    item 1, Shinichi's decision.
  So the gate's EXPECT ("exactly 1 failed being test_em_louis.jl:127") is not satisfied, and the one
  extra failure is the decision itself. Nothing else in 16,341 tests regressed under S8.
  `test/test_grouped_analytic_grad.jl` now executes INSIDE the suite and passes there
  ("grouped analytic outer gradient vs finite differences | 8 | 8 | 0.5s"); until this session added
  the include line it had never run in the suite or in CI at all.
  Run 1 is recorded as a caution rather than deleted. It reported
  `3847 pass, 2 fail, 1 error, 3853 total, 11m12.4s`, which reads like an almost-clean suite and is
  not one: the single error was `ArgumentError: Package Printf not found in current path`, thrown
  while loading the new gate file, and it propagated out of the `@testset` at `test/runtests.jl:48`,
  so **every file after `runtests.jl:192` never ran**. 3847 was a partial count. Cause: the gate file
  `using`s `Printf`, which resolves fine under `julia --project=.` but is absent from the test
  environment `Pkg.test()` builds. Fixed by adding the stdlib to `test/Project.toml`, following this
  repo's own precedent `f15ae2f52`.
  **OWNS EXTENSION, declared rather than slipped in:** `test/Project.toml` is not in this leaf's OWNS
  list and was edited anyway, for one stdlib line. The alternative was rewriting fifteen `@printf`
  calls in a file whose job is to print evidence legibly. Checked before editing: no live lane has
  touched that file since origin/main, and its last change on this lineage was 2026-09-17.
  **Timing correction for whoever runs this next:** a full unsharded local `Pkg.test()` here is a
  **95-minute** job. The pre-run estimate was 20 to 45 minutes, extrapolated from CI's "8 shards is
  about 1 h of runner time"; that was wrong, and it overran. Estimate from 95 minutes, not from the
  shard arithmetic.

ABANDON: GA.2 Not a runnable gate and never was. Its CHECK is "read the TSV written by GA.1 and state the three section shares", a human decision, and its EXPECT is the decision RULE rather than any program output, so gate-check can only ever report it unmet. The decision itself WAS taken and is recorded: the FD-attributable share on the larger fixture cleared 25 per cent, the arc proceeded to GATE B, and S8 and S9 both shipped on that basis. Kept as a decision record, marked so the ledger stops claiming a machine can check it.

## STOP conditions (report, never smooth over)

An identity failing at rtol 1e-8. A tolerance that would need widening to pass. GA.2 below 25%. Any need to touch HSquared.jl, hsquared, PR #781, or src/takahashi_selinv.jl. Two Julia suites at once.
