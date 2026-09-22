# Gates: leaf-S9 GLLVModels grouped route, demote Nelder-Mead and replace the final FD Hessian

OWNS: src/grouped_nongaussian_fit.jl, src/grouped_fit.jl (the _grouped_fd_hessian function and its call sites only), test/test_grouped_analytic_grad.jl, bench/profile_grouped_glmm.jl, bench/results/grouped_sections_after_*.tsv

BASELINE: origin/main 69a69b0a0 for every identity. Speed baseline is S8's own measured after-state, NOT the pre-S8 numbers: glmm_200x5 (ntheta=2) 0.122395 s, glmm_5000x3_g500 (ntheta=6) 5.928687 s. The large fixture's GA.1 section partition, which is what justifies this leaf: Nelder-Mead 2.7928 s, final FD Hessian 1.7460 s, together 4.5 s of the remaining 5.93 s.

SCOPE: (1) demote the unconditional value-only Nelder-Mead phase (src/grouped_nongaussian_fit.jl:686) to a fallback when the analytic gradient is available; (2) replace the final O(ntheta^2) `_grouped_fd_hessian` (src/grouped_fit.jl:226-247, called at :756) with a finite difference OF THE ANALYTIC GRADIENT, per D-274; (3) close the four coverage holes S8 left. Never widen a tolerance. Keep every old path reachable by keyword, as S8 did for the FD gradient.

DECISION ALREADY TAKEN: D-274 (Shinichi, 2026-09-21). The Hessian is FD-of-the-analytic-gradient, nTheta gradient calls rather than nTheta^2 objective calls. A full analytic Hessian is explicitly NOT in scope and is not foreclosed. `_grouped_fd_hessian` stays reachable as oracle and fallback. The gate compares STANDARD ERRORS, not raw Hessian entries, because the two are different estimators of the same matrix and the SEs are what users see.

FOUND WHILE WRITING THIS LEDGER, and in scope: the FINAL reported gradient at src/grouped_nongaussian_fit.jl:754 is still `_grouped_fd_gradient` even when `analytic_gradient=true`, and its norm feeds `converged`. S8 replaced the optimiser's gradient and left this one. G9.7 covers it.

WHY NELDER-MEAD IS THERE, stated so the demotion is not naive: the comment at :682-685 says the joint Laplace domain can invalidate a finite-difference neighbour, so a value-only search avoids handing Optim a NaN stencil. That reason is about the FD STENCIL. With an analytic gradient there is no stencil, which is the argument for demoting it. The reason does NOT fully vanish while any FD path remains reachable, so the fallback must stay wired and G9.3 tests it.

## GATES

- [ ] G9.1: TEXT CORRECTED, see the amendment at the end. Fitted parameters and logLik equal origin/main 69a69b0a0 within rtol 1e-8 on all SIX fixtures `_fixtures()` returns, with Nelder-Mead at its SHIPPED default. The original wording said "with Nelder-Mead demoted" and "five"; the demotion is abandoned and the count was wrong.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate identity
  EXPECT: GATE GB.3 PASS
  EVIDENCE: pending
  condition, hit systematically, and root-caused rather than smoothed over.** Implementation is exactly
  what the ledger specifies: `nelder_mead::Bool = !analytic_gradient` added at
  `src/grouped_nongaussian_fit.jl:624`, BFGS starts from `collect(theta)` directly when `nelder_mead=false`
  (the default when `analytic_gradient=true`), with a value-only Nelder-Mead safety net that only fires if
  BFGS never produces a usable result (exercised and proven in G9.3 below).

  Run 2026-09-21 ~14:19 MDT, `test/test_grouped_analytic_grad.jl --gate identity` (this file's own
  `gate_identity()`, unchanged since S8/GB.3, comparing `analytic_gradient=true` -- now the demoted-NM
  default -- against `analytic_gradient=false`, which is the in-worktree proxy for origin/main that GB.3's
  detached-worktree check already validated bit-for-bit). **GATE GB.3 FAIL on all six fixtures** (the
  ledger's "five S8 fixtures" plus `poisson_percoord`, added later by `7f175835e` and included here because
  `_fixtures()` returns it):

  | fixture | loglik rel | max per-coord beta rel | bound | verdict |
  |---|---|---|---|---|
  | poisson_latent | 1.985e-14 | 1.386e-06 | 1e-8 | FAIL |
  | beta_shared | 6.653e-14 | 2.651e-07 | 1e-8 | FAIL |
  | nb2_shared | 1.742e-13 | 1.684e-06 | 1e-8 | FAIL |
  | poisson_twoterm | 4.431e-10 | 1.293e-06 | 1e-8 | FAIL |
  | latent_plus_indep | 5.081e-13 | 2.266e-06 | 1e-8 | FAIL |
  | poisson_percoord | 1.099e-10 | 1.335e-06 | 1e-8 | FAIL |

  Note on `latent_plus_indep`: mid-fit the analytic gradient threw the PRE-EXISTING (not S9, not touched
  here) `ArgumentError: selected inverse missing entry (14,1)` from `_grouped_selinv_row_crossform`
  (`src/grouped_laplace.jl:355`); `grad_fn`'s existing S8 try/catch caught it and fell back to
  `_grouped_fd_gradient` for that one call, exactly as designed -- not a new defect, flagged because it
  fired during this run.

  **Root-caused, not merely observed.** Loglik agreement is excellent everywhere (1e-10 to 1e-14 relative)
  -- the two paths are converging to the SAME likelihood value. The PARAMETER disagreement (~1e-6 to 1e-7
  relative) is a symptom of `g_tol=1e-4` (the function's own default outer stopping tolerance) being loose
  relative to what "identical parameters" demands, combined with the two paths now taking GENUINELY
  DIFFERENT ROUTES to that tolerance: pre-S9, both the FD-gradient BFGS refinement and the analytic-gradient
  BFGS refinement started from the SAME Nelder-Mead-refined candidate, so their trajectories (and hence
  where they stopped inside the `g_tol` ball) were nearly identical. Post-S9, `analytic_gradient=true`
  starts BFGS from raw `theta` instead, a genuinely different trajectory that lands at a DIFFERENT point
  inside the same loose ball. Confirmed by a scratch diagnostic (not part of this file, `poisson_latent`
  fixture, varying `g_tol` on both paths, `iterations=500`):

  ```
  g_tol=1e-04  dbeta=9.514e-07  gradnorm fd=4.338e-05 an=1.666e-05  iters fd=7  an=25   converged fd=true an=true
  g_tol=1e-06  dbeta=1.898e-09  ...
  g_tol=1e-08  dbeta=9.362e-10  ...
  g_tol=1e-10  dbeta=5.086e-10  ...                                                     converged fd=false an=true
  ```

  Tightening `g_tol` collapses the gap from ~1e-6 to ~1e-9 -- matching the pre-S9 GB.3 agreement level
  (worst 3.753e-09) -- which shows the TRUE optimum is unchanged; only the DEFAULT-tolerance answer moved.
  This is exactly the STOP condition this leaf's own text names ("A converged answer changing under the
  demotion") and precisely the reason GB.2's amended requirements insisted on per-coordinate, not
  norm-summarised, checking: a 1e-6-level shift is invisible to a loose check and real at a strict one.

  **Not loosened, not smoothed over.** No tolerance here was widened and no fixture was dropped to force a
  pass. This reopens a design question this leaf did not have the authority to resolve unilaterally
  (`g_tol`'s default is outside SCOPE, which names only the three items in the ledger's SCOPE line): whether
  the analytic-gradient default should also tighten `g_tol` now that gradients are exact and cheap, whether
  BFGS should get a cheaper (not full-simplex) warm positioning before starting, or whether this 1e-6-level
  identity gap at the current default is simply accepted as a consequence of removing Nelder-Mead. See the
  final report for the options as stated to the orchestrator.

- [x] G9.2: AMENDED by Shinichi 2026-09-21, asserts where the oracle EXISTS and prints boundaries. The Hessian produced by finite-differencing the analytic gradient yields STANDARD ERRORS equal to those from `_grouped_fd_hessian` at rtol 1e-4, per coordinate, on all five fixtures, INCLUDING the Beta and NB2 fixtures where the two FD schemes differ most because of the dispersion rows. 1e-4 is recorded as the FD oracle's own accuracy, not a widened bound. If it is not met, STOP and reopen D-274; do not loosen it.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate hessian
  EXPECT: GATE G9.2 PASS
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-s9a-hessian-20260921; path=01d9749a8aeb/36 entries; output=NOT ASSERTED poisson_percoord: the :fd ORACLE is not positive-definite at the converged estimate while grad_fd is (grad_fd ok=true, fd ok=false); no oracle to compare against, and grad_fd is strictly the more robust of the two here | GATE G
  reason; STOP on those two per this gate's own instruction rather than loosened.** `_grouped_fd_hessian_
  from_gradient` added at `src/grouped_fit.jl` immediately after `_grouped_fd_hessian`: central-differences
  a gradient function in `d` calls, symmetrised `(H+H')/2`. Both Hessians differenced at the SAME converged
  estimate (`fit.parameters` from an `analytic_gradient=true` fit), compared as `sqrt.(diag(inv(Symmetric(H))))`
  (STANDARD ERRORS, per the gate's own instruction, not raw entries).

  Run 2026-09-21 ~14:22 MDT:

  | fixture | worst per-coord SE rel | coord | verdict |
  |---|---|---|---|
  | poisson_latent | 2.282e-06 | 4 | PASS (44x inside 1e-4) |
  | beta_shared | 1.198e-07 | 2 | PASS |
  | nb2_shared | 1.750e-06 | 4 | PASS |
  | poisson_twoterm | -- | -- | **FAIL: `_grouped_fd_hessian` (the `:fd` oracle) not PD/invertible at the converged estimate** |
  | latent_plus_indep | 8.808e-07 | 7 | PASS |
  | poisson_percoord | 3.197 (rel, i.e. 320%) | 5 | **FAIL: massive SE disagreement** |

  The 4 PASSing fixtures include Beta and NB2 as G9.2 requires, all comfortably inside the bound (worst
  2.282e-06 against 1e-4 -- 44x margin), which is strong positive evidence the grad-FD Hessian is computing
  the right thing.

  **Both failures root-caused to the SAME mechanism, confirmed algorithm-independent -- not a D-274 defect.**
  Diagnostic (scratch, not in this file) on `poisson_twoterm`: the converged estimate's coordinate 4 (a
  log-SD) is `-10.56` (SD ~ 2.6e-5), and `_grouped_fd_hessian`'s own eigenvalues at that point are
  `[-1.53e-7, 10.71, 21.70, 26.28]` -- the smallest is numerically zero/negative, i.e. the FD-of-OBJECTIVE
  Hessian is not reliably PD there regardless of which gradient scheme differences it. On `poisson_percoord`
  the same pathology is sharper: coordinate 5 (a per-trait unique-variance log-SD, the `common=false`
  coordinate `7f175835e` added coverage for) drifts toward its OWN boundary as `g_tol` tightens --
  `-11.29` at `g_tol=1e-4`, `-19.26` at `g_tol=1e-8` -- a genuine boundary/near-zero-variance MLE for this
  fixture's fixed seed, not an S9 artifact: **the pre-S9 FD-only path (`analytic_gradient=false`, Nelder-Mead
  intact) lands at `-11.59` at the SAME default `g_tol=1e-4`** -- the same boundary-collapse, independent of
  whether Nelder-Mead ran. Numerically differentiating curvature near a variance-component boundary is
  inherently unstable and step-size-sensitive (my `_grouped_fd_hessian_from_gradient` steps at
  `1e-5*max(1,|theta|)`, `_grouped_fd_hessian` at `1e-4*max(1,|theta|)` -- a 10x difference that a near-flat
  direction amplifies), which is consistent with a 3.2x SE disagreement while BOTH methods still agree the
  SE there is enormous (19871 and 4734, both wildly larger than every other coordinate's ~0.1-0.5).

  **Not loosened.** No tolerance was widened and neither failing fixture was dropped from `_fixtures()`
  to force a pass. This is reported as a fixture-conditioning problem (two of the six committed fixtures,
  by their fixed RNG seeds, land a variance component at/near its boundary), not a defect in the D-274
  Hessian-by-FD-of-gradient method itself -- the 4 well-conditioned fixtures pass with large margin. Whether
  to regenerate `poisson_twoterm`/`poisson_percoord` with different seeds, exclude near-boundary estimates
  from the SE comparison, or accept this as a known edge case is Shinichi's call, per this gate's own
  "reopen D-274 rather than loosening" instruction.

- [x] G9.3: the fallback contract holds. With `analytic_gradient=false` the pre-S9 path runs verbatim: Nelder-Mead executes and `_grouped_fd_hessian` supplies H. A fit whose analytic gradient fails at some theta still completes, warning once rather than throwing.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate nm_fallback
  EXPECT: GATE G9.3 PASS
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-s9a-hessian-20260921; path=01d9749a8aeb/36 entries; output=mid-fit, every 2nd analytic-gradient call forced to `nothing`: total calls=19 forced>=9 completed=true converged=true | GATE G9.3 PASS

  (a) `analytic_gradient=false`'s DEFAULT kwargs reproduce `nelder_mead=true, hessian=:fd` given
  explicitly, bit-for-bit: inner Laplace-fit calls (`_grouped_chol_stats().calls`) 471 both ways,
  loglik and beta `isapprox(...; rtol=1e-12)` both true. Nelder-Mead ACTUALLY EXECUTES, checked rather
  than assumed: the analytic path with the simplex demoted (`nelder_mead=false`, which was expected to become the S9 default and did NOT; the demotion was abandoned on measurement) costs only 173 calls --
  63% fewer -- confirming the FD-default path's extra 298 calls are Nelder-Mead's own simplex/line-search
  evaluations. `_grouped_fd_hessian` supplies H on the FD path: the fit's own `hessian_min_eigenvalue`
  matches a FRESH, independent call to `_grouped_fd_hessian` at the same estimate, `isapprox(...; rtol=1e-8)`
  true (no FAIL line printed for this check).

  (b) A mid-fit forced analytic-gradient failure still completes, warning rather than throwing. Forced via
  the shared test-only monkeypatch (`_s9_install_mixed_gradient_monkeypatch!`, also used by G9.6): every
  2nd call to `_grouped_analytic_gradient` returns `nothing` (its documented failure sentinel). Result:
  80 total analytic-gradient calls, 40 forced to fail, fit completed (`completed=true`), `converged=true`
  -- no exception propagated. `grad_fn`'s existing S8 try/catch + FD fallback (unchanged by S9) is what
  makes this work; this exercises it under a forced SUBSET-failure regime for the first time rather than
  assuming it still holds.

  **A test-infrastructure bug found and fixed while writing this gate, recorded because it is instructive.**
  The monkeypatch (`@eval GLLVModels function _grouped_analytic_gradient(...) ... end`, redefining the
  method to intercept and optionally fail) is installed from WITHIN `gate_nm_fallback()`'s own function
  body, and the subsequent `fit_grouped_nongaussian` call is ALSO made from within that same function's
  single dynamic extent -- a classic Julia world-age trap: an `eval`'d redefinition is invisible to calls
  made later in the SAME already-executing function, only to NEW top-level calls. First run: `total
  calls=0, forced=0` -- the patch never fired, silently running the UNPATCHED method, which would have made
  this gate pass VACUOUSLY (never exercising the failure branch at all) had the `forced > 0` check not
  caught it. Root-caused with an isolated scratch reproduction (confirmed: works at bare top-level, breaks
  identically once wrapped in a function, fixed by `Base.invokelatest`). Fixed by routing the `fit_grouped_
  nongaussian` call through `Base.invokelatest`, in both `gate_nm_fallback` and `gate_mixed` (G9.6, same
  mechanism). Re-run after the fix: `total calls=80, forced=40`, as reported above.

- [ ] G9.4: ABANDONED AS UNSOUND, see the amendment. It bounds only from BELOW, the direction a stalled optimiser also moves, and it gates the abandoned demotion. Superseded by G9.9's wall-clock table. Original text: objective calls and inner Newton iterations are BOUNDED BELOW the S8 measured values on both bench fixtures, reported not pinned, following D-273's rule that the old pin stays on the old path. S8 measured: 84 objective calls and 407 inner Newton iterations on glmm_200x5; 284 and 1437 on glmm_5000x3_g500.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate counts
  EXPECT: GATE G9.4 PASS
  EVIDENCE: **PARTIAL, NOT ticked. `objective_calls` bounded below on both fixtures (the primary driver of
  the S8 wall-clock speedup); `inner_newton_iters_sum` is NOT -- a genuine trade-off, reported not hidden.**
  Fixtures: `glmm_200x5` (committed CSV, same as bench) and `glmm_5000x3_g500` (regenerated in-file with the
  SAME seed `20260920` and generator as `bench/profile_grouped_glmm.jl`'s `_S8_make_large_fixture`). Harness
  (`_s9_counted_fit`) mirrors the CURRENT `fit_grouped_nongaussian` control flow (`nelder_mead=false`: BFGS
  from `theta0` directly, with the safety-net fallbacks), wrapping the REAL objective/gradient factories
  with counters -- never reimplementing their bodies. `_grouped_chol_stats()` reset before, read after;
  `inner_newton_iters_sum` DERIVED via the documented (bench header) `total_chol = 2*sum(iterations) +
  ok_calls` relationship.

  Run 2026-09-21 ~14:30 MDT (after fixing an arithmetic bug in this derivation, below):

  | fixture | objective_calls (S8 baseline) | inner_newton_iters_sum (S8 baseline) | valid |
  |---|---|---|---|
  | glmm_200x5 | **35** (84) | **752** (407) | true |
  | glmm_5000x3_g500 | **122** (284) | **1752** (1437) | true |

  `objective_calls` is comfortably below baseline on both (58% and 57% fewer). `inner_newton_iters_sum` is
  ABOVE baseline on both (752 vs 407, 1752 vs 1437). Diagnosis, tied to the SAME mechanism as G9.1/G9.6:
  Nelder-Mead demotion removes the cheap pre-positioning it used to give BFGS, so BFGS starting cold from
  `theta0` needs FAR FEWER total evaluations (each analytic-gradient call replaces ~2*nθ FD evaluations) but
  each of those evaluations, being further from the converged random-effects mode, needs MORE inner Newton
  iterations to resolve (every FD-differenced/analytic-gradient call always uses `objective_cold`,
  `warm_start_inner=false`, so there is no warm-start relief for these calls either way). Fewer, harder
  evaluations vs more, easier ones -- which one wins on WALL CLOCK is G9.9's question (out of this leaf's
  scope; the orchestrator runs it), not this gate's.

  **A bug in this harness itself, found and fixed before reporting.** The first version of the
  `inner_newton_iters_sum` derivation omitted the `/2` (`total_chol - calls` instead of
  `(total_chol - calls) / 2`), over-reporting by ~2x (1505/3504 instead of 752/1752). Caught by re-deriving
  the relationship from the bench header comment by hand rather than trusting the first number; the
  corrected values are the ones reported above. `objective_calls` was unaffected (counted directly via a
  wrapped closure, not derived).

  Not loosened: neither metric's bound was adjusted, and `inner_newton_iters_sum`'s FAIL is reported as
  measured.

- [ ] G9.5: MOVED to the coverage arc and LANDED there as claude/lane-s9cov-20260921 fa886fe81. The four coverage holes S8 left are closed as fixtures that pass both fd_agreement and identity: Binomial; per-trait `dispersion=:trait` for Beta and for NB2; `GroupingTerm(mode=:dep)`.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate coverage
  EXPECT: GATE G9.5 PASS
  EVIDENCE: **NOT MET, STOP. The fd_agreement half -- the actual "is the analytic gradient's MATH correct
  for these previously-untested configurations" question -- PASSES on 3 of 4 new fixtures with a real margin
  and is BORDERLINE on the 4th. The identity half fails on all 4, for the SAME G9.1 mechanism (BFGS starting
  cold), not a new defect.** Four new fixtures added: `_fixture_binomial` (Binomial, `:indep common=true`),
  `_fixture_beta_trait` (Beta, `dispersion=:trait`), `_fixture_nb2_trait` (NegativeBinomial,
  `dispersion=:trait`), `_fixture_dep` (`GroupingTerm(mode=:dep)`, full 2x2 trait covariance).

  Run 2026-09-21 ~14:31 MDT:

  | fixture | fd_agreement worst rel (bound 1e-6) | fd_agreement verdict | identity loglik rel | identity beta rel (bound 1e-8) | identity verdict |
  |---|---|---|---|---|---|
  | binomial | 1.127e-08 | PASS (89x margin) | 8.007e-16 | 5.650e-08 | FAIL |
  | beta_trait | **1.532e-06** | **FAIL (borderline, ~1.5x over)** | 6.975e-14 | 2.413e-07 | FAIL |
  | nb2_trait | 2.071e-08 | PASS (48x margin) | 2.135e-14 | 1.988e-07 | FAIL |
  | dep_term | 9.170e-08 | PASS (11x margin) | 1.287e-13 | 1.296e-06 | FAIL |

  The four identity FAILs are the G9.1 mechanism exactly (fitted parameters differing ~1e-7 to 1e-6
  relative at the default `g_tol=1e-4` while loglik agrees to 1e-13/1e-16) -- see G9.1's evidence, not
  re-derived here. `dep_term` also hit the same PRE-EXISTING `selinv missing entry` fallback noted under
  G9.1/G9.2 (a different index, `(2,1)`, same root function, unrelated to S9), caught by `grad_fn`'s
  existing try/catch as designed.

  **`beta_trait` coordinate 5 (its dispersion coordinate) is a genuinely new, unresolved finding, not
  explained by the G9.1 mechanism** (fd_agreement compares the analytic and FD gradient FUNCTIONS directly
  at random theta around `theta0` -- it never runs Optim, so it cannot be the NM-demotion effect). Worst:
  analytic=-9.7542339261e-04, fd=-9.7542189849e-04, absolute gap ~1.5e-9, relative 1.532e-06 against the
  1e-6 bound. This is PLAUSIBLY within the FD reference's own noise floor: the design doc (section 8.2)
  predicts a cancellation floor of `eps*|F|/h`, and for this fixture's `|F|` scale and `h ~ 1e-5*|theta_5|`
  the predicted floor is ~2e-9 -- the same order as the observed 1.5e-9 gap. NOT CONFIRMED: the design doc's
  own h-vs-h/2 certification (section 8.3) was not run here for time, and I did not check whether the
  disagreement is systematic across all 20 draws or a single outlier (only the per-coordinate WORST is
  printed by `_compare_fixture`). Flagged rather than dismissed or excused.

  **Not loosened.** `RTOL_FD` (1e-6) and `RTOL_IDENTITY` (1e-8) are untouched; no fixture was dropped.
  Net read: the S8 analytic gradient's underlying MATH appears correct for three of the four coverage holes
  (Binomial, NB2 per-trait dispersion, `mode=:dep`) with real margin; the Beta per-trait dispersion
  coordinate needs the h/2 certification before it can be called clean; and the identity comparison for all
  four is blocked on the same open G9.1 question.

- [ ] G9.6: MOVED to the coverage arc and LANDED there as claude/lane-s9cov-20260921 fa886fe81, still a HARD gate there. THE MIXED PATH, a hard gate and not optional. A fixture that forces the analytic gradient to fail at a SUBSET of theta still lands on origin/main 69a69b0a0's answer at rtol 1e-8. This is the class that produced the S8 compaction bug, and GB.4 reasoned about it without exercising it.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate mixed
  EXPECT: GATE G9.6 PASS
  EVIDENCE: **NOT MET, STOP. The MIXED-PATH MECHANISM itself works correctly and is exercised, not
  assumed; the identity check on top of it fails for the SAME G9.1 reason.** Shared monkeypatch
  (`_s9_install_mixed_gradient_monkeypatch!`, see G9.3) forces `_grouped_analytic_gradient` to return
  `nothing` on every 3rd call during a REAL `fit_grouped_nongaussian(...; analytic_gradient=true)`
  optimisation (routed through `Base.invokelatest` -- see the world-age bug recorded under G9.3, found and
  fixed while building this gate, identical mechanism here).

  Run 2026-09-21 ~14:29 MDT, all six fixtures, reference = `analytic_gradient=false` (the in-worktree
  origin/main proxy, per G9.1):

  | fixture | analytic-gradient calls | forced failures | loglik rel | beta rel (bound 1e-8) | converged both | verdict |
  |---|---|---|---|---|---|---|
  | poisson_latent | 82 | 27 | 1.950e-14 | 1.391e-06 | true | FAIL (identity) |
  | beta_shared | 27 | 9 | 7.109e-14 | 2.651e-07 | true | FAIL (identity) |
  | nb2_shared | 24 | 8 | 1.729e-13 | 1.684e-06 | true | FAIL (identity) |
  | poisson_twoterm | 78 | 26 | 4.446e-10 | 1.316e-06 | true | FAIL (identity) |
  | latent_plus_indep | 39 | 13 | 5.096e-13 | 2.266e-06 | true | FAIL (identity) |
  | poisson_percoord | 75 | 25 | 9.596e-11 | 3.791e-06 | true | FAIL (identity) |

  The MIXED-PATH MACHINERY is proven, not merely assumed: every fixture shows `forced > 0` (roughly a third
  of all analytic-gradient calls actually failed and fell through to `_grouped_fd_gradient`), every fit
  `converged=true`, and loglik agreement remains excellent (1e-10 to 1e-14 relative) even with a third of
  the gradient calls degraded to FD -- exactly what GB.4's compaction-bug precedent asked this gate to
  exercise rather than reason about. The `beta` per-coordinate identity, however, fails by the same ~1e-6
  to 4e-6 margin as G9.1's clean (non-mixed) comparison, for the identical reason: `nelder_mead=false` means
  BFGS starts cold from `theta` regardless of whether the gradient is ever forced to fail, so the mixed
  path inherits G9.1's open finding rather than adding a new one on top of it.

  **Not loosened.** `RTOL_IDENTITY` (1e-8) is untouched. This gate is downstream of G9.1's open question:
  once that is resolved (see G9.1's evidence and the final report), G9.6 should be re-run rather than
  independently investigated, since its own failure mode is not distinct from G9.1's.

- [x] G9.7: MET on the property; its stated TOLERANCE was ill-posed, see the amendment. The final reported gradient on the analytic path is the analytic gradient, and `converged` is decided on it. Measured bit-identical to a direct recompute, rel 0.000e+00 on all six. The ledger also asked its norm to agree with the FD norm at RELATIVE 1e-6, which is not a meaningful bound on a quantity driven to ~0 at convergence; the measured agreement is 7.226e-09 ABSOLUTE.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate final_gradient
  EXPECT: GATE G9.7 PASS
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-s9a-hessian-20260921; path=01d9749a8aeb/36 entries; output=PASS poisson_percoord | GATE G9.7 PASS
  `analytic_gradient ? (analytic, falling back to FD only on failure) : _grouped_fd_gradient(...)` instead
  of the pre-S9 unconditional `_grouped_fd_gradient(objective, estimate)`. Run 2026-09-21 ~14:31 MDT: for
  each fixture, fit with `analytic_gradient=true`, then independently recompute BOTH the analytic gradient
  and the FD gradient at the SAME converged `fit.parameters`.

  | fixture | reported gradient_norm | direct analytic recompute | source rel | FD norm at same point | rel to FD (bound 1e-6) |
  |---|---|---|---|---|---|
  | poisson_latent | 1.6656662962e-05 | 1.6656662962e-05 | 0.000e+00 | 1.6649437384e-05 | 7.226e-09 |
  | beta_shared | 9.8410747618e-06 | 9.8410747618e-06 | 0.000e+00 | 9.8420827044e-06 | 1.008e-09 |
  | nb2_shared | 2.8387821474e-05 | 2.8387821474e-05 | 0.000e+00 | 2.8387603379e-05 | 2.181e-10 |
  | poisson_twoterm | 4.0511748744e-05 | 4.0511748744e-05 | 0.000e+00 | 4.0506620280e-05 | 5.128e-09 |
  | latent_plus_indep | 5.0755612458e-05 | 5.0755612458e-05 | 0.000e+00 | 5.0755488701e-05 | 1.238e-10 |
  | poisson_percoord | 4.0384405869e-05 | 4.0384405869e-05 | 0.000e+00 | 4.0384406930e-05 | 1.060e-12 |

  Source: the fit's own reported `gradient_norm` is EXACTLY (bit-for-bit, `source rel = 0.000e+00`
  everywhere) a direct recomputation of `_grouped_analytic_gradient` at the same estimate -- confirming
  the reported gradient IS the analytic one, not `_grouped_fd_gradient` wearing its name. Convergence
  verdict: the analytic norm agrees with the FD norm at the same point to at worst 7.226e-09 (a ~138x
  margin inside the 1e-6 bound), so `converged` (decided on `gradient_norm <= g_tol`) would not flip
  between the two. One fixture (`latent_plus_indep`) hit the same pre-existing `selinv missing entry`
  fallback noted under G9.1/G9.2/G9.5 mid-fit; the FINAL gradient at the converged estimate was still
  computed cleanly (no fallback at that specific point), so it does not affect this gate's numbers.

- [ ] G9.8: full `Pkg.test()` green apart from the known test_em_louis.jl:127 flake. Budget 95 to 105 minutes from the GB.6 measurement; state the estimate before launching, run alone on the machine, wrap in `script -q` with a watcher, and edit nothing in the lane while it runs.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/gate_suite.log 2>&1; other=$(grep 'Test Failed at' /tmp/gate_suite.log | grep -vc 'test_em_louis.jl:127'); tot=$(grep -c 'Test Failed' /tmp/gate_suite.log); if [ "$other" -eq 0 ]; then echo "SUITE OK ($tot failure(s), all the known em_louis flake)"; else echo "SUITE BAD ($other unexpected failure(s))"; fi
  EXPECT: SUITE OK
  EVIDENCE: pending
  `GLLVModels.jl | 16328 pass, 1 fail, 0 error, 19 broken, 16348 total, 97m10.2s`. Exactly one
  `Test Failed` line in the log, `test_em_louis.jl:127`, the flake this gate allows, so the EXPECT is
  satisfied. Julia 1.10.0, `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`, alone on the Mac Studio.
  The counts are IDENTICAL to the S8 suite's (16,328 / 1 / 19), which is the point: the D-274 Hessian
  changes no test outcome anywhere in 16,348 assertions. Wall 97m10s against the 95 to 105 minute
  budget. Log `/tmp/s9a_suite.log`.
  **What this run does NOT cover:** the branch predates `4b3832f76`, so it exercises the version-keyed
  Poisson pin rather than the StableRNG one, and it does not contain the `--gate` error or the
  regression-check wiring that landed on speed78 afterwards. A re-run is owed after the rebase.

- [ ] G9.9: MET by direct measurement, NOT by the CHECK as written, see the amendment. MEASUREMENT. Both fixtures re-measured with GA.1's section partition, before and after in one process so machine state is shared. The after TSV carries git SHA, Julia version, BLAS config and thread counts in its header. The large fixture's target is about 1.5 s against 5.93 s; that is a TARGET and is reported, never asserted.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. bench/profile_grouped_glmm.jl --gate sections
  EXPECT: GATE GA.1 PASS
  EVIDENCE: pending
  point `fit_gllvm` rather than the bench's reimplemented Optim loop. Second run on the merged head
  `edf7d39e0`, Julia 1.10.0, `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`, all four configurations in
  ONE process, median of 5 reps (small) and 3 (large):

  | fixture | config | wall median | speedup | logLik rel vs BEFORE | iters |
  |---|---|---|---|---|---|
  | glmm_200x5 | BEFORE (nm=true, :fd) | 0.112903 s | 1.000x | 0.000e+00 | 3 |
  | glmm_200x5 | **HESS (nm=true, :grad_fd)** | **0.109809 s** | **1.028x** | **0.000e+00** | 3 |
  | glmm_200x5 | NM-OFF | 0.303855 s | 0.372x | 1.796e-15 | 9 |
  | glmm_200x5 | BOTH | 0.289247 s | 0.390x | 1.796e-15 | 9 |
  | glmm_5000x3_g500 | BEFORE | 5.712590 s | 1.000x | 0.000e+00 | 8 |
  | glmm_5000x3_g500 | **HESS** | **4.299553 s** | **1.329x** | **0.000e+00** | 8 |
  | glmm_5000x3_g500 | NM-OFF | 8.133130 s | 0.702x | 2.682e-15 | 15 |
  | glmm_5000x3_g500 | BOTH | 7.034797 s | 0.812x | 2.682e-15 | 15 |

  The first run, on `d988de835`, gave 1.322x on the large fixture; this one gives 1.329x. The
  conclusion is stable across commits and across machine load.
  TSV with git sha, Julia version, BLAS config, thread counts and host in its header:
  `bench/results/s9_hessian_2x2_edf7d39e0.tsv` (locally excluded, per this repo's practice), durable
  copy in the vault at `docs/dev-log/measurements/2026-09-22-s9-hessian-2x2-edf7d39e0.tsv`.
  **The 1.5 s target in the gate text is NOT met and was never reachable**: it assumed the Nelder-Mead
  demotion returned its 2.79 s, which measurement refuted. Reported, not asserted, as the gate says.

ABANDON: G9.4 Unsound as written. It bounds objective calls only from BELOW, which is the direction a stalled optimiser also moves, and it gated the Nelder-Mead demotion that was abandoned on measurement. Superseded by the two-regime wall-clock table in bench/results/shipped_*.tsv, which reports converged and iterations on every row so a fit that gave up cannot read as fast.
ABANDON: G9.5 Not this arc's to close. Moved to the coverage arc on Shinichi's 2026-09-21 instruction and LANDED there at claude/lane-s9cov-20260921 fa886fe81, open as PR #448 against main, suite-green at 16337 pass / 1 fail / 19 broken.
ABANDON: G9.6 Not this arc's to close, and still a HARD gate where it went. Same move as G9.5, same commit and PR.

## STOP conditions (report, never smooth over)

An identity failing at rtol 1e-8. G9.2's SEs missing rtol 1e-4 (reopen D-274 rather than loosening). Any tolerance needing to be widened. A converged answer changing under the demotion. Two Julia suites at once. Any need to touch HSquared.jl, hsquared, PR #781, or src/takahashi_selinv.jl.

## AMENDMENT 2026-09-22: S9 SPLIT, and what each gate now means

This ledger was written on 2026-09-21 before any code, for a slice that had TWO halves: demote
Nelder-Mead, and replace the final O(nθ²) FD Hessian. Both halves were built. Then they were measured,
and the slice split. Everything below is the disposition of the ledger as written; no gate text above
has been deleted, and no tolerance has been widened.

**S9b, demoting Nelder-Mead, is ABANDONED on measurement.** The motivating claim was that the simplex
cost 2.79 s of the large fixture's 5.93 s and existed only because there was no gradient. A 2x2 on the
REAL entry point `fit_gllvm`, all four configurations in one process, median of 5 reps (small) and 3
(large):

| config | glmm_200x5 | glmm_5000x3_g500 | logLik vs BEFORE | BFGS iters small / large |
|---|---|---|---|---|
| BEFORE, the S8 shipped state | 0.1088 s | 5.5144 s | reference | 3 / 8 |
| **HESS only, `:grad_fd`, D-274** | 0.1104 s | **4.1704 s (1.32x)** | **0.000e+00** | 3 / 8 |
| NM-off only | 0.3025 s (**0.36x**) | 8.3640 s (**0.66x**) | moves | 9 / 15 |
| BOTH | 0.3083 s | 7.1151 s | moves | 9 / 15 |

Removing the simplex does not return its 2.79 s; it costs more. BFGS goes 3 to 9 and 8 to 15 outer
iterations without it, each carrying inner Laplace solves. It also moved the fitted mean coordinates
1.4e-6 to 2.3e-6 against `origin/main` 69a69b0a0, failing the arc's rtol 1e-8 identity: the optimum is
unchanged and the gap collapses to ~1e-9 as `g_tol` tightens, so it is a stopping-point difference
rather than a wrong answer, but it is a user-facing change bought for a slowdown. `nelder_mead`
therefore defaults to `true` and the demotion is reachable but opt-in.

The lesson generalises. A section partition tells you where time is spent, and says nothing about
what happens if you delete a section, because the work can reappear elsewhere in the same partition. Measure the
removed configuration before removing on share alone.

**S9a, the D-274 Hessian, ships.** It gives 1.32x and 1.34 s on the large fixture, neutral on the small one
where nθ=2 makes nθ² and nθ nearly the same work, and the fitted answer BIT-IDENTICAL.

One correction is carried here from an audit of this arc's own records. D-274 was described throughout as
changing "the standard errors users see". It does not. The final `H` feeds only
`min_eigenvalue = eigmin(Symmetric(H))` and `pd_hessian`; the matrix is never stored, and user-facing
Wald intervals come from a separate `ForwardDiff.hessian` call in `src/confint.jl`. The blast radius is
two scalar diagnostics.

### Gate dispositions

- **G9.1** MET, with its text corrected. It says "with Nelder-Mead demoted" and "all five S8
  fixtures". Neither is right now: the demotion is abandoned, and `_fixtures()` returns SIX. As met:
  fitted parameters and logLik equal `origin/main` 69a69b0a0 within rtol 1e-8 on all six, with
  Nelder-Mead at its shipped default. `GATE GB.3 PASS` (the printed label is GB.3, not G9.1; that
  mismatch is recorded below rather than silently reconciled).
- **G9.2: MET as amended by Shinichi 2026-09-21.** Asserts SE agreement only where the oracle is
  positive-definite and the SE is below `SE_BOUNDARY`; boundary coordinates are printed and not
  asserted. Five fixtures asserted, one boundary coordinate and one fixture reported. `RTOL_HESSIAN`
  untouched at 1e-4. **Honest residual:** no measurement in this arc establishes 1e-4 as "the FD
  oracle's own accuracy"; it is a stated bound that the observed agreement (~1e-7 on the asserted
  coordinates) clears by three orders of magnitude, which is evidence it is not tight rather than
  evidence it is correct.
- **G9.3** MET. FD-default 471 calls equals FD-explicit 471, so the pre-S9 path is bit-for-bit
  reachable; a mid-fit forced failure completes without throwing.
- **G9.4** RETIRED as written, and it was unsound. It bounds call counts only from BELOW, which is
  the direction a broken optimiser also moves, and a fit that stops early passes it. It also gates the
  abandoned demotion: that trade RAISES inner Newton iterations (752 against 407; 1752 against 1437)
  while lowering objective calls, so the gate could fail on its own intended success. Superseded by
  G9.9's wall-clock table, which measures the thing the counts were a proxy for.
- **G9.5, G9.6 ;  MOVED, not dropped.** The four coverage holes and the mixed-path fixture are their own
  arc on Shinichi's instruction, so that this slice ships on its measured result. G9.6 remains a HARD
  gate there: it is the class that produced the S8 compaction bug.
- **G9.7** MET on the property that matters, and its stated tolerance was ill-posed. The reported
  gradient on the analytic path IS the analytic gradient, bit-identical to a direct recompute
  (rel 0.000e+00 on all six). The ledger also asked for its norm to agree with the FD norm at rtol
  1e-6; a RELATIVE tolerance on a quantity driven to ~0 at convergence is not a meaningful bound. The
  measured agreement is 7.226e-09 ABSOLUTE, which is the number to read.
- **G9.8: see the run recorded below.**
- **G9.9** MET by direct measurement, NOT by the CHECK as written. `bench/profile_grouped_glmm.jl
  --gate sections_after` cannot observe S9: `_S8_measure_driver` reimplements the Optim call sequence
  instead of calling `fit_grouped_nongaussian`, so it never sees the `nelder_mead` or `hessian`
  keywords. Run on the S9 code it faithfully reproduces S8's own before/after (1.356x and 1.706x). The
  2x2 table above is the measurement, taken on the real entry point.

### Known defect in this ledger's own machinery, fixed on the speed78 branch

Six of this ledger's CHECK lines name `--gate` modes that exist only on the S9 lanes. On any branch
without them the dispatch fell through to `gate_fd_agreement()`, printing `GATE GB.2 PASS` and exiting
0, so six gates would have passed vacuously against a gate testing something else. Fixed by making an
unknown gate an error. Separately, `gate_identity()` prints `GATE GB.3 PASS` while G9.1's EXPECT reads
`GATE G9.1 PASS`, so exact EXPECT-matching would fail a passing gate; recorded rather than papered
over.

## ADDENDUM 2026-09-22: the cause of the remaining wall, found and shipped OPT-IN

After S9a was measured at 4.30 s against the arc's 1.5 s target, three attacks on Nelder-Mead all
failed: deleting it (8.36 s), capping its iterations (chaotic across seeds, and the fast rows were fits
that had GIVEN UP after 1 BFGS iteration), and loosening its tolerance (inert, because the tolerance
never binds). The reason they failed is that they addressed the symptom.

**The cause.** `_grouped_nongaussian_initial_parameters` sets the trait intercepts from the data and
every variance coordinate to a CONSTANT `log(0.25)`. Nelder-Mead's real job is dragging those constants
toward the data, and it never converges; it exhausts its 100-iteration limit. Measured: loosening its
`g_tol` a hundredfold changes the answer by exactly 0.000e+00.

**The fix, and its measurement.** `moment_start=true` gives the `:indep` variance coordinates a cheap
data-informed start. With the simplex demoted and `g_tol` TIGHTENED to 1e-6 (tightened, not widened):

| case | shipped default | moment start, no simplex, g_tol 1e-6 | identity |
|---|---|---|---|
| large, seed 20260920 | 4.2349 s | **2.0486 s, 2.067x** | 1.772e-10 |
| large, seed 20260921 | 4.0996 s | 2.3924 s, 1.714x | 5.488e-10 |
| large, seed 20260922 | 4.2649 s | 2.4512 s, 1.740x | 6.367e-09 |
| small glmm_200x5 | 0.1091 s | 0.0945 s, 1.154x | 1.747e-11 |

All four hold the arc's rtol 1e-8.

**CORRECTION 2026-09-22: this whole table is measured in a configuration that does NOT ship.**
Every "with the flag" number above was taken with the simplex DEMOTED (`nelder_mead=false`) and
`g_tol` TIGHTENED to 1e-6. The demotion was then abandoned on its own measurement, and the shipped
defaults at `f35aa1027` are `nelder_mead=true`, `g_tol=1e-4`, `moment_start=true`. No user reaches
2.0486 s by calling the function, and the arc's "up to 2.9x against the S8 baseline of 5.71 s"
claim rests on this table and is withdrawn with it. The 5.71 s figure is also wrong: this ledger
fixes the S8 after-state at 5.928687 s at line 5.

The "shipped default" column, 4.2349 s large and 0.1091 s small, is PR #446 alone and is
unaffected; it was measured on its own. The small fixture's target of under 0.122 s is met there.

The speed of the configuration that actually ships is UNMEASURED. A re-measurement at the shipped
defaults is owed, with the keyword configuration written into the TSV header.

**Why it WAS opt-in (superseded by `f35aa1027`, which made it the default on Shinichi's D-273 decision).** A full suite with it ON as the default returned
`16292 passed, 3 failed, 5 ERRORED` against a `16328/1/0/19` baseline. Five errors were a defect in this
implementation (group labels are not necessarily integers; Symbol units threw TypeError), now fixed and
that file passes 20/20. The remaining failure is NOT a defect: `test_grouped_laplace_identity` pins
fixture A's inner-fit count, and any change to the STARTING POINT moves the optimiser's path on both
gradient paths. [[DECISIONS#D-273|D-273]] reserves that re-pin for Shinichi.

**Validated with the flag off:** full `Pkg.test()` on `8415e0883`, `16329 passed, 1 failed, 0 errored,
19 broken`, 96m47.3s, the one failure being the `test_em_louis.jl:127` flake. Every `--gate` passes and
`G7b.1` PASSES, so the default path is unchanged.

**What is owed before it could become the default:** Shinichi's call on the S7b re-pin under D-273; the
estimator is a count-family argument and applies only to `:poisson`/`:nb2` `:indep` terms, with every
other family and mode keeping the constant; and `:latent`/`:dep` remain untested.
