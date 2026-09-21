# Gates: leaf-S9 GLLVModels grouped route, demote Nelder-Mead and replace the final FD Hessian

OWNS: src/grouped_nongaussian_fit.jl, src/grouped_fit.jl (the _grouped_fd_hessian function and its call sites only), test/test_grouped_analytic_grad.jl, bench/profile_grouped_glmm.jl, bench/results/grouped_sections_after_*.tsv

BASELINE: origin/main 69a69b0a0 for every identity. Speed baseline is S8's own measured after-state, NOT the pre-S8 numbers: glmm_200x5 (ntheta=2) 0.122395 s, glmm_5000x3_g500 (ntheta=6) 5.928687 s. The large fixture's GA.1 section partition, which is what justifies this leaf: Nelder-Mead 2.7928 s, final FD Hessian 1.7460 s, together 4.5 s of the remaining 5.93 s.

SCOPE: (1) demote the unconditional value-only Nelder-Mead phase (src/grouped_nongaussian_fit.jl:686) to a fallback when the analytic gradient is available; (2) replace the final O(ntheta^2) `_grouped_fd_hessian` (src/grouped_fit.jl:226-247, called at :756) with a finite difference OF THE ANALYTIC GRADIENT, per D-274; (3) close the four coverage holes S8 left. Never widen a tolerance. Keep every old path reachable by keyword, as S8 did for the FD gradient.

DECISION ALREADY TAKEN: D-274 (Shinichi, 2026-09-21). The Hessian is FD-of-the-analytic-gradient, nTheta gradient calls rather than nTheta^2 objective calls. A full analytic Hessian is explicitly NOT in scope and is not foreclosed. `_grouped_fd_hessian` stays reachable as oracle and fallback. The gate compares STANDARD ERRORS, not raw Hessian entries, because the two are different estimators of the same matrix and the SEs are what users see.

FOUND WHILE WRITING THIS LEDGER, and in scope: the FINAL reported gradient at src/grouped_nongaussian_fit.jl:754 is still `_grouped_fd_gradient` even when `analytic_gradient=true`, and its norm feeds `converged`. S8 replaced the optimiser's gradient and left this one. G9.7 covers it.

WHY NELDER-MEAD IS THERE, stated so the demotion is not naive: the comment at :682-685 says the joint Laplace domain can invalidate a finite-difference neighbour, so a value-only search avoids handing Optim a NaN stencil. That reason is about the FD STENCIL. With an analytic gradient there is no stencil, which is the argument for demoting it. The reason does NOT fully vanish while any FD path remains reachable, so the fallback must stay wired and G9.3 tests it.

## GATES

- [ ] G9.1: with Nelder-Mead demoted, fitted parameters and logLik equal origin/main 69a69b0a0 within rtol 1e-8 on all five S8 fixtures.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate identity
  EXPECT: GATE G9.1 PASS
  EVIDENCE: **NOT MET. STOP, escalated -- this is the "converged answer changing under the demotion" STOP
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

- [ ] G9.2: the Hessian produced by finite-differencing the analytic gradient yields STANDARD ERRORS equal to those from `_grouped_fd_hessian` at rtol 1e-4, per coordinate, on all five fixtures, INCLUDING the Beta and NB2 fixtures where the two FD schemes differ most because of the dispersion rows. 1e-4 is recorded as the FD oracle's own accuracy, not a widened bound. If it is not met, STOP and reopen D-274; do not loosen it.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate hessian
  EXPECT: GATE G9.2 PASS
  EVIDENCE: **PARTIAL. NOT ticked -- 4 of 6 fixtures PASS comfortably, 2 FAIL for a root-caused, non-D-274
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
  EVIDENCE: **GATE G9.3 PASS**, run 2026-09-21 ~14:29 MDT, `poisson_latent` fixture.

  (a) `analytic_gradient=false`'s DEFAULT kwargs reproduce `nelder_mead=true, hessian=:fd` given
  explicitly, bit-for-bit: inner Laplace-fit calls (`_grouped_chol_stats().calls`) 471 both ways,
  loglik and beta `isapprox(...; rtol=1e-12)` both true. Nelder-Mead ACTUALLY EXECUTES, checked rather
  than assumed: the analytic-default path (`nelder_mead=false`, the S9 default) costs only 173 calls --
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

- [ ] G9.4: objective calls and inner Newton iterations are BOUNDED BELOW the S8 measured values on both bench fixtures, reported not pinned, following D-273's rule that the old pin stays on the old path. S8 measured: 84 objective calls and 407 inner Newton iterations on glmm_200x5; 284 and 1437 on glmm_5000x3_g500.
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

- [ ] G9.5: the four coverage holes S8 left are closed as fixtures that pass both fd_agreement and identity: Binomial; per-trait `dispersion=:trait` for Beta and for NB2; `GroupingTerm(mode=:dep)`.
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

- [ ] G9.6: THE MIXED PATH, a hard gate and not optional. A fixture that forces the analytic gradient to fail at a SUBSET of theta still lands on origin/main 69a69b0a0's answer at rtol 1e-8. This is the class that produced the S8 compaction bug, and GB.4 reasoned about it without exercising it.
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

- [ ] G9.7: the final reported gradient on the analytic path is the analytic gradient, and `converged` is decided on it. Its norm agrees with the FD gradient's norm at the converged point to rtol 1e-6, so the convergence verdict does not change.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate final_gradient
  EXPECT: GATE G9.7 PASS
  EVIDENCE: **GATE G9.7 PASS, all six fixtures.** `src/grouped_nongaussian_fit.jl:754`'s `gradient` is now
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
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Pkg; Pkg.test()'
  EXPECT: tests passed, or exactly 1 failed being test_em_louis.jl:127
  EVIDENCE: pending

- [ ] G9.9: MEASUREMENT. Both fixtures re-measured with GA.1's section partition, before and after in one process so machine state is shared. The after TSV carries git SHA, Julia version, BLAS config and thread counts in its header. The large fixture's target is about 1.5 s against 5.93 s; that is a TARGET and is reported, never asserted.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. bench/profile_grouped_glmm.jl --gate sections
  EXPECT: GATE GA.1 PASS, and bench/results/grouped_sections_after_<sha>.tsv non-empty with a full header
  EVIDENCE: pending

## STOP conditions (report, never smooth over)

An identity failing at rtol 1e-8. G9.2's SEs missing rtol 1e-4 (reopen D-274 rather than loosening). Any tolerance needing to be widened. A converged answer changing under the demotion. Two Julia suites at once. Any need to touch HSquared.jl, hsquared, PR #781, or src/takahashi_selinv.jl.
