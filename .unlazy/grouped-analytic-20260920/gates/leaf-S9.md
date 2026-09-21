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
  EVIDENCE: pending

- [ ] G9.2: the Hessian produced by finite-differencing the analytic gradient yields STANDARD ERRORS equal to those from `_grouped_fd_hessian` at rtol 1e-4, per coordinate, on all five fixtures, INCLUDING the Beta and NB2 fixtures where the two FD schemes differ most because of the dispersion rows. 1e-4 is recorded as the FD oracle's own accuracy, not a widened bound. If it is not met, STOP and reopen D-274; do not loosen it.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate hessian
  EXPECT: GATE G9.2 PASS
  EVIDENCE: pending

- [ ] G9.3: the fallback contract holds. With `analytic_gradient=false` the pre-S9 path runs verbatim: Nelder-Mead executes and `_grouped_fd_hessian` supplies H. A fit whose analytic gradient fails at some theta still completes, warning once rather than throwing.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate nm_fallback
  EXPECT: GATE G9.3 PASS
  EVIDENCE: pending

- [ ] G9.4: objective calls and inner Newton iterations are BOUNDED BELOW the S8 measured values on both bench fixtures, reported not pinned, following D-273's rule that the old pin stays on the old path. S8 measured: 84 objective calls and 407 inner Newton iterations on glmm_200x5; 284 and 1437 on glmm_5000x3_g500.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate counts
  EXPECT: GATE G9.4 PASS
  EVIDENCE: pending

- [ ] G9.5: the four coverage holes S8 left are closed as fixtures that pass both fd_agreement and identity: Binomial; per-trait `dispersion=:trait` for Beta and for NB2; `GroupingTerm(mode=:dep)`.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate coverage
  EXPECT: GATE G9.5 PASS
  EVIDENCE: pending

- [ ] G9.6: THE MIXED PATH, a hard gate and not optional. A fixture that forces the analytic gradient to fail at a SUBSET of theta still lands on origin/main 69a69b0a0's answer at rtol 1e-8. This is the class that produced the S8 compaction bug, and GB.4 reasoned about it without exercising it.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate mixed
  EXPECT: GATE G9.6 PASS
  EVIDENCE: pending

- [ ] G9.7: the final reported gradient on the analytic path is the analytic gradient, and `converged` is decided on it. Its norm agrees with the FD gradient's norm at the converged point to rtol 1e-6, so the convergence verdict does not change.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate final_gradient
  EXPECT: GATE G9.7 PASS
  EVIDENCE: pending

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
