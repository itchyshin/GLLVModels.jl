# Gates: leaf-S8 GLLVModels grouped route, analytic outer gradient (overnight arc 2026-09-20)

OWNS: src/grouped_laplace.jl, src/grouped_nongaussian_fit.jl, src/grouped_fit.jl, test/test_grouped_analytic_grad.jl, test/runtests.jl (one include line), bench/profile_grouped_glmm.jl, bench/results/grouped_sections_*.tsv, docs/design/grouped-analytic-gradient.md

BASELINE: origin/main 69a69b0a0. Latte 200x5 fixture, banked walls: 0.192 s (pre-S7c), 0.184721 s before / 0.150383 s after S7c (bench/results/grouped_warm_68c2f067c.tsv, git-ignored). Latte.jl 0.015 s. Per fit at that fixture: 118 objective calls, 711 summed inner Newton iterations, nθ = 2.

SCOPE: replace the finite-difference outer gradient with an analytic one derived through the implicit function theorem, reusing src/takahashi_selinv.jl for the log-det trace term; then unconfine the S7c warm start, demote Nelder-Mead to a fallback, and drop the final O(nθ²) FD Hessian. Every step an identity against 69a69b0a0. Never widen a tolerance.

## GATE A (the arc's own go/no-go, runs BEFORE any src/ change)

- [x] GA.1: `bench/profile_grouped_glmm.jl` gains a second fixture with nθ >= 6 and a section partition (Nelder-Mead evaluations, BFGS line search, FD gradient calls, final FD Hessian, inner Newton iterations, GLM state/score/curvature, CHOLMOD factorisation, log-det). Sections sum to within 10% of the measured wall on both fixtures.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. bench/profile_grouped_glmm.jl --gate sections
  EXPECT: GATE GA.1 PASS
  EVIDENCE: PASS. New fixture `glmm_5000x3_g500`: 3 traits, one :indep :unit term, Poisson, n=5000, G=500 -> nθ=6 (3 trait intercepts + 3 per-trait log_sd), generated deterministically (Random.Xoshiro(20260920)) in the bench script, no CSV committed. Section timing measured by wrapping the REAL `_grouped_nongaussian_objective`/`_grouped_fd_gradient`/`_grouped_fd_hessian` closures with bench-side `Base.@elapsed` counters and driving `Optim.optimize` with the verbatim call sequence from `fit_grouped_nongaussian` (not a reimplementation of the objective body, unlike the S4/S7b shadow) -- driver loglik matched a real `fit_gllvm` call exactly (loglik_gap_rel=0.000e+00) on both fixtures. Sections-vs-wall: glmm_200x5 gap=8.0% (0.1546s/0.1680s), glmm_5000x3_g500 gap=0.04% (10.8869s/10.8879s) -- both within the 10% bound. Inner Newton iterations (derived from the existing `_grouped_chol_stats()` counter, 2n+1 relationship): 563 summed over 118 calls (small), 2748 summed over 514 calls (large). GLM-state/CHOLMOD/log-det split (sampling `Profile` over real `fit_gllvm` calls, a share of the objective-call time already counted above, not summed again): small 20.9%/1.3%/0.0%, large 22.4%/0.6%/0.0% -- the remaining inner-solve time (triangular solves, sparse `W` construction, Newton bookkeeping) is not attributed to any of the three named buckets. TSV: bench/results/grouped_sections_a2dc58557.tsv (git-ignored).

- [x] GA.2: THE DECISION. On the LARGER fixture, the FD-attributable share (FD gradient calls + final FD Hessian + Nelder-Mead evaluations) is >= 25% of the fit wall.
  CHECK: read the TSV written by GA.1 and state the three section shares and their sum
  EXPECT: share >= 0.25 -> proceed to GATE B; share < 0.25 -> STOP the arc, the measurement is the deliverable, write the finding into the progress record and the after-task
  EVIDENCE: PROCEED. Large fixture (glmm_5000x3_g500, driver wall 10.8879s): FD gradient 5.8507s (53.7%) + FD Hessian 1.7460s (16.0%) + Nelder-Mead 2.7928s (25.6%) = 10.3895s -> share = 0.954. FD gradient alone dominates (53.7% of wall); BFGS line search itself is only 0.4973s (4.6%). Second data point, small fixture (glmm_200x5, driver wall 0.1680s): FD gradient 0.0604s + FD Hessian 0.0135s + Nelder-Mead 0.0656s = 0.1395s -> share = 0.830.

## GATE B (only if GA.2 says proceed)

- [ ] GB.1: docs/design/grouped-analytic-gradient.md derives the gradient before any code, with the symbolic-alignment table: implicit db̂/dθ, the tr(A⁻¹ dA/dθ) log-det term, and the observed-curvature dependence on y. Every symbol maps to the function that will compute it.
  CHECK: manual read by the orchestrator
  EXPECT: PASS
  EVIDENCE: pending

- [ ] GB.2: the analytic gradient agrees with the existing central-difference gradient (`_grouped_fd_gradient`, src/grouped_fit.jl:213-224) at 20 random theta on BOTH fixtures, rtol 1e-6, which is the FD reference's own accuracy and not a widened bound.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate fd_agreement
  EXPECT: GATE GB.2 PASS
  EVIDENCE: pending

- [ ] GB.3: fitted parameters and logLik equal origin/main 69a69b0a0 within rtol 1e-8 on both fixtures, and the existing grouped identity fixtures A, B and D still pass.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate identity && env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_laplace_identity.jl --gate identity
  EXPECT: both GATE ... PASS
  EVIDENCE: pending

- [ ] GB.4: with the warm start UNCONFINED (the S7c restriction to Nelder-Mead removed), fixture D's regression test still passes and the converged answer is unchanged at rtol 1e-8. This is the gate that S7c could not pass with an FD gradient.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_laplace_identity.jl --gate warm_identity
  EXPECT: GATE G7c.1 PASS
  EVIDENCE: pending

- [ ] GB.5: objective calls and summed inner Newton iterations are reported before and after (118 and 711 banked at fixture A); the wall on fixture A is recorded against 0.150383 s and Latte's 0.015 s; the larger fixture against its own GA.1 baseline. Numbers reported whatever they are, no claim beyond them.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. bench/profile_grouped_glmm.jl --gate sections_after
  EXPECT: GATE GB.5 PASS and a TSV at bench/results/grouped_sections_after_<sha>.tsv
  EVIDENCE: pending

- [ ] GB.6: full `Pkg.test()` green apart from the known pre-existing test_em_louis.jl:127 flake; test/test_grouped_laplace.jl unchanged from 69a69b0a0.
  CHECK: test -z "$(git diff --name-only 69a69b0a0 -- test/test_grouped_laplace.jl)" && env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Pkg; Pkg.test()'
  EXPECT: tests passed, or exactly 1 failed being test_em_louis.jl:127
  EVIDENCE: pending

## STOP conditions (report, never smooth over)

An identity failing at rtol 1e-8. A tolerance that would need widening to pass. GA.2 below 25%. Any need to touch HSquared.jl, hsquared, PR #781, or src/takahashi_selinv.jl. Two Julia suites at once.
