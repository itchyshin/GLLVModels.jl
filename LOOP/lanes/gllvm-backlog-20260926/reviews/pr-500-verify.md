# PR #500 verify (fresh adversarial verifier)

PR: fix(twopart) per-site mode search, Fixes #484. Head acb0563a9 (branch claude/twopart-mode-search-484).
Compared against the first-review head 1700f6c30 and origin/main b90641c97.
Environment: macOS aarch64, Julia 1.10.0, OPENBLAS_NUM_THREADS=1, JULIA_NUM_THREADS=1-2.
Throwaway worktrees ~/local-scratch/verify-500-{head,old,main} (removed afterwards). Read-only on GitHub.

## Verdict: MERGE_AFTER_FIXES

The two first-review BLOCKING items are resolved, and so is the stale census figure. The reproducer
is red on 1700f6c30 and green on the head. But I could break the new step-acceptance rule: it
creates a new false -Inf path at healthy, near-converged sites with Lambda_z != 0. In every case I
found, a halved step would have raised q, so the fix is small. No fitter can reach this path, since
every fitter uses Lambda_z = 0. The brief requires "could not break the rule" before MERGE, so the
verdict is MERGE_AFTER_FIXES. CI: Documenter is green. The 8 Julia shards were still running when
I checked (see section 6).

## BLOCKING

None of the first review's blocking items remain.

## REQUIRED before MERGE (it gates the verdict under this brief; its severity is SHOULD-FIX because no fitter reaches it)

R1. **The new small-step "fail" exit returns -Inf at healthy modes that a halved step would reach.**
`src/families/twopart.jl` (`_twopart_mode_stage`, the Lambda_z != 0 `:newton` small-step branch).
A small step that lowers q is followed by `return z, true` if `max|g| < sqrt(tol)`, and otherwise
by `return z, false`. There is no halving. With high curvature, `|g|` can exceed sqrt(tol) = 3.2e-5
while the iterate is already about 1e-8 from the mode, so the gradient threshold depends on scale.
- Test: start the `:newton` stage at a healthy mode (full-Hessian refine, |g| < 1e-9, negative-definite
  Hessian) plus a perturbation of 1e-2 to 1e-5, over the 400-site random scan per family and scale
  (p = 6, K = 2, Lambda_z != 0). This mimics a Fisher stage that stops just short of tolerance.
  Small-step-fail exits: ZINB(2) x3 4/1500; ZIP x2 1/1344; ZIP x3 5/1356; ZIP x5 6/1288;
  ZIB 0. **In all 16 cases a halved step (2^-k Delta, k <= 30) raises q.**
  Example: ZIP x2 site 223, y = [420,400,0,0,0,0]. The search fails at |g| = 8.45e-5 with
  |Delta| = 1.4e-8 and |z - z_mode| = 1.4e-8, so the point is at the mode and still gets -Inf.
- Against 1700f6c30 on the same starts, the total false count falls from 28 to 24. But ZIP x2
  (0 -> 1) and ZIP x3 (4 -> 5) are new failures from this exit. Every head failure of this kind
  comes from the new branch.
- Natural path (Fisher then Newton from z = 0, 4800 sites): this branch never fired. The two
  remaining false -Inf sites (ZINB x5 site 322 and ZIP x5 site 331) are pre-existing and identical
  on 1700f6c30. The first is a slow Newton stage that converges at maxiter = 1000. The second
  is a halving failure at a clamped-eta point with |g| = 25.7.
- Fix (small): for the Lambda_z != 0 `:newton` case, send a q-decreasing small step into the same
  halving loop the large-step path uses, rather than `return z, false`. Keep the relaxed
  "converged" exit only when halving also fails. Add one perturbed-start case, for example the
  ZIP site above, as a relation test: `isfinite(value)` and the refined |grad q| is below a bound.

## SHOULD-FIX

S1. **The relaxed-convergence exit weakens the documented contract and is used often.** It fires
at 19 to 92 of 400 sites per family and scale on the natural path, with |g| up to 5.7e-6. The
docstring says the value is -Inf unless the search "converge[s] to `tol`" (1e-9, a step-size test).
Here, "converged" means max|g| < 3.2e-5. On my scan, the worst value error against the value at
the fully refined mode rose from 1.2e-8 on 1700f6c30 to 1.7e-6 on the head (ZIP x5; 3.8e-7 for
ZIB x5). The error is first order in dz because of the log-det term. The reproducer's own site is
off by 1.5e-7. None of this is fitter-reachable, but either tighten the exit (R1's halving fix
mostly removes the need for it) or state it in the docstring.

S2. **The reproducer test's final assertion does not test what its comment claims, and its margin
is thin.** At y = [0,13,0,0,0,1], the `:newton` stage takes the relaxed exit on its first
iteration and returns the Fisher iterate unchanged (|g| = 6.05e-7 in both). So
"must land at a genuine stationary point of q, not merely stop early" in fact tests that the
Fisher iterate already meets |g| < 1e-6, with a 1.65x margin. Fix: once R1 is fixed, assert that
the Newton result's |g| is well below 1e-6 (for example 1e-8), or re-word the comment.

S3. **The CHANGELOG contradicts itself on runtime.** The #484 bullet says "Healthy fits are not
slower after this ... hurdle-Poisson was unchanged" and "within 2 to 23 percent of their
pre-#484 (main) time". The next bullet says hurdle Poisson is +20% and ZIP +25% against main.
Presumably "unchanged" means relative to 1700f6c30, which the text should say. That bullet also
carries process narrative ("caught by re-running the existing test suite before merge, not by
the reviewer"). The narrative belongs in the after-task, not the CHANGELOG.

S4. **The PR body (paragraph 4) understates the rule.** It says "when it would [lower q], the
search keeps the current iterate rather than declaring failure". The code keeps the iterate only
if max|g| < sqrt(tol). Otherwise it declares failure, which is R1.

S5. **(Carried, platform) Fitted-outcome assertions on committed fixtures, inherited from
1700f6c30.** These are not new in this delta and not seed-dependent, because the data are
committed fixtures with SHA checks. But they assert optimiser outcomes: `fit_zip_gllvm` base-s1
`loglik >= -920.7` (the surface is multimodal; the old fit stopped at -935.3), and ZINB/ZIB
logLik within 1e-6 of fixed values. The 1.13 CI legs are the evidence for these. If a leg fails
there, change them to relations (for example, logLik at least that of the pre-#484 point
re-scored with converged modes, or `converged` together with |grad| small).

## Checks

### 1. First-review items on acb0563a9
- B1 Documenter: RESOLVED. The docstring now sits directly above `function twopart_loglik_site`.
  `GLLVModels.twopart_loglik_site` is added to docs/src/api.md, and the @ref is qualified. CI
  Documenter passes on acb0563a9.
- B2 HurdleNB disclosure: RESOLVED. There is a separate CHANGELOG bullet, and the 1e-8 claim is
  narrowed to ZIP and hurdle Poisson (plus the two converged ZINB audit fits). The bullet states
  that ZIB and the delta families were not re-verified. Wording is accurate (see 4).
- SF1 small-step bypass: PARTLY. The reproducer is fixed, and no q-decreasing step is ever
  accepted. The rule then introduces R1.
- SF2 runtime: RESOLVED. q0 is taken from the pieces loop, so there is no second
  `_twopart_logpost` call per large step. ℓ0 uses the same clamped eta and the same per-t
  `_tp_pieces_at` summation order as `_twopart_logpost`. The cost is disclosed, though the
  wording has a problem (S3).
- SF3 census comment: RESOLVED (361%, with a re-measurement note).
- SF4 docstring placement: RESOLVED (same change as B1).
- SF5 PR body path claim: RESOLVED ("only when no step was halved ... 256 of 320").

### 2. Reproducer and adversarial cases
- Reproducer ZINB(2), y = [0,13,0,0,0,1]: 1700f6c30 gives a site value of -Inf, with Newton
  failing at |g| = 1.38e-3. Head gives -11.788067, finite, with a negative-definite Hessian
  (eigenvalues -11.67 and -2.44). The test file passes 41/41 on the head (50 s).
- Natural-path scan, 4800 sites (ZINB(2), ZIB(10), ZIP; loadings x1/x2/x3/x5; Lambda_z != 0),
  checked against an independent full-ForwardDiff-Hessian damped Newton refine:
  - False -Inf, head (1700f6c30): ZINB x5 1 (1), ZIB x1 0 (1), ZIP x5 1 (4). Both head cases are
    pre-existing (see R1).
  - Finite but non-stationary: none apart from high-curvature ZIP x5 sites with |g| of 1e-5 to 2e-5
    and |dz| of about 1e-9 (the same set on 1700f6c30), plus the new relaxed-exit site
    (ZIP x5 site 45: |g| = 1.2e-6, |dz| = 2.7e-7, dv = 1.7e-6; see S1).
  - Every finite value sits at a negative-definite stationary point.
- Perturbed-start stress (about 13,000 runs): R1.

### 3. Runtime (fit_hurdle_poisson_gllvm; simulated p = 6, n = 150, K = 2; min of 3 warm runs, two alternating rounds)
main 1.551 s / 1.515 s against head 1.877 s / 1.895 s, so **+21% to +25%**. logLik is identical to
1e-8 (-1410.45164502) and both fits converged, in 34 against 35 iterations. This agrees with the
claimed +20%.

### 4. HurdleNB CHANGELOG bullet
Its numbers (-1766.970 at r = 1.32e7 over 173 iterations, moving to -1746.342 at r = 3.576 over 48
iterations; Laplace at truth -2003.994 moving to -1758.200; diag(Lambda Lambda') 1.292 moving to
0.952) match the first review's measurement exactly. The mechanism (a = r/(r+mu), which moves
both the mode and the log-det weight) is stated correctly. "Essentially every HurdleNB fit" is
right, because the score changes at every positive count. I did not re-run that fit, since the
first review's scratch data were not kept.

### 5. Platform robustness
- The delta adds one testset, the reproducer. It uses fixed literal data and parameters, with no
  RNG and no optimiser, so it is deterministic up to floating-point rounding. Fisher needs between
  100 and 150 iterations at this site (it fails at 100, with |g| = 6.05e-7, and converges by 150),
  so the `!ok_fisher` premise has a sound margin. The last |g| < 1e-6 assertion has a 1.65x margin
  (S2). It is not seed-dependent, so it is **not BLOCKING**.
- No new test asserts a seed-drawn fitted outcome. The inherited fixture-based fit assertions are
  under S5.

### 6. CI on acb0563a9
Documenter SUCCESS. Frozen R 0.7.0 smoke FAILURE (advisory, excluded). The Julia 1.10 shards 1-4
and Julia 1 shards 1-4 were IN_PROGRESS (started 18:24Z) at my final check; see the update line below.

## Scope notes
I did not run R parity (not applicable) or the full suite. Scripts are in the session scratchpad:
scan.jl, drill.jl, drill2.jl, perturb.jl, perturb2.jl, stage_instr.jl (an instrumented copy of
`_twopart_mode_stage` loaded via `Base.include(G, ...)`; it counts the branches), repro.jl, rt.jl.

CI update (19:05Z): Julia 1.10 shards 1,2,4 SUCCESS, shard 3 IN_PROGRESS; Julia 1 shard 3 SUCCESS, shards 1,2,4 IN_PROGRESS. Documenter SUCCESS; Frozen R FAILURE (advisory). Stopped waiting at the 10-minute cap.
