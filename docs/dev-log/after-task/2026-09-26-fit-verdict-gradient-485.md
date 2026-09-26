# After-task: `_fit_verdict` requires the gradient criterion (#485, 2026-09-26)

Lane: Claude, branch `claude/fit-verdict-gradient-485`, worktree
`~/local-scratch/gllvm-verdict-485`. Cut from `origin/main` @ `d89179d41` (a stale
local checkout of the same branch name was fast-forwarded to it; includes #481, #483,
#487, #488, #492, #494, #495). [FILL: commit / push / PR status]

## 1. Goal

Fix #485. A previous builder, blocked mid-task by a now-released lease, left a full
diagnosis: `_fit_verdict(res)` (`src/fit_verdict.jl:53-55` on `origin/main`) reports
`converged = true` whenever `Optim.converged(res)` is true. Optim 1.13.3's
`converged = x_converged || f_converged || g_converged`, with
`x_abstol = x_reltol = f_abstol = f_reltol = 0.0` by default, so a single zero-length
line-search step trips `x`/`f` convergence even when `Optim.g_residual(res)` is far
above `Optim.g_tol(res)`.

Evidence on file (unmerged, `docs/dev-log/core070/class-audit-20260924/` on
`origin/claude/lane-true-parity-finish-20260925`, and issue #485 itself): 21 fits
across 7 default routes (ZIP, NB1-grouped, Poisson, Binomial, ordinal, truncated
Poisson, Gaussian), seeds 101-103. `fit_nb1_gllvm_grouped` reported `converged = true`
on 3/3 seeds with Optim's own gradient residual 5.68, 214, 17.9 against `g_tol = 1e-5`
(reproduced verbatim below); one ZIP seed reported `converged = true` with residual 45.
An independent "skeptic" re-implementation (own log-density, own guarded Newton mode
search, a different line search) confirms the NB1 fits sit on a genuinely
slow-converging ridge: re-optimising its own smooth objective from the same point
gains only 0.03-0.48 log-likelihood units over 60+ seconds and still does not reach
its own gradient tolerance. The issue additionally notes `fit_nb1_gllvm` (the
scalar/shared-dispersion route, not grouped) shows the same class with residuals up
to 1.35e11 on other seeds.

Required outcome: `_fit_verdict(res)` additionally requires the scale-aware gradient
criterion `_tweedie_verdict`/the Beta grouped verdict (#480/#483) already use, without
touching the 3-argument form (no production caller currently uses it; it is exercised
only by `test/test_known_sentinel_defects.jl`).

## 2. Implemented

- `_gradient_criterion_met(res)` in `src/fit_verdict.jl`: `gres <= max(g_tol, g_tol *
  |Optim.minimum(res)|)`, reading `Optim.g_residual`/`Optim.g_tol` directly off the
  result (no caller-supplied `g_tol` needed, unlike `_beta_grouped_g_met`, which takes
  it as an argument for its own convenience but computes the same value).
- `_fit_verdict(res)` now ANDs this into `Optim.converged(res)` before dispatching to
  the 3-argument form. The 3-argument form (`_fit_verdict(nll, converged, iterations)`)
  is untouched: no `Optim` result to judge the gradient from.
- New regression test `test/test_fit_verdict_gradient.jl`:
  - Reproduces the audit's `route_nb1` exactly (`MersenneTwister(101)`,
    `fit_nb1_gllvm_grouped`, p=5/n=80/K=2/φ=1.0 defaults): before the fix,
    `converged = true` at `loglik = -1038.5503750549412` (matches the audit log to 10
    decimals); the test asserts `!fit.converged`.
  - A second, independent, genuinely-stationary NB1-grouped smoke fit (tiny p=4/n=50)
    stays `converged = true` (guards against over-correction).
  - Wired into `test/runtests.jl` next to `test_beta_grouped_convergence.jl`.
- Flip-list (targeted regression sweep, `GLLVM_VERDICT_FLIP_LOG=1` instrumentation
  printing every call where `Optim.converged(res)` disagrees with the new
  gradient-gated verdict): across the 15-file, 211-assertion clean run (section 5)
  plus the new test, 3 distinct `_fit_verdict`-internal flips, all traced to a
  specific cause:
  1. This PR's own `test_fit_verdict_gradient.jl` (NB1 grouped, seed 101):
     `gres = 5.68` vs threshold `0.0104` - the reported defect, fixed as intended.
  2. `test_nb_boundary_restart.jl` (#477's own regression test), an NB2
     grouped-dispersion fit at a per-group dispersion boundary: `gres = 19.5` vs
     threshold `0.00223`. **Not observable**: that file's own
     `fit_nb_gllvm_grouped`/`_cov` already force `converged = conv &&
     !any(boundary)`, and the test already asserts `@test !fit.converged   # a
     boundary group is still flagged` - `.converged` was `false` before this fix
     too, for an independent reason.
  3. `test_beta_grouped_convergence.jl`'s `d01` case (#480/#483's own regression
     test, deliberately set `g_tol = 1e-12` to probe exactly this criterion):
     `gres = 4.41e-7` vs threshold `2.73e-10`. **Not observable**: that fitter
     already ANDs `_beta_grouped_g_met(res, g_tol)` after `_fit_verdict(res)`
     (the identical formula), so `.converged` was `false` before this fix too.
  No other flip appeared in 211 assertions across the 15 mainstream/family test
  files run (NB1, NB2, Beta grouped, Gamma grouped, Poisson, Binomial, ordinal,
  Tweedie - section 5), and 0 files errored. A larger, 139-file sweep (matched to
  every `_fit_verdict` caller by family/function name) was attempted twice but not
  completed cleanly inside the time available (section 9); the ~135 files it did
  reach before being interrupted printed no flip line, though an interrupted
  process cannot rule out a buffered, unflushed print (section 10).
  **Classification: honesty correction, not a regression.** Every flip is
  `old = true → new = false` by construction (`_gradient_criterion_met` can only
  narrow, never widen, what already-passed `Optim.converged`), and every case
  found has `gres` one to five orders of magnitude above its own scale-aware
  threshold - not a borderline miscalibration.

## 3a. Decisions and Rejected Alternatives

- **Chosen: reuse the exact `_beta_grouped_g_met`/`_tweedie_verdict` formula**
  (`gres <= max(g_tol, g_tol * |nll|)`), read directly off the `Optim` result via
  `Optim.g_residual`/`Optim.g_tol`, rather than inventing a new threshold or taking
  `g_tol` as a caller-supplied argument. This is the already-shipped, already-reviewed
  rule (#480/#483); a third variant would be a third thing to keep in sync.
- **Rejected: a caller-supplied `g_tol` parameter to `_gradient_criterion_met`** (as
  `_beta_grouped_g_met` does). `Optim.g_tol(res)` returns the identical value the
  caller passed into `Optim.Options(g_tol = ...)`, so the parameter would be pure
  duplication; `_fit_verdict(res)` only ever has `res`.
- **Rejected: also touching the 3-argument form.** It has no `Optim` result to read a
  gradient from, and no production caller uses it (only
  `test/test_known_sentinel_defects.jl` does, directly). Changing it would be
  speculative.
- **Rejected: a complementary restart-on-stall**, as #480/#483 add for Beta. Out of
  scope for #485, whose diagnosis and required outcome are the verdict flag itself;
  a restart is a separate, per-family policy decision (as #479's report also declined
  to add a "gradient-criterion guard" to Gamma, for the same reason in reverse).
- **Verified before generalising: no `_fit_verdict` caller uses a gradient-free
  optimizer.** 94 of 100 call sites use `Optim.LBFGS()`; the remaining 6
  `Optim.NelderMead()` calls (phylo_*_xlv constrained-refit helpers) never feed their
  result into `_fit_verdict` (see section 8). Had any done so, `Optim.g_residual`
  would read the simplex diameter (`state.nm_x`), not a gradient norm - a different
  quantity the same formula would still apply to sensibly, but it was checked, not
  assumed.

## 4. Files Touched

- `src/fit_verdict.jl`
- `test/test_fit_verdict_gradient.jl` (new)
- `test/runtests.jl` (one `_shard_include` line)
- `CHANGELOG.md`
- this report

Outside the repo: a read-only baseline worktree
`~/local-scratch/gllvm-verdict-485-mainbase` (detached `origin/main` @ `d89179d41`,
the same commit this branch was cut from), used only for before/after comparison;
scratch scripts and logs in
`/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-glmmTMB/.../scratchpad/`.

## 5. Checks Run

All runs used Julia 1.10.12/1.10.0 (juliaup default), `JULIA_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1`, single Mac Studio, `--project=.` (the package's own
`Project.toml`; no separate test environment needed for these files).

- New test: 4/4 assertions pass on this branch; the `!fit.converged` assertion (and
  only that one) fails on unfixed `origin/main`.
- `test/test_known_sentinel_defects.jl` (the only file exercising the 3-argument
  form directly): 25/26 pass, 1 broken - unchanged from main (the 3-arg form is
  untouched by this fix).
- Targeted regression sweep, 15 files matched to `_fit_verdict`'s highest-risk
  callers (NB1 scalar + grouped, NB2 grouped-boundary, Beta grouped, Gamma grouped,
  Poisson, Binomial, ordinal, Tweedie, plus `test_known_sentinel_defects.jl` and
  this PR's own test): **211 pass, 1 broken (pre-existing), 0 errors, 0 files
  failed.** 3 internal verdict flips, all classified in section 2's flip-list; none
  observable at the public `.converged` field beyond this PR's own test.
- A wider, 139-file sweep (every `_fit_verdict` caller's test file, matched by
  family/function name) was attempted but not completed inside the time budget -
  see section 9.

## 6. Tests of the Tests

- The new test fails on unfixed `origin/main`: only `!fit.converged` fails (the
  `loglik ≈ -1038.5503750549412` assertion passes on both), confirming this
  reproduces the real reported defect and is not a fabricated Optim result.
- The reproduction uses the exact same data-generation code and seed
  (`MersenneTwister(101)`, `route_nb1` in the audit's
  `fit_verdict_classB_probe.jl`) as the audit evidence, and its log-likelihood
  matches the audit's `_fit_verdict_run1.log` to 10 decimals
  (`-1038.5504` there, `-1038.5503750549412` here).
- The independent skeptic script (`skeptic/skeptic_zip_nb1.jl`,
  `skeptic/run1.log`, same audit branch) reaches the same conclusion from a
  wholly separate implementation (own log-density, own Newton mode search,
  a different Optim line search).

## 7a. Issue Ledger

- #485: addressed on this branch. PR opened (not merged, not marked ready by this
  lane beyond opening it - see PR body); not closed and not commented on beyond the
  PR itself.
- Found in passing, for the orchestrator to file or route. None is fixed here:
  1. The same undamped per-site inner mode search that #479 (Gamma) and #480 (Beta)
     fixed for their families is unfixed in several siblings (per the prior audit's
     `notes.md`, stress-probe non-mode rates): Poisson `_laplace_mode_off` (420/4000),
     NB1 grouped kernel (108/3000), ZIP (616/1500), ZINB (328/1500), HurdlePoisson
     (147/1500), DeltaGamma (274/1500), COMPoisson (19/800), OrderedBeta (212/1500),
     BetaBinomial (103/1500), StudentT shared/grouped (246/1500, 273/1500), GP1
     generic (120/1000), mixed (42/1200), Tweedie grouped (13/300), NB2 grouped
     kernel (147/1500). `ordinal pertrait` is clean (0/1500).
  2. `confint_family.jl`'s bootstrap (`_family_bootstrap`, ~line 2904) counts any
     finite `θ` as converged and never reads the fit's own `.converged`/`.loglik`;
     its profile refit (`_family_profile_refit`, ~line 2817) accepts any finite
     `nmin`, including the `1e12` failure sentinel, and ignores `Optim.converged`
     entirely. Neither goes through `_fit_verdict`.
  3. The NB1 grouped kernel's finite-difference gradient at the fitter's own warm
     start is highly step-size sensitive at the audit's stress points (roughly
     0.4 to 530 across `h` = 1e-4/1e-5/1e-6), consistent with (1): the inner
     per-site search is not tight enough for a clean outer gradient signal.
  4. `_phylo_verdict` (`src/fit_phylo.jl:126`, used by `fit_phylo_gaussian`) has the
     identical root-cause defect - it screens only the failure-sentinel plateau
     (`isfinite(nll) && nll < _PHYLO_PENALTY`) and passes `Optim.converged` straight
     through with no gradient check at all - but it is a separate function from
     `_fit_verdict`, so it is untouched by this fix and outside #485's stated scope.

## 8. Consistency Audit

- 94 `_fit_verdict(res)` call sites use `Optim.LBFGS()`; all support
  `Optim.g_residual`/`Optim.g_tol` natively, so the fix applies uniformly.
- 6 `Optim.NelderMead()` calls exist (the phylo_*_xlv family's constrained-refit
  helpers for profile confidence intervals: phylo_beta_xlv.jl, phylo_binomial_xlv.jl,
  phylo_gamma_xlv.jl, phylo_nb_xlv.jl, phylo_ordinal_xlv.jl, phylo_poisson_xlv.jl).
  None of their results are passed to `_fit_verdict` - each computes its own
  `converged` locally (`Optim.converged(last_res) || abs(constraint_error) <= 1e-3`).
  Unaffected.
- Beta grouped (`fit_beta_gllvm_grouped`/`_cov`, `grouped_dispersion.jl`) already
  ANDs `_beta_grouped_g_met(res, g_tol)` after `_fit_verdict(res)` (#480/#483, same
  formula). After this fix that second check is redundant (idempotent given the
  same `g_tol`), not wrong - a harmless double-application, not simplified here
  (out of scope; a Karpathy-discipline "found, not silently fixed" item, folded into
  this section rather than the ledger since it changes nothing observable).
- Tweedie grouped fits do not call `_fit_verdict` for their convergence decision at
  all; `_tweedie_verdict` is called directly with the same formula. Unaffected
  either way.
- `grouped_fit.jl`, `grouped_nongaussian_fit.jl`, `source_fit.jl`,
  `precision_multivariate_fit.jl`, `joint_phylo_grouped_fit.jl` already carry their
  own gradient guard (per the prior audit's `notes.md`) and do not call
  `_fit_verdict` at all - a separate, already-guarded code path.
- The 3-argument form (`_fit_verdict(nll, converged, iterations)`) has no current
  production caller; it is exercised only by `test/test_known_sentinel_defects.jl`,
  which is unaffected (25/26 pass, 1 pre-existing broken, unchanged from main).

## 9. What Did Not Go Smoothly

- **The 139-file sweep ran far longer than estimated and was interrupted twice.**
  A single-shard timing sample (18 files, 4m56s) extrapolated to ~35-40 minutes
  for 139 files; the actual run passed 45 minutes still executing. Both attempts
  were killed while inside `test_zero_inflated.jl` (a ZIP fit whose per-site mode
  search does a dense `p x p` solve per Newton step - the same twopart-family cost
  the prior audit's `notes.md` already flagged as expensive). Cause: the earlier
  18-file timing sample happened not to include a twopart/ZIP file, so its
  per-file average badly underestimated the tail. Recovery: a fast 23-file subset
  excluding the twopart family was tried next and also ran past estimate (12+
  minutes, killed at the same `test_zero_inflated.jl`); the final, successful run
  further excluded `test_zip_x_identity.jl`/`test_zero_inflated.jl`/
  `test_twopart_substrate.jl`/`test_zinb_x_identity.jl` and completed cleanly in
  4m04s. The ZIP/twopart family itself is untested here beyond what the prior
  audit's evidence already covers (one seed, `converged = true`, `gres = 45`);
  section 10.
- **The killed process's own output could not be trusted as "no flips found."**
  Julia buffers stdout when writing to a file; a `SIGTERM` mid-run does not
  guarantee a flush, so a `VERDICT_FLIP` line from an earlier, already-processed
  file could have been silently lost. The interrupted 139-file run's apparent
  "no flips" is reported as a lower-confidence, secondary observation, not
  evidence on the same footing as the clean 15-file run.
- **Two other agent lanes share this session's lane-lease identity.** A sibling
  subagent's `--claim` call twice overwrote this lane's file claim mid-task (the
  lease tool derives identity from the orchestrating session's PID, which this
  subagent shares with its siblings); re-claimed both times. A genuinely separate
  lane (`claude:GLLVM.jl:wave3-fix500`) also holds `CHANGELOG.md` live at
  write time - handled per the shared-file protocol (claim just before editing,
  retry on refusal; section 4/this section).

## 10. Known Residuals

- Targeted regression sweep covered 15 of the ~59 distinct source files that call
  `_fit_verdict` (the highest-risk ones: NB1, NB2-grouped-boundary, Beta-grouped,
  Gamma-grouped, Poisson, Binomial, ordinal, Tweedie), not the full ~85 call sites
  or the 347-file suite. CI's full matrix and Documenter are the cross-check for
  the remainder.
- The twopart/ZIP family (`fit_zip_gllvm`, `fit_zinb_gllvm`, `fit_zib_gllvm`,
  hurdle/delta variants) was excluded from every completed sweep because its
  tests are individually expensive (dense per-site solves). This is exactly the
  family the prior audit found a real flip in outside the shipped test suite
  (one seed, `gres = 45`) - not re-verified against the shipped fixtures here.
- The R-parity suite (`test/parity/`, RCall + the frozen `gllvmTMB` library) was
  not run. It directly asserts `.converged` in 24 of 46 files and the issue names
  it as the test to check first, but none of its files cover the two families
  where the audit found a real flip (NB1, ZIP/twopart); it is a heavier, optional
  (`GLLVM_PARITY_TESTS=1`), RCall-gated suite kept separate from
  `test/runtests.jl`; CI runs it separately.
- One platform (macOS aarch64, Julia 1.10.0/1.10.12).
- Ledger item 1 (the undamped inner mode search across many siblings) is a
  materially larger fix than #485's scope; #479 and #480 each took one family at a
  time for exactly this reason.

## 11. Team Learning

- A shared verdict helper is worth exactly as much as its weakest disjunct: adding
  the sentinel guard (the FIRST defect `_fit_verdict` was built to catch) did not
  also catch the x/f-only zero-length-step case, because `Optim.converged` itself
  ORs three different criteria and the helper trusted all of them equally. The fix
  for #485 is one line at its root because the scale-aware gradient formula already
  existed (twice, independently, in `_tweedie_verdict` and `_beta_grouped_g_met`) -
  reuse it rather than re-deriving a fourth version.
- A single instrumented run beats two full runs: rather than diffing a full test
  suite on `origin/main` against the fix branch (which only surfaces a flip where an
  EXISTING assertion happens to depend on `.converged`'s exact value - many test
  files construct fits and check other properties without asserting on
  `.converged` at all), a one-line, env-var-gated diagnostic print inside
  `_fit_verdict` itself (`old != new`) catches every flip across every fit any test
  file constructs, whether or not a test happens to look at it.
- `notes.md` from a prior, unmerged audit branch (`origin/claude/
  lane-true-parity-finish-20260925`) is real, load-bearing evidence even though it
  never reached `main` - it named exact function names, line ranges, and a
  stress-probe census for issues #485's own diagnosis only sketched informally.
  When a task cites an unmerged branch's `docs/dev-log/`, read the whole folder, not
  just the one file named.

## 12. Cross-Product Coverage

This change is a single shared verdict helper feeding roughly 85 fit constructors
across every response family in the package (Gaussian, Poisson, Binomial, NB1, NB2,
Beta, BetaBinomial, Gamma, Tweedie, Student-t, ordinal, multinomial, COM-Poisson,
GP1, exponential, ordered Beta, censored/truncated Poisson, truncated NB2, twopart
(ZIP/ZINB/ZIB/hurdle/delta), variational variants, phylogenetic xlv variants,
coevolution, missing-predictor, SPDE, two-level, random-effects, and row/random/
mixed/species-covariate/constrained-ordination/fourth-corner/quadratic/RRR).

**Covers:**
- Every `_fit_verdict(res)` call site (LBFGS-based; 94 of 100 total call sites):
  its `converged` field now additionally requires the scale-aware gradient
  criterion. Confirmed on 15 files/211 assertions across NB1, NB2-grouped, Beta-
  grouped, Gamma-grouped, Poisson, Binomial, ordinal and Tweedie with 0 errors and
  0 observable flips beyond this PR's own test (section 2).
- Downstream consumers of `.converged` that only read the field (AIC/BIC gating,
  `Base.show`'s NOT-CONVERGED tag, boundary-forcing `conv && !any(boundary)`
  patterns in the grouped-dispersion families): these read whatever
  `_fit_verdict` now returns, so they inherit the fix automatically. Not
  independently re-verified per family beyond the sweep.

**Does NOT cover:**
- The 6 `Optim.NelderMead()` call sites (phylo_*_xlv constrained-refit helpers):
  they never call `_fit_verdict` and are unaffected either way (section 8).
- `_tweedie_verdict`, `_phylo_verdict`, and the already-gradient-guarded
  `grouped_fit.jl`/`grouped_nongaussian_fit.jl`/`source_fit.jl`/
  `precision_multivariate_fit.jl`/`joint_phylo_grouped_fit.jl` family: separate
  verdict logic, not `_fit_verdict`, untouched by this change (some already
  correct, one - `_phylo_verdict` - sharing the same unfixed defect; ledger
  item 4).
- `confint_family.jl`'s bootstrap and profile-refit machinery, which never reads
  `.converged` at all (ledger item 2) - a fit whose flag now correctly flips to
  `false` will still be silently counted as converged there.
- The R-parity suite (`test/parity/`) and any downstream code that branches on
  `.converged` inside `confint`/prediction paths not exercised by the targeted
  sweep.
- The underlying inner per-site mode-search noise that produces some of these
  large gradients in the first place (ledger item 1) - this fix reports the
  defect honestly; it does not repair the objective.

Memory receipt: no shinichi-brain lookup run in this slice; the task's own brief,
the cited audit evidence files, and the shipped #480/#483 precedent were sufficient.
Golden Set: no `tools/memory_regression.py` in this repo; the applicable
repeated-failure check run instead was the sentinel-defect regression test
(`test/test_known_sentinel_defects.jl`, unaffected) and the #480/#483 precedent's
own regression tests (`test_beta_grouped_convergence.jl`, in the sweep).
