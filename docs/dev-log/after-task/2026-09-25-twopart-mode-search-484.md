# After-task: Two-part mode search is damped and fails loudly (#484, 2026-09-25)

Lane: Claude, branch `claude/twopart-mode-search-484`, worktree `~/local-scratch/gllvm-twopart-484`.
Handed off from a previous session with two commits already on the branch: `5cb477e7d` (regression
test, committed failing) and `deb8ff154` (WIP kernel fix, "not reviewed, not tested to completion").
This session claimed a narrow lane lease, rebased the branch onto `origin/main` (past #481 Gamma,
#483 Beta, #492 and #494 covariate-kernel), reviewed the WIP fix, verified it, finished the commit
message, added CHANGELOG/check-log/after-task records, and opened the PR.

## 1. Goal

Fix #484. The per-site mode search shared by all two-part families (ZIP, ZINB, ZIB, Hurdle-Poisson,
Hurdle-NB, Delta-Gamma, Delta-lognormal, Beta-hurdle) took full Fisher-scoring steps with no damping
and stopped at `maxiter` with no convergence signal; `twopart_loglik_site` then scored whichever `z`
the loop held, converged or not. The value was finite, so a fit could report `converged = true` far
below the optimum: the public base-s1 ZIP fit reported -935.296, where a restart reaches -920.603,
and at the ZIP seed-101 simulation truth 9 of 80 sites ran out of iterations with the package
objective at -10085.3 against -930.4 at converged modes.

Required outcomes (from the regression test the previous session committed failing):
- A search that does not converge must never produce a finite site value.
- The public base-s1 ZIP fit must reach at least -920.7, or must not claim convergence.
- Site values where the old search converged must be unchanged (to 1e-8).

## 2. What I did this session

- Claimed a lane lease narrowed to this task's paths (a wider claim conflicted with a live lease
  another lane held on `docs/dev-log/core070/`, `docs/dev-log/owed/`, `tools/parity_ledger.py`).
- Created the worktree (it already existed from the previous session at the branch's tip) and
  rebased onto `origin/main` @ `d4da31544` (the #494 merge). Only `CHANGELOG.md` conflicted (two
  independent `### Fixed` entries added in the same place); resolved by keeping both, ours after
  #480's. `src/families/twopart.jl` had not changed on `main` since the branch's base, so the kernel
  commit applied with no conflict.
- Reviewed the WIP kernel fix (`deb8ff154`) against the precedent this repository already set for
  the same class of bug: `_gamma_grouped_mode` (#479/#481) and `_laplace_mode_off` (#494/d0e99f6d0).
  The two-stage structure (Fisher with step-halving first, damped Newton fallback on observed
  curvature second, `-Inf` on total failure) matches those exactly; the one addition specific to
  two-part families is dropping negative observed weights in the Newton stage when they make the
  step matrix indefinite (ZIP's zero-inflated mixture can bend the observed curvature negative at a
  zero count, which Gamma/log never does). Read this as a correct, non-mechanical extension of the
  established pattern, not a divergence from it.
- Verified the independent HurdleNB score fix (missing NB2/log chain-rule factor `a = r/(r+μ)`) by
  hand against the `TruncatedNegBin2` derivative used elsewhere in the same file, and by the new
  ForwardDiff cross-check test, which fails without the factor (see §6).
- Ran the required red/green checks (§5) and the full two-part-reaching test battery (§5).
- Amended the WIP commit's message into a finished, reviewed one (no code changes — the diff is
  byte-identical to what the previous session left); kept both `Co-Authored-By` trailers and added
  my own.
- Added the CHANGELOG entry already drafted by the WIP commit (carried through the rebase unchanged),
  a `docs/dev-log/check-log.md` entry (prepended — this file orders newest-first), and this report.
- Pushed the branch and opened the PR.

No kernel code was changed in this session; the WIP fix was reviewed, verified, and landed as-is.

## 3a. Decisions and Rejected Alternatives

Carried over from the previous session's WIP commit (I verified rather than re-derived these):

- **Fisher first, then Newton, then -Inf** — same choice #479/#481 made, for the same reason: Fisher
  scoring alone left many sites unconverged (1327 of 3600 stress-site searches), so Newton-only would
  change the path on every site (breaking bit-identity with the old loop where it was already
  correct); Newton-only-as-fallback keeps every previously-converged site on its exact old path.
- **Negative-weight dropping in the Newton stage, `:fisher`/`:newton` selectable.** Specific to
  two-part families: ZIP's observed positive-part curvature can be negative at a zero count under a
  large Poisson mean (the zero-inflated mixture bends the wrong way), which would make
  `Λc'diag(Wc)Λc + I` indefinite. Dropping negative weights keeps the matrix `>= I`, so the step is
  still an ascent direction and halving still works. The new stress test exercises this at `z = -1.13`.
- **HurdleNB score fix bundled with the damping fix, not filed separately.** The two bugs interact:
  without the correct score, "the mode search converged" is not enough, because the point it
  converges to is not the actual mode. Fixing damping alone would have left HurdleNB's `_tp_pieces`
  silently wrong while its search "converged" to the wrong point.
- **This session's own decision: verify before re-deriving.** Given the WIP fix already matched
  established precedent and its own regression test, I chose independent verification (revert-src
  red check, full two-part battery) over rewriting, per the instruction to review critically rather
  than restart.

## 4. Files Touched

- `src/families/twopart.jl` — `_twopart_mode_stage`/`_twopart_mode_search` (new), `_tp_newton_Wc`
  (new), `_twopart_logpost` (new), `_twopart_mode` (now a thin wrapper), `twopart_loglik_site`
  (returns `-Inf` on non-convergence), HurdleNB's `_tp_pieces` (NB2 chain-rule factor), docstring
  updates.
- `src/families/beta_hurdle.jl` — one docstring sentence (the mode search no longer depends on
  `hessian`).
- `test/test_twopart_mode_search.jl` — extended with the score-derivative cross-check and the
  stress-site test (this session verified only; did not edit).
- `test/fixtures/twopart_mode_search_484.toml` — unchanged from the previous session.
- `CHANGELOG.md` — the entry drafted by the WIP commit, carried through the rebase.
- `docs/dev-log/check-log.md` — new entry, this session.
- `docs/dev-log/after-task/2026-09-25-twopart-mode-search-484.md` — this report, this session.

Not touched: `Project.toml`, any frozen-contract file, any required-cell assertion, any ledger
signature, `gllvmTMB` (the R twin, never edited from this repo).

## 5. Checks Run

All runs used Julia 1.10.0 (juliaup default on this Mac; `julia +1.10` not separately needed since it
was already the active channel), `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, `--project=.` (the
package's own environment; `test/Project.toml` deps resolve through Julia's stacked test-environment
convention, so the whole test file runs directly with `include`).

- **Red on main.** `src/families/twopart.jl` and `src/families/beta_hurdle.jl` temporarily replaced
  with `git show origin/main:...` (test file untouched): 14 of 35 assertions in
  `test/test_twopart_mode_search.jl` fail. Restored the fixed files afterward (`git diff` against the
  committed tree came back empty, confirming an exact restore).
- **Green on branch.** The same test file on the finished commit: 35 of 35 pass (52.8 s).
- **Every existing test file that reaches a two-part family.** Found by grepping `test/*.jl` for the
  two-part family constructors, fitters and kernel entry points (ZIP/ZINB/ZIB/ZIPoisson, HurdlePoisson
  /HurdleNB, DeltaGamma/DeltaLogNormal, BetaHurdle, `fit_*_gllvm` for each, `twopart_loglik*`,
  `_twopart_mode`), 25 files beyond the new one:

  `test_beta_hurdle.jl`, `test_bridge_x.jl`, `test_bridge_zib.jl`, `test_bridge_zip_nox.jl`,
  `test_confint_family.jl`, `test_curvature_census.jl`, `test_delta_disp_group.jl`,
  `test_delta_fit.jl`, `test_delta_gamma.jl`, `test_delta_postfit.jl`,
  `test_delta_shared_predictor.jl`, `test_formula.jl`, `test_hurdle_nb.jl`,
  `test_hurdle_poisson.jl`, `test_offset.jl`, `test_postfit_zib_tweedie.jl`,
  `test_second_order_delta_followup.jl`, `test_twopart_alloc_equiv.jl`,
  `test_twopart_hessian_kwarg.jl`, `test_va_vs_laplace.jl`, `test_variational_dgamma.jl`,
  `test_zero_inflated.jl`, `test_zib_x_identity.jl`, `test_zinb_x_identity.jl`,
  `test_zip_x_identity.jl`.

  Run together (each in its own nested `@testset`, on the finished commit): **2059 pass, 1 broken, 0
  fail, 0 error, 2060 total, 24m55.6s.** The one "broken" result is a pre-existing `@test_skip` in
  `test_second_order_delta_followup.jl` line 183 (an opt-in live-R-parity test, gated on
  `GLLVM_PARITY_TESTS=1`), present on `origin/main` and unrelated to this change.
- Trial rebase: clean, one conflict (`CHANGELOG.md`, resolved by keeping both entries).

## 6. Tests of the Tests

- **The regression test fails on the unfixed kernel**, measured directly (not inferred from the
  commit message): 14 of 35, listed in full in the run log. This includes the score-derivative
  cross-check (which would fail even without the mode-search damping change, since it targets the
  HurdleNB chain-rule bug specifically) and every mode-search-specific assertion (seed-101 truth,
  the `-Inf`-on-failure guard, the two stress sites, the public base-s1 fit).
- **The independent evaluator shares no code with the kernel.** `_tp484_zip_logf` /
  `_tp484_indep_site` (in the test file) reimplement the ZIP density and its own Newton search from
  scratch; the "where the old loop converged, unchanged to 1e-8" testset (5/5) and the stress-site
  testset compare the package kernel against this independent path, not against a stored expected
  value.
- **The score-derivative check is itself falsifiable.** It failed loudly on unfixed `origin/main`
  (`worst < 1e-10` evaluated to `1.49` before the fix) rather than passing vacuously, confirming it
  exercises the actual chain-rule bug.
- I did not write or modify any test in this session — the above is verification that the test suite
  the previous session left behind does what its own commit message claims, not a claim about tests I
  authored.

## 7. Issue Ledger

- #484: fixed on this branch; PR opened (not merged — merging is out of scope for this lane).
- Nothing new found in passing this session. (The previous session's WIP commit message and this
  report both note the HurdleNB chain-rule bug as part of the same fix, not a separate finding.)

## 8. Consistency Audit

The two-part kernel is now damped and reports convergence honestly; every OTHER per-site mode search
in this repository was already fixed or left as known debt before this branch started:

| Path | Damping | Convergence reported | Status |
|---|---|---|---|
| `_twopart_mode_stage`/`_twopart_mode_search` (this fix) | Fisher + halving, then damped Newton | yes | fixed here |
| `_gamma_grouped_mode` (#479/#481) | Fisher + halving, then damped Newton | yes | fixed, prior branch |
| `_beta_grouped_*` convergence gate (#480/#483) | gradient-criterion gate + restart | yes (by gate, not the mode search itself) | fixed, prior branch |
| `_laplace_mode_off` (covariates.jl, #494) | Fisher + halving, then damped Newton | yes | fixed, prior branch |
| `_grouped_laplace_mode` (getLV, generic core) | halving only | no explicit flag | unchanged, out of scope |
| `_nb_grouped_loglik_site` (NB2), `_nb1_grouped_loglik_site`, `_tweedie_grouped_loglik_site` | none / partial | no | unchanged, out of scope (recorded by #479's after-task) |

I did not re-audit these siblings myself this session; the table reflects what the #479 after-task
already recorded and what this branch's own commits changed. No new sibling work was done.

## 9. What Did Not Go Smoothly

- The lane-lease claim was refused on the first try (a live lease on `docs/dev-log/core070/` and
  `docs/dev-log/owed/` held by a differently-scoped lane using the same session identity prefix);
  narrowed the path list and it was granted immediately after.
- `docs/dev-log/check-log.md` is 20,551 lines and orders newest-first, not append-at-bottom the way
  I assumed from a first `tail`; caught by reading a prior PR's (`#479`) actual diff to the file
  before writing, not by guessing the convention.
- Running the two-part-reaching test battery unsharded took about 25 minutes (`test_bridge_x.jl`,
  `test_bridge_capabilities.jl`-style files run full optimizer fits), longer than expected for a
  single-file estimate; no correctness surprises, just wall time.

## 10. Known Residuals

- **Runtime.** Not separately measured this session for the two-part kernel (the previous session's
  commit message does not claim a number either); the Gamma precedent (#479) measured 20-25 percent
  slower healthy fits from the extra log-posterior evaluations on large steps, and the two-part
  kernel's damping is structurally the same, so a comparable cost is plausible but unverified here.
- **Sibling kernels not revisited.** NB2, NB1 and Tweedie's grouped per-site searches remain
  undamped, as recorded by #479's after-task; this branch touches only the two-part family kernel.
- **The R parity suite was not run.** This fix has no R/gllvmTMB counterpart to compare against (the
  two-part mode search is a Julia-side numerical-search change, not a likelihood-formula change from
  TMB's perspective); `test/parity/` was not touched or run.
- **One platform.** All runs were on macOS aarch64, Julia 1.10.0. CI's Linux shards are the
  cross-platform check; not reproduced locally on Linux or Windows.
- **`maxiter`/`tol` sensitivity beyond what the test covers.** The test exercises the default
  `maxiter = 100, tol = 1e-9` and `maxiter = 1` (to force the `-Inf` path); no sweep over other
  values.
- **A masked-data case where the Newton fallback fires was not specifically tested.** Masks pass
  through the same code (`_tp_pieces_at` is called per-site regardless), but nothing in the new test
  forces the fallback under a mask specifically.

## 11. Team Learning

- Reviewing a WIP fix against the repository's OWN recent precedent (here: #479/#481's
  `_gamma_grouped_mode`) is a fast, concrete way to critically evaluate an unreviewed change: the
  structure either matches an established, already-scrutinized pattern, or it doesn't, and either
  answer is informative.
- Read a file's actual recent history (via a real prior commit's diff) before assuming its edit
  convention (append-at-bottom vs. prepend-at-top) from a single `tail`.
- A finite-value sentinel is not enough on its own; this is the same lesson #479's after-task
  recorded, now confirmed to generalize across two independent per-site search implementations in
  this repository.

## 12. Cross-Product Coverage

This work does NOT cover:
- the NB2, NB1 or Tweedie grouped kernels, or the generic `_grouped_laplace_mode`/`_laplace_mode`
  (getLV) paths;
- non-default `maxiter`/`tol` combinations beyond the two the test forces;
- a masked-data case that specifically exercises the Newton fallback;
- runtime cost measurement for the two-part kernel specifically;
- the R parity suite (not applicable to this change; not run);
- Julia versions other than 1.10.0/1.10 LTS, or non-macOS platforms;
- merging the PR (left open, as instructed).

Memory receipt: no shinichi-brain lookup was run in this slice; inputs were the orchestrator's brief,
the previous session's two commits and their messages, and repository sources (`grouped_dispersion.jl`,
`laplace.jl`, `families/covariates.jl`) for the precedent comparison in §3a and §8.

Golden Set: no memory-regression run. Applicable repeated-failure checks were run: failure recorded
before the fix (§5 red-on-main), the fix verified against an independent evaluator that shares no
code with the kernel (§6), and no tolerance was widened.
