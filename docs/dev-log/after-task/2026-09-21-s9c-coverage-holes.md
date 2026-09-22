# After-task, leaf-S9c: the four coverage holes leaf-S8 left open

Date: 2026-09-21 · Platform: Claude Code · Lane: `claude/lane-s9cov-20260921` (own worktree)
PR: itchyshin/GLLVModels.jl#445 → `claude/lane-speed78-20260919` · Base: eb48adcc2

## 1. Goal

Close the four coverage holes leaf-S8 left in the grouped analytic outer gradient (Binomial,
per-trait dispersion for Beta and NB2, `GroupingTerm(mode=:dep)`, and the mixed analytic/FD-fallback
path), each with a fixture passing both `fd_agreement` (rtol 1e-6) and `identity` (rtol 1e-8 vs
`origin/main` 69a69b0a0). Also fix the `--gate` dispatch that let a typo pass vacuously. Split out of
S9 on Shinichi's instruction so S9 could ship on its speed result alone.

## 2. Implemented

Test-only; no `src/` file touched. In `test/test_grouped_analytic_grad.jl`:

- four fixtures (`_s9c_fixture_binomial`, `_s9c_fixture_beta_trait`, `_s9c_fixture_nb2_trait`,
  `_s9c_fixture_dep`), added to `_fixtures()` itself, so `fd_agreement`, `identity`, `mixed` and the
  in-suite `@testset` all pick them up rather than a side gate reaching them;
- `_s9c_assert_per_trait_dispersion`, asserting inside the gate that each per-trait fixture really owns
  2 dispersion coordinates, so it cannot silently collapse to the shared case;
- `--gate coverage` (holes 1-3) and `--gate mixed` (hole 4), the latter forcing
  `_grouped_analytic_gradient` to return its `nothing` sentinel every 3rd call during a real fit, with
  `forced > 0` asserted;
- `_s9c_fd_certificate` (design doc section 8.3, h vs h/2) and `_s9c_richardson_check` (a higher-order
  reference) as two independent instruments;
- `_s9c_flat_direction_check` / `_s9c_theta_verdict`, adjudicating any over-tolerance theta coordinate
  by measuring whether the objective moves along it;
- the `atol` term section 8.3's own criterion specifies and this gate lacked, measured per draw per
  coordinate; `rtol` is always tried first;
- `gate_identity` strengthened to compare the FULL theta vector, not just `beta` and `loglik`;
- `_S9C_GATES` dispatch: an unknown `--gate` name exits 2 instead of running `fd_agreement`.

## 3a. Decisions and Rejected Alternatives

- Based on speed78 rather than speed9. The brief said "speed9 if it has landed, else speed78"; it has not
  (PR #430 still open). It is also right on the merits, and this is the finding of the task. Leaf-S9 records
  G9.5's identity half FAILING on all four of these fixtures; on speed78 it passes at 1e-10 to 1e-16.
  The S9 identity failures were its Nelder-Mead demotion, not these fixtures.
- Rejected, widening `RTOL_FD` to clear `beta_trait` coord 5. Explicitly forbidden, and wrong: the
  measurement shows the instrument, not the gradient, is what runs out.
- Rejected, reseeding `beta_trait` away from the near-zero-gradient draw. That is tolerance-shopping
  by another route, and it would hide a real property (a relative-only bound is unreachable at a
  near-stationary coordinate) rather than fix it.
- Chosen instead, the `atol` the design doc's own section-8.3 formula already contains, measured
  rather than chosen, with every use printed and each one cross-checked against Richardson.
- Rejected, exempting the two flat theta coordinates by naming them. Adjudicated by measuring the
  objective along the coordinate instead, so a genuinely wrong coordinate still fails.
- Rejected, merging PR #430 to main despite "merge everything", see §10.

## 4. Files Touched

- `test/test_grouped_analytic_grad.jl`, modified (410 to 974 lines)
- `.unlazy/grouped-analytic-20260920/gates/leaf-S9c.md`, created (force-added; `.unlazy` is
  gitignored but its sibling ledgers are tracked)
- `docs/dev-log/after-task/2026-09-21-s9c-coverage-holes.md`, created (this file)

Scratch only, not committed: step-sweep, Richardson and baseline-probe scripts in the session scratchpad.

## 5. Checks Run

All with `env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=.`, Julia 1.10.0, BLAS 1 thread.

| check | result |
|---|---|
| `--gate coverage` | GATE G9c.1 PASS, exit 0 |
| `--gate mixed` | GATE G9c.2 PASS, exit 0, 10/10 fixtures |
| `--gate fd_agreement` | GATE GB.2 PASS, exit 0, 10/10 |
| `--gate identity` | GATE GB.3 PASS, exit 0, 10/10, now incl. full theta |
| in-suite `include(...)` | 20 pass / 20 total, 7.1 s |
| two-file shared `Main` include | both testsets pass, no collision |
| `--gate coverag` (typo) | exit 2, no gate ran |
| `--badflag x` | exit 2 |
| real 69a69b0a0 baseline diff | 4/4 logliks, 20/20 theta coords within rtol 1e-8 |
| full `Pkg.test()` | **see §10, deferred because another lane held the machine** |

## 6. Tests of the Tests

Each new gate was checked for the way it could pass without testing anything:

- The mixed gate could pass vacuously if the monkeypatch never fires (Julia world age: an `@eval`'d
  redefinition is invisible to a call made later in the same function's dynamic extent). `forced > 0` is
  asserted, and `Base.invokelatest` is load-bearing. Observed forced counts 3 to 13 per fixture.
- The per-trait fixtures could collapse to the shared case and still go green.
  `_s9c_assert_per_trait_dispersion` asserts a 2-coordinate dispersion block; the 69a69b0a0 baseline
  independently prints `log_phi[1]`/`log_phi[2]` and `log_r[1]`/`log_r[2]`.
- The dispatch fix was tested by exercising the trap rather than by reading the code: a typo'd gate exits 2.
- The `atol` escape could mask a real error. It is cross-checked against an independent higher-order
  instrument at the exact offending point, and the control coordinate on the same draw agrees to 4e-11,
  so the method demonstrably discriminates.
- The flat-direction exemption could excuse a wrong coordinate. It is decided by re-evaluating the
  objective with only that coordinate swapped; a wrong value moves it.

## 7a. Issue Ledger

| # | issue | status |
|---|---|---|
| 1 | Binomial uncovered | closed |
| 2 | per-trait dispersion (Beta, NB2) uncovered | closed |
| 3 | `mode=:dep` uncovered | closed |
| 4 | mixed path reasoned about, never exercised | closed |
| 5 | `--gate` fall-through → vacuous pass | closed (still live on speed9) |
| 6 | `beta_trait` coord 5, S9's "NOT CONFIRMED" | closed by measurement |
| 7 | two S8 fixtures with a boundary variance component | characterised, not fixed (§10) |
| 8 | full `Pkg.test()` | deferred (§10) |

## 8. Consistency Audit

The neighbourhood sweep, having found one instance and looked for the same class:

- A `_GATES` collision. Checked whether my new top-level names exist elsewhere in `test/`.
  `_s9c_`/`_S9C_`, `gate_*` and `_compare_fixture` are unique, but `test_grouped_laplace_identity.jl:369`
  binds `const _GATES` unconditionally at top level. Renamed mine `_S9C_GATES` and verified by including
  both files into one `Main`.
- The dispatch trap in sibling gate files. Checked all three; `test_laplace_grad_identity.jl` and
  `test_sparse_phy_identities.jl` already `error(...)` on an unknown gate, as does
  `test_grouped_laplace_identity.jl`. This file was the only outlier, so the fix restores the repo's own
  convention rather than inventing one.
- The theta-vs-beta gap. `gate_identity` compared only `beta` and `loglik`, so I applied the same
  full-theta check there as in the mixed gate, which is how the two boundary fixtures surfaced at all.
- Whether the `atol` weakened S8's gate. Confirmed all six inherited fixtures still pass on rtol
  alone (worst 1.986e-07).

## 9. What Did Not Go Smoothly

- `@printf` rejects a concatenated format string; three call sites had to be rewritten.
- `const _GATES` inside a top-level `if` block needed to be a plain binding.
- The first Richardson script aborted because including the test file runs its `@testset`, which was
  (correctly) red at that moment, had to wrap the include.
- The `--text` flag of `agent_mention_check.py` takes file paths, not a string.
- A first attempt to guess the dispersion-count helper name (`_dispersion_length`) was wrong; verified
  against source before using `_grouped_nongaussian_dispersion_count`.

## 10. Known Residuals

- Full `Pkg.test()` was not run by this lane. Another lane (`GLLVM.jl-s9a-hessian-20260921`) started one
  at 18:36 MDT and the machine allows only one. I waited rather than break that rule. If it completes in
  this session the suite is run and this line updated. A suite-wide regression outside the edited file is
  therefore unproven. The file itself was run standalone and as an include, plus a two-file `Main` check.
- PR #430 was NOT merged to main, despite the instruction to "merge everything". My branch sits on
  speed78, which is itself PR #430's head, so merging to main would land the whole S4/S7/S7b/S7c/S8 perf
  change rather than my work. That PR's own GB.6 records 1 failure and 19 broken, and leaf-S9 has several
  unmet STOP conditions. Landing it is a judgement call that belongs to Shinichi, so PR #445 targets
  speed78 and #430 is left for him.
- `beta_trait` coord 5 is certified to the best available instrument's resolution (1.350e-09
  absolute), not to 1e-6. A gradient error smaller than that is undetectable there by any FD method;
  catching one would need AD or exact differentiation.
- Two S8 fixtures keep an unidentified variance component at the default `g_tol=1e-4`. Whether to
  reseed them is open, and is the same question leaf-S9's G9.2 raised from the Hessian side.
- The mixed path is driven by a forced failure rather than a naturally failing theta.
- The `--gate` fall-through is still live on `claude/lane-speed9-20260921`; that lane should take
  this fix when it rebases.

## 11. Team Learning

- A gate that falls through to a default gate can report PASS for a gate that never ran. Test the
  trap rather than the code. Three sibling files here already had it right; the outlier was the newest.
- "The gate is red" starts the diagnosis rather than ending it. The rule is never to widen a
  tolerance, but the honest alternative is not simply to fail either: it is to certify the instrument. The
  design doc already specified h-vs-h/2 and an `atol`; the gate simply had not implemented them.
  Measuring with a second, higher-order instrument turned S9's "NOT CONFIRMED" into a closed question in
  about ten minutes.
- A relative-only per-coordinate bound is unreachable wherever the true value is near zero. With
  20 random draws, some coordinate eventually sits near stationarity. `rtol*|g| + atol` is not a
  weakening; omitting `atol` leaves a latent unpassable gate.
- Compare the whole parameter vector rather than the headline quantities. `loglik` and `beta` agreed to
  1e-13; two collapsed variance components were only visible in full `theta`.
- Check what is running before claiming the machine. A full suite was already in flight from another
  lane, and the one-suite rule caught it.

## 12. Cross-Product Coverage

Families × dispersion × term mode, after this leaf:

| | Poisson | Binomial | Beta | NB2 |
|---|---|---|---|---|
| `:indep common=true` | ✅ S8 | ✅ new | ✅ S8 shared, ✅ new per-trait | ✅ S8 shared, ✅ new per-trait |
| `:indep common=false` | ✅ S8 (`poisson_percoord`) | ❌ | ❌ | ❌ |
| `:latent rank=1` | ✅ S8 | ❌ | ❌ | ❌ |
| `:dep` | ✅ new | ❌ | ❌ | ❌ |
| two terms | ✅ S8 | ❌ | ❌ | ❌ |

Gradient paths: pure analytic ✅, pure FD ✅, mixed ✅ new. The negative space is deliberate and
named. `:dep`, `:latent`, `common=false` and multi-term are each covered for Poisson only, so a
family-specific bug in a non-Poisson `:dep` or `:latent` term would not be caught. Dispersion-bearing
families (Beta, NB2) are covered only under `:indep common=true`. Extending the grid is the obvious
next slice, and is not claimed here.
