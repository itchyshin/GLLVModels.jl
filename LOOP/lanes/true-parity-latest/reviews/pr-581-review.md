# Review: PR #581 — Laplace breakdown guard on `fit_truncated_nbinom2_gllvm`

Branch `claude/fix-truncnb2-laplace-breakdown`, head `5cb29b01d`, merge-base `880cad4c7`
(GitHub lists the base as `main` @ `1214e948e`; the three commits are test / fix / docs).
Reviewer worktree: detached at `5cb29b01d` under `local-scratch/lanes/GLLVM.jl-review-581`
(plus a second detached worktree at `880cad4c7` for the red-on-main check); both removed.
Julia 1.10.12 and 1.13.0, `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1`, four processes,
about 35 min of compute. Scripts in the session scratchpad (`common.jl`, `regress.jl`,
`diag.jl`, `floor.jl`, `ci.jl`, `rescue301.jl`, `diag604.jl`).

## Verdict: NON-BLOCKING

The guard does what the PR says: on every draw I tried, main's silent breakdowns (converged
= true with a smallest site eigenvalue of 1e-6 to 2e-4 and a Laplace value tens to hundreds
of units off the exact marginal) become either a healthy optimum or an honest
`converged = false`. Red-on-main and green-here reproduce on both Julia versions. No
blocking defect. Two things should be corrected before merge (findings 1 and 2 are
wording / a small reporting choice), and one pre-existing defect found in passing (finding
6) deserves its own issue.

## Findings

### 1. The "healthy fits end 5e-4 lower" regressions are not early stops of the guard; it is main's fits that stop early, on a discontinuity (NON-BLOCKING, fix the note and CHANGELOG)

The brief's key question was whether a guarded fit ending 5e-4 below main's value with
`converged = true` is a silent early stop. I reproduced seeds 103, 101 and 111 on Julia
1.13.0 and inspected Optim's convergence flags and the finite-difference gradient at both
optima (`diag.jl`, a replica of the fitter's own start and L-BFGS options):

| seed (1.13) | run | loglik | Optim flags | `g_residual` | wall hits |
|---|---|---|---|---|---|
| 103 | main path (`eigmin_floor = -Inf`) | -1352.048581 | `f_converged` only | **45.3** | 0 |
| 103 | guarded (this PR) | -1352.049080 | `g_converged` | 8.4e-6 | 1 |
| 101 | main path | -1597.341775 | `f_converged` only | 0.16 | 0 |
| 101 | guarded | -1597.341777 | `g_converged` | 4.7e-6 | 1 |
| 111 | main path | -1578.782500 | `f_converged` only | 0.056 | 13 |
| 111 | guarded | -1578.782501 | `g_converged` | 4.1e-6 | 9 |

The guarded fits are the stationary points (gradient below `g_tol`). Main's fits stopped
on `f_converged` with a gradient of 45 (seed 103) at a point where the objective is
discontinuous: on seed 103, `f(θ̂ + 1e-5 e_r) - f(θ̂) = +5.49e-4` while
`f(θ̂ - 1e-5 e_r) - f(θ̂) = +1.5e-8`, i.e. main's fit sits on the upper lip of a 5.5e-4
cliff in the Laplace objective (the inner mode solve or a clamp switches on one site).
Central differences at h = 1e-5 give a gradient of 27 there and 1.5e-3 at h = 1e-6. The
guarded fit sits 2e-3 away in a smooth region. Polishing the guarded optimum with the
floor off takes 0 iterations (it is already stationary). So the 5e-4 "loss" is the height
of a pre-existing cliff, not a stall caused by the wall, and nothing needs fixing in the
guard for it. On the 1.13 sweep below the same effect goes the other way on seed 121
(guarded +6.1e-4 higher, two polish iterations).

Suggested fix: reword the decision note and CHANGELOG line "three stop up to 5e-4 short of
the same optimum, within the optimiser's g_tol" to say that main's fits on those draws
stopped on `f_converged` at a discontinuity with a large gradient, and the guarded fits are
the gradient-converged points. The discontinuity itself (a pre-existing property of the
observed-curvature Laplace objective with the Fisher-scoring mode solve) is out of scope;
it is the likely cause of finding 6 too.

### 2. When both the first fit and the retry end at the guard, the worse of the two is reported (NON-BLOCKING, small reporting choice)

`at_guard(f2) || (f = f2)` keeps the first fit whenever the retry is also at the guard, the
#557 rule. On a hard draw (`genTNB` with r = 0.05, β = 0.5 + randn, seed 604, Julia 1.10)
the first run ends at the wall at loglik -3659.3 with r̂ = 6e-42 (45 walled evaluations,
`x_converged`/`f_converged`), and the retry ends at the wall at -2008.5 with r̂ = 0.024.
The fit reported is the -3659.3 one. Both are flagged `converged = false`, so nothing is
silent; but a user who inspects the flagged fit gets the least informative of the two.
Suggested fix: when both are at the guard, report the one with the higher loglik (still
flagged), or say in the docstring that the first fit is reported by design. Same rule
exists in #557; if changed here, change it there.

Also on this draw, main's default start reaches a point with a healthy eigenvalue (1.38,
loglik -2836.0), which every guarded start (eight tried, including the moment r with
loadings x 1, x 0.5, x 0.1) misses, ending at the wall around -2007. That is not a lost
healthy optimum: the wall points beat main's point by 830 Laplace units and its Laplace
value is itself 79 units below the exact marginal (r̂ = 0.0025). It does show that
"retry once from the moment r" does not guarantee a healthy end point; the honest flag is
the right outcome there.

### 3. Floor 0.1 is safe on the evidence; the eigenvalue distribution is bimodal (CLEAN)

Smaller-r and lower-count settings, Julia 1.10.12, p = 4, n = 150, K = 1 (`floor.jl`;
"raw" = `eigmin_floor = -Inf` = main's path; gap = Laplace minus exact 4001-point
quadrature):

| setting | draws | raw silent breakdowns (conv, min eig < 0.1) | guarded: healthy optimum | guarded: flagged | healthy min eig range |
|---|---|---|---|---|---|
| r = 0.15, β = 1 + randn | 10 (301-310) | **7** (min eig 8e-6 to 1.7e-4; gap 31 to 278) | 5 (gap 1.9 to 4.0) | 2 (301, 306; both at 0.100) | 0.886 to 0.973 |
| r = 0.10, β = 1 + randn | 6 (501-506) | 4 (gap 73 to 787) | 3 (gap 1.5 to 6.6) | 1 (502, at 0.102) | 0.80 to 1.02 |
| r = 0.05, β = 0.5 + randn | 6 (601-606) | 3 (601, 602, 606) | 1 | 3 (602, 604, 606) | 0.88 to 29.3 |
| r = 0.30, β = -0.5 + randn (low counts, mean 3 to 4) | 3 (401-403) | 0 | 3 (unchanged bit for bit) | 0 | 1.04 to 1.34 |
| r = 0.30, β = 1 + randn, Julia 1.13 | 20 (121-140) | 2 (126, 130; min eig 7e-5, 2e-7) | 2 (gap not computed; min eig 1.09, 1.08) | 0 | 0.716 to 1.98 |

Across 45 draws the smallest site eigenvalue at a fit that is not a breakdown never fell
below 0.716, and every breakdown was below 2e-4. Nothing lands between 0.11 and 0.7, so the
floor with its 10% band is not cutting into healthy optima on these data. Note the main-path
breakdown rate at r = 0.15 is 7 of 10, far above the audit's 4 of 20 at r = 0.3: this
guard matters more the smaller r is. The flagged fits are honest: on seed 301 no start of
six reaches a point off the wall, and the wall points (exact marginal -2088 to -2095) are
still far better than main's breakdown point (exact -2472).

Seeds 603 and 605 (r = 0.05) show the guard's limit: raw and guarded agree, min eig 29.3
and 1.09, `converged = true`, but the Laplace value is 75 units below the exact marginal on
603 with r̂ = 0.00101 (a lower clamp on r, it seems). That is ordinary Laplace error the
floor is not meant to catch; the decision note already says so.

### 4. Red then green, fixture guard, tolerances, scope (CLEAN)

- Red on main: the PR's test file and fixture copied into a worktree at `880cad4c7`:
  8 passed, 8 failed, 1 errored (the mechanics testset errors on the `eigmin_floor`
  keyword), matching the PR body.
- Green here: 27/27 on Julia 1.10.12 (85 s) and 27/27 on 1.13.0 (71 s).
- The sha256 in the fixture matches a recomputation over `Float64.(Y)` bytes for all three
  seeds (88e407419c3e…, 23cfe8e665c7…, 1d78bc32503a…). The guard is real.
- No `@test_broken`, no tolerance change in any existing test; the only `atol`/`rtol` in
  the diff are in the new file.
- Diff touches exactly `src/families/truncated_nbinom2.jl`, the new test and fixture,
  `test/runtests.jl` (one `_shard_include` next to `test_truncnb2_precision.jl`),
  `CHANGELOG.md`, and the decision note. `laplace.jl`, `zi_twin.jl`, `formula.jl`,
  `Project.toml`, CI untouched.
- Struct field: `TruncatedNegBin2Fit` is constructed positionally in exactly one place
  (the fitter); `postfit.jl` `_nparams`, `confint_family.jl`, the `show` method and the
  docs use field names or are unaffected; no serialization / StructTypes usage. `SHA` and
  `TOML` are already in `test/Project.toml` and used by other tests (so the quick
  `julia --project=. test/runtests.jl` route needing them is pre-existing).
- Public marginals: `truncated_nbinom2_marginal_loglik_laplace` at `raw.θ` equals
  `raw.loglik` to 1e-6 (the test), and the site kernel returns before touching
  `eigmin(A)` when the floor is `-Inf`, so the per-trait fitter and the CI adapter are
  bit-identical to main.
- No agent `@handle` in the PR body, commits or docs (the only `@` strings are
  `Co-Authored-By` trailers and `@test` macros).

### 5. Per-trait fitter and the CI adapter: leave out of scope, with one measured data point (NON-BLOCKING)

On the fixture draw 104 (the spurious-global-maximum case) I profiled r by hand over the
unguarded objective (fix log r, re-optimise β and Λ from the guarded optimum), which is
what `_family_ci(::TruncatedNegBin2Fit)` evaluates:

| r | Laplace | exact | min eig | D = 2(ℓ̂ − ℓ_p) |
|---|---|---|---|---|
| 0.133 (r̂) | -1686.333 | -1686.370 | 1.09 | 0 |
| 0.10 | -1686.403 | -1687.429 | 1.02 | 0.14 |
| 0.05 | -1686.786 | -1689.740 | 0.917 | 0.91 |
| 0.033 | -1686.990 | -1690.484 | 0.888 | 1.31 |
| 0.15 | -1686.348 | -1685.982 | 1.12 | 0.03 |

The profile stays healthy (min eig at least 0.89) across the range a 95% interval would
cover, so on this draw the adapter does not walk into the breakdown region. One draw is not
a proof; the PR body's "not probed" is now "probed on one draw, no breakdown". Keeping the
adapter and the per-trait fitter unguarded in this PR is reasonable; if the guard is later
lifted into the shared helper the PR proposes, they are the next two callers.

### 6. Found in passing, pre-existing: `confint(fit, Y; method = :profile, parm = "r")` on this family takes 12 minutes and returns `status = :failed`, and the Wald interval for r is far too narrow (OUT OF SCOPE, file an issue)

On draw 104 (guarded fit, r̂ = 0.1334): `method = :wald` returns [0.131, 0.136] (se 0.0096
on the log scale) in 3 s; `method = :profile` ran 698 s and returned `lower = upper = NaN`,
`status = :failed`. My hand profile above says D(r = 0.033) is only 1.31, i.e. the 95%
profile interval extends below r = 0.03, while the Wald interval implies D(0.033) of about
2e4. The FD Hessian is almost certainly picking up the cliff of finding 1 (the objective is
discontinuous at scale 1e-5, and Optim's central-difference step is 6e-6). None of this is
changed by the PR (the adapter runs main's code path), and `docs/src/response-families.md`
line 406 still says the family has no `confint` dispatch although `_family_ci` exists for
it. Worth its own issue: the r interval from Wald on this family is not trustworthy.

### 7. Small things (NON-BLOCKING)

- `converged && at_guard(f)`: a fit that hit the iteration cap at the wall is reported
  `converged = false` without the warning. Harmless; the warning could be issued on
  `at_guard(f)` alone.
- `_truncnb2_min_site_eigen` runs n mode solves after every L-BFGS run even with
  `eigmin_floor = -Inf`. About one objective evaluation; fine.
- The warning message prints the floor via string interpolation of a `Real`; with the
  default it reads `floor 0.1`, fine.
- Decision note "healthy optima 0.79 to 2.04": my sweeps add 0.716 (1.13, seed 124) and
  0.80 (r = 0.1, seed 504). Still well above 0.11; the note could quote the wider range.

## What I did not check

- The full `Pkg.test()` suite (Aqua, JET, the other 100-odd files). I ran only the new
  file on both Julia versions and the red-on-main reproduction; the builder's 2343-pass
  count over 23 truncation files was not re-run.
- Julia 1.10 healthy sweep beyond seeds 102 and 108 (both reproduce the PR body: 102 moves
  to the higher healthy optimum -1215.971 with a gradient-converged end point; 108 identical
  to 2.3e-12). The 20-draw healthy sweep was repeated only on 1.13 (seeds 121-140, a
  different set from the builder's 101-120): 16 of 18 healthy draws bit-identical, seed
  129 within 3e-12, seed 121 +6.1e-4 (the cliff, finding 1), and both breakdowns rescued.
- K > 1, masks, offsets, `hessian = :fisher` (where W is non-negative, so the guard is
  inert by construction; not exercised).
- The per-trait fitter on breakdown-prone draws (only the r-profile of the shared-r
  adapter, finding 5).
- The parity file and anything needing R.
- Whether the 1e-5 scale discontinuity of finding 1 is in `_grouped_laplace_mode`'s
  backtracking, `_clamp_mu`, or the observed-weight formula; I measured it, not its source.
