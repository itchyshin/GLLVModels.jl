# Review: PR #557 (second pass) — zi_* twin, Laplace breakdown guard

Head reviewed: `d0a57e05d39a1f8bcd454d1c3e1a5300f987ee63` (branch `claude/twin-zi`, three commits on
top of `8bab78578`: b24aeb8bd fix, 544872698 tests, d0a57e05d docs). Reviewer worktree:
`/Users/z3437171/local-scratch/lanes/GLLVM.jl-review-557b` (detached, removed after review).
Julia 1.10.0, `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`; R 4.6.0 with gllvmTMB 0.7.1 built from the
P1 checkout (`scratchpad/gllvmTMB-p1`, library `scratchpad/rlib2`). Scratch scripts and logs:
`/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-GLLVM-jl/5b26a91c-0091-4434-b99f-c6632beb6982/scratchpad/r557b/`.

## Verdict: NON-BLOCKING

The blocking item from the first review (a `converged = true` fit at a Laplace-artefact maximum,
-3012.80 against R's -3311.45) is fixed, and the fix is evidenced red-then-green on the same data.
The guard does not change the R-semantics objective at any R optimum I evaluated (three P1 fixtures,
twelve R-converged sweep draws, four R-converged probe draws), the twin tests compare to R's recorded
values, and the shared substrate is untouched. What remains is a set of documentation and robustness
items (findings 2 to 9), none of which reintroduces a silent wrong answer.

## Findings

### 1. (Resolved) Spurious NB2 maximum: red on 8bab78578, green on d0a57e05d — evidence

Same sha256-guarded CSV (`test/fixtures/zi_nb2_reviewer_seed12.csv`), `fit_gllvm(Y; family = zi_nbinom2(), K = 1)`:

```
# old head (--project=GLLVM.jl-review-557b-old @ 8bab78578), red.jl
loglik=-3012.804691829346 R=-3311.454733466863 |diff|=298.65 converged=true phi=[0.62, 0.59, 41.85, 1.80]
PASSES 1e-4 bound: false
# fix head (d0a57e05d)
loglik=-3311.45473336782  R=-3311.454733466863 |diff|=9.9e-8 converged=true phi=[0.899, 2.394, 1.185, 3.497]
PASSES 1e-4 bound: true
```

R refit of all three literal fixtures reproduces the TOML values to the printed digits
(`r_literal.log`): seed 12 conv 0 / -3311.454733; boundary conv 0 / -3385.594830 (phi1 4.91e7);
seed 6 conv 1 / -3192.196822 (Lambda [1.32, -0.55, 0, 0], phi1 146.9).

Test files on the head (test env, `Pkg.develop` of the worktree): `test_zi_twin.jl` 60/60,
`test_zi_recovery.jl` 58/58, `test_zero_inflated.jl` 29/29. No `@test_broken`; the only `@test_skip`
is the pre-existing fixture-absent branch in the twin test.

### 2. (Non-blocking, document) Guarded fits can stall at the wall on data where R converges to a sensible optimum

At the recovery test's own NB2 DGP (p = 4, n = 350, beta = [2.0, 1.6, 2.2, 1.8], lambda = [0.6, -0.5, 0.4, 0.5],
phi = [1.5, 2, 1, 2]) with `MersenneTwister` instead of StableRNG (`legit.jl`, `fromtruth.jl`, `rpar.R`, `eig_at_R.jl`):

```
zi_nbinom2_s1.0_seed2: default start -> flagged (converged=false), ll=-3749.88, mineig=0.10000
                        from truth    -> ll=-3820.2665, mineig=0.727, Optim converged
                        R (P1)        -> conv=0, logLik=-3820.266506, Julia's guarded value at R's point -3820.266506, mineig@R=0.727
```

So a sensible optimum exists, R reaches it, Julia's start hardening does not: the fit stops at the
1e12 cliff with a Laplace value 70 units ABOVE R's optimum (still inflated at eig = 0.1) and is flagged.
This is honest (not silent), and it is 1 of 15 draws at that DGP (the other 14: 12-seed sweep
`sweep_rec.log` + `r_rec.log`, where Julia's 10 converged fits match R's logLik to about 1e-6 each and the
2 flagged draws, seeds 11 and 21, are draws R also fails: seed 11 R errors, seed 21 R conv = 1 with
phi = 431; plus probe seed 1, R conv = 1, min eig at R's point 6e-4). But the docs and the PR body
say the start "is what keeps fits in the sensible basin"; that is overstated.

Suggested fix: when the default-start fit ends at the guard, retry once from a shrunk start (for
example loadings at a quarter of the SVD scale, or Lambda = 0 with the moment phi) and keep the best
un-flagged optimum; state the measured flag rate in the docs ("about 1 in 15 draws at the recovery
setting was Julia-only; the rest are draws gllvmTMB fails on too").

### 3. (Non-blocking, document) Strong loadings: default starts (Julia and R alike) land far below the truth, sometimes with converged = true

Poisson, lambda scaled x2 and x3 (|lambda| 0.8 to 1.8), n = 350 (`legit.log`, `fromtruth.log`, `r_probe.log`):

```
case                   ll_truth   Julia default (guarded)          R (P1)                 Julia from truth
zi_poisson_s2.0_seed1  -3008.04   flagged, -3520.77, eig 0.100     conv=0, -3520.472306   -3001.75, eig 1.84
zi_poisson_s3.0_seed2  -3189.36   converged=true, -3900.59, eig 1.49   conv=0, -3900.591281   -3185.22, eig 1.32
zi_poisson_s3.0_seed1  -3185.51   flagged, -4108.18                conv=0, -4115.261853   -3178.22, eig 1.72
```

On `s3.0_seed2` Julia and R converge to the identical point (-3900.5913, min eig 1.49 at R's point)
that is 711 log-likelihood units below the truth's Laplace value, and both report convergence. This is
not a guard artefact (eigenvalue 1.49) and it is twin-consistent, so it is out of this PR's scope, but
the recovery test's loading range (|lambda| <= 0.6) is the only range where recovery has been shown.
Suggested: say so in `docs/src/response-families.md` next to the recovery-test pointer, and log it as
a start-quality gap (multi-start would fix it on both sides).

### 4. (Non-blocking, code comment) Optim reports `converged = true` at the 1e12 cliff; the 1.1x-floor rule is the only thing catching it

`fromtruth.jl` column `default_optim_conv`: `true` on every wall-stalled fit (min eig 0.10000x):
`zi_nbinom2_s1.0_seed1/2`, `s2.0_seed3`, `zi_poisson_s2.0_seed1`, `s3.0_seed1/3`. LBFGS with
`BackTracking` treats a failed line search into the sentinel as a zero step and declares convergence.
So the "within 10% of the floor -> not converged" rule is load-bearing, not belt-and-braces, and the
-Inf/1e12 wall is safe for the optimiser only because of it. Worth stating in the comment above
`ZI_LAPLACE_EIGMIN_FLOOR` and in the decisions note, so nobody later relaxes the 1.1x rule as
redundant.

### 5. (Non-blocking) The 0.1 floor is justified on a thin measured base; the binomial margin is 2x

`equiv.log`: min site eigenvalue at R's optimum is 0.618 (ZIP), 0.904 (ZINB), **0.202** (ZIB). The
flag threshold is 0.11, so the binomial fixture has 1.8x headroom, and the "half the smallest sensible
eigenvalue" rationale rests on three fixtures plus NB2 draws only. No binomial or Poisson sweep was
run by the builder. My 12-seed NB2 recovery sweep and 20-seed reviewer sweep show all R-converged
points at eig >= 0.39 (one at 0.399, seed 13, with R's phi at 6.4e7). Suggested: a short zi_binomial
sweep (or note that the floor's margin on the binomial family is the least measured), and expose the
measured minimum in the decisions note as the basis.

### 6. (Non-blocking) Missing-typed matrices hit a `MethodError`, not the documented refusal

```
julia> fit_gllvm(Matrix{Union{Missing,Int}}(Y); family = zi_poisson(), K = 1)
MethodError: no method matching fit_zi_gllvm(::Matrix{Union{Missing, Int64}}; family::ZiPoisson, K::Int64)
```

The new ArgumentError ("gllvmTMB masks missing responses row by row; this route does not yet") fires
only for `NaN` in a `Float64` matrix (which is what the test checks). Suggested: accept
`AbstractMatrix{<:Union{Missing,Real}}` in `fit_zi_gllvm` and route `missing` to the same refusal, or
say "NaN" in the docs.

### 7. (Non-blocking) PR body is stale on one mechanism sentence; no parity overclaims; no @handles

- PR body, "How the R-semantics route is built": "it calls the existing `twopart_loglik_site` unchanged".
  Since b24aeb8bd the route calls its own `_zi_loglik_site` (a copy of the site Laplace with the eigen
  guard; it reuses `_twopart_mode_search` and `_tp_pieces`). `equiv.jl` shows the copy is bit-identical
  to `twopart_loglik_site(hessian = :observed)` with the floor off (max site difference 0.0 over 20
  random parameter draws per family), so the twin is unaffected, but the sentence should be updated.
- `grep -nE "@[A-Za-z]" prbody.md` matches only `@details` (R roxygen) and `@formula`; no agent handle.
- CHANGELOG / response-families / decisions note: "twin", "matches to 1.4e-8", "exact marginal
  (quadrature)"; no "parity" claim. Fine.

### 8. (Non-blocking) `test_zi_recovery.jl` carries R-pinned literal values but no `# gllvm-parity-tag: P1`

`grep -n "gllvm-parity-tag" test/test_zi_twin.jl test/test_zi_recovery.jl` -> only the twin file. The
P1 job (`.github/workflows/parity-p1-twin.yml`) discovers tagged files and runs them with
`julia --project=.`, so tagging the recovery file as-is would break that job (it needs StableRNGs from
`test/Project.toml`; 11 other test files already do, so the quick core run is not newly affected).
Either move the two literal R-pinned cases into `test_zi_twin.jl` (tagged), or leave untagged and say
in the file header that the R values are P1-pinned but the file runs under `Pkg.test()` only.

### 9. (Non-blocking) Recovery bounds are tuned to the two StableRNG draws

Other draws at the identical DGP exceed the bounds while matching R's optimum exactly:
seed 14 `e_zi = 0.119` (bound 0.08; R logLik identical to 1e-6), seed 13 `e_logphi = 14.5` (R also puts
phi at 6.4e7). The header's "recovery bounds for one dataset each, not coverage claims" is accurate, but
the rationale that higher means avoid the zi/phi trade-off is only partly true at n = 350. Suggested: one
sentence in the header noting the observed spread across draws (zi error up to 0.12, phi boundary
about 1 in 6 draws, matching R), so the next person does not tighten or "fix" the bounds.

## Scope and claims checked

- `git diff 8bab78578 d0a57e05d --stat`: 16 files; `src/` touches only `src/GLLVModels.jl` (export) and
  `src/families/zi_twin.jl`; `src/families/laplace.jl` diff is empty (0 lines); `_laplace_mode`
  untouched. STOP item respected.
- Twin receipts on the head (`test_zi_twin.log`): d_opt 1.35e-8 / 1.04e-8 / 1.35e-9, d_cross_j
  4.25e-9 / 4.0e-10 / 3.5e-9, d_cross_r 4.3e-9 / 4.1e-10 / 4.4e-9 — unchanged from the PR body; the
  comparisons are against `r_reference.loglik` and `r_at_julia.r_objective_at_julia_optimum` read from
  `zi_p1.toml`, not Julia-self values.
- Guarded vs unguarded objective at R's three optima: identical to all digits (`equiv.log`).
- Builder's 20-draw sweep reproduced (`sweep.log`): 0 silent breakdowns, flagged seeds 6 and 10
  (mineig 0.1000, Lambda_max 1.32 / 1.29), seed 16 -3371.265, seed 18 -3386.720, seed 11 -3268.552,
  7/20 with one phi at ~1e6-1e7 — all as claimed.
- Fix commit does not modify any test tolerance; the new tests have no `@test_broken`.

## What I did not check

- Julia 1.13 (only 1.10.0 run here); Documenter build; the full `Pkg.test()` suite (only the three
  named files plus the scripts above).
- The quadrature numbers in the decisions note (-3376.08 / -3310.37 / -3315.87); I did not rerun the
  6001-point grid.
- A zi_binomial or zi_poisson breakdown sweep (finding 5 is about that absence); K >= 2 behaviour of the
  guard (all evidence is K = 1).
- Whether gllvmTMB at P1 itself ever returns `convergence = 0` at a sub-0.1 eigenvalue point on other
  data (every R conv = 0 point I evaluated had min eig >= 0.20; every sub-0.1 point had R conv = 1 or an
  R error).
- Interaction with the `@formula` front door, `predict`/`confint` (declared out of scope by the PR).
