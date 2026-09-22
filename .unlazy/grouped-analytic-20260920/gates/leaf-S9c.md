# Gates: leaf-S9c — the four coverage holes leaf-S8 left open

Split out of leaf-S9 on Shinichi's instruction 2026-09-21, so S9 could ship on its measured speed
result alone and this coverage work would not be rushed alongside a performance change.

OWNS: test/test_grouped_analytic_grad.jl (only). No `src/` file is touched by this leaf.

BRANCH: `claude/lane-s9cov-20260921`, branched from `claude/lane-speed78-20260919` at eb48adcc2.
**Why speed78 and not speed9:** the brief said "speed9 if it has landed, else speed78". It has NOT
landed — PR #430 is still open on speed78, and speed9 exists only as its own branch. That is also the
substantively right base, and it is the single most important decision in this leaf. On speed9 the
Nelder-Mead demotion is in force, and leaf-S9's own ledger records G9.5's identity half FAILING on all
four of these fixtures (~1e-7 to 1e-6). On speed78 the demotion does not exist, and **the identity half
passes on all four at 1e-10 to 1e-16**. The S9 identity failures were the demotion, not these fixtures.

BASELINE: `origin/main` 69a69b0a0, checked DIRECTLY, not only through the in-worktree proxy — see G9c.3.

ENVIRONMENT for every run below: Julia 1.10.0, `JULIA_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`,
BLAS `LBTConfig([ILP64] libopenblas64_.dylib)` at 1 thread, macOS arm64. Runs 2026-09-21 ~18:00-18:40
MDT, sole Julia process on the machine (checked before starting).

SCOPE: the four holes, plus the dispatch trap the brief named. Never widen a tolerance:
`RTOL_FD = 1e-6` and `RTOL_IDENTITY = 1e-8` are untouched, and both are still the literal constants at
the top of the file.

## The fixtures added

All four are in `_fixtures()`, not a side list. Closing a coverage hole means every gate that loops the
fixtures — `fd_agreement`, `identity`, `mixed`, and the in-suite `@testset` — picks them up
automatically. `_s9c_coverage_fixtures()` names them as a subset so `--gate coverage` can report on
them alone. Prefix `_s9c_`/`_S9C_`, distinct from this file's `_S8_`/`_s8_` and from the S9 lane's
`_s9_`, because the suite shares one `Main`.

| fixture | hole | what it exercises | nθ |
|---|---|---|---|
| `binomial` | 1 | Binomial, trials=8, `:indep common=true` | 3 |
| `beta_trait` | 2 | Beta, `dispersion=:trait`, phi=[7,11] → `log_phi[1]`,`log_phi[2]` | 5 |
| `nb2_trait` | 2 | NegativeBinomial, `dispersion=:trait`, r=[4,6] → `log_r[1]`,`log_r[2]` | 5 |
| `dep_term` | 3 | `GroupingTerm(mode=:dep)`, off-diagonal `Ltrue=[0.6 0; 0.3 0.5]` → 3 loading coords | 5 |

Hole 2 is about the SHAPE of the dispersion block, so `_s9c_assert_per_trait_dispersion` asserts, inside
the gate, that each per-trait fixture really owns 2 dispersion coordinates. Without it the fixture could
silently collapse to the shared case the S8 fixtures already cover and the gate would still go green.
Measured: both report `mode=trait length=2 (require 2) OK`.

## GATES

- [x] G9c.1: holes 1-3 pass BOTH fd_agreement (per-coordinate, rtol 1e-6) and identity (rtol 1e-8).
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate coverage
  EXPECT: GATE G9c.1 PASS
  EVIDENCE: **GATE G9c.1 PASS**, exit 0.

  | fixture | fd worst per-coord rel (bound 1e-6) | fd verdict | identity loglik rel | identity max theta rel (bound 1e-8) | identity |
  |---|---|---|---|---|---|
  | binomial | 1.127e-08 | PASS (89x margin) | 2.669e-16 | 2.389e-10 | PASS |
  | beta_trait | 1.532e-06 at coord 5 | PASS, coord 5 instrument-limited (below) | 1.119e-15 | 6.070e-11 | PASS |
  | nb2_trait | 2.071e-08 | PASS (48x margin) | 1.405e-16 | 2.529e-09 | PASS |
  | dep_term | 9.170e-08 | PASS (11x margin) | 1.270e-15 | 9.871e-10 | PASS |

  **`beta_trait` coord 5 — leaf-S9 flagged this "NOT CONFIRMED"; it is now measured and closed.**
  S9's G9.5 reported 1.532e-06 against the 1e-6 bound and said the h-vs-h/2 certification "was not run
  here for time". It was run here. Two independent instruments say the same thing:

  1. **Section 8.3 certificate** (`docs/design/grouped-analytic-gradient.md`), h=1e-5 vs h/2=5e-6:
     coords 1-4 certify at 1.98e-08, 2.26e-09, 1.68e-10, 3.98e-09. **Coord 5 does NOT certify: 3.864e-06**
     — the FD reference disagrees with ITSELF by 2.5x more than the gap it is being asked to adjudicate.
  2. **Richardson cross-check** (a higher-order instrument, in the gate, at the exact offending point —
     `beta_trait` coord 5, draw 7). Extrapolants `(4g(h/2)-g(h))/3` down a halving ladder from h=2e-4:
     their own worst consecutive gap is 1.350e-09, and the analytic value sits 8.456e-10 from the worst
     of them — INSIDE the better instrument's own resolution, at every rung (2.350e-10, 3.168e-10,
     7.583e-10, 5.041e-10, 8.456e-10).

  Control, same fixture, same draw, coord 4: the Richardson extrapolants agree with each other to
  ~1e-12 and the analytic gradient matches them to **4.041e-11**. So the instrument is excellent where
  the gradient is large and poor where it is small, which is the signature of the FD cancellation floor,
  not of a bad gradient. Mechanism, quantified: |g₄| = 31.37 but |g₅| = 9.75e-04, while the absolute FD
  floor `eps*|F|/h` is the same for both (|F| = 14.79, h = 1e-5 → ~3.3e-10). Relative to coord 4 that is
  ~1e-11 (observed ~1e-11); relative to coord 5 it is ~3.4e-07, so a PURE RELATIVE 1e-6 bound is
  unreachable at that coordinate by construction, whatever the gradient does.

  **What changed in response, and what did not.** `RTOL_FD` is untouched. What was added is the `atol`
  term that section 8.3's own criterion already contains and this gate had been missing:
  `abs(g_fd_h - g_fd_halfh) <= rtol*abs(g_fd_h) + atol`. The atol is MEASURED at every draw and every
  coordinate (the h-vs-h/2 absolute difference at that point), never chosen, and rtol is tried FIRST
  always. Every use is recorded and printed. Across all ten fixtures the atol term is used at exactly
  **one** (coord, draw) point in the whole suite — `beta_trait` (5, 7) — and the six inherited S8
  fixtures still pass on **rtol alone**, so S8's own gate is not weakened. Each atol-reliant point is
  then cross-checked against Richardson and FAILS the gate if it does not survive that.

- [x] G9c.2: hole 4, THE MIXED PATH. A fit whose analytic gradient fails at a SUBSET of theta still
  lands on the all-FD answer at rtol 1e-8, and really did fall back.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate mixed
  EXPECT: GATE G9c.2 PASS
  EVIDENCE: **GATE G9c.2 PASS**, exit 0, all TEN fixtures. `_grouped_analytic_gradient` is monkeypatched
  to return its documented `nothing` sentinel on every 3rd call during a REAL
  `fit_grouped_nongaussian(...; analytic_gradient=true)`. `forced > 0` is ASSERTED, not merely printed,
  because without it the gate passes vacuously — and that is not hypothetical: the S9 lane hit exactly
  that (a Julia world-age trap where the `@eval`'d patch is invisible to a call made later in the same
  function's dynamic extent). `Base.invokelatest` is load-bearing here for that reason, not defensive.

  | fixture | analytic calls | forced | loglik rel | max beta rel | verdict |
  |---|---|---|---|---|---|
  | poisson_latent | 16 | 5 | 1.045e-15 | 7.001e-10 | PASS |
  | beta_shared | 12 | 4 | 1.957e-15 | 2.654e-10 | PASS |
  | nb2_shared | 12 | 4 | 2.677e-16 | 3.449e-10 | PASS |
  | poisson_twoterm | 34 | 11 | 1.922e-13 | 3.398e-09 | PASS (2 flat coords, below) |
  | latent_plus_indep | 18 | 6 | 3.548e-16 | 1.948e-09 | PASS |
  | poisson_percoord | 40 | 13 | 9.715e-13 | 7.854e-09 | PASS (1 flat coord, below) |
  | binomial | 10 | 3 | 8.007e-16 | 6.148e-11 | PASS |
  | beta_trait | 14 | 4 | 2.611e-15 | 4.476e-11 | PASS |
  | nb2_trait | 21 | 7 | 5.619e-16 | 1.184e-10 | PASS |
  | dep_term | 17 | 5 | 0.000e+00 | 6.469e-10 | PASS |

  **A real finding, surfaced by checking the FULL theta vector.** This gate compares every parameter,
  not just `beta` and `loglik` (leaf-S9's G9.6 compared only those two). Doing so exposed two inherited
  S8 fixtures whose max theta disagreement is 2.580e-05 and 4.660e-04 — far over 1e-8. Root-caused, and
  **the mixed path is NOT implicated**:

  - `poisson_twoterm` coord 4 = `cluster.log_sd_common` = **-8.4506** (SD ~ 2.1e-4, collapsed). The plain
    analytic-vs-FD fit, with NO forced failure at all, already disagrees there by **4.409e-05** — LARGER
    than the mixed path's 2.580e-05.
  - `poisson_percoord` coord 5 = `unit.log_sd[2]` = **-9.3198** (SD ~ 8.9e-5). Plain analytic-vs-FD:
    6.266e-06.
  - All three paths report `gradient_norm` ~2.5e-5 against the default `g_tol=1e-4`, and identical
    iteration counts.

  These are variance components driven to their boundary by the fixtures' own fixed RNG seeds — flat,
  unidentified directions, pre-existing, and the same class leaf-S9 root-caused under its G9.2. They are
  adjudicated by MEASUREMENT, never waved through: `_s9c_flat_direction_check` substitutes the other
  path's value for that ONE coordinate into the reference estimate and re-evaluates the objective. On a
  flat direction the objective does not move; a genuinely wrong coordinate moves it. Measured relative
  objective change: **3.041e-15**, **1.950e-13** (poisson_twoterm coords 3 and 4) and **9.697e-13**
  (poisson_percoord coord 5), all against the 1e-8 bound. The two paths are the same point on the
  likelihood. This is the ONLY reason any coordinate is ever exempted, it is per coordinate, and it is
  re-measured every run.

- [x] G9c.3: the four new fixtures land on the REAL `origin/main` 69a69b0a0 answer, not only on the
  in-worktree `analytic_gradient=false` proxy.
  CHECK: a self-contained fixture script run in BOTH trees and diffed —
    (baseline) cd /Users/z3437171/local-scratch/lanes/GLLVM.jl-s8-baseline-69a69b0a0 && env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. <script>
    (lane)     cd /Users/z3437171/local-scratch/lanes/GLLVM.jl-s9cov-20260921     && env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. <script> --analytic
  EXPECT: every loglik and every theta coordinate within rtol 1e-8
  EVIDENCE: **PASS — all 4 fixtures, all 20 theta coordinates, all 4 logliks, plus `converged` on each.**
  Worth doing separately because GB.3 validated the `analytic_gradient=false` proxy for the OLD fixtures;
  these fixtures are new code and inherited nothing from that check. The baseline worktree is detached at
  69a69b0a0, clean, idle, and verifiably pre-S8 (`grep -c analytic_gradient src/grouped_nongaussian_fit.jl`
  = 0), so the script cannot pass the kwarg there and exercises the as-shipped path.

  Worst relative difference per fixture (bound 1e-8): binomial **2.389e-10**, beta_trait **6.070e-11**,
  nb2_trait **2.529e-09**, dep_term **9.871e-10**. Logliks agree to 2.669e-16, 1.119e-15, 1.405e-16,
  1.270e-15. The baseline's own labels confirm the fixtures reach the intended code: `beta_trait` reports
  `log_phi[1]` and `log_phi[2]`, `nb2_trait` reports `log_r[1]` and `log_r[2]`, `dep_term` reports
  `unit.loading[1..3]`.

- [x] G9c.4: the `--gate` dispatch errors on an unknown name instead of falling through.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate coverag; echo $?
  EXPECT: exit 2, and NO gate runs
  EVIDENCE: **PASS**, exit 2, printing `unknown gate "coverag". Known gates: coverage, fd_agreement,
  identity, mixed` and `Refusing to run a different gate under this name: that is a vacuous pass.`
  A malformed invocation (`--badflag x`) also exits 2.

  The pre-S9c dispatch was `ok = gate == "identity" ? gate_identity() : gate_fd_agreement()`, so ANY
  unrecognised name silently ran `fd_agreement` under the wrong name and could report PASS for a gate
  that never executed. **The S9 lane's expanded `elseif` chain kept the same `else gate_fd_agreement()`
  fall-through, so the trap is still live on `claude/lane-speed9-20260921`.** Note the three sibling gate
  files (`test_grouped_laplace_identity.jl`, `test_laplace_grad_identity.jl`,
  `test_sparse_phy_identities.jl`) already `error(...)` on an unknown gate — this file was the outlier.
  Exit 2 (usage) rather than 1 (gate failed) so a script can tell a typo from a real failure.

- [x] G9c.5: the six inherited S8 fixtures are not weakened, and the file still behaves in the suite.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate fd_agreement
         env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate identity
         env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. -e 'include("test/test_grouped_analytic_grad.jl"); include("test/test_grouped_laplace_identity.jl")'
  EXPECT: GATE GB.2 PASS, GATE GB.3 PASS, in-suite testset green, no name collision
  EVIDENCE: **PASS.** GB.2 PASS over all ten fixtures, exit 0; all six S8 fixtures pass on **rtol alone**
  (worst 1.986e-07, nb2_shared), confirming the added atol term changes nothing for them. GB.3 PASS over
  all ten, exit 0, now including the full theta check. In-suite `@testset`: **20 pass / 20 total, 7.1s**.

  Name collision checked rather than assumed: `_s9c_`/`_S9C_`, `gate_*` and `_compare_fixture` appear in
  no other test file, but `_GATES` is bound as a top-level `const` by
  `test_grouped_laplace_identity.jl:369`, so this file's dispatch table is named `_S9C_GATES`. Both files
  were then included into ONE `Main` in a single process and both testsets pass.

- [x] G9c.6: the full suite is green apart from the known `test_em_louis.jl:127` flake.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Pkg; Pkg.test()'
  EXPECT: the known baseline, plus this leaf's new tests, and no new failure
  EVIDENCE: **PASS.** Run on the merged state 462c2675f (this branch with speed78's 4b3832f76 merged
  in), 20:15 to 21:52 MDT 2026-09-21, 97 minutes, wrapped in `script -q /tmp/s9c_suite.log`, sole suite
  on the machine, nothing edited in the lane while it ran.

  | | baseline (leaf-S8 GB.6, and the s9a lane's own run earlier tonight) | this run | delta |
  |---|---|---|---|
  | passed | 16328 | **16336** | **+8** |
  | failed | 1 | 1 | 0 |
  | broken | 19 | 19 | 0 |

  The +8 is exactly this leaf's own additions: the in-suite `@testset` went from 6 fixtures x 2
  assertions to 10 x 2, and it reports `grouped analytic outer gradient vs finite differences | 20 20
  2.0s`. Nothing else moved: broken is unchanged at 19, and the failure count is unchanged at 1.

  The single failure is the KNOWN flake this arc has named all along, at the same file and the same line
  leaf-S9's G9.8 predicted: `test_em_louis.jl:127`, `SE PRIMARY GATE: EM-SEM SEs match dense-Hessian SEs
  (p=10)`, `rel = 0.0010560201922229443 <= 0.001` — marginally over its own 1e-3 bound, in a file this
  leaf never touched. It is NOT counted as green-by-assertion here; it is reported as the pre-existing
  failure it is, and it reproduced identically in the s9a lane's independent run at 20:14 tonight.

  Machine discipline: a full suite from lane `GLLVM.jl-s9a-hessian-20260921` was already running when
  this leaf was ready for one (started 18:36, finished 20:14). Only one may run at a time, so this one
  waited rather than starting beside it.

## What this leaf does NOT cover

- **No `src/` change is validated here.** This leaf is test-only, on a base (eb48adcc2) that contains
  S8 but NOT S9. It says nothing about the Nelder-Mead demotion, the D-274 Hessian, or S9's speed result.
- The full `Pkg.test()` WAS run, see G9c.6: 16336 pass / 1 fail / 19 broken in 97 min, the known
  baseline plus this leaf's +8. The one failure is the pre-existing `test_em_louis.jl:127` flake.
- **`beta_trait` coord 5 is certified to the resolution of the best instrument available, not to 1e-6.**
  Two instruments agree the analytic value is right, and the Richardson reference's own resolution there
  is 1.350e-09 absolute. A gradient error SMALLER than that would not be detected at that coordinate by
  any FD method. Detecting one would need a different instrument (exact/AD differentiation of the
  objective), which is out of scope here.
- **The mixed path is exercised by a FORCED failure**, every 3rd call, not by a naturally failing theta.
  It proves the fallback wiring and the answer; it does not prove the analytic gradient fails only where
  it should. The pre-existing `selinv missing entry` fallback that leaf-S9 saw fire mid-fit did not fire
  in these runs.
- **The two flat coordinates are exempted from rtol 1e-8 on a measured flatness argument.** They remain
  unidentified variance components in `poisson_twoterm` and `poisson_percoord` at the default
  `g_tol=1e-4`. Whether to reseed those two S8 fixtures away from the boundary is open, and is the same
  question leaf-S9's G9.2 raised from the Hessian side.
- Not touched: `src/takahashi_selinv.jl`, HSquared.jl, hsquared, PR #781, and any other lane's files.

## STOP conditions honoured

No tolerance was widened: `RTOL_FD = 1e-6` and `RTOL_IDENTITY = 1e-8` are the same literals as before.
No fixture was dropped or reseeded to force a pass. One Julia suite at a time; no other Julia process ran
during these measurements.
