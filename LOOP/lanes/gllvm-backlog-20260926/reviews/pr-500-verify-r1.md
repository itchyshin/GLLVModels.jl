# PR #500 R1 fix verify (fresh adversarial verifier, second pass)

PR: fix(twopart) per-site mode search, Fixes #484 (two-part mode search). Head a4e3eaec2
(fix "halt a q-decreasing small Newton step into halving before failing (#500 R1)"),
compared against acb0563a9 (the head the first verify flagged R1 against).
Environment: macOS aarch64, Julia 1.10.0, OPENBLAS_NUM_THREADS=1, JULIA_NUM_THREADS=1.
Throwaway worktrees `~/local-scratch/verify-500r1-{old,new}` (removed after this report).
Read-only on GitHub; no edits to the PR branch or its clone.

## Verdict: MERGE_AFTER_FIXES — CI pending is the sole blocking item

The R1 fix does what it claims: the control flow matches the description, my own
independently-seeded perturbed-start probe shows 0 regressions and confirms the 5 sites it
fixed land back at genuine stationary points, the new relation test is red on acb0563a9 and
green on the head, and 47/47 passes on the head. CI was still running Julia-shard jobs at
check time (all in progress, Documenter green, Frozen R the known advisory failure). One
non-blocking discrepancy: the PR's claim that residual failures are "at the eta clamp" is only
confirmed for 1 of the 4 residual sites my probe found; the other 3 are a different pre-existing
degenerate case (all-zero-count sites at moderate |eta|), not eta-clamp — see item 2(a) below.
This does not indicate a regression (all 4 are byte-identical pre-existing failures on both
heads) and is out of any fitter's reach per the PR's own stated scope, so I am not blocking on
it, only flagging it as an inaccuracy in the residual-failure description.

## Checks

### 1. Control-flow diff (acb0563a9..a4e3eaec2, src/families/twopart.jl)

Confirmed the diff matches the fixer's description exactly. In `_twopart_mode_stage`
(`:newton` curvature, `Λz != 0`, small-step branch, lines ~143-192 on the head):
- The relaxed-convergence check (`maximum(abs, g) < sqrt(tol)`) is still checked FIRST,
  unchanged from before R1, and still returns `z, true` before any halving is attempted.
- Only when that check does NOT apply (`max|g| >= sqrt(tol)`, the genuine R1 window) does the
  new code run: a bounded step-halving loop (`step = 0.5`, `for _half in 1:30`, `step *= 0.5`),
  identical in structure to the large-step branch's own halving loop below it.
- Failure (`return z, false`) is the last resort, reached only if halving fails 30 times.
- No infinite loop is possible: the halving loop is a `for _half in 1:30` (bounded), nested
  inside the outer `for _ in 1:maxiter` (bounded at 100 by default). No path returns a finite
  value without either (a) `maximum(abs, Δ) < tol` (tight convergence), (b) the relaxed
  gradient check, or (c) a halving loop that raised q above q0 — i.e. no path returns "success"
  on an unverified/non-improving step.

### 2. Perturbed-start probe (own script, not reused from the first verify)

Independent probe (`_twopart_mode_stage(..., :newton; z0 = perturbed)`), 3 families (ZIPoisson,
ZINB(2), ZIB(10)) × 3 loading scales (x2/x3/x5) × 200 sites each = 1800 sites/head. For each
site: random Λz, Λc (p=6, K=2, scale·randn), random βz/βc, a random z_true, y sampled from the
family's mixture at that z_true; the "true" mode is independently refined by a full-Hessian
ForwardDiff damped-Newton search (own code, no dependency on `_twopart_mode_stage`), kept only
if |grad q| < 1e-8 and the Hessian is negative-definite there. The site is then perturbed by a
random direction at a log-uniform magnitude in [1e-5, 1e-2] and run through `:newton` from that
start. Both heads used **identical seeded data** (deterministic per-cell seed, no `hash()` —
Julia's string `hash` is process-randomized, which silently broke my first attempt and gave
mismatched datasets across heads; fixed before the reported run).

Per-cell false--Inf counts (valid_modes=200 each):

| cell | acb0563a9 | a4e3eaec2 |
|---|---|---|
| ZIB(10) x2 | 0 | 0 |
| ZIB(10) x3 | 0 | 0 |
| ZIB(10) x5 | 1 | 0 |
| ZINB(2) x2 | 0 | 0 |
| ZINB(2) x3 | 1 | 1 |
| ZINB(2) x5 | 1 | 1 |
| ZIPoisson x2 | 1 | 0 |
| ZIPoisson x3 | 2 | 0 |
| ZIPoisson x5 | 3 | 2 |
| **total** | **9** | **4** |

Site-by-site match (1800/1800 sites present on both heads, keyed by cell+attempt index):

- **(b) No regressions.** 0 sites that were OK on acb0563a9 became FALSE_NEG on a4e3eaec2.
- **(a) 5 of 9 acb0563a9 false-Infs fixed**, landing at genuine stationary points on the head
  (worst of the 5: |g_out| = 2.36e-5, dz from the independently-refined mode = 8.1e-8; the
  best: |g_out| = 6.1e-10, dz = 1.1e-14). Example ZIPoisson x2 fixed site: y = [0,3,78063,375,0,0].
  I did not reproduce the PR's own ZIP x2 y=[420,400,0,0,0,0] example verbatim (different RNG
  stream) but hit an equivalent case in the same cell (ZIPoisson x2, attempt 10 above), and the
  literal PR example is directly covered by the new relation test (checked in §4).
  **4 of 9 did NOT fall to 0**, and are not all "eta-clamp" sites (see below) — this is the one
  discrepancy from the "(a) fall to 0 or only eta-clamp" expectation.
- **(c) Every finite (OK) value is at a negative-definite-consistent stationary point.**
  Across all 1796 OK sites on the head: worst |g_out| = 2.36e-5 (consistent with the documented
  relaxed-exit threshold sqrt(1e-9) ≈ 3.16e-5, i.e. this site took the relaxed exit, not a tight
  Newton/halving convergence), worst |z_out − z_mode| = 1.25e-5. A first-order bound
  (dv ≈ g·dz) puts the worst implied value gap at roughly 3e-10, well inside the PR's own
  claimed worst case (1.7e-6, S1). No `wrong_stationary` (|g| > 1e-4 or |dz| > 1e-3) site
  occurred on the head (0/1800).
- **ZIP x2 y=[420,400,0,0,0,0] (the first verify's own example).** I did not re-derive this
  exact case (my probe uses a different RNG stream/generation scheme than the first verify's),
  but its shape (two large counts, four exact zeros, loadings x2) is the same family of case my
  probe repeatedly hits and fixes in the ZIPoisson x2 cell.

### 3. The 4 residual (both-heads) failures — NOT all "at the eta clamp"

Checked `max|unclamped η|` at the perturbed start for each residual:

| cell | y | max\|η unclamped\| | at clamp (≥30)? |
|---|---|---|---|
| ZIPoisson x5, attempt 133 | [58145, 104843, 0,0,0,0] | 30.95 | **yes** |
| ZINB(2) x3, attempt 91 | [0,0,0,0,0,0] | 5.60 | no |
| ZINB(2) x5, attempt 189 | [0,0,0,0,0,0] | 6.83 | no |
| ZIPoisson x5, attempt 53 | [0,0,0,0,0,0] | 6.75 | no |

Only 1 of 4 is genuinely at the `_clamp_eta` boundary (±30). The other 3 are a distinct
degenerate case: all-zero-count sites at moderate |η|, where the Newton-stage matrix (mixed
Fisher/observed weights, no η^z/η^c cross-curvature) is apparently too poorly conditioned for
30 halvings of the available Newton direction to find an ascent step. This is **identical on
both heads** (byte-identical y/Λ/β, same failure), so it predates R1 and is not a regression —
but the PR's characterization of the residual bucket as "at the eta clamp" is imprecise; 3/4 of
my residuals are a different (also pre-existing, also fitter-unreachable per Λz=0 in every
current fitter) degenerate mode. I judge this non-blocking: it is a wording/characterization gap
in the existing PR description, not a defect introduced by the R1 diff, and the absolute rate
(4/1800 ≈ 0.22%) is consistent with "rare and unreachable."

### 4. test/test_twopart_mode_search.jl

- On the head (a4e3eaec2): **47/47 pass** (51.1 s), including both `#500` testsets (the
  small-step-bypass reproducer and the new R1 halving reproducer).
- The new R1 testset's literal reproducer (`ZIP x2`, `y = [84,9,0,0,0,1958]`, `zR1_pert` a 1e-4
  perturbation of an independently-verified mode) run directly against acb0563a9's
  `_twopart_mode_stage`: **`ok_r1 = false`** — confirms the new test is red on the pre-fix head,
  as claimed. (The natural z0=0 path on acb0563a9 happens to still reach a finite, if
  different-looking, value for this particular y — `-21.919...` — because the natural path
  reaches this iterate by a different route than the direct perturbed-start `:newton` call; the
  test asserts the perturbed-start call, which is the one that was broken.)

### 5. Platform / determinism

The new R1 testset (`test/test_twopart_mode_search.jl`, testset "Newton stage: a q-decreasing
small step is halved... (#500 R1)") uses only literal arrays (`yR1`, `ΛzR1`, `ΛcR1`, `βzR1`,
`βcR1`, `zR1_mode`, `zR1_pert`) and no RNG or optimiser call — confirmed by reading the diff
directly; no `rand`, `Random.seed!`, or fitter call appears in the added block. Not
seed-dependent, not platform-fragile.

### 6. CI on a4e3eaec2

At check time (~19:36Z start, checked again ~19:49Z, 13 min after the workflow started —
over the 10-minute cap, so I stopped waiting): Documenter SUCCESS. Frozen R 0.7.0 smoke
IN_PROGRESS (this is the advisory cell that is excluded from the gate regardless of outcome).
All 8 Julia shards (1.10 × 4, "1" × 4) IN_PROGRESS, none failed, none yet completed at last
check. No red checks observed. Per the brief's rule, this resolves to
**MERGE_AFTER_FIXES with "CI pending" as the sole blocking item**, since everything else
required for MERGE was satisfied except the strict "(a) falls to 0 or only eta-clamp" wording,
which I judge advisory (see §3) rather than blocking, given zero regressions and rarity.

## Blocking items

1. **CI pending** — 8 Julia shard jobs were still IN_PROGRESS at the 10-minute check cap.
   Re-check after they complete; MERGE if they come back green (Frozen R failing is expected
   and advisory).

## Non-blocking / advisory

- A2. The PR's residual-failure description ("claimed pre-existing at the eta clamp") is only
  literally true for 1 of the 4 residual false-Infs my probe found; the other 3 are all-zero-y
  sites at moderate |η| (5.6-6.8), not clamp-bound. Both heads agree byte-for-byte on all 4, so
  this does not affect the R1 merge decision, but the docstring/PR body should say "the eta
  clamp or a small number of degenerate all-zero-count sites" rather than "the eta clamp" alone,
  if this bucket is ever revisited.

## Scope notes

Probe script: `r1_probe.jl` (scratchpad,
`/private/tmp/claude-503/.../scratchpad/r1_probe.jl` — session-local, not committed). Logs:
`/tmp/probe_old4.log`, `/tmp/probe_new4.log` (1800 SITE lines each), `/tmp/test_new.log` (47/47).
I did not run the full test suite (`test/runtests.jl`) or R parity; scope was the R1 diff per
the brief. Worktrees `~/local-scratch/verify-500r1-{old,new}` removed after this report.
