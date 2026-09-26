# Ordered-beta verification: reproduction + optimum validity (2026-09-26)

Lens: reproduce a subset of the sibling screen's ordered-beta finding
(`sibling-screen-2026-09-26.md`) exactly, then independently check whether the
flagged fits are genuinely non-stationary and whether the restart lands on a
real, valid optimum. **Default posture was to try to refute the finding.**

**Verdict: NOT refuted — CONFIRMED, and strengthened.** All three flagged seeds
reproduce exactly (same loglik, same restart gap, to 4 decimal places). The
restart points are valid, and — at two of three seeds fully, and effectively at
the third — genuinely stationary (FD gradient stable and near-zero across five
step sizes spanning four orders of magnitude). The reported "converged=true"
first-fit points are not merely steep, they sit at genuine **jump
discontinuities** in the negll(θ) surface (gradient scales as exactly 1/h as
the FD step shrinks — the signature of a finite jump in the function value, not
a smooth-but-large derivative) — a specific, falsifiable mechanism the original
report did not have. Refitting from the true DGP parameters reaches the
restart's basin, not the first fit's, in all three cases.

## Setup

Totoro, via the existing `cm-` socket (no fresh login). Own directory
`~/hsq_work/obeta-verify-reproduce-20260926/` (repo + env copied from the
screen's `~/hsq_work/gllvm-sibling-screen-20260926/`, then re-`Pkg.develop`'d
to point at the copy so nothing here reads or writes the screen's directory).
`JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`. Commit `d4da31544` (matches the
screen). Estimate stated before running: a few minutes (5 datasets, ~1s/fit,
a couple of restarts each) — actual total Totoro compute was **under 2
minutes** across all runs, well inside the 30-min/run and 60-min job caps. No
process left running (`ps aux | grep julia` empty after each step).

## (1) Reproduction

Reused `sibling_screen.jl`'s exact RNG stream, DGP (`rand_orderedbeta`,
`lowtri`), and fitter call (`fit_ordered_beta_gllvm(Y; K=2)`) for 3 of the 7
flagged seeds (2003, 2005, 2010 — chosen to span the range of gap sizes) and 2
of the 3 clean seeds (2001, 2008).

| seed | reported ll (screen) | reproduced ll | reported drestart | reproduced drestart |
|---|---|---|---|---|
| 2003 (flagged) | -196.8406 | **-196.840579** | +21.71 | **+21.714** |
| 2005 (flagged) | -3041.3466 | **-3041.346600** | +2858.5 | **+2858.537** |
| 2010 (flagged) | -372.3648 | **-372.364800** | +199.7 | **+199.748** |
| 2001 (clean) | -169.0038 | **-169.003800** | ~0 | **+6.0e-12** |
| 2008 (clean) | -192.2081 | **-192.208100** | ~0 | **+2.5e-12** |

Exact match to the digits the screen reported. Reproduction is not in
question.

## (2) Independent recompute, validity, and gradient at first-fit and restart

For each of the 3 flagged seeds: `negll(θhat)` was recomputed by calling
`GM.ordered_beta_marginal_loglik_laplace` directly (the fitter's own public
wrapper, at the fitter's own default inner-Newton settings) — **not** by
reading `fit.loglik` — and compared: `|negll(θhat) - (-fit.loglik)| = 0.0` for
all 5 seeds, so the reported log-likelihoods are honest (the bug is not in the
loglik bookkeeping). Parameters at every point checked (first fit, restart,
true-start refit) were finite, cutpoints ordered (`c1 > c0`, gap 1.7–2.4),
precision `φ` finite and positive (5.1–18.7), and loadings bounded
(`max|Λ| < 6.3`, nowhere near a degenerate blow-up) — the "converged" point is
not diverging to a boundary, it is a bounded, plausible-looking parameter
vector that simply has the wrong value.

Central-FD gradient (independent step `h=1e-6`, distinct from the screen's own
`1e-5`) at theta_hat:

| seed | first-fit gscaled | restart gscaled (best of perturbed/true start) |
|---|---|---|
| 2003 | 9.56e7 | 0.128 (see multi-h check below) |
| 2005 | 1.77e5 | 8.2e-6 |
| 2010 | 5.44e7 | 1.6e-5 |

Both clean seeds: first-fit gscaled ≈ 1.1–1.9e-5 (small, as expected).

**A multi-step-size FD sweep (h = 1e-4 … 1e-8) separates real gradient from FD
artifact.** This is the key diagnostic beyond what the screen ran:

- At **every flagged seed's first-fit theta_hat**, the estimated gradient
  scales *exactly* as `1/h` (e.g. seed 2003, coordinate 2: raw gradient 4.854e5
  at h=1e-4, 4.854e6 at h=1e-5, …, 4.854e9 at h=1e-8 — each step exactly 10×
  the last as h shrinks 10×). `raw_g × 2h` is constant (≈97.1 log-lik units)
  across all five step sizes. That is the signature of a **genuine jump
  discontinuity** in `negll(θ)` at that point, not a smooth, steep, but finite
  derivative — a true derivative would converge to a fixed value as h shrinks,
  not blow up proportionally to 1/h. Seeds 2005 and 2010 show the same
  divergent pattern (magnitude climbing each decade), though at shifting
  coordinates (12→17→17→15→4, and 15→3→4→9→10 respectively), consistent with
  *multiple* such discontinuities near those points (unsurprising given how far
  those two fits are from the true optimum — -3041 vs -183, -372 vs -173).
- At the **restart points**, the picture is different. Seeds 2005 and 2010:
  gscaled is flat and tiny across all five step sizes (8.2–14e-6 and
  1.6–2.4e-5 respectively, same coordinate throughout) — a genuinely
  stationary point, full stop. Seed 2003's restart is more interesting: at
  h=1e-4/1e-5 it also shows large, coordinate-hopping values (an FD step that
  size still clips a nearby discontinuity), but at h=1e-6, 1e-7, 1e-8 it
  locks onto a **stable, non-blowing-up** raw gradient of -0.082 (same
  coordinate, β2) — three decades of h agreeing to 4 significant figures is
  the signature of a real, small residual gradient, not FD noise or a jump. A
  follow-up polish (re-running LBFGS from this point with `g_tol` tightened
  from 1e-5 to 1e-9) moved **zero iterations** and changed the loglik by
  exactly 0.0 — Optim's own convergence check agrees this point is settled.
  So seed 2003's restart is a real optimum with a modest residual gradient
  (0.08, small relative to the first-fit's 1e7–1e9, but technically still
  above the screen's own 1e-3 scale-aware flagging bar) — a limitation of that
  bar's strictness relative to `g_tol=1e-5`, not evidence of a second instance
  of the bug.
- The two **clean seeds'** theta_hat gradients are flat and tiny (~1e-5)
  across all five step sizes with no 1/h divergence anywhere — the clean
  fits are genuinely, unambiguously converged, in sharp contrast to the
  flagged ones.

This points to a specific mechanism the original report flagged as needing
follow-up instrumentation: `_ordered_beta_mode`'s per-site inner Newton solve
(nonconvex per-site conditional log-density with point masses at 0/1) is
plausibly flipping between two local modes for at least one site as θ crosses
a threshold, producing an actual jump in the *outer* marginal `negll(θ)`
surface. The outer optimizer's own `autodiff=:finite` gradient, sampled right
at or near such a jump, would return degenerate/arbitrary values — plausibly
tripping `_fit_verdict`'s x/f-based convergence checks (the `#485`-class
"converged on a effectively-zero step" failure the screen already suspected)
without the true log-likelihood surface ever being explored. This is
consistent with, and sharpens, the screen's own hypothesis; it does not
contradict it.

## (3) Refit from the true DGP parameters

Same optimizer config the fitter itself uses (`LBFGS` + `BackTracking(order=3)`,
`g_tol=1e-5`, `iterations=500`, `autodiff=:finite`), starting exactly at the
packed true generating parameters (not perturbed):

| seed | ll from true start | vs first fit `\|Δ\|` | vs restart `\|Δ\|` | closer to |
|---|---|---|---|---|
| 2003 | -175.1269 | 21.71 | 0.0000 | **restart** |
| 2005 | -182.8099 | 2858.5 | 0.0000 | **restart** |
| 2010 | -172.6168 | 199.75 | 0.0000 | **restart** |

In all three cases the true-parameter refit lands at **exactly** the
restart's basin (to the printed precision), not anywhere near the reported
"converged" first fit. This rules out the alternative explanation that the
first fit is a legitimate second mode near the true parameters and the
restart is some unrelated, spurious high-likelihood region: the true DGP
parameters themselves refit straight into the restart's optimum.

## Distinguishing defect from genuine multimodality (repeated here, independently)

A real second local optimum requires a *small* gradient at *both* points. Here
the reported point has a gradient that is not merely large but literally
diverges as the FD step shrinks (a jump), which cannot be a stationary point
under any interpretation. The restart/true-start optimum is (at minimum, for 2
of 3 seeds fully, and for the third to within a small, step-size-stable
residual well below the first fit's magnitude by 6–9 orders of magnitude)
genuinely stationary. This is squarely the same fitter-defect class as
#480/#485, not multimodality — confirming the screen's own conclusion, with a
sharper mechanism.

## Files

- `obeta-verify-reproduce-raw/obeta_verify_reproduce_lens.jl` — main
  reproduction + validity + gradient + true-start-refit script (5 seeds).
- `obeta-verify-reproduce-raw/main_run.log` — its output.
- `obeta-verify-reproduce-raw/obeta_multi_h_check.jl` /
  `multi_h_check_output.log` — the multi-step-size FD sweep at first-fit and
  restart theta for the 3 flagged seeds (the jump-discontinuity diagnostic).
- `obeta-verify-reproduce-raw/local_check_clean.jl` /
  `multi_h_clean_output.log` — the same sweep for the 2 clean seeds (contrast
  case).
- `obeta-verify-reproduce-raw/obeta_followup_seed2003.jl` /
  `followup_seed2003_output.log`, and `obeta_followup_seed2003b.jl` /
  `followup_seed2003b_output.log` — the seed-2003 restart polish
  (`g_tol`-tightening, 0 further movement) and per-coordinate multi-h scan
  that located the stable residual at β2.

Totoro worktree left in place at `~/hsq_work/obeta-verify-reproduce-20260926/`
(own directory, not the screen's) in case of follow-up; no process left
running.
