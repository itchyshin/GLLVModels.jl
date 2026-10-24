# DRAFT: convergence-parity rule for C3 campaign rows (W1-11)

**Status: DRAFT. Needs the maintainer's signature (N9). Not a ruling, not in `GATES.md`, changes no tolerance,
no receipt and no case-map row.** Nothing in this file is read by `tools/true_parity_check.mjs` or
`tools/true_parity/campaign/write_receipts.py`.

## Why this exists

Two C3 campaign rows, `ORDINAL-LOGIT-RSZ` and `ISDM-HEADLINE-RSZ`, do not bind. In the ordinal row the loadings
product `LLt` differs by 1.299e-4 against a tolerance of 1e-4, while the log-likelihoods agree to 2.1e-7 (tolerance
1e-6). On the same objective, R's `nlminb` stops early: it reports "relative convergence (4)" with a gradient
max-abs of 4.3e-4, and Julia's L-BFGS stops at 7.8e-6 (a 55-fold gap; the ISDM row's gap is reported by the
maintainer as larger, up to about 2000-fold, and was not re-measured here). Both engines call themselves converged. A flat
likelihood ridge turns "converged" into a few units in the fourth decimal of `LLt`. The row then measures how
early each optimiser stopped, not whether the two engines fit the same model.

## What was measured (scratch, ordinal cell only, local Mac, not a receipt)

Same data bytes as the committed row (sha256 `df6a45d4...c0c1`), P1 gllvmTMB 0.7.1 (9539352f6), Julia at origin/main
8f33addfc. Harness: `run_R.R` and `run_J.jl` with `CAMPAIGN_POLISH=1`, which write separate
`<cell>_R_polish.json` and `<cell>_J_polish.toml` files and leave the files `write_receipts.py` reads untouched.

| quantity | R before | R after | Julia before | Julia after |
|---|---|---|---|---|
| gradient max-abs | 4.314e-4 | 4.709e-7 | 7.822e-6 | 2.365e-6 |
| logLik | -11506.9285615104 | -11506.9285612997 | -11506.928561299732 | -11506.92856129974 |

After polish the two engines differ by 4.0e-11 in logLik and 1.85e-7 in `LLt` (max-abs), against the row's existing
tolerances 1e-6 and 1e-4. Before polish the `LLt` difference was 1.29e-4. R's own `LLt` moved 1.29e-4 under its
Newton step; Julia's moved 2.2e-6. The row's failure is R's early stop, not a model or likelihood difference.

Two notes on the measurement. Restarting `nlminb` from its own optimum with `rel.tol = 1e-13` did nothing
(1 iteration, "singular convergence (7)", gradient unchanged): the stall is at the noise floor of the Laplace
gradient, which a tighter tolerance cannot cross. The Newton step does cross it. And Julia's gradient after the step
(2.4e-6) is limited by the central-difference gradient, not by the point; Julia's fitter has no analytic gradient for
this family.

## Proposed rule (for N9)

A C3 campaign comparison counts only when BOTH engines satisfy a convergence bound at the point whose outputs are
compared. The bound is checked on the engine's own objective, in the engine's own coordinates, and is recorded in
the receipt beside the existing legs.

1. Each engine records the gradient max-abs of its own objective at the returned point (`max_abs_gradient` in the
   R raw output already exists; Julia's raw output gains the same field).
2. If an engine's gradient max-abs is above **1e-5**, that engine takes one Newton step with its own gradient and
   Hessian (R: TMB gradient, and the Hessian TMB's `sdreport` reports; Julia: `tools/true_parity/campaign/polish_J.jl`),
   and the comparison uses the stepped point. Both the before and the after record are kept.
3. The comparison counts when the gradient max-abs after the optional step is at or below 1e-5 for both engines.
   If an engine cannot get there (for example a finite-difference gradient whose noise floor is above the bound),
   the alternative test is that the Newton step's size, max-abs of `H^-1 g`, is at or below 1e-5 at that point. That
   quantity is the first-order distance to the optimum in the same coordinates as `LLt` and `beta`, so it bounds
   what an early stop can do to the compared outputs.
4. A row that still fails after step 3 fails on the substance of the comparison, with the tolerance unchanged.

Open choices the maintainer must make: the 1e-5 value (the measurement above would also pass at 1e-4 for Julia and
fail it for R's default, so the value decides which rows are re-opened, and is not derived from data here); whether
step 3's alternative is allowed; and whether R's and Julia's gradients, which are in different parameter bases, may
share one number at all. The Newton-step size in step 3 is basis-free to first order and is the safer common
quantity.

## Why this changes no tolerance

The tolerances (`logLik` 1e-6, `LLt` 1e-4, and every other quantity on every row) are not edited, loosened or
re-derived. The rule adds a precondition on the inputs to the comparison: that each engine is at its optimum to a
stated numerical accuracy before its outputs are compared. A comparison between two points that are each some
distance from the optimum is not a comparison of the two models, and widening the tolerance to absorb that distance
would be the change to a tolerance; this rule does the opposite and moves the points. A row that is within tolerance
only because of a loose stopping rule is not made to pass by it, and a row that fails with both engines converged is
not rescued by it.

## Not covered

One cell, one seed, ordinal only. The ISDM row was not run. The stepped point is a diagnostic polish, not a refit: it is
not what `gllvmTMB()` returns to a user, so a passing polished comparison says the engines agree at the optimum, not
that the unpolished default fits agree to the row's tolerance. Whether the row's receipt should cite the default or
the polished point is part of what N9 decides.
