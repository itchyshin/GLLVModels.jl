# P1 Newton polish: step-acceptance rule

Date: 2026-11-10

Status: AWAITING-SIGNATURE

Signed by: (empty; only the maintainer fills this, in a later separate act)

## What changes

The N9 Newton polish loop (Julia `tools/true_parity/campaign/polish_J.jl`, R
`tools/true_parity/campaign/run_R.R`) used to reject any step with
`fn > f_current`, where f is the negative log-likelihood the loop minimises.
The new rule accepts a step when

    fn <= f_current + 1e-12 * max(1, |f_current|)

A step above that bound is still rejected and recorded. An accepted step with
`fn > f_current` is flagged `accepted_uphill = true` in its point record. The
same rule applies to both engines.

## What does not change

The gradient bar (max-abs strictly below 1e-5 with a positive definite Hessian
at the same point), the step cap of 3, the finite-difference steps, every
comparison tolerance (logLik 1e-6, beta 1e-4, LLt 1e-4, thresholds 1e-4,
predict_* 1e-6, eta 1e-6) and every test assertion.

## This is a new threshold

`1e-12` relative is about 4500 machine epsilons. It is not a rounding-unit
quantity. It was chosen from two observed rejected steps (Julia iSDM, rise 5e-11
on |f| 2.4e4; R fungi, rise 1.9e-9 on |f| 5.8e3, a size set by the Laplace
inner-solve noise) and was first checked on the same data it was set from. It
has not been validated on other cells. A pure rounding bound (about 100
epsilons) would pass the Julia iSDM rise but leave the R fungi step rejected.

Largest total effect: three accepted steps can lower the final logLik below
the pre-step logLik by at most 3e-12 relative, about 1.7e-8 at |f| = 5.8e3,
well inside the logLik tolerance of 1e-6.

## Text changed with the rule

`POLISH_PROCEDURE` in `write_receipts.py` used to say that a step that raises
the objective is not taken. That sentence would be false under the new rule,
so it is changed in the same commit, with the matching header comments and
rejection reason strings in `polish_J.jl` and `run_R.R`.

## Approval record

"D1 rerun: yes" (decision card 2026-11-10 10:02 MT) approves one rerun of the
ordinal and iSDM cells under this rule on a local draft branch. It is not a
signature on the rule. Fungi is not rerun.

## Signature

Pending. Until signed, receipts produced under this rule are measurements, not
accepted evidence.
