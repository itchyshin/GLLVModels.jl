# Temporal source, slice 1: likelihood, parameterisation and provenance

Date: 2026-09-27. Design spec: `docs/design/temporal-port-spec.md` (draft PR #535).
R reference: gllvmTMB `9539352f66f2db2cc26b1c393e67212a359b60c9` (P1, 0.7.1), read only.

## Model

Long data, one row per (series g, occasion t, trait j[, replicate]). States are the
ordered (series, time) pairs, sorted by series label then time (R/temporal.R:378-388).

    y ~ N(X beta, Z (K_blockdiag ⊗ Sigma_T) Z' + sigma_eps^2 I)

- `K` is block diagonal over series: `phi^|t - u|` (AR1, integer exponent) or
  `exp(-kappa |t - u|)` (OU). This is the stationary marginal of R's recursion
  (src/gllvmTMB.cpp:1798-1835): unit innovation variance at a series start,
  `a_s z_{s'} + sqrt(1 - a_s^2) z_innov` after, with `a_s = phi^gap` or
  `exp(-kappa elapsed)`.
- `phi = (1 - 1e-6) tanh(theta_time)`, `kappa = exp(theta_time)` (cpp:1795-1796).
- `Sigma_T` by mode: `diag(psi)` (indep), `L L'` with lower-triangular `p × p` `L`
  (dep), `lambda lambda'` (+ `diag(psi)` when unique) for latent rank one;
  `psi_j = exp(2 theta_diag_j)`; loadings unpack with gllvmTMB's convention
  (`unpack_lambda`, packing.jl, equal to gll_unpack_rr_loadings).
- The whole model is linear Gaussian, so TMB's Laplace objective is this exact
  marginal (spec 1.5). Julia evaluates it as a sum of dense per-series Gaussian
  terms (Cholesky per series); no Laplace, no mode search, no `_laplace_mode`.

## Parameter vector

`[b_fix (q); log_sigma_eps; theta_temporal_time; theta_temporal_rr; theta_temporal_diag]`,
with the rr block present for dep (`p(p+1)/2`) and latent (`p`) and the diag block for
indep and latent-unique (`p`). This is R's `names(opt$par)` order, measured on the P1
install: TMB orders fixed parameters by template declaration (cpp:1161-1199), so
`log_sigma_eps` is second, not last.

**Spec deviation.** Spec section 3.3 lists `log_sigma_eps` last. That is wrong about
R at P1; the implementation follows R's measured order, and every receipt test asserts
R's recorded name vector against the Julia layout before evaluating (spec section 5
already required that assertion).

Starting values follow R: OLS `beta`, `log(max(sd(residual), 1e-3))`, `theta_time = 0`,
`init_rr_theta` (0.5 diagonal, 0 lower), `theta_diag = 0` (R/fit-multi.R:4277-4288,
5785-5791, 9831-9835).

## Optimiser

Optim LBFGS with ForwardDiff gradients (Hager-Zhang line search). A boundary fit can
send a trial step to a singular covariance; the objective is then `Inf` and Hager-Zhang
asserts, so the fit reruns from the same start with cubic backtracking. The verdict
recomputes the gradient and Hessian at the returned point.

## Provenance

Ported semantics (no code copied): the pre-pass checks and messages of
R/temporal.R:1-453, the extractor of R/temporal.R:495-542 and the sign anchor of
R/temporal.R:466-483, at gllvmTMB P1. The receipts are produced by
`test/fixtures/temporal_p1/generate_temporal_p1.R` against a temporary install of P1.
