# Review of PR #525, iSDM port spec (head 832d5983f), 2026-09-27: NOT READY for the build

R reading sound: 31 of 31 spot-checked citations confirmed (admitted laws Poisson-log and Bernoulli-cloglog only; shared Lambda_B and b_fix; no source random effect or cross-arm term; loadings unconstrained in sign, compare Lambda Lambda'; cloglog kernel tails; predict semantics; contract refusals). Per-cell block-diagonal Laplace confirmed correct.

1. HIGH. Twin table maps the wrong R files: test-isdm-public-door.R uses the legacy `isdm_family` route (which Q4 says is not twinned); the public door (isdm_sources() validation, observation formulas, source-masked design, QR rank retention, name collisions) is tested in tests/testthat/test-isdm-multisource.R (7 blocks, 20 expects) and test-isdm-source-formula.R (7 blocks, 27 expects), absent from the spec. Fix: add both (14 blocks), relabel public-door rows as legacy-shape twins via the core predicate; +1 day to 1c.
2. HIGH. Q5 (ForwardDiff through the damped mode search) contradicts src/laplace_grad.jl:1-26, which records that naive AD through the inner Newton fails and implements the one-step implicit gradient. Fix: one-step implicit gradient as the 1b default with the OBSERVED A in the dual step; carry the offset through s(z;theta) explicitly (laplace_grad.jl falls back to FD with offsets); AD-friendly `_pois_logpmf` (laplace_grad.jl:30); keep the 3-point FD check.
3. MEDIUM. Detection-row observed weight taken from `_glm_obs_weight` (curvature of Distributions' clamped binomial) while the log-density is the copied gll_dbinom_cloglog; they differ in the tails. Fix: weight = minus the second ForwardDiff derivative of `_isdm_dbinom_cloglog` itself (written Dual-safe with ifelse); pin value and weight on the 24-point eta grid.
4. LOW. Fisher steps plus observed logdet consistent with R; #514's decrement floor and cond > 1e16 -Inf are inherited, per-cell converged field in receipts is the guard.
5. LOW. test-isdm-predict.R fixture comes from set.seed(7): export once from R at P1 as a hash-pinned CSV.
6. LOW. Error-text twins must use cli-rendered text (`family` must be an R <family> object.).
7. LOW. R/fit-multi.R:1440-1445 refusal (declared source with no rows) fires before the core predicate; add to the refusal table.
8. LOW. Q8 code path known (species column from object$data[["species"]]); measure in 1c stands.

Estimates: 1b about 8 to 10 days after fix 2; 1c 4 to 5 days; 1d 2 to 3 days as a gllvmTMB PR.
