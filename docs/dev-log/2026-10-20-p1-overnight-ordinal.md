# ORDINAL-LOGIT-RSZ overnight check

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Verdict: **NO_PROVEN_BUG**. No likelihood code was edited. The tolerance 0.0001 is unchanged. ORDINAL-LOGIT-RSZ is not admitted. The scoreboard was not edited.

## The failing entry

The receipt comparison is `Lambda Lambda'` traits `t01` and `t15`.

| Side | Entry (t01, t15) |
| --- | ---: |
| R | 0.243906750881311 |
| Julia receipt | 0.24403664215890872 |
| Absolute difference | 0.00012989127759771568 |

The bar is 0.0001. Log-likelihood and cutpoints in that receipt are already inside their bars. The returned-fit reconstruction error on the captured Julia point is 0. This note does not rerun `test/test_ordinal_logit_twin.jl` and does not refit either engine.

## The two objectives are the same function

The captured parameter vector in `docs/dev-log/2026-10-20-p1-ordinal-logit-rsz-capture.toml` (99 coordinates, the campaign layout: 20 intercepts, 39 lower-triangular loadings, 40 log-spacings) was unpacked with `unpack_lambda` and `_unpack_cutpoints_pertrait`. `ordinal_marginal_loglik_laplace_pertrait` at that point, inner maxiter 100 and tol 1e-9, returns -11506.928561299756. That matches the captured reported log-likelihood with absolute error 0.

R's `beta`, R's cutpoints, and the lower-triangular factor of R's stored `Lambda Lambda'` were then passed to the same function. The factor reproduces R's Gram matrix to a maximum absolute error of 4.44e-14, and its `(t01, t15)` entry is 0.24390675088131103. Julia's log-likelihood at that point is -11506.928561507913. R's reported log-likelihood is -11506.9285615101. Absolute difference: 2.1864252630621195e-9.

Julia's captured point is 2.0815787138417363e-7 higher than R's point on this one function. That gap is inside the log-likelihood bar of 1e-6. The covariance bar is a separate check, and the one pair above 0.0001 remains.

## Contract checks

Link. Both engines use the logistic CDF. `P(y = k) = F(tau_k - eta) - F(tau_{k-1} - eta)` with `eta = beta + Lambda z`. `Ordinal()` defaults to `LogitLink`, and `fit_gllvm(Y; family = Ordinal(), K = 2)` calls `fit_ordinal_gllvm_pertrait`. gllvmTMB family id 20 uses the same expression. On the captured vector the cutpoints differ from R's `extract_cutpoints` by at most 5.463890465184917e-6 (`t19`).

Sign. Intercepts differ by at most 3.5761540941869896e-6 (trait 4). `tau_1` is fixed at 0 and the free cutpoints stay positive and ordered on both sides. Those two facts sit with the shared linear-predictor sign.

Parameter count. Both fits have 99 parameters: 20 intercepts, 39 loadings, 40 free log-spacings (2 per trait, 4 categories, `tau_1 = 0`). The formula is `latent(..., d = 2, unique = FALSE)`. R records `LLt_from_Lambda_B_max_abs_diff = 0`, so the compared matrix is `Lambda_B Lambda_B'`. Julia's matrix is `fit.Λ * fit.Λ'`. In the lower-triangular factor `Λ[1, 2] = 0`, and the failing entry equals `Λ[1, 1] * Λ[15, 1]`. That product is a free entry of the rank-2 Gram matrix.

Which pair. Rebuilding `Lambda Lambda'` from the captured loadings and subtracting R's matrix leaves the same symmetric pair on top, absolute difference 0.00012955959650953752. The next pairs are `(t01, t20)` at 9.504236825683132e-05 and `(t01, t04)` at 9.35755104297109e-05. In column 1 of the shared lower-triangular factor, trait 15 is the largest loading gap: Julia -0.5252792969653081, R factor -0.5249948460916806, absolute difference 0.0002844508736274953. Trait 1's first loading differs by 4.936625652041116e-6. The covariance gap is that trait-15 gap times a first loading of about -0.4646. The median absolute column-1 loading gap is 0.00010837233612559505. The failing pair is the largest member of that shift.

Probability floor and clamp. Julia takes `log(max(P, 1e-12))`. At the captured mode the smallest raw cell probability on the 10,000 cells is 0.027272857576204457. Zero cells are below 1e-8. The largest absolute linear predictor is 4.964221835653216, under the clamp of 30. The 2.186e-9 agreement with R's log-likelihood is the same statement: the floor and the clamp are idle on this fit.

Ridge. The campaign control does not name `aghq_ridge`. The 2.186e-9 agreement is the evaluated objective at R's published point, penalty included or not as R stored it in `-opt$objective`.

## Where the two fits sit

The captured central-difference gradient (h = 1e-5) has maximum absolute value 9.367795428261161e-6. The two loadings in the failing product have gradients -4.274625098332763e-6 (`Λ[1, 1]`) and 2.819433575496077e-6 (`Λ[15, 1]`). R's maximum absolute gradient is 0.000430188546080568, with `convergence` 0 and `pdHess` true. R's `cond(H)` is 1250.13808805707. The engines stopped a short distance apart on a flat surface of one likelihood.

No ordinal source file was changed. There is no before/after likelihood number from a code edit. The receipt difference 0.00012989127759771568 stays above 0.0001.
