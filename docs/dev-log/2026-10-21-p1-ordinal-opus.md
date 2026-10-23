# ORDINAL-LOGIT-RSZ second read

Date: 2026-10-21. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc67542`. Read only. No Julia was started, no fit was run, nothing was committed. The tolerance 0.0001 is unchanged. ORDINAL-LOGIT-RSZ is not admitted.

Verdict: **NO_NEW_BUG**. This read agrees with the first pass (`docs/dev-log/2026-10-20-p1-overnight-ordinal.md` and `docs/dev-log/2026-10-20-p1-miss-ordinal-llt.md`). It adds one number that explains where the gap comes from: R's optimizer stop rule.

## The five suspects

Each suspect below would change the log-likelihood at R's own point, or is cancelled by the compared quantity. The first pass evaluated Julia's Laplace marginal at R's intercepts, R's cutpoints, and a factor of R's `Lambda Lambda'` and got -11506.928561507913 against R's -11506.9285615101, a gap of 2.2e-9.

Trait order. `wide_from_long` at `tools/true_parity/campaign/run_J.jl:40-46` sorts the trait strings. The names are zero padded (`t01` to `t20`), so the string sort equals R's factor level order. A row swap would also move the log-likelihood at R's point far beyond 2.2e-9, because each trait carries its own intercept and cutpoints.

Cutpoint sign. Julia uses `P(y <= c) = F(tau_c - eta)` (`src/families/ordinal.jl:6`, `_ord_prob` at `src/families/ordinal.jl:129-133`). A reversed convention would need negated intercepts and reflected cutpoints to reproduce R's likelihood. R's cutpoints plugged in unchanged reproduce it, and the receipt cutpoints agree to 5.585e-6.

`tau_1 = 0`. `_unpack_cutpoints_pertrait` sets `tau[t, 1] = 0.0` and builds later cutpoints by positive increments (`src/families/ordinal.jl:421-435`). The receipt lists two free cutpoints per trait (40 values) on both sides, the same count as R's `extract_cutpoints`.

Sign of `beta + Lambda z`. The per-trait linear predictor is `beta .+ Lambda * z` (`src/families/ordinal.jl:256`, `294`, `312`). Intercepts agree to 3.6e-6. A flipped sign on `beta` would fail the plug-in check above.

Loading rotation. The compared object is `Lambda Lambda'` on both sides: R takes `extract_Sigma(..., part = "shared", link_residual = "none")` and records `max(abs(Lambda_B Lambda_B' - S)) = 0` (`tools/true_parity/campaign/run_R.R:143-146`); Julia writes `fit.Λ * fit.Λ'` (`tools/true_parity/campaign/run_J.jl:118`). That product is unchanged by any orthogonal rotation or sign flip of the latent axes, so rotation cannot produce the gap.

Two further checks. No loading ridge is active: at the reference commit `9539352f6`, `gllvmTMBcontrol()` defaults `aghq_ridge = 2` but records `aghq_ridge_explicit <- !missing(aghq_ridge)`, and the Laplace route sets `laplace_ridge_tau` only when that flag is true (`R/gllvmTMB.R:2306`, `R/fit-multi.R:7927-7934`). The campaign call `gllvmTMBcontrol(n_init = 1L, se = FALSE)` does not name it. The observed curvature weight clamp `max(..., 0)` (`src/families/ordinal.jl:155`) is idle for the logit link, because a difference of two logistic CDFs is log-concave in `eta`. The 2.2e-9 plug-in agreement covers this too.

## Where the gap comes from

R's raw record (`raw/ordinal_R.json.gz`) reports 292 iterations, message `relative convergence (4)`, and maximum absolute gradient 4.30e-4. `.gllvmTMB_nlminb_call_args` in `R/fit-multi.R` (around line 9846) at the reference commit sets only `eval.max = 2000` and `iter.max = 1500` and merges any caller `optArgs$control`, so with the campaign control `rel.tol` keeps the nlminb default of 1e-10. Message 4 is the relative function convergence test. At `|f| = 11506.93` that test allows a stop when the predicted remaining gain is under about 1.15e-6.

Julia's point is 2.08e-7 higher on the same function, with maximum absolute gradient 9.37e-6 (central differences, `docs/dev-log/2026-10-20-p1-ordinal-logit-rsz-capture.md`). That gain sits inside R's stop window, so R stopping there is the expected behaviour of its rule.

A quadratic estimate (AGENT-INFERRED, no Hessian was read): a remaining gain `0.5 g' H^-1 g = 2.08e-7` with `|g|` about 4.3e-4 implies a curvature near 0.44 along the step direction, and a step length near 1e-3 in parameter units. The measured shift is smaller than that bound: the trait 15 first loading moves 2.84e-4, and `0.4646 * 2.84e-4 = 1.32e-4` reproduces the failing `(t01, t15)` entry. The column 1 loading gaps share one direction (median 1.08e-4), which is what an early stop along one flat direction looks like.

## What this does and does not show

It shows no defect in Julia's ordinal likelihood, data mapping, or compared quantity. It does not prove that Julia's point is the exact optimum; it shows that Julia's point is higher on R's own objective and has a gradient 46 times smaller. Confirming the explanation needs one R refit from R's returned point with a tighter `rel.tol` (for example `optArgs = list(control = list(rel.tol = 1e-14))`, which that helper merges) and a re-read of the `(t01, t15)` entry. That refit was not run here. Moving the bar is a maintainer decision and was not touched.
