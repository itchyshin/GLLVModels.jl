# Worse optimum in Julia bootstrap refits: characterisation, cause, fix (2026-11-01)

Read-only on the repo (driver and tests read from `origin/main` a2fc5ce03; Totoro copy verified byte-identical by sha256). No commits, pushes or PRs; #990 untouched; no R repo edited. Runs on Totoro in `~/claude-bsig-opt/` only (copies of the earlier setups), JULIA_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, at most 4 of my processes. Estimate I held: 30-40 min total, under 3 h; actual about 35 min wall. All my processes have exited. Totoro load average was 22-30 from other users, so all millisecond figures are inflated by an unknown factor; compare them with each other, not with the "1-10 ms" nominal.

Files (all under `/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-GLLVM-jl/eb56923c-569d-427b-bdc5-d1f448428f52/scratchpad/investigate-optimum/`): `OPTIMUM.md`, `bootstrap_sigma_r_refit_opts.patch` (proposed, never applied), `scripts/` (`opt_batch.jl`, `r_refit.R`, `analyse.py`, `tl_opt.jl`, `tl_opt2.jl`, `tl_analyse.py`, `tl2_analyse.py`, `find_repro.jl`, `repro_check.jl`), `out/` (per-replicate tables), `logs/analyse_*.txt` (full analysis output).

## Premise check (the first thing to know)

The "about 1% worse optimum" is NOT in `fit_gaussian_gllvm`. It comes from the TwoLevel route (`fit_twolevel_gaussian`, fixture B, K_B=2, K_W=1), a different fitter. On the GllvmFit fixtures (rr-only, has_diag K=1, has_diag K=2) Julia never lands at a converged-but-worse optimum; the only deficit is the already-known 500-iteration cap on fixture c (0.4% of replicates, logLik short by at most 2e-3). Both are reported below.

## Q1. Characterisation

Method: Julia simulates the driver batch (`MersenneTwister(seed)`, `n_boot = 999`, seeds 101 and 102, fixtures a/b/c from `~/claude-bsig-vs-r/gllvmfit/data`), the same Y matrices are refit by stock `fit_gaussian_gllvm` and by R `gllvmTMB` (`se = FALSE`, pinned oracle library). logLik definitions agree (point fits equal to 1e-8; the logLik gap R minus Julia is -1e-7 at worst when both converge). 6 batches x 999 = 5994 replicates.

### GllvmFit fixtures (fit_gaussian_gllvm), R logLik minus stock Julia logLik, same data

| fixture | seeds | R > Julia by >1e-3 | R > Julia by >1e-4 | stock non-converged (R: 0 everywhere) | Julia better than R by >1e-4 |
|---|---|---|---|---|---|
| a rr-only K=1 | 101, 102 | 0 / 1998 | 0 | 0 | 0 |
| b has_diag K=1 | 101, 102 | 0 / 1998 | 0 | 0 | 0 |
| c has_diag K=2 | 101 | 2 / 999 | 5 / 999 | 5 | 0 |
| c has_diag K=2 | 102 | 2 / 999 | 3 / 999 | 3 | 0 |

- Every fixture-c deficit is a replicate where stock stopped at the 500-iteration cap (`converged = false`); all 8 have a deficit of 1e-4 to 1.8e-3 logLik, none above 2e-3. There is no replicate where Julia converged at a worse optimum. Julia never beats R by more than 1e-7.
- Best of 10 Julia starts (stock, warm, tight, tight12, iter2000, 5 random) reaches R's logLik (within 1e-3) in all 8; the logLik surface here is not multimodal near the PPCA start. Random starts are much worse than the PPCA start: of 999 refits 6-21 land more than 1e-3 below R on fixture c, some by 10-340 logLik (real local optima, plus cap failures), so a naive random multi-start is not the right device.
- Bounds distortion (fixture c, 42 entries: 36 Sigma_B plus 6 communality_B; percentile 2.5/97.5, SE = SD of draws), stock survivors against R's bounds on the same data: seed 101 up to 3.2e-3 (lower), 2.2e-3 (upper), SE up to 1.32% off; seed 102 up to 1.2e-3, 1.4e-3, SE 0.86%. When the failed replicates are rescued, the same comparison gives 6e-6 to 1e-5 and SE 0.00%. Fixtures a and b: 1e-5 or less either way.

### TwoLevel fixture B (fit_twolevel_gaussian), where the ~1% lives

Uses the saved batch 2001 from the earlier investigation (R-simulated and Julia-simulated, both refit by R and Julia; 999 replicates each; replicates where both converged: 992 and 978).

| batch | R > Julia by >1e-3 | by >1e-2 | by >0.1 | max deficit | Julia better than R by >1e-3 |
|---|---|---|---|---|---|
| R-simulated | 9 / 992 (0.91%) | 7 | 1 | 0.34 | 0 |
| Julia-simulated | 6 / 978 (0.61%) | 3 | 1 | 0.87 | 0 |

- These are genuine converged local optima: Julia reports `converged = true` in all of them, 125-331 iterations, gradient below 1e-8 (g_tol), so the driver cannot flag them. Typical deficit 0.003-0.04; two outliers (0.34, 0.87). `iterations = 3000` and a second LBFGS pass from the end point change nothing (deficits identical to 3 digits).
- Bounds/SE distortion: stock draws against the best logLik found over 10 starts differ by at most 1.6e-3 in bounds and 0.23% in SE (50 entries of Sigma_B and Sigma_W). Dropping the affected replicates instead changes bounds by up to 1.7e-2 and SE by 1.0%, i.e. the harm would come from filtering, not from keeping them. Against R's bounds the stock gap is 3.8e-3 / 6.5e-3 and SE 0.68% (partly R's own differences).

## Q2. Cause

GllvmFit, fixture c (CONFIRMED, consistent with the earlier investigation): a flat ridge. sigma2_B, sigma2_W and sigma_eps^2 enter only as a sum, Heywood draws run along it, and LBFGS needs 600-1800 iterations (g_tol 1e-6) against a 500 cap. Not the start: warm start from the original fit's parameters fails 6 of 999 (stock 5); 5 random perturbed starts fail (non-converged) 8-23 each. It is the convergence criterion plus the cap, not local optima: with `iterations = 2000` alone the logLik reaches R's (deficit 0 to 1e-6) but `converged` stays false in 3-5 replicates (the gradient test at 1e-6 is still not met), so that is not enough for the failure count.

TwoLevel: local optima of the likelihood, not the cap and not the start scale. Evidence: (i) `polish` (restart from the end point) and `iter3000` leave every deficit unchanged; (ii) different starts rescue different replicates: sigma^2 init 0.1 rescues some, 2.0 others, perturbed (sd 0.3) starts others, warm start from the original fit others; (iii) no single start rescues all, and best-of-10 starts leaves 2/992 (max 0.028) and 1/978 (0.003) still below R; (iv) sigma^2 init 0.1 alone is worse (18 and 15 deficits >1e-3). A closed-form ANOVA start (eigen-decomposition of the between and within covariances) does not help: better than stock in 2-3 replicates, worse in 6-7. R finds the better optimum with its own start, which I did not inspect. I did not identify what separates the two basins (a boundary solution is the obvious candidate; not checked).

## Q3. Candidate fixes, measured

Cost is mean seconds per replicate under Totoro load; "rescued" counts replicates below R by >1e-3 (GllvmFit: of the 4 with that deficit; non-converged: of the 8).

GllvmFit fixture c (1998 refits; 8 non-converged, 4 with deficit >1e-3 under stock):

| fix | non-converged left | deficit >1e-3 left | max deficit | mean ms/rep (stock 77) | bound change vs stock-rescued set |
|---|---|---|---|---|---|
| stock | 8 | 4 | 1.8e-3 | 77 | reference |
| warm start from original fit | 9 | 1 | 1.1e-3 | 80 | none beyond 1e-5 |
| 2 starts (stock + warm), keep best ll | 8 | 1 | 1.1e-3 | 158 | no |
| 3 starts (stock + 2 random), keep best | n/a (best ll) | 0-1 | 1.6e-3 | 295 | no |
| `iterations = 2000` only | 8 | 0 | 1e-6 | 83 | no (but flags still false) |
| `iterations = 2000, f_tol = 1e-10` | 0 | 2 | 1.3e-3 | 51 | up to 4e-5 |
| **`iterations = 2000, f_tol = 1e-12`** | **0** | **0** | **1e-4** | **60** | **6e-6 to 1.1e-5** |

`iterations = 2000, f_tol = 1e-12` fixes all 8 replicates, stops by the relative objective test (like R), and is cheaper than stock on all three fixtures (a 1.0 vs 1.4 ms; b 18 vs 25 ms; c 60 vs 77 ms), because the stock gradient test at 1e-6 keeps iterating on the flat ridge. Iterations: median 53, max 1755 (so the 2000 cap is not reached in 1998 refits, but the margin is 12%). `f_tol = 1e-10` (the earlier investigation's choice) leaves deficits up to 1.3e-3 and 2 deficits >1e-3; `f_tol = 1e-12` is the better setting. On a and b the same options change nothing material (bounds 1e-5; a and b were not failing).

Caution on `f_tol`: the `fit_gaussian_gllvm` docstring records that a relative `f_tol = 1e-10` stopped iterative phylo-unique and covariate fits with gradients near 3e-3 and Wald SEs up to 0.25% off. This is why the proposed patch restricts the options to the bootstrap's built-in refit for `has_diag` or `K_W > 0`, leaves the fitter's defaults alone, and leaves a user `_refit` untouched. The fixtures use a six-intercept `X`, so the covariate-profile path was exercised (a, b, c all fine). Only 2 seeds and one DGP per fixture.

TwoLevel fixture B (two batches; deficits >1e-3 under stock: 9 and 6):

| fix | deficits >1e-3 left (R-sim / J-sim) | cost x stock | max bound change vs stock |
|---|---|---|---|
| warm start from original fit (single start) | 10 / 6 | 1.0 | n/a |
| 2 starts (stock + warm) | 6 / 4 | 2.0 | well under 2e-3 |
| 2 starts (stock + sigma^2 0.1) | 5 / 4 | 2.1 | |
| 3 starts (stock + sigma^2 0.1 + sigma^2 2) | 2 / 2 | 3.1 | |
| 3 starts (stock + warm + 1 random) | 4 / 4 | 3.1 | |
| 5 starts (stock + 4 random) | 5 / 2 | 5.2 | |
| 10 starts, best of all | 2 / 1 | ~10 | max 1.6e-3 in bounds, 0.23% in SE |
| closed-form ANOVA start (single) | 12 / 11 | 0.9 | |
| stock + ANOVA | 6 / 4 | 1.9 | |

No cheap TwoLevel fix removes the deficit: even 10 starts leaves 1-2 replicates, and 3 starts cost 3x to move bounds by under 2e-3 and SE by under 0.25%. Also the public `fit_twolevel_gaussian` has no start-value keyword (`θ_init`), so warm and perturbed starts need a fitter change, not just a driver change (my experiments used a verbatim copy of the fitter body with an explicit start; `stockcopy` equals stock exactly).

## Recommendation (cheapest that removes the deficit)

1. GllvmFit: adopt `iterations = 2000, f_tol = 1e-12` for the built-in refit when `has_diag || K_W > 0` (patch below). Removes all 8 failures and the 1e-3 deficits, lowers runtime, moves c's bounds by up to 3.2e-3 (back to R's, residual 1e-5), SE by up to 1.3%. Single-start, no change to the driver's draw logic.
2. TwoLevel: no code change. The effect on percentile bounds and SEs is below 0.25% (SE) and 1.6e-3 (bounds) even when fully corrected, which is below the Monte Carlo noise of a 999-draw bound; the cheapest partial fix (3 starts, 3x cost) is not worth it unless the maintainer wants it, and it needs a `θ_init`/restart keyword in `fit_twolevel_gaussian` first. State the ~0.6-0.9% worse-local-optimum rate in the docstring (the driver cannot detect it because the fit reports `converged = true`).
3. Do not filter the affected replicates: that moves bounds by more than keeping them (1.7e-2 against 1.6e-3).

### Patch (never applied): `bootstrap_sigma_r_refit_opts.patch`

`patch -p1 --dry-run` applies cleanly to `origin/main`. Source change (the full test and data hunks are in the file):

```diff
--- a/src/bootstrap_sigma_r.jl
+++ b/src/bootstrap_sigma_r.jl
@@ -162,9 +162,18 @@
         end
         return Ys
     end
+    # Built-in refit only (a user `_refit` keeps its own controls). A has_diag / K_W fit has a flat
+    # ridge ... (comment as in the patch file)
+    refit_opts = (_refit === nothing && (m.has_diag || m.K_W > 0)) ?
+        (iterations = 2000, f_tol = 1e-12) : NamedTuple()
     refit = Y -> refit_fn(Y; K = m.K, K_W = m.K_W, has_diag = m.has_diag,
                           K_phy = m.K_phy, has_phy_unique = m.has_phy_unique,
-                          Σ_phy = nothing, X = Xm)
+                          Σ_phy = nothing, X = Xm, refit_opts...)
     return simulate, refit
 end
```

### Tests that pin it (in the patch; verified on Totoro)

Testset 18 in `test/test_bootstrap_sigma_r.jl` plus `test/data/bsig_c_s101_b65.csv` (replicate 65 of bootstrap seed 101 on fixture c, 200 x 6, 15.8 KB; a deterministic bad optimum, no RNG needed):
1. Stock `fit_gaussian_gllvm` on that Y: `converged == false` and `n_iter == 500` (pins the cause).
2. The driver's built-in refit (via `_bsig_setup`) on the same Y: `converged`, `logLik >= stock`, and `logLik ≈ -1422.97095347239` (R gllvmTMB on the same data, convergence 0) to 1e-4. Measured: -1422.9709544, 1754 iterations.
3. A user-supplied `_refit` receives no `iterations`/`f_tol` (seam unchanged).

Verified: with the patch, the whole file passes (217/217, 1 m 29 s on Totoro, including the existing 16 testsets). Against the unpatched source the same testset fails at tests 2 (`converged`, logLik) and passes test 1, so it is a true red-then-green pin. The existing end-to-end seed (fixture c, `seed = 101, n_boot = 999`: `n_failed` 5 before, 0 after) is the 999-replicate acceptance check; I measured the equivalent per-replicate (0 non-converged in 1998) but did not rerun the patched driver end to end.

TwoLevel pin, if ever wanted: replicate 196 of the Julia-simulated batch 2001 (deficit 0.868) or 252 of the R-simulated one (0.34); data are in `out/jsim_Y_2001.csv` row 196 and `out/rsim_Y_2001.csv` row 252 on Totoro `~/claude-bsig-opt/out/`. Not written as a test because there is no fix to pin.

## Not covered / caveats

- One DGP per fixture, p = 6, n = 200, Gaussian; GllvmFit seeds 101 and 102 only; TwoLevel one batch per simulator (batch 2001), p = 5, K_B=2, K_W=1. 8 GllvmFit events and 15 TwoLevel events: the rates (0.4%, 0.6-0.9%) are rough.
- GllvmFit R refits were run on Julia-simulated data only; R-simulated batches were not used for fixtures a/b/c.
- `f_tol = 1e-12` was tested only inside the bootstrap's `has_diag` refit, not on the fitter generally (docstring warns against it for covariate/phylo fits).
- Why TwoLevel basins differ (boundary versus interior) and what start R uses were not established.
- Timings are under heavy shared load; relative order is reliable, absolute values are not.
