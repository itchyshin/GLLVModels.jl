# Where does wall time go in `bootstrap_Sigma_r`? (origin/main 79857ff87)

Read-only profiling. No tracked file edited, nothing pushed, no PR. Scratch copy of `git archive origin/main` on Totoro (`~/prof_scratch/pkg*`; the only side effect in the repo was `git fetch origin`). Patched variants live only in scratch: `profile_ss.jl`, `twolevel_ss.jl`, `pmv_nohess.jl`, `pmv_cholre.jl` in this directory.

Setup: one Julia thread, `OPENBLAS_NUM_THREADS=1`, `taskset -c 200,201`, Julia 1.12.6. Totoro load average was about 296 on 384 cores, so absolute seconds are inflated 2-4x. The same code varied by 20-40% between identical repeats, and even CPU-time clocks were not stable. **Trust the percentages (sampling profiler) and the bit-identical / not-identical checks. Treat the speedup ratios as order of magnitude.**

Synthetic cases (my own seeds, one dataset per class, so generalisation is untested):

| Case | Setup | Parameters | LBFGS iterations per replicate |
|---|---|---|---|
| GllvmFit | p=30, n=200, K=2, has_diag (so 2p+rr = ~119 free parameters), `y` passed | ~119 | mean 180, max 307; 20/20 converged |
| TwoLevelFit | p=10, 80 individuals x 5 obs, K_B=K_W=1 | 41 | mean 80.7, max 90; 20/20 converged |
| PrecisionMultivariateFit | 30 tips, 2 obs per tip, d=5, rank 1, `:explicitunique` | 20 | mean 161, max 400; 19/20 converged (one replicate hit the 400-iteration cap and is a lost replicate in every variant) |

## 1. Component breakdown, n_boot = 20

Driver pieces timed separately (`_bsig_build` simulate batch, `refit`, `_bsig_entries` extraction), then the full `bootstrap_Sigma_r` call.

| Case | simulate batch | refit | extraction | per replicate (full driver, loaded) | alloc per refit | GC share |
|---|---|---|---|---|---|---|
| GllvmFit | 0.0007 s (0.0%) | 73.7 s (**100.0%**) | 0.004 s (0.01%) | 2.59 s (breakdown loop 3.69 s; earlier unloaded figure 1.24 s) | **13.8 GB** (276.5 GB / 20) | 45-47% of wall; 49% of profile samples in GC frames |
| TwoLevelFit | 0.0012 s (0.0%) | 6.31 s (**100.0%**) | 0.0008 s (0.01%) | 0.27-0.70 s | 0.78 GB | 12-20% of wall; 31% of samples |
| PMV | 0.0005 s (0.0%) | 29.1 s (**100.0%**) | 0.0007 s (0.00%) | 0.95-1.46 s | 2.07 GB | 14% of wall; 29% of samples |

Simulation, extraction (`_bsig_entries`, `_cov2cor`, dict building), result aggregation and the driver's own overhead are each below 0.02%. Nothing to gain there, including "fewer allocations in extraction" (total extraction allocation is 0.05-1.4 MB over 20 replicates).

PPCA init is also below the profiler's resolution for GllvmFit (p=30 eigendecomposition, once per refit), so reusing PPCA across replicates would gain under 1%.

## 2. Profile (inclusive samples, GLLVModels source lines)

**GllvmFit** (22.7k samples). 99.3% in `gaussian_profile_nll` called by the ForwardDiff gradient inside Optim LBFGS. 97.1% of all time is five lines of `src/profile.jl`, the non-phy pass-through branch:

| profile.jl line | statement | share |
|---|---|---|
| 261 | `Ainv_r = Dinv_r .- DinvL * z` | 45.0% |
| 262 | `quad = sum(resid .* Ainv_r)` | 22.3% |
| 258 | `Dinv_r = d_inv .* resid` | 18.4% |
| 260 | `z = cA \ LtDr` | 6.6% |
| 259 | `LtDr = L_U' * Dinv_r` | 4.8% |

Cause: `d_inv` is a vector of Dual numbers (about 12 partials), so every evaluation builds several p x n = 30 x 200 Dual matrices from the Float64 data `y`, about 13.8 GB of garbage per refit. The data enter only through `y*y'`.

**TwoLevelFit** (6.6k samples). 98.4% in `_twolevel_loglik`, 94.9% under the ForwardDiff gradient. Line 83 (`quad_centered = sum(abs2, cW.L \ Yic)`, a Dual triangular solve repeated for each of the 80 individuals) is **55.3%**; line 92 (`quad_mean = ni * dot(mi, cMean \ mi)`) is 15.2%. Per-individual `Matrix{T}` copies and centring (lines 78-80, 86) add about 20%. Function-level: `LinearAlgebra.\` is 51.7% inclusive.

**PMV** (19.3k samples). 98.0% in `_precision_multivariate_nll`; 96.1% in `_multivariate_phylo_precision_evaluate`, of which `precision_multivariate.jl:147` (`joint_chol = cholesky(Symmetric(joint_precision))`, sparse CHOLMOD, a 354 x 354 system) is **64.2%**. 85.5% of time is the central finite-difference gradient `_pmv_fd_gradient!` (2 x 20 = 40 objective evaluations per gradient); the post-fit finite-difference Hessian diagnostics `_pmv_hessian_diagnostics` (about 800 evaluations) is **10.0%**. One objective evaluation is 0.124 ms (minimum of 300).

## 3. Top three costs and tested speedups

### Cost 1 (GllvmFit): Dual-matrix temporaries in `gaussian_profile_nll`, 97% of time

Idea: use the sufficient statistic `S = y y'` (Float64, p x p). For q = 0 and no phylo block, `quad = sum_t d_inv[t] S[t,t] - tr(A^-1 M' S M)` with `M = D^-1 L`. The data are never multiplied by Duals.

Measured in the scratch copy (profile.jl non-phy branch, q == 0 only; `S` recomputed on every call, so caching it would be faster still), same 20 simulated datasets, same optimiser settings (`iterations = 2000, f_tol = 1e-12`):

| Run pair | stock | patched | ratio |
|---|---|---|---|
| wall clock | 70.2 s (3.51 s/rep) | 3.08 s (0.154 s/rep) | 22.8x |
| CPU clock, later | 93.8 s (4.69 s/rep) | 2.89 s (0.144 s/rep) | 32.5x |

Draw changes (**not bit-identical, 0/20 identical**): max abs difference in any Sigma entry 5.2e-8, max relative 1.8e-6 (relative floor 1e-8); max |logLik diff| 1.1e-10; iteration counts identical (0/20 changed); 20/20 converged both ways. The same two stock runs repeated are bit-identical (0 difference), so the changes are due to the patch, not noise. Allocation after the patch was not measured.

Not tested, and the risk that matters most here: the Heywood / flat-ridge replicates (fixture c) that motivated `f_tol = 1e-12`. The new form subtracts aggregated quantities, so conditioning near sigma_eps -> 0 could differ from the old elementwise form. Needs the fixture-c grid (about 2000 refits) before anyone adopts it.

### Cost 2 (TwoLevelFit): per-individual Dual solves, 70% of time

Idea: precompute once outside AD the pooled within-cross-product `Sc = sum_i Yic Yic'` and, per distinct group size, `sum_i n_i m_i m_i'` plus the count; then the log-likelihood needs one p x p Dual Cholesky solve per group size instead of one per individual. Uses dense Cholesky solves, so it keeps the no-subtractive-Woodbury rule stated in `_twolevel_loglik`.

| Run pair | stock | patched | ratio |
|---|---|---|---|
| wall clock | 7.77 s (0.389 s/rep) | 0.83 s (0.042 s/rep) | 9.4x |
| CPU clock, later | 13.86 s | 0.88 s | 15.7x |

Draw changes: max abs difference 1.3e-7, max relative 2.8e-5; max |logLik diff| 4.6e-12; iteration counts changed in 16/20 replicates (-5 to +5); no convergence flag changed. Drift is rounding-level but real: the optimiser stops at slightly different points.

### Cost 3 (PMV): the objective itself (sparse Cholesky inside a 40-evaluation finite-difference gradient)

Two zero-draw-change candidates tested, both **bit-identical to stock (20/20 draws, max difference 0, same iterations, same convergence flags)**:

1. Skip `_pmv_hessian_diagnostics` in bootstrap refits. The profile gives 10.0% of PMV time. The bootstrap uses only `converged`, which does not depend on the Hessian (`converged = Optim.converged && valid && gradient_norm <= g_tol`). Wall and CPU timings could not resolve a 10% gain: identical stock repeats differed by 22%, and the patched runs came out 0.75-1.0x of stock, i.e. inside noise and in the wrong direction. So the 10% rests on the profile share, not on a clean timing.
2. Reuse the CHOLMOD symbolic factorisation (`cholesky!(F, A)` with a pattern-equality guard). The microbenchmark shows no gain: objective evaluation minimum 0.124 ms stock vs 0.124 ms patched (a second patched run read 0.196 ms, worse). **Rejected**: the 64% on that line is numeric factorisation of a 354 x 354 system, not the symbolic analysis.

A real PMV speedup needs an analytic or AD gradient replacing the 40-evaluation central difference. That would change the optimisation path and so the draws. Not built, not measured.

## 4. Ranked proposals

| # | Change | Gain (measured) | Draw change | Risk | Needs |
|---|---|---|---|---|---|
| 1 | Sufficient-statistic objective in `gaussian_profile_nll` (GllvmFit, q = 0 path) | ~23-32x on refit | rounding-level, not bit-identical (5e-8 abs) | Medium: core fitter shared by every Gaussian fit; Heywood/flat-ridge conditioning untested | Maintainer decision (below) |
| 2 | Sufficient-statistic loglik in `fit_twolevel_gaussian` | ~9-16x | rounding-level (1.3e-7 abs); iteration counts shift | Medium: same, plus iteration-count shifts | Maintainer decision |
| 3 | PMV: skip Hessian diagnostics in bootstrap refits | ~10% (profile share; not resolvable by timing) | none (bit-identical) | Low: needs a `diagnostics = true` keyword on `fit_precision_multivariate`; do not apply to the JointPhyloGrouped refit (the driver reads its `hessian_positive_definite`) | Small API decision |
| - | PMV CHOLMOD symbolic reuse | 0% | none | - | Rejected |
| - | Faster extraction / simulate / PPCA reuse | < 0.02% / 0.0% / < 1% | none | - | Nothing to gain |

## 5. Not recommended, with reasons

- **Warm start from the original fit**: not run and not recommended. Earlier evidence (as you reported) is that it changes which optimum is reached; the source comments in `bootstrap_sigma_r.jl` also record that R refits cold and that a warm start would shrink failure rates and tails. It would change draws, so it needs its own evidence, which I did not produce.
- **Looser `f_tol` / `g_tol` / fewer iterations**: not re-run. The source comment says `f_tol = 1e-10` stops early (logLik up to 1.3e-3 low) and must not be loosened. GllvmFit already uses `f_tol = 1e-12`, mean 180 and max 307 iterations against a cap of 2000, so the cap is not the cost. Any tolerance change alters draws.
- **Forward instead of central differences for PMV**: halves gradient cost but changes the path and draws. Not tested.

## 6. What needs a maintainer decision

1. Whether rounding-level draw changes (about 1e-7) in `fit_gaussian_gllvm` / `fit_twolevel_gaussian` are acceptable. The alternative is a bootstrap-only internal path that reuses a precomputed `S`; the draws still change, only the blast radius shrinks. Either way every pinned bootstrap value and seed-reproducibility claim needs rechecking, and the bootstrap tests that compare with R tolerances should be run.
2. Before adopting #1 or #2, run them on the fixture-c Heywood/flat-ridge grid (the `f_tol = 1e-12` evidence) to show failure counts and logLik versus the optimum are unchanged. I did not do this.
3. Scope of #1: I patched only the q == 0 non-phy branch (it also covers K_W > 0 and has_diag). Fits with X keep the old path, and the profile-out-beta branch is unchanged.
4. API for #3: an optional `diagnostics::Bool = true` on `fit_precision_multivariate`, passed `false` only from the PMV bootstrap refit.

## 7. Caveats

- Load: all timings were taken under load average ~300. Ratios for #1 and #2 come from two independent run pairs each (22.8x and 32.5x; 9.4x and 15.7x), which already shows the spread.
- Single synthetic dataset and seed per class; replicate draws are 20, so no coverage conclusions follow from the matching of draws.
- The profiler's GC-frame share (29-49%) counts samples with GC frames anywhere on the stack; it is an indicator, not a precise wall fraction.
- Files: scripts `prof.jl`, `prof2.jl`, `exp2.jl`, `exp3.jl`, `cmp.jl`, `bench_obj.jl`; outputs on Totoro in `~/prof_scratch/` (`out_*.txt`, `out2_*.txt`, `out3_*.txt`, `expo_*.txt`, `ex3_*.txt`, `exp2_*.jls`, `exp3_*.jls`). Nothing deleted.
