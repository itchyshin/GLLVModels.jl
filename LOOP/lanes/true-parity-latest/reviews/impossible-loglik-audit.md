# Audit: fits reported `converged = true` at an impossible or spurious log-likelihood

Date: 2026-09-27. Read-only audit. Nothing in `src/` was edited and no PR was opened.

- Code: detached worktree `/Users/z3437171/local-scratch/GLLVM.jl-ll-audit` at `origin/main` `880cad4c7` (removed after the audit).
- Julia 1.10.0, `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`.
- Probe scripts are kept in `/Users/z3437171/local-scratch/ll-audit-probes/` (`p1_*.jl` to `p6_*.jl`). Each one runs with
  `~/.juliaup/bin/julialauncher --project=<worktree> <script>`.
- Total probe compute was about 6 minutes, within the 30-minute budget.

## What this pattern is

A fit is at risk when an objective can return a finite value that no model could produce, and the fitter then accepts it as a log-likelihood. Three different numerical failures lead there. Each needs its own gate.

| Mechanism | Symptom | Known case | Gate that catches it |
|---|---|---|---|
| M1. Cancellation in the conditional density | log-pmf > 0, or a huge positive loglik | beta-binomial #522 | loglik above the family's bound (≤ 0 for a discrete family) |
| M2. Laplace log-det at a near-singular site precision `A = I + Λ'W_obsΛ` (observed curvature W < 0) | loglik inflated by tens to hundreds of units. It can be a spurious global maximum | ZI twin #557 | floor on the smallest eigenvalue of `A` at the optimum |
| M3. Subtractive Woodbury quadratic form at extreme variance ratios | negative quadratic form, loglik ~ +1e22 | two-level #576 (draft) | quadratic form ≥ 0, or a dense solve when the variance ratio is extreme |

**The key finding on M2.** A PD check cannot catch M2. At a certified local maximum of the site log-posterior, `A` is the negative Hessian, so it is positive semi-definite automatically. Probe 5 confirms this: 0 of 19,412 converged mixed-family sites had an indefinite `A`. The failure is a PD but near-singular `A`, where `-½ logdet A → +∞`. That is why #557 needed a floor on the smallest eigenvalue and not an `issuccess(cholesky)` test. Every existing PD guard in the Laplace kernels (`laplace.jl`, `covariates.jl`, `studentt.jl`, `aghq_grid.jl`, `twopart.jl`) tests only non-PD, so none of them can catch M2.

## Summary table

"Guard" means: does the fitter check for an impossible value before it reports `converged`?

| Fitter / path | Mechanism exposure | Guard present? | Probe run? | Result | Risk |
|---|---|---|---|---|---|
| `fit_truncated_nbinom2_gllvm` (own kernel `truncated_nbinom2.jl:320-338`, `hessian = :observed` default) | M2: observed weight < 0 for small `r` (min −0.033 at r = 0.2) | No PD guard, no floor. `_fit_verdict` checks only the sentinel | Yes (P4, P4b, P5) | **3 of 12 fits converged with min site A ≈ 1e-5 and Laplace 40 to 139 units above the exact marginal. In seed 4 the breakdown point beats the healthy optimum: a spurious global maximum, exactly as in #557** | **High** |
| `fit_twolevel_gaussian` / `_twolevel_loglik` (`twolevel.jl:36-110`) | M3 | None on main (#576 draft adds a sign check) | Yes (P2 C0, C) | #576 replicate on main: F64 **+7.85e21**, exact −622.8. Random: 0/1000 overstated at ratio spans ≤ 21 decades, 57/1000 at 30 decades (7 positive where exact < 0), 196/1000 at 52 decades (103 positive) | **High** (known) |
| `gaussian_marginal_loglik` non-phy Woodbury (`likelihood.jl:170-205`) with per-species `σ²_B`/`σ²_W`. Used by `gaussian_nll_packed` in `confint_profile.jl:185,250`, `confint_derived.jl:679`, `loading_profile_confirmatory_internal.jl:256`, `reml.jl:78` | M3 | None | Yes (P2 A, P2b) | 0/1000 overstated at spans ≤ 14 decades, 4/1000 at 21, 62/1000 at 30, 99/1000 at 52 (worst: F64 +1.5e19 vs exact −7.8e19). Equal-d case (σ_eps only, X_lv path): 0/3000 | **Medium**: reachable only if a constrained or profile refit runs the variances apart, which is what happened in #576 |
| `gaussian_marginal_loglik_sparse_phy` (`likelihood_sparse_phy.jl:205-216`, same subtractive Woodbury, Float64 only because of CHOLMOD) | M3 | None | No | Same formula as the path above. AGENT-INFERRED, same exposure | Medium (inferred) |
| `gaussian_profile_nll` (main `fit_gaussian_gllvm` route, `profile.jl:150-275`) | M3 in principle | None | Yes (P2 B) | 0/3000 overstated. `d̃ = 1 + τ ≥ 1` keeps D well conditioned even with \|L\| up to 1e8 | Low |
| `em_fa` (`em_fa.jl:75-95`, explicit `quad_a − quad_b`, ψ floored at `eps()`) | M3 in principle | None | Yes (P2 D, P2b D) | F64 vs BigFloat within 1e-11, including a near-collinear (Heywood-type) variable. EM stalls honestly (`conv = false`) before ψ reaches the floor | Low |
| `fit_studentt_gllvm` (`studentt.jl:270-300`, observed default) | M2 (W < 0 for \|r\| > σ√ν, min −0.156) | PD guard only | Yes (P4, P4b) | 6 of 12 converged fits have a site with A in [3e-5, 8e-4]. That site's Laplace value is 4.2 above exact. Whole-fit Laplace stays 3 to 22 below exact, so no spurious global maximum was seen | Medium-low |
| `fit_nb1_gllvm` (grouped NB1 kernel, observed default) | M2 (observed weight min −10.7 at φ = 0.1) | No floor | Yes (P3b, P4) | 0/12 breakdowns, min A ≥ 1.46, \|gap\| ≤ 3.6 | Low-medium (mechanism present, not triggered) |
| `fit_beta_gllvm` (core `laplace_loglik_site`, observed default) | M2 (min −3.0 at φ = 5) | PD guard only | Yes (P3c, P4) | 0/12 fits, 0/40,000 random sites with A < 0.1 | Low |
| `fit_hurdle_nb_gllvm` (twopart, observed default, truncated-NB2 count part) | M2 (same count curvature as TruncNB2) | twopart mode-search PD fallback only | Yes (P6) | 0/10 breakdowns, min A ≈ 1 | Low-medium (mechanism present) |
| `_mixed_loglik_site` (`mixed.jl:340-393`), `grouped_dispersion.jl` kernels (`hessian = :observed`), `quadratic.jl`, `ordered_beta.jl` (W clamped ≥ 1e-8) | M2. Also, `logdet(Symmetric(A))` of an indefinite A with an even number of negative eigenvalues returns a finite value (P3a: `logdet(Symmetric(diag(−0.5,−0.2))) = −2.30`) | No PD guard, no floor | Partly (P5) | 0 indefinite A at certified modes in 19,412 sites. The finite-logdet trap needs a non-maximum mode | Low (near-singular risk as M2 in general) |
| `phylo_glm.jl`, `spde_latent.jl`, `coevolution_glm.jl`, `phylo_*_xlv.jl` | M2 when the default hessian is observed | `cholesky` throws on non-PD (caught). No floor | No | Not probed | Low-medium (inferred) |
| Julia-native ZIP / ZINB / ZIB (`twopart.jl`) | M2 at y = 0 (the #557 mechanism) | #557 applies its floor only to the new `zi_twin.jl` route | No | Not probed. The #557 note says these routes use the Fisher log-det (W ≥ 0), which would make them immune. Unverified | Medium (needs a check) |
| `fit_beta_binomial_gllvm` | M1 | **Yes** (`_beta_binomial_verdict`: loglik ≤ 1e-6, φ boundary) | P1 | log-pmf ≤ 0 across φ ∈ [1e-6, 1e30] | Fixed (#522) |
| Discrete conditional pmfs: Poisson, NB2, NB1, TruncNB2, TruncPoisson, COM-Poisson, Binomial | M1 | n/a (density level) | Yes (P1) | max log-pmf ≤ 0 and Σ pmf ≤ 1 + 7e-14 across extreme μ and dispersion (NB2 r to 1e30, NB1 φ to 1e-20, COM ν ∈ [0.01, 50]) | Low |
| GP-1 with α < 0 | M1 (improper mass) | none | P1 | Σ pmf up to 1.0016 at α = −0.1. Where 1 + αy < 0 the density throws a DomainError, which becomes the sentinel | Low |
| Shared `_fit_verdict` (`fit_verdict.jl`, used by about 60 files) | all | Upper sentinel only (`nll ≥ 1e11` or non-finite) | n/a | No lower bound, no curvature and no quadratic-form checks | This is the structural gap |

## Findings with evidence

### F1. Truncated NB2 converges at a Laplace breakdown point; seed 4 is a spurious global maximum (High)

Probe `p4_fits.jl`. The data follow the model: p = 4, n = 150, K = 1, r_true = 0.3, seeds `MersenneTwister(100 + s)`. Each fit uses the default `fit_truncated_nbinom2_gllvm(Y; K = 1)`. "exact" is a 4001-point quadrature over z ∈ [−12, 12] using the package's own `_glm_logpdf` with its clamps.

```
== TruncNB2/log r_true=0.3
  seed  4 conv=true  loglik= -1663.100 exact= -1741.215 gap=  78.115 minA=7.86e-06
  seed  6 conv=true  loglik= -2221.222 exact= -2359.845 gap= 138.623 minA=0.000128
  seed 10 conv=true  loglik= -1787.275 exact= -1827.147 gap=  39.871 minA=1.28e-05
  (seeds 1 and 3: gap 5.3 and 8.5 with minA 0.91 and 0.79, which is ordinary Laplace error and not breakdown)
```

Per-site decomposition (`p4b_site.jl`, seed 6: r̂ = 0.0020, λ̂ = [1.13, −2.14, 6.96, 0.09]). The 18 sites with A < 0.01 carry +49.5 of excess; the 29 sites with A > 0.5 carry +10.1. Example: site 28, A = 1.3e-4, Laplace −10.49, exact −14.65.

Refit from a shrunk start (`Λ_init = 0.1 Λ̂`, `r_init = 0.3`; `p5_mixed_refit.jl`):

```
seed 4: default conv=true r̂=0.0145 loglik=-1663.100 exact=-1741.215 minA=7.9e-06 | shrunk refit conv=true r̂=0.133 loglik=-1686.335 exact=-1686.390 minA=1.1
seed 10: default conv=true r̂=0.0505 loglik=-1787.275 exact=-1827.147 minA=1.3e-05 | shrunk refit conv=true r̂=0.195 loglik=-1637.256 exact=-1638.673 minA=0.97
seed 6: default loglik=-2221.222 | shrunk refit conv=true loglik=-1888.350 exact=-1889.895 minA=1.01
```

In seed 4 the Laplace objective is 23 units *higher* at the breakdown point than at the healthy optimum, but 55 units *lower* in exact terms. A better optimiser would find the artefact more often, not less. In seeds 6 and 10 the fit stalls at a local breakdown point below the healthy optimum and still reports `converged = true`. This is #557's mechanism in a family #557 does not touch. Cause: `_glm_obs_weight(::TruncatedNegBin2)` goes negative when r is small (P3b: −0.033 at r = 0.2, y = 1, μ = 7.4). r̂ → 0 makes it more negative, and the kernel at `truncated_nbinom2.jl:338` has no floor.

### F2. Two-level Gaussian: confirmed on main (High, fix in draft #576)

`p2_gauss_woodbury.jl` C0. The verbatim #576 replicate on origin/main gives `F64=7.853e+21 exact=-622.8`.

`p2b_range.jl` C, with Λ ~ N(0, 1) and log10 σ² uniform on the span shown:

```
C twolevel log10 range [-15,6]:  overstated=0/1000   positive-where-exact-negative=0
C twolevel log10 range [-20,10]: overstated=57/1000  positive-where-exact-negative=7
C twolevel log10 range [-37,15]: overstated=196/1000 positive-where-exact-negative=103
```

The #576 sign check (quad < 0 → −Inf) catches the positive cases. It does not catch every overstated case: some keep a negative but inflated value, for example F64 −2.3e25 vs exact −2.6e25. Those are far from any optimum and are unlikely to matter.

### F3. `gaussian_marginal_loglik` has the same Woodbury hazard, and it feeds profile and derived CIs (Medium)

`p2b_range.jl` A, with per-species `σ²_B`, `σ_eps = 1e-8`, and data drawn near the model:

```
A σ²_B per species log10 range [-10,4]:  overstated=0/1000
A σ²_B per species log10 range [-15,6]:  overstated=4/1000
A σ²_B per species log10 range [-20,10]: overstated=62/1000
A σ²_B per species log10 range [-37,15]: overstated=99/1000
A equal d (σ_eps only) log10 range [-16,-10]: overstated=0/1000
```

`p2_gauss_woodbury.jl` A, the worst case: F64 +1.459e19 vs exact −7.849e19 (p = 6, K = 2, σ_eps = 3e-18, σ²_B spanning 1e-36 to 2e9).

The healthy range is wide, so a point fit is unlikely to walk there. The exposure is in constrained refits. `confint_profile.jl` optimises `gaussian_nll_packed` with one coordinate pinned, and the #576 bootstrap showed that one trait's variance can run to 1e-37 while another's runs to 1e15. The main profiled fitter is not exposed (`p2_gauss_woodbury.jl` B: 0/3000). The explicit-σ_eps X_lv path uses equal d and is also not exposed. `gaussian_pervar.jl:17` already chose a dense Cholesky deliberately "because subtractive Woodbury solves can lose the quadratic form". That is the same lesson, applied in one file only.

### F4. Student-t, NB1, Beta: the mechanism is present but did not trigger in these probes

- Student-t (P4, P4b): 6 of 12 fits converged with one site at A ≤ 8.5e-4. The site's excess over exact is about +4.2 (≈ −½ log A); the whole fit still sits below exact. Bounded per site, so this is not a spurious optimum here. With more outlier sites the excess would scale with their number.
- NB1: the observed curvature is strongly negative (−10.7 at φ = 0.1, y = 60, μ = 1.65) and NB1 defaults to `:observed`, yet 0 of 12 fits broke down (min A ≥ 1.46). Beta: 0 of 12 fits and 0 near-singular sites in 40,000 random sites.
- These negatives come from a modest DGP and 12 seeds. They do not show that the families are immune.

### F5. Discrete conditional densities are sound (Low)

`p1_pmf_bound.jl` excerpt:

```
NB2       max logpmf = -5.8e-06 | max Σpmf(0:400) = 1.0000000000000189 at (μ=148, r=1e12)
NB1       max logpmf = 0.0 (y=0, μ=3e-4, φ=1e-20) | max Σpmf = 1.0000000000000673
COMPoisson max logpmf = -4.5e-05 | max Σpmf(0:3000) = 1.0000000000156
BetaBinom max logpmf = -1.7e-05 (φ up to 1e30) | Σpmf = 1.0000285 at φ = 1e-6
GP1       max Σpmf = 1.0015713 at (μ=7.39, α=-0.1) | errors=6363 (DomainError where 1+αy<0)
```

M1 is closed for everything except the GP-1 α < 0 improper-mass corner. That corner is small, and the GP-1 fitter uses the Fisher log-det by default.

## Ranked fixes

1. **TruncNB2: add the #557 guard now.** Take the site-level smallest eigenvalue of `A` in the kernel at `truncated_nbinom2.jl:320-338`. Report `converged = false` with a reason when the optimum's min site eigenvalue is below a floor. Retry once from a shrunk start (Λ × 0.1, moment-based r). Probe P5 shows that retry recovers the healthy optimum in all 3 breakdown seeds. Test: the three literal datasets `MersenneTwister(104/106/110)` from `p4_fits.jl::genTNB`. Assert (a) no converged fit with min A < floor, and (b) seed 4 reaches loglik ≈ −1686.3. Effort: small.
2. **Merge #576**, then apply the same sign check, `quad < 0 → -Inf`, to `gaussian_marginal_loglik` (`likelihood.jl:202`) and `gaussian_marginal_loglik_sparse_phy` (`likelihood_sparse_phy.jl:215-216`). Better still: when `maximum(d)/minimum(d) > 1e12`, fall back to a dense Cholesky of `ΛΛ' + D`, as `gaussian_pervar.jl` already does. This costs p³ only in the pathological region, so the headline speed path is untouched. The sign check alone leaves some negative-but-inflated values; the dense fallback removes the cause. Test: the P2b `[-20,10]` sweep with fixed seeds, asserting F64 ≤ exact + 1e-6·|exact|.
3. **Extend the `_fit_verdict` contract with an opt-in family bound.** Add a keyword `loglik_max = Inf` and have every discrete-family fitter pass `0.0` plus a floating-point margin (the `_BB_LOGLIK_MAX = 1e-6` precedent). The exact marginal of a discrete model is ≤ 0 whatever curvature the Laplace step uses, so a positive value is always an artefact. This generalises the #522 guard from one family to about 25 call sites at zero runtime cost. It catches only gross M1/M2 failures, since the TruncNB2 breakdowns sat at about −1700.
4. **One shared post-fit Laplace health pass**, `_laplace_site_eigmin`, for the kernels that use observed curvature: core `laplace.jl`, `grouped_dispersion.jl`, `truncated_nbinom2.jl`, `mixed.jl`, `studentt.jl`, `twopart.jl`, `zi_twin.jl`. Evaluate the smallest eigenvalue of each site's `A` once at the returned optimum (n sites × one K×K eigen-solve), then hand it to the verdict. Priority order: TruncNB2 (done in item 1), Hurdle-NB, NB1, Student-t, Beta, mixed.
5. **Check whether the Julia-native ZIP/ZINB/ZIB routes (`twopart.jl`) use observed or Fisher weights in the log-det.** If observed, they carry the exact #557 defect, and #557 only guarded `zi_twin.jl`. Effort: a 10-minute probe, re-using `p4_fits.jl` with `fit_zip_gllvm`/`fit_zinb_gllvm`.
6. Low priority: in the unguarded kernels, replace `logdet(Symmetric(A))` with `logdet(cholesky(A; check = false))` plus `issuccess`. An indefinite `A` with two negative eigenvalues currently returns a finite value (P3a). It is unreachable at a certified mode, but it is a trap if a mode search ever returns a non-maximum.

## The Rose-principle question: one shared gate?

**Where it can live.** The natural home is `_fit_verdict` in `src/fit_verdict.jl`. About 60 files already route `Optim` results through it. Its docstring states the governing rule, "refusing to report a failure sentinel as a log-likelihood", and its design note already argues for one shared helper over ninety per-fitter verdicts. The three family-specific verdicts (`_beta_binomial_verdict`, `_tweedie_verdict`, `_phylo_verdict`) would become callers of the extended shared form.

**What it can check cheaply.** A generic gate has three checks. Each needs something different from its caller.

| Check | Needs from the caller | Runtime cost | Catches |
|---|---|---|---|
| loglik finite and ≤ the family bound | one number (`loglik_max`: 0 for discrete, `Inf` for continuous) | none | #522, gross M2 |
| min site eigenvalue of the Laplace precision ≥ floor (the #557 rule: refuse, and flag when within 10% of the floor) | the kernel must expose `A` or its eigenvalue at the optimum | one pass over sites, K×K eigen, far below one objective evaluation | #557, F1 |
| Gaussian quadratic form ≥ 0, or a well-conditioned solve | must live inside the Woodbury helpers, not the verdict: the value is already summed by the time the verdict sees it | none (sign check) or p³ in the pathological region (dense fallback) | #576, F3 |

**What it would cost.** Checks 1 and 2 can share the verdict. Check 1 is a keyword plus about 25 one-line call-site edits. Check 2 needs a per-kernel "site eigmin at θ̂" function for the 7 observed-curvature kernels. The verdict itself stays generic: `_fit_verdict(res; loglik_max, site_eigmin, eigmin_floor)`. Check 3 cannot sit in the verdict. It belongs in the 3 Woodbury helpers (`_woodbury_core` in twolevel, the non-phy block of `gaussian_marginal_loglik`, and `_woodbury_apply` in sparse phylo), ideally one shared helper that returns `-Inf` on a negative quadratic form.

**Caveats.**
- Where the M2 check sits changes what it does. A floor *inside* the objective changes the surface (the #557 note says it is safe only together with the within-10% rule). A floor applied *post-fit* only relabels the fit. Post-fit plus one shrunk-start retry is the less invasive design, and P5 shows the retry recovers the healthy optimum for TruncNB2.
- The floor value 0.1 comes from ZI data. Healthy optima in this audit had min A of 0.79 to 9.6 (TruncNB2, NB1, Beta, Hurdle-NB) and 0.3 to 0.5 in #557's sweeps. 0.1 looks safe across families, but that is AGENT-INFERRED from these probes.
- None of the three checks catches ordinary Laplace error: the gaps of 5 to 8 units at healthy A in TruncNB2 seeds 1 and 3, or Student-t's 3 to 22 units below exact. That error belongs to the approximation, not to numerical breakdown, and is out of scope here.

## What this audit does not cover

- Only probe results were measured. The ZIP/ZINB/ZIB routes, sparse phylo, the phylo/spde/coevolution GLMs and the grouped-dispersion kernels were read, not run.
- The fit sweeps are small (10 to 12 seeds per family, one DGP each). A negative result does not show immunity.
- No parametric bootstrap was run beyond reproducing the #576 replicate. The TruncNB2 rate (3 of 12 default fits) is a point estimate from one DGP.
- Nothing was run on Julia 1.13.
