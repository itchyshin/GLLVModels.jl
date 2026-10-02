# GLLVModels.jl parity suite (`test/parity/`)

Opt-in comparison tests between GLLVModels.jl (Julia) and R's `gllvmTMB`
(primary twin), using RCall.jl to drive R from within Julia. Optional light
CRAN `gllvm` cross-checks are secondary and must not block the twin cell.

## Why isolated from the default test suite

`RCall.jl` requires a working R installation to precompile.  CI runners that
lack R would fail at `Pkg.instantiate()` if RCall appeared in
`test/Project.toml`.  To keep `julia --project=. test/runtests.jl` always
runnable — on any machine, with or without R — this directory is a
**completely separate Julia project**.  The default suite never includes or
references any file here.

## How to run

```sh
GLLVM_PARITY_TESTS=1 julia --project=test/parity test/parity/runparity.jl
```

Running without `GLLVM_PARITY_TESTS=1` exits cleanly with a skip notice.

`runparity.jl` resolves its own environment (it calls `Pkg.develop` before loading
RCall), so the command above works on a fresh checkout. There is no committed
`Manifest.toml` — it is gitignored. If you instead load this project **directly**
(e.g. `julia --project=test/parity -e 'using RCall'`) on a machine that has never run
the suite, instantiate it first, or `using RCall` fails with a missing-dependency
error:

```sh
julia --project=test/parity -e 'using Pkg; Pkg.instantiate()'
```

Point `GLLVM_PARITY_R_LIBS` at the library holding the twin. The historical default
(`/tmp/R-gllvmtmb-x-parity-20260802`) is a `/tmp` path and does not survive a reboot;
when it is missing, `_parity_prepend_twin_lib!()` silently falls back to
`.libPaths()`. That often still works, but by accident — set it explicitly:

```sh
export GLLVM_PARITY_R_LIBS="$(Rscript -e 'cat(.libPaths()[1])')"
```

## R prerequisites

1. R ≥ 4.2 installed and on `PATH` (or `R_HOME` set).
2. The **`gllvmTMB`** twin package (local checkout / GitHub), not only CRAN `gllvm`.
3. RCall built against that R:
   ```sh
   julia --project=test/parity -e 'using Pkg; Pkg.build("RCall")'
   ```

## What is compared and why

Raw loadings `Λ` are **not** compared (rotation / sign non-identifiability).

Only **rotation-invariant** quantities are tested:

| Quantity | Invariance |
|---|---|
| Marginal log-likelihood | Fully invariant (scalar objective) |
| Fitted covariance `Σ_y = ΛΛᵀ + σ²I` | Invariant under `Λ → ΛQ`, `Q'Q = I` |
| Residual SD `σ_eps` | Invariant |

## Live call shape

Primary oracle path (see
`docs/dev-log/plans/scratch/2026-08-01-gaussian-rcall-shape.md` and
`test/parity/parity_helpers.jl`):

- Fit with **`gllvmTMB::gllvmTMB`**, formula
  `value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE)`.
- **`unique = FALSE`** so Ψ is off (Gaussian: `Σ = ΛΛᵀ + σ²I`).
- **Gaussian only:** centre Y per trait (Julia zero-mean J1 vs R `0+trait`).
- **Binomial / Poisson:** do **not** centre — Julia already estimates per-trait `β`.
- **NB2:** per-trait dispersion: the parity tests call
  `fit_gllvm(...; family=NegativeBinomial(), disp_group=:species)`
  → `NBGroupedFit` (observed Laplace Hessian) to match R's
  `log_phi_nbinom2[p]` (#132). Bare `NegativeBinomial()` fits one shared `r`
  (`fit_nb_gllvm`) since #615 (maintainer decision 2026-10-01).
- **Beta:** per-trait precision `φ` — Julia public default
  `fit_gllvm(...; family=Beta())` → `BetaGroupedFit` to match R's
  `log_phi_beta[p]` (#148). Shared-φ remains via named `fit_beta_gllvm`.
- Extractors: `as.numeric(logLik(fit))` (= `-opt$objective`); Gaussian also
  compares `report$sigma_eps` and `extract_Sigma(..., part="shared")`.

Cells **green** (light logLik oracles):
Gaussian; Binomial (Bernoulli); Poisson; NB2 (public `fit_gllvm` per-trait φ);
Beta (public `fit_gllvm` per-trait φ + observed Beta/logit Laplace Hessian);
Ordinal **probit** + observed Hessian.

**Shared site-X cohort** (`test_x_covariate_parity.jl`, after no-X cells):
Gaussian / Binomial / Poisson with `q=1` shared site covariate
(`X[t,s,1]=x[s]`). R formula uses bare `+ x` (shared slope), not
`(0 + trait):x`. Julia: `fit_gaussian_gllvm(; X=)` / `fit_gllvm_cov`.
Prefer twin R lib via `GLLVM_PARITY_R_LIBS` (default
`/tmp/R-gllvmtmb-x-parity-20260802`).
**NB2+X / Beta+X (Arc 2, `test_x_covariate_parity.jl` cohort 2):** shared
site-X slope γ + **per-trait** dispersion (`group=collect(1:p)`), twin API B
under X — matches R's `gllvmTMB::nbinom2()` / `gllvmTMB::Beta()` per-trait
default. Julia: `fit_nb_gllvm_grouped_cov` / `fit_beta_gllvm_grouped_cov`,
**default `hessian=:observed`**. See
`docs/dev-log/after-task/2026-08-02-nb2-beta-x-arc2-parity.md` (#177 merged).

**Gamma+X** (Arc 2 cohort 3): `fit_gamma_gllvm_grouped_cov` with
`group=collect(1:p)` and default `hessian=:observed` (TMB Laplace curvature;
OH unblocker vs Fisher-only) vs `stats::Gamma(link="log")` / per-trait
`log_phi_gamma`. Δ ≈ 3.03e-8 at seed=46 (rtol 1e-6). See
`docs/dev-log/after-task/2026-08-03-gamma-x-arc2-parity.md`.

**Ordinal+X** (Arc 2 cohort 4, #181): `fit_ordinal_gllvm_pertrait_cov` +
`ProbitLink` vs `ordinal_probit()`; Δ ≈ 5.38e-9 at seed=47.

**Species-specific XB**: R `(0 + trait):x` vs Julia `fit_gllvm_speciescov`
— Poisson (#190, Δ≈4.20e-9) + Binomial Bernoulli N=1 (S1). See
`test_species_x_parity.jl`.

**Not claimed here:** multi-family species-XB cohort; Gaussian species-XB;
X_lv;
shared-φ-Julia-vs-per-trait-R comparisons; shared-dispersion NB2/Beta via
named fitters as twin default; ordinal-logit; ADEMP; coverage; “full family
parity.” `n_drift=0` ≠ these cells.

## Tolerances

| Quantity | Target |
|---|---|
| logLik | rtol ≤ 1e-6 |
| σ_eps (Gaussian) | rtol ≤ 1e-4 |
| Σ_y (Gaussian) | atol ≤ 1e-4 |

Do not silently widen. If a live cell fails, fix call shape / model identity.

### Binomial oracle inputs

The no-X and shared-X helpers accept `N` (traits×units) and `binomial_link`
(`:logit`, `:probit`, `:cloglog`). Supplied trials become R weights; omitted
trials preserve Bernoulli behavior. Complete-data fixtures require finite
integer successes and positive integer trials, exactly representable as Float64.
These controls are input-transport tested; new fitted link/trial parity remains
unqualified. They do not widen the public R→Julia bridge.
