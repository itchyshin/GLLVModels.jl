# GLLVModels.jl legacy R bridge scaffold — `gllvm_julia()`

This directory contains a historical **gllvm-style direct R scaffold** for
low-level parity smoke checks. It is not the current `gllvmTMB(...,
engine = "julia")` admission surface; the current R package bridge is guarded
by `GLLVModels.bridge_capabilities()` and lives in `gllvmTMB`. You call something
that looks like `gllvm::gllvm(...)` — same family strings, `num.lv`, `row.eff`,
`disp.formula` — and the scaffold runs the fit in GLLVModels.jl via
[JuliaConnectoR](https://github.com/stefan-m-lenz/JuliaConnectoR), then returns
a list in **gllvm parameter conventions** (e.g. NB dispersion as `phi = 1/r`).

> **Status: SCAFFOLD, ONE PARITY SMOKE GREEN.** The JuliaConnectoR path starts
> Julia, activates the selected local project when `GLLVM_JL_PATH` or `jl_path`
> is supplied, loads canonical `GLLVModels` (falling back only for an installed
> legacy `GLLVM`) plus `Distributions`, constructs family markers, and
> extracts scalar/vector fields that JuliaConnectoR may already have converted to
> R values. A live Poisson `method="LA"` no-row-effect smoke check on 2026-06-14
> matched R `{gllvm}` to tight tolerances (`|ΔlogLik| = 2.09e-11`, max beta diff
> `1.76e-7`, Procrustes loading diff `6.56e-7`) after scaling R `{gllvm}` loadings
> by `sigma.lv`. **Full numerical parity is still open** for other families,
> dispersion structures, covariates, missingness, and CI payloads.

> **Transition correction (2026-09-18).** The canonical package is now
> `GLLVModels`; the loader selects it first and falls back to an installed legacy
> `GLLVM` package only when the canonical package is absent. `GLLVM_JL_PATH`
> remains the accepted environment variable for local-checkout compatibility.

## Files

| File | Purpose |
|------|---------|
| `gllvmjl.R`        | low-level accessor wrappers (`coef_table`, `getLV`, `getLoadings`, `predict`, …) calling GLLVModels.jl per-family fitters directly. |
| `gllvmtmb_julia.R` | the `gllvm_julia(...)` front door + dispersion conversions; **builds on** `gllvmjl.R`. |
| `parity_check.R`   | `compare_gllvm(y, ...)` numerical-parity harness (R gllvm vs the bridge). |
| `README_bridge.md` | this file. |

## Setup (once)

1. **R side.**
   ```r
   install.packages("JuliaConnectoR")
   install.packages("gllvm")          # only needed for parity_check.R
   ```
2. **Julia side.** Install GLLVModels.jl (Julia ≥ 1.10):
   ```julia
   using Pkg
   Pkg.add(url = "https://github.com/itchyshin/GLLVModels.jl") # or:
   Pkg.develop(path = "/path/to/checkout")                    # for local dev
   ```
3. **Point R at Julia** (if `julia` isn't on PATH) by setting `JULIA_BINDIR` to the
   directory containing the `julia` binary *before* loading JuliaConnectoR:
   ```r
   Sys.setenv(JULIA_BINDIR = "/path/to/julia/bin")   # e.g. ~/.juliaup/bin
   ```
4. **Point R at this checkout** when validating local code:
   ```r
   Sys.setenv(GLLVM_JL_PATH = "/path/to/checkout")
   ```

## Calling `gllvm_julia`

```r
source("r/gllvmtmb_julia.R")     # also sources r/gllvmjl.R for the accessors
gllvm_jl_init()                  # activates GLLVM_JL_PATH when set; imports once

# y is n x p: SITES in rows, SPECIES in columns (the gllvm orientation).
set.seed(1)
y <- matrix(rnbinom(150 * 8, mu = 4, size = 2), nrow = 150)   # 150 sites x 8 species

fit <- gllvm_julia(y, family = "negative.binomial", num.lv = 2,
                   method = "LA", disp.formula = ~1)          # shared dispersion
print(fit)
fit$dispersion          # list(name = "phi", value = ...)  — already 1/r converted
fit$loadings            # p x K
fit$lvs                 # n x K site scores

gllvm_julia_coeftable(fit)               # tidy Wald table
pred <- gllvm_julia_predict(fit, "response")   # n x p fitted means
```

## Family / option mapping

| gllvm call | bridge → Julia fitter | notes |
|------------|----------------------|-------|
| `family="gaussian"`, `disp.formula=~1` | `fit_gaussian_gllvm` | shared σ (profiled) |
| `family="gaussian"`, `disp.formula=NULL` | `fit_gaussian_pervar_gllvm` | per-species variances |
| `family="poisson"` | `fit_gllvm(family=Poisson())` / `fit_poisson_gllvm_va` | no dispersion |
| `family="negative.binomial"` (NB2) | `fit_nb_gllvm` / `_grouped` / `_va` | dispersion `r`; gllvm `phi = 1/r` |
| `family="negative.binomial1"` (NB1) | `fit_nb1_gllvm` / `_grouped` | dispersion `phi` (identity) |
| `family="binomial"` | `fit_binomial_gllvm` / `_va` | pass `N` for counts |
| `family="beta"` | `fit_beta_gllvm` / `_grouped` / `_va` | precision `phi` (identity) |
| `family="Gamma"` (or `"gamma"`) | `fit_gamma_gllvm` / `_grouped` / `_va` | shape `alpha` → gllvm `phi` (relabel) |
| `family="exponential"` | `fit_exponential_gllvm` | no dispersion |
| `family="ordinal"` | `fit_ordinal_gllvm` | common ordered cutpoints; logit/probit |
| `family="tweedie"` | `fit_tweedie_gllvm` / `_grouped` | power `p`, `phi` (identity); set `p_init=1.1` |
| `row.eff="fixed"` | `fit_roweffect_gllvm(family=…)` | per-site fixed intercepts |
| `row.eff="random"` | `fit_row_random_gllvm(family=…)` | `ρ_s ~ N(0, σ_row²)` |
| `disp.formula=NULL` (default) | `fit_*_gllvm_grouped(group = 1:p)` | **gllvm's per-species default** |
| `disp.formula=~1` | `fit_*_gllvm` (shared scalar) | one dispersion for all species |
| `method="LA"` (default) | Laplace fitters | |
| `method="VA"` | `fit_*_gllvm_va` | poisson / NB2 / binomial / beta / gamma only |

**Orientation:** gllvm uses `y` as **n × p** (sites × species); GLLVModels.jl uses **p × n**
(species × sites). The bridge transposes internally and returns loadings (p × K) and
scores (n × K) in gllvm orientation.

## Dispersion-conversion table (engine → gllvm convention)

Source of truth: `docs/src/gllvmtmb-parity.md` → "R bridge: parameterization map".
The bridge applies these **on the way out**, so `fit$dispersion` is already in gllvm units.

| Quantity | gllvm (R) | GLLVModels.jl | Bridge rule (applied in `.convert_dispersion`) |
|----------|-----------|----------|-----------|
| NB2 dispersion | `φ`, `Var = μ + μ²φ` | `r` (size), `Var = μ + μ²/r` | **`φ = 1/r`** (also ZINB / Hurdle-NB / grouped-NB) |
| NB1 dispersion | `φ`, `Var = μ + μφ` | `φ`, `Var = μ(1+φ)` | identity |
| Gamma dispersion | `φ` = **shape**, `Var = μ²/φ` | `α` = **shape**, `Var = μ²/α` | relabel `α → φ` (no inversion) |
| Beta precision | `φ`, `Var = μ(1−μ)/(1+φ)` | `φ` (same) | identity |
| Tweedie | power `ν`, `Var = φμ^ν` | power `p`, `Var = φμ^p` | identity; set `p_init=1.1` to match gllvm's optimiser path |
| Gaussian dispersion | per-species SD `φ_j` | per-species **variances** (pervar fit) | `φ_j = sqrt(variance_j)` |
| Dispersion **structure** | per-species by default (`disp.formula=NULL`) | shared scalar by default | route via `fit_*_gllvm_grouped(group=1:p)`, or set gllvm `disp.formula=~1` |

## Method note (LA vs VA)

gllvm's default estimation method is **`"VA"`** (variational); GLLVModels.jl's default path
is **Laplace**. They differ in finite samples, so for a parity check **pin matching
methods** on both sides (`compare_gllvm(..., method=)` passes the same string to each).
VA fitters in GLLVModels.jl exist for **poisson, negative.binomial, binomial, beta, gamma**
only, and only for the plain (shared-dispersion, no-row-effect) case.

## Known-unsupported combos in this legacy direct scaffold

This `r/` directory is a historical direct `gllvm_julia()` scaffold for parity
smoke tests. It is not the current `gllvmTMB(..., engine = "julia")` admission
surface. The current R package bridge is guarded by `GLLVModels.bridge_capabilities()`
and admits a subset of fixed-effect `X` models through `gllvmTMB`; keep this
scaffold conservative unless it is deliberately rewired to that same contract.

- **Site covariates `X`** — not yet wired through `gllvm_julia` (the engine *has*
  `fit_gllvm_cov` / `@formula`, but converting an R design matrix into Julia's
  `p × n × q` array needs a live-session check). Passing non-NULL `X` errors.
- **`method="VA"` + (grouped dispersion | row effects | unsupported family)** — errors;
  use `method="LA"`.
- **Families with no GLLVModels.jl path** (`ZNIB`, correlated LVs `lvCor`, structured row
  effects `corAR1/corExp/corCS`) — out of scope per `docs/src/gllvmtmb-parity.md`.
- **gllvmTMB temporal sources** (`temporal_indep/dep/latent`) — not bridged. The
  temporal source, alone or beside ordinary `unit` / `unit_obs` terms, has a
  native Julia counterpart, `fit_temporal_gllvm`
  (`docs/src/temporal.md`); fit it in R with gllvmTMB or in Julia directly.
- **Ordinal species-specific cutpoints** — GLLVModels.jl uses common ordered cutpoints.

## Validating parity

```r
source("r/gllvmtmb_julia.R"); source("r/parity_check.R"); gllvm_jl_init()
set.seed(1); y <- matrix(rpois(150 * 8, 3), nrow = 150)
res <- compare_gllvm(y, family = "poisson", num.lv = 2, method = "LA")
# prints logLik / beta / loadings (Procrustes-aligned) / dispersion diffs.
```

`compare_gllvm()` reports max abs / relative differences. Loadings are identifiable
only up to rotation/sign, so they are **Procrustes-aligned** before differencing.
Live 2026-06-14 smoke result (`family = "poisson"`, `num.lv = 1`, `method = "LA"`,
30 sites x 4 species): after activating the local Julia project via
`GLLVM_JL_PATH` and scaling R `{gllvm}` loadings by `sigma.lv`, the smoke passes
the tight parity gate (`|ΔlogLik| = 2.09e-11`, max beta diff `1.76e-7`,
Procrustes-aligned loading diff `6.56e-7`). Earlier same-day failed numbers
(`|ΔlogLik| = 0.619`, max beta diff `0.0486`) were traced to harness activation
and loading-scale drift.
