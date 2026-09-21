# Tutorial: general-model interface tour

This is a reference tour of the general latent-variable model interface. It is
not one serial, copy-and-run analysis: later sections deliberately use symbolic
data names (`Y`, `N`, `Yp`, `Yc`, and `Yo`) whose required support is described
next to each family. Do not run the snippets as one workflow or substitute an
undefined matrix into them.

If you are new to the package, choose a complete, defined-data route first:

- **General latent-variable question:** [General latent-variable first fit](quickstart.md)
  simulates a Gaussian `p × n` response matrix and fits it end to end.
- **Phylogenetic comparative question:** [First phylogenetic Gaussian model](vignettes/phylogenetic-gllvm.md)
  supplies both a tree and an aligned continuous trait vector.
- **Community or species-distribution question:** [First community abundance model](vignettes/community-abundance.md)
  supplies a count matrix and a Poisson community fit.

Return here after that first fit to identify a response family, an interface,
or a supported extension. The matrix convention for the multivariate snippets
is **`Y` is `p × n`**: `p` species or responses (rows) by `n` sites or
observations (columns).

```julia
using GLLVModels, Distributions, Random
```

## 1. Core family interfaces

The most direct entry is the **unified** `fit_gllvm`, which dispatches on a
`Distributions.jl` family marker (the GLM.jl convention) and forwards `K` and any
family-specific keywords to the underlying fitter:

```julia
fit = fit_gllvm(Y; family = Poisson(), K = 2)   # counts, log link, Laplace marginal
```

`fit_gllvm` dispatches `Normal()`, `Poisson()`, `Binomial()`,
`NegativeBinomial()`, `Beta()`, `Gamma()`, `Exponential()`, and `Ordinal()`.
Each is equivalently reachable through its **family-specific driver**, which is
where the family-specific keyword arguments live:

```julia
fp = fit_poisson_gllvm(Y; K = 2)                       # Poisson (counts)
fn = fit_nb_gllvm(Y; K = 2)                            # NB2: Var = μ + μ²/r
fb = fit_binomial_gllvm(Y; K = 2, N = N)               # Binomial, N = trial counts
fβ = fit_beta_gllvm(Yp; K = 2)                         # Beta, proportions in (0,1)
fg = fit_gamma_gllvm(Yc; K = 2)                        # Gamma, positive continuous
fe = fit_exponential_gllvm(Yc; K = 2)                  # Exponential (no dispersion)
fo = fit_ordinal_gllvm(Yo; K = 2)                      # ordered categories 1:C
```

`fit_binomial_gllvm` takes `N` (a `p×n` integer matrix of trial counts; default
all-ones = Bernoulli). The default links are the canonical ones — `LogLink()`
for counts / positive continuous, `LogitLink()` for Binomial / Beta / Ordinal —
and can be overridden with `link = ...` (`LogitLink`, `ProbitLink`,
`CLogLogLink`, `LogLink`, `IdentityLink`). For ordered categories the cumulative
link can be switched to probit: `fit_ordinal_gllvm(Yo; K = 2, link = ProbitLink())`.

For binomial counts that are **over-dispersed** relative to `Binomial(N, μ)`, the
beta-binomial lets the per-trial success probability itself vary
(`p ~ Beta(μφ, (1−μ)φ)`), jointly estimating the Beta precision `φ`:

```julia
fbb = fit_gllvm(Yb; family = BetaBinom(), K = 2, N = N)  # per-species φ → fbb.φ is length p
fbs = fit_beta_binomial_gllvm(Yb; K = 2, N = N)          # one shared φ
```

`N` is a `p×n` matrix of trial counts; as `φ → ∞` the family reduces to
`Binomial(N, μ)`. Links: `LogitLink()` (default), `ProbitLink()`, `CLogLogLink()`.
Through `fit_gllvm` (and `gllvm`) `N` is **required** — at `N = 1` the family
collapses to `Bernoulli(μ)` and `φ` cannot be identified — while the named fitters
keep their all-ones default. The marker's own `φ` argument is inert, so
`BetaBinom()` is the usual call.

Two negative-binomial variances are available. The default `fit_nb_gllvm` is
**NB2** (quadratic, `Var = μ + μ²/r`); `fit_nb1_gllvm` is **NB1** (linear,
`Var = μ(1 + φ)`, quasi-Poisson-like) for communities whose overdispersion grows
proportionally with the mean:

```julia
f2 = fit_nb_gllvm(Y;  K = 2)    # NB2, dispersion r (shared)
f1 = fit_nb1_gllvm(Y; K = 2)    # NB1, dispersion φ (shared); Var = μ(1+φ)
```

NB1 also has a marker, so it goes through the unified entry point and the
`@formula` front end. Both default to **per-species** `φ`; the marker's own `φ`
argument is inert (dispersion is always estimated), so `NB1()` is the usual call:

```julia
fit_gllvm(Y; family = NB1(), K = 2)               # per-species φ → NB1GroupedFit
gllvm(@formula(y ~ 1), Y, site_data; family = NB1(), K = 2)   # same fit
```

For biomass / abundance with exact zeros and continuous positives, **Tweedie**
(compound Poisson–Gamma, `1 < p < 2`) fits the power mean–variance model
directly, estimating both the dispersion `φ` and the power `p`:

```julia
ft = fit_tweedie_gllvm(Y; K = 2)    # Tweedie; fitted φ and power p
ft.φ, ft.p
```

For continuous responses with occasional gross outliers, **Student-t** is an
outlier-robust drop-in for `Normal()` on the identity link — the heavy tail bounds
each cell's influence, so a few extreme values barely move `β̂`. The degrees of
freedom `ν` sets the tail weight. A finite positive value on the marker fixes
it; the empty marker requests estimation. The scale `σ` is estimated:

```julia
fit_gllvm(Y; family = StudentTFamily(4.0), K = 2)   # ν = 4; fitted σ
fit_gllvm(Y; family = StudentTFamily(), K = 2)      # estimate ν and σ
gllvm(@formula(y ~ 1), Y, site_data; family = StudentTFamily(4.0), K = 2)
```

Student-t is a no-X surface for now: covariates and row effects are not admitted.

For counts that are under- or over-dispersed relative to Poisson,
**Conway–Maxwell–Poisson** estimates a shared dispersion exponent `ν`
(`ν = 1` ⇒ Poisson). The marker's `ν` is a tag payload (always estimated) —
unlike a numeric Student-t marker, whose `ν` fixes the degrees of freedom:

```julia
fit_gllvm(Y; family = COMPoisson(), K = 2)          # ν estimated
fit_gllvm(Y; family = COMPoisson(2.0), K = 2)       # same fit — marker ν never read
gllvm(@formula(y ~ 1), Y, site_data; family = COMPoisson(), K = 2)
```

COM-Poisson is a no-X surface: covariates and row effects are not admitted.
The twin has no CMP family, so there is no R-parity claim.

For proportions / cover in `[0,1]` with point masses at **0 and 1**,
**ordered-beta** (Kubinec 2023) estimates shared cutpoints `c0 < c1` and a
shared Beta precision `φ`. All three marker fields are tag payloads (always
estimated) — they are not Ordinal's `τ₁ = 0` pin:

```julia
fit_gllvm(Y; family = OrderedBeta(), K = 2)                 # c0, c1, φ estimated
fit_gllvm(Y; family = OrderedBeta(0.0, 2.0, 3.0), K = 2)    # same fit — tags never read
gllvm(@formula(y ~ 1), Y, site_data; family = OrderedBeta(), K = 2)
```

Ordered-beta is a no-X surface: covariates and row effects are not admitted.
The twin has no ordered-beta family, so there is no R-parity claim.

For **NB2, NB1, Beta, and the beta-binomial**, `fit_gllvm` defaults to
**per-species** dispersion (matching gllvmTMB). Shared dispersion remains available
via the named fitters `fit_nb_gllvm` / `fit_nb1_gllvm` / `fit_beta_gllvm` /
`fit_beta_binomial_gllvm`. Other dispersion families
still default to one shared parameter; to vary by species (or by groups —
gllvm's `disp.group`), use
`disp_group = :species` on `fit_gllvm`, or the matching `_grouped` driver with a
length-`p` `group` vector (default `1:p`):

```julia
fit_gllvm(Y; family = NegativeBinomial(), K = 2) # NB2 per-species r (default)
fit_nb_gllvm(Y; K = 2)                           # NB2 shared r (named)
fit_nb_gllvm_grouped(Y;  K = 2, group = group)   # NB2 r per custom group
fit_gllvm(Y; family = NB1(), K = 2)              # NB1 per-species φ (default)
fit_nb1_gllvm_grouped(Y; K = 2)                  # NB1 φ, default per-species
fit_beta_gllvm_grouped(Yp;    K = 2)             # Beta precision φ per species
fit_gllvm(Yb; family = BetaBinom(), K = 2, N = N) # beta-binomial per-species φ (N required)
fit_gamma_gllvm_grouped(Yc;   K = 2)             # Gamma shape α per species
fit_tweedie_gllvm_grouped(Y;  K = 2)             # Tweedie φ per species (shared power p)
```

For an intercept-only NB2 model, the formula interface retains the generic
fitter's **per-species size** parameters:

```julia
site_data = (site = collect(1:size(Y, 2)),)
f_formula = gllvm(@formula(y ~ 1), Y, site_data;
    family = GLLVModels.NegativeBinomial(), K = 2, g_tol = 1e-7, iterations = 800)
```

The `1` means a separate intercept for each species; `K` gives the latent rank.
Every supplied table column must have one entry per column of `Y`, even if the
formula does not use it. An empty `NamedTuple()` is also valid when there are no
covariates. The long-format call `gllvm(@formula(y ~ 1), long_data; ...)` sorts
species and site keys and requires a complete, duplicate-free grid. The original
NB2 data qualification checks both shapes against the native model; broader formula
models and the public R bridge still require separate qualification.

For continuous Gaussian data with **unequal residual variance across species**
(heteroscedastic; gllvm's default), `fit_gaussian_pervar_gllvm` estimates a
separate variance per species instead of the single shared `σ_eps` of
`fit_gaussian_gllvm`:

```julia
fpv = fit_gaussian_pervar_gllvm(Y; K = 2)   # per-species residual variances
```

For fixed effects, pass a complete `X` design of shape `(p, n, q)`. The fitter
profiles its `q` coefficients by GLS, adds no implicit intercept, and returns
them in `coef(fpv)` for a fit made with that design. See the per-species Gaussian
section of [Response families](response-families.md) for a worked design. Its
explicit `fixed_residual_sd` option retains unique and total diagonal variances
separately when the model has a known residual scale; it does not change defaults.

Displaying any fit prints a summary (family, dimensions, log-likelihood, AIC,
convergence):

```julia
fp            # rich REPL summary
```

## 2. Zero-inflated and two-part families

When zeros are *more* frequent than the count family predicts, use a
zero-inflated or hurdle/delta model. These carry **two** linear predictors — an
occurrence/zero part (`βz`) and a positive/count part (`βc`, with loadings `Λc`).
Hurdle-Poisson and the Delta families are reachable through no-X `fit_gllvm`;
the others still have dedicated drivers:

```julia
# Zero-inflated (a structural-zero mixture)
fzip  = fit_zip_gllvm(Y;  K = 2)              # zero-inflated Poisson
fzinb = fit_zinb_gllvm(Y; K = 2)             # zero-inflated NB2 (shared r)
# fzinbx = fit_zinb_gllvm_cov(Y; X = X, K = 2)  # ZINB + shared site-X (γz/γc; Λz=0)
fzib  = fit_zib_gllvm(Y;  K = 2, N = 10)     # zero-inflated Binomial — N trials (Int)

# Hurdle (Bernoulli occurrence × zero-truncated positive count)
fhp = fit_gllvm(Y; family = HurdlePoisson(), K = 2)
# named fitter remains: fit_hurdle_poisson_gllvm
fhn = fit_gllvm(Y; family = HurdleNB(), K = 2)
# named fitter remains: fit_hurdle_nb_gllvm

# Delta / two-part continuous (occurrence × positive continuous)
fdl = fit_gllvm(Yc; family = DeltaLogNormal(), K = 2)  # Bernoulli × lognormal
fdg = fit_gllvm(Yc; family = DeltaGamma(), K = 2)      # Bernoulli × Gamma
# named fitters remain: fit_delta_lognormal_gllvm / fit_delta_gamma_gllvm
fbh = fit_gllvm(Yp; family = BetaHurdle(), K = 2)   # Bernoulli × Beta (point mass at 0)
# named fitter remains: fit_beta_hurdle_gllvm
```

Delta-lognormal, Delta-Gamma, Hurdle-Poisson, Hurdle-NB, and Beta-hurdle are
**no-X** surfaces on `fit_gllvm` / `gllvm(@formula(y ~ 1), …)`. The Delta
markers' `σ` / `α`, the Hurdle-NB marker's `r`, and the Beta-hurdle marker's
`φ` are tag payloads (always estimated); `HurdlePoisson()` is an empty marker.
Covariates and row effects are not admitted on these routes.

`fit_zib_gllvm` needs the number of trials `N` as a scalar `Int` (a shared trial
count for all entries). The two-part fits expose `βz` (occurrence/zero logits),
`βc` (positive-part intercepts), `Λc` (positive-part loadings), and the relevant
dispersion (`r` for ZINB / hurdle-NB, `σ` for delta-lognormal, `α` for
delta-Gamma).

## 3. Ordination and post-fit

All single- and two-part fits share one post-fit API. The headline ecology
summary is `ordination`, which returns the site/species coordinates in the shared
latent space as a named tuple:

```julia
o = ordination(fp, Y)        # Y must match the matrix passed to the fitter
o.sites                       # n×K site scores (the ordination point cloud)
o.species                     # p×K species loadings (the "arrows")
o.rotation                    # K×K canonical (principal-axis) rotation
```

The two coordinate sets are also available directly:

```julia
getLV(fp, Y)                  # n×K conditional latent scores (site ordination)
getLoadings(fp)               # p×K species loadings, canonically rotated
getLoadings(fp; rotate = false)   # raw fitted Λ
rotation(fp)                  # the canonical K×K rotation alone
```

Latent factors are identified only up to a `K×K` orthogonal rotation, so
`getLV` / `getLoadings` / `ordination` apply a canonical principal-axis,
sign-fixed rotation by default (`rotate = false` returns the raw fitted
orientation).

Fitted values come from `predict`, on the link or the response scale; `fitted` is
the response-scale shorthand:

```julia
predict(fp, Y; type = :link)        # linear predictor η = β + Λẑ
predict(fp, Y; type = :response)    # μ = linkinv(η) (e.g. exp(η) for counts)
fitted(fp, Y)                       # == predict(fp, Y; type = :response)
```

The standard goodness-of-fit check is the **Dunn–Smyth** randomized quantile
residual — approximately `N(0,1)` under a correct model and comparable across
families (a normal Q–Q plot is the usual diagnostic):

```julia
residuals(fp, Y)                    # Dunn–Smyth (default)
residuals(fp, Y; type = :pearson)   # Pearson, for comparison
```

For discrete families the Dunn–Smyth randomization draws on an RNG; pass a seeded
`rng` (e.g. `residuals(fp, Y; rng = MersenneTwister(1))`) to reproduce.

Information criteria come off a single fit; `bic` needs the site count passed
explicitly (the fit does not store the data):

```julia
aic(fp)                # 2k − 2·logLik
bic(fp, size(Y, 2))    # k·log(n_sites) − 2·logLik
```

To choose `K`, `select_lv` sweeps `K = 1:Kmax`, fits each, and reports the
criteria:

```julia
sel = select_lv(Y; family = Poisson(), Kmax = 3)
sel.aic; sel.bic; sel.best_k; sel.best     # sel.best is the fitted model at best_k
```

Lower AIC/BIC is better; BIC penalises extra factors more and tends to pick a
smaller `K`. Use `criterion = :aic` to switch.

Finally, `simulate` draws a fresh response matrix from scalar-mean GLM-style,
Tweedie, and covariate fits (useful for posterior-predictive checks):

```julia
Ysim = simulate(fp, size(Y, 2))                 # p×n new draw
Ysim = simulate(fb, size(Y, 2); N = N)          # Binomial needs N
```

## 4. Inference

The non-Gaussian family fits share one confidence-interval entry,
`confint(fit, Y; method = ...)`, with three flavours:

```julia
confint(fp, Y; method = :wald)                              # observed-information Wald
confint(fp, Y; method = :profile, parm = "beta[1]")         # LRT-inversion profile
confint(fp, Y; method = :bootstrap, n_boot = 500, parallel = true)  # parametric bootstrap
```

Wald is one finite-difference-Hessian solve (cheapest, locally quadratic);
profile inverts the likelihood-ratio test (respects skew); bootstrap resamples
from the fitted model (no quadratic assumption, but slowest). `parm` subsets
terms by name (`"beta[1]"`, `"Lambda[2,1]"`, `"r"`) or by group (`"beta"`,
`"Lambda"`); `N = N` supplies Binomial trial counts.

All three methods accept the scalar-μ GLM families (`PoissonFit`, `BinomialFit`,
`NBFit`, `BetaFit`, `GammaFit`, `ExponentialFit`), the two-part families
(`ZIPFit`, `ZINBFit`, `ZIBFit`, the hurdle/delta fits), `OrdinalFit`, and
`GllvmCovFit` when the fitted design is supplied as `X = X`. Structural rows are
narrower: `QuadraticFit` and `RowEffectFit` have Wald/profile intervals but no
bootstrap route, while species-covariate, fourth-corner, RRR, and constrained
ordination fits use dedicated Wald helpers because their designs are not stored
inside the fit object. (The Gaussian `GllvmFit` uses the separate `confint` /
`profile_ci` / `bootstrap_ci` interface — see [Confidence
intervals](confidence-intervals.md).)

For the headline regression-style summary, `coef_table` wraps the Wald entry and
adds the `z` statistic and two-sided p-value:

```julia
coef_table(fp, Y)                       # term, estimate, std_error, z, pvalue, lower, upper
coef_table(fp, Y; parm = "beta", level = 0.90)
```

Any extra keywords flow through to `confint`, so `X = X` (covariate fits) and
`N = N` (Binomial) work unchanged.

## 5. Covariates and structure

Real surveys carry site environment and species traits. GLLVModels.jl exposes several
fixed-effect front ends, all taking the same `family` marker. The `(p, n, q)`
covariate array `X` follows the engine contract `X[t, s, k]` = covariate `k` for
species `t` at site `s`:

```julia
# Shared environmental slope γ (one coefficient per covariate, all species)
fit_gllvm_cov(Y; family = Poisson(), X = X, K = 2).γ

# Species-specific slopes B (one row per species)
fit_gllvm_speciescov(Y; family = Poisson(), X = X, K = 2).B

# Community row effects ρ_s (per-site intercepts; ρ[1] ≡ 0 reference)
fit_roweffect_gllvm(Y; family = Poisson(), K = 2).ρ
```

Row effects can instead be treated as **random** (gllvm's `row.eff = "random"`),
`ρ_s ~ N(0, σ_row²)`, which shrinks the per-site intercepts and estimates a single
row-effect SD rather than `n − 1` free intercepts:

```julia
fr = fit_row_random_gllvm(Y; family = Poisson(), K = 2)
fr.σ_row              # estimated row-effect SD
row_effects(fr, Y)    # per-site row-effect BLUPs ρ̂_s
```

The **fourth-corner** model structures the species × environment interaction
through measured traits — `Xenv` is the `n×q` site-by-covariate matrix, `TR` the
`p×r` species-by-trait matrix, and the fitted `q×r` coefficient matrix `C`
couples them (far fewer parameters than free per-species slopes):

```julia
fit_fourthcorner_gllvm(Y; family = Poisson(), Xenv = Xenv, TR = TR, K = 2).C
```

Predictor-informed latent-score means keep the latent axes random but shift
their mean by site covariates. The rotation-stable effect is the trait-scale
product `B_lv = Λ * alpha_lv'`:

```julia
fit_poisson_gllvm(Y; K = 1, X_lv = X_lv) |> extract_lv_effects
fit_ordinal_gllvm(Yo; K = 1, X_lv = X_lv) |> extract_lv_effects
```

For **constrained ordination**, the latent axes are driven by site covariates.
`fit_constrained_gllvm` (= `fit_concurrent_gllvm`, gllvm's `num.lv.c`) keeps a
residual random effect, `z_s ~ N(B' x_s, I_K)`; `fit_rrr_gllvm` (gllvm's
`num.RR`, reduced-rank regression) makes the axes a deterministic `z_s = B' x_s`
(no residual integral). Both take a 2-D `n×q` site-covariate matrix `X`:

```julia
fc = fit_constrained_gllvm(Y; family = Poisson(), X = X, K = 2)
fr = fit_rrr_gllvm(Y;         family = Poisson(), X = X, K = 2)
fr.B               # q×K constrained ordination axes (environment → latent)
fr.Λ               # p×K species loadings on those axes
getLV(fr, X)       # n×K deterministic site scores z_s = B' x_s
```

For a familiar R-`gllvmTMB`-style interface, the `@formula` front end maps a
formula plus a site-level data table onto the engine (v1: an intercept +
continuous main effects; dispatches to `fit_gaussian_gllvm` for `Normal()` and
`fit_gllvm_cov` otherwise):

```julia
gllvm(@formula(y ~ 1 + temp + depth), Y, site_data; family = Poisson(), K = 2)
```

For Gaussian trait-specific variances, add `pervar=true`. This route preserves
trait intercepts in `y ~ 1 + temp`, shared site slopes, and the zero-mean choice
`y ~ 0`. An explicit `fixed_residual_sd` can separate known residual variation
from estimated unique variance. See the executed
[per-variance examples](response-families.md#Gaussian-with-per-species-variance-—-fit_gaussian_pervar_gllvm).
For this decomposition, AGHQ requests retain exact Gaussian/Laplace and report
why unique effects are ineligible; `aghq=1` is quiet. The R bridge and intervals
remain under development.

## 6. Structured latent fields

The latent variables can themselves be given spatial or phylogenetic structure.

### Spatial SPDE fields

`fit_spde_latent_gllvm` makes the `K` latent variables **spatially smooth**
Matérn-GMRF fields over a triangular mesh (gllvm's `corLV = "spatial"`). Build a
mesh from the site coordinates with `spde_mesh_delaunay` (or `spde_mesh_grid`),
then fit with the observation locations `locs` (`M×2`):

```julia
nodes, tris = spde_mesh_delaunay(locs)          # mesh from site coordinates
fs = fit_spde_latent_gllvm(Y, nodes, tris, locs; family = Poisson(), K = 1)
fs.κ, fs.τ                                       # fitted Matérn range / precision params
```

The headline capability is **kriging** to new, unobserved locations:
`predict_spatial` finds the field mode from the training data, then interpolates
the fitted Matérn field to `new_locs`:

```julia
μ_new = predict_spatial(fs, Y, locs, new_locs; type = :response)   # p×M′
```

### Phylogenetic GLLVM

`fit_phylo_glm` fits a per-species phylogenetic random intercept correlated
across species by a tree, via an augmented-state joint Laplace over the sparse
phylogenetic precision. Build the augmented tree from a Newick string with
`augmented_phy` (its leaf order must match the rows of `Y`, and
`p == phy.n_leaves`):

```julia
phy = augmented_phy("((A:0.1,B:0.2):0.3,C:0.5);")   # p = phy.n_leaves
fph = fit_phylo_glm(Y, phy; family = Poisson())      # Y is p×n
fph.σ²_phy                                            # estimated phylogenetic variance
```

`family` accepts the usual markers (`Poisson()`, `NegativeBinomial()`,
`Binomial()`, …); supply `N` for Binomial. As `σ²_phy → 0` the fit reduces to the
independent-family marginal.

#### R-convention precision (`PrecisionPhy`) and the `correlation` estimand

`AugmentedPhy` (from `augmented_phy`/`make_phy`) is the native-Julia
representation: leaves-first, root included, raw branch lengths — the
convention `σ²_phy` fits under by default. R's `gllvmTMB` twin instead ships a
**precision** already reduced to one canonical form (root dropped,
internal-first/tips-last, unit root-to-tip height baked in) — the shape
`animal_*`/pedigree/kernel inputs arrive in on the R side, and what a future
bridge slice will hand across for phylo/animal parity fits.
`PrecisionPhy(phy::AugmentedPhy)` builds that R-convention bundle from a
native tree without any inversion (only a row/col drop and a permutation),
and feeds the identical `gaussian_marginal_loglik_sparse_phy` kernel as
`AugmentedPhy` itself:

```julia
phy = augmented_phy("((A:0.1,B:0.2):0.3,C:0.5);")
pp  = PrecisionPhy(phy)                 # correlation = false: same estimand as phy
recomputed, shipped, abs_diff = precision_logdet_check(pp)   # per-fit log-det checksum
```

Passing `correlation = true` — to `augmented_phy`, `make_phy`, or
`PrecisionPhy` — rescales the precision to **unit root-to-tip height** (R's
fit-path convention: `σ²_phy` then absorbs the raw tree scale). It requires
an ultrametric tree (root-to-tip heights equal within `sqrt(eps())`); a
non-ultrametric tree raises `GJL-GATE-PHYLO-NONULTRAMETRIC` rather than
silently fitting an estimand that has no R fit to pair against. `σ²_phy`
fitted with `correlation = true` is exactly `height` times the `σ²_phy`
fitted with `correlation = false`, for the same actual model — the two give
**identical log-likelihoods**, only the variance parameterisation differs.
Native Julia fits default to `correlation = false` (opt-in — a default flip
would silently change every existing user's `σ²_phy` by a tree-dependent
factor). This keeps established Julia estimates on the same scale while making
the R-compatible convention an explicit choice.

## 7. Choosing a family

Match the family to the response support and its mean–variance behaviour:

- **Counts.** Start with `Poisson()`. If the data are overdispersed (variance
  grows faster than the mean), move to `NegativeBinomial()` — NB2
  (`fit_nb_gllvm`, `Var = μ + μ²/r`) for quadratic overdispersion, NB1
  (`fit_nb1_gllvm`, `Var = μ(1+φ)`) when it grows linearly.
- **Excess zeros.** If zeros are more common than the count family predicts,
  use a zero-inflated model (`fit_zip_gllvm`, `fit_zinb_gllvm`) for a structural-
  zero mixture, or a hurdle model (`fit_hurdle_poisson_gllvm`,
  `fit_hurdle_nb_gllvm`) when presence and abundance are governed by distinct
  processes.
- **Presence/absence and trials.** `Binomial()` (with `N` trials; `N ≡ 1` is
  Bernoulli for presence/absence).
- **Proportions in (0,1).** `Beta()`. If there are point masses at 0 (or at 0 and
  1), use `fit_gllvm(Y; family = BetaHurdle())` (zero) or
  `fit_gllvm(Y; family = OrderedBeta())` (zeros and ones).
- **Positive continuous (biomass, size).** `Gamma()` (or `Exponential()` with no
  dispersion). With exact zeros mixed in, use a delta model
  (`fit_delta_gamma_gllvm`, `fit_delta_lognormal_gllvm`) or `fit_tweedie_gllvm`
  (which estimates the power `p` and handles the zeros in one model).
- **Ordered categories.** `Ordinal()` (proportional-odds cumulative logit).

When a Laplace fit looks unstable (a degenerate Hessian, implausible dispersion),
the variational (`fit_*_gllvm_va`) drivers optimise an ELBO instead — slower but
steadier; see [Response families](response-families.md).

See also: [Get started](quickstart.md) · [Working with a fit](working-with-a-fit.md) ·
[Response families](response-families.md) · [Structured dependence](structured-dependence.md) ·
[Confidence intervals](confidence-intervals.md) · [Reference](api.md).
