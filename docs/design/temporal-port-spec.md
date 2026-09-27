# Temporal port spec: gllvmTMB temporal models at P1 into GLLVModels.jl

Status: DRAFT design spec, no code. Written 2026-09-27 against the signed P1
boundary (vault D-295: temporal is inside P1; port R's semantics; Julia designs
stay documented extras; twin R's public API at R's scope and no further).

Pins:

- R side, P1 = gllvmTMB commit `9539352f66f2db2cc26b1c393e67212a359b60c9`
  (`DESCRIPTION` Version 0.7.1). Every `R/...`, `src/...`, `tests/...`,
  `docs/...` citation below is `path:line` at that commit, read with
  `git show 9539352f6:<path>`. Nothing in the R repo was checked out or edited.
- Julia side, `origin/main` = `d286ac4c5`.
- The eight exports under contract (draft PR #526, class `required_core`,
  signed by D-297): `temporal_dep`, `temporal_indep`, `temporal_latent`,
  `extract_temporal`, `forecast_temporal`, `bootstrap_temporal`,
  `profile_temporal`, `compare_temporal`. The S3 method `update.gllvmTMB_multi`
  (R/methods-gllvmTMB.R:14) is `required_core` too and depends on these.

The plan on file recorded R's temporal scope as "Gaussian, rank-1 latent score,
AR1/OU correlation; extract, forecast, bootstrap, profile, compare". Section 1
checks that against R. The short verdict: correct on family, rank and
correlation, but incomplete. R has three covariance modes (`indep`, `dep`,
`latent`), of which the rank-one latent score is one; the four helpers are bounded to
unreplicated Gaussian `temporal_indep` fits only; and R admits four
cross-source cells and ordinary `unit` / `unit_obs` composition that the plan
did not mention. Section 7 asks the maintainer how much of that is in scope.

## 0. Reading guide

- Section 1: the model R fits, its parameterisation and identifiability.
- Section 2: the public API contract for the eight exports.
- Section 3: the Julia design.
- Section 4: block-by-block map of the 91 R test blocks.
- Section 5: receipts and case-map rows.
- Section 6: symbolic alignment table.
- Section 7: open questions with a recommendation each.
- Section 8: estimate.

Notation: `g` indexes series, `t` indexes occasions, `j` indexes traits
(`p` traits), `r` indexes replicates, `o` indexes long-format rows. A
"state" is one `(series, occasion)` pair; `S` is the number of states.

## 1. The model R fits at P1

### 1.1 Data layout and the state index

R fits long data with one row per `(series, occasion, trait[, replicate])`.
The temporal term is captured by a pre-pass before generic formula
desugaring (R/gllvmTMB.R:1138-1141 calls `.parse_temporal_latent_formula`,
R/temporal.R:100-453). The pre-pass:

- accepts exactly one temporal term (R/temporal.R:121-123);
- requires the bar `0 + trait | series` literally (R/temporal.R:340-351);
- requires `time` to be a bare column of finite numerics (R/temporal.R:326-360),
  integer-valued for AR1 (R/temporal.R:361-363);
- requires complete `series` and `trait` identifiers (R/temporal.R:364-366),
  at least three traits (R/temporal.R:367-370), and at least three strictly
  ordered occasions per series (R/temporal.R:371-377);
- builds the state table `pair_table(pair_id, series, time)` ordered by
  series then time (R/temporal.R:378-388);
- unreplicated workflow: one complete trait panel per state, no duplicate
  `(state, trait)` rows (R/temporal.R:408-417). Replicated workflow
  (`replicate = <col>`): each replicate carries a complete panel and every
  state has at least two measurements (R/temporal.R:391-407);
- strips the marker to `0` so the generic parser never sees it
  (R/temporal.R:428-439) and returns `spec` (R/temporal.R:440-452).

The engine receives the state index as `temporal_state_id` (one per row),
`temporal_predecessor` (one per state, `-1` at a series start),
`temporal_gap` (integer AR1 gaps) and `temporal_elapsed` (numeric OU gaps)
(R/fit-multi.R:5508-5545; validated in C++ at src/gllvmTMB.cpp:621-653).
Irregular spacing is therefore admitted in both structures: AR1 uses the
integer gap as a power, OU uses the elapsed difference.

### 1.2 Fit-level admission

A temporal fit must be `engine = "tmb"`, `REML = FALSE`, `estimator = "ml"`,
`family = gaussian()` with identity link (R/gllvmTMB.R:1144-1147), Laplace
integration with `aghq = FALSE` (R/gllvmTMB.R:1148-1155), and no
`known_V`, `mesh`, `phylo_vcv`, `phylo_tree`, column-coefficient or
structured-rho argument (R/gllvmTMB.R:1156-1163). Other covariance providers in
the formula are refused (R/temporal.R:296-303) except the four replicated AR1
`temporal_indep()` cross-source cells (`kernel_indep`, `phylo_indep`,
`animal_indep`, `spatial_indep`; R/temporal.R:159-166, R/temporal.R:419-426).
Ordinary `unit` and `unit_obs` terms (`indep`, `dep`, `latent`) are admitted
beside the temporal term (R/temporal.R:154-158) with a partition check when a
stable-unit component is present (R/gllvmTMB.R:1253-1264).

### 1.3 Latent process (C++)

The dedicated temporal source is src/gllvmTMB.cpp:1770-1842. Parameters
(src/gllvmTMB.cpp:1197-1200): scalar `theta_temporal_time`, vector
`theta_temporal_rr` (packed lower-triangular loadings), vector
`theta_temporal_diag` (length `p`), and innovation matrices `z_temporal`
(`rank × S`) and `q_temporal` (`p × S`). Transforms:

- AR1 persistence `phi = (1 - 1e-6) * tanh(theta_temporal_time)`
  (src/gllvmTMB.cpp:1795);
- OU rate `kappa = exp(theta_temporal_time)` (src/gllvmTMB.cpp:1796);
- temporal SDs `exp(theta_temporal_diag)` (src/gllvmTMB.cpp:1797), so the
  reported variance is `psi_j = exp(2 theta_temporal_diag_j)`
  (R/temporal.R:518);
- `Lambda_temporal = gll_unpack_rr_loadings(theta_temporal_rr, p, rank)`,
  length `p*rank - rank*(rank-1)/2` (src/gllvmTMB.cpp:1787-1794).

For each state `s` with predecessor `s'` (src/gllvmTMB.cpp:1798-1835):

```
a_s   = phi^gap_s                      (AR1, integer power, cpp:1804, 62-75)
      = exp(-kappa * elapsed_s)        (OU, cpp:1806)
sd_s  = sqrt(1 - a_s^2)                (AR1, cpp:1813)
      = sqrt(1 - exp(-2 kappa elapsed_s))   (OU, stable helper, cpp:1812, 77-90)
z_s   = z_innov_s                      at a series start
      = a_s z_s' + sd_s z_innov_s      otherwise            (rank > 0, cpp:1815-1824)
q_js  = psi_j^{1/2} q_innov_js         at a series start
      = a_s q_js' + psi_j^{1/2} sd_s q_innov_js  otherwise   (unique == 1, cpp:1825-1834)
```

with standard-normal priors on every innovation. The marginal of the states
is therefore stationary: within a series, `Cov(z_s, z_u) = K(t_s, t_u)` and
`Cov(q_js, q_ju) = psi_j K(t_s, t_u)`, where `K = phi^{|t-u|}` (AR1) or
`exp(-kappa |t-u|)` (OU); across series the covariance is zero. R's own
forecast code uses exactly this closed form (R/temporal-forecast.R:147-162).

The states enter the linear predictor (src/gllvmTMB.cpp:3102-3112):

```
eta_o = x_o' beta + sum_k Lambda_temporal[j_o, k] z_{k, s_o} + q_{j_o, s_o}
```

and the Gaussian observation density is `y_o ~ N(eta_o, sigma_eps^2)`
(src/gllvmTMB.cpp:3294). `log_sigma_eps` stays free for a temporal fit; the
per-row auto-suppression of `sigma_eps` is bypassed for the unreplicated
temporal workflow (R/fit-multi.R:6959-6960).

### 1.4 The three modes

Mode and rank are set at R/fit-multi.R:5541-5544: `temporal_rank = p` for
`dep`, `d` (= 1) for `latent`, `0` for `indep`; `unique` is `TRUE` for
`indep`, the user flag for `latent`, `FALSE` for `dep` (R/temporal.R:38-39).
The resulting trait block `Sigma_T` (the covariance of the temporal effect
across traits at one state) is:

| Mode | `Sigma_T` | Free temporal coordinates |
| --- | --- | --- |
| `temporal_indep` | `diag(psi_1..psi_p)` | `theta_time`, `theta_diag[1:p]` |
| `temporal_dep` | `L L'`, `L` lower-triangular `p × p` | `theta_time`, `theta_rr[1:p(p+1)/2]` |
| `temporal_latent(d=1, unique=FALSE)` | `lambda lambda'` | `theta_time`, `theta_rr[1:p]` |
| `temporal_latent(d=1, unique=TRUE)` | `lambda lambda' + diag(psi)` | `theta_time`, `theta_rr[1:p]`, `theta_diag[1:p]` |

Parameter maps confirm the table: unused blocks are mapped off at
R/fit-multi.R:6421-6435, and `z_temporal` / `q_temporal` are declared random
only when their rank or `unique` flag is active (R/fit-multi.R:7262-7263).

### 1.5 Marginal likelihood (the object Julia must reproduce)

Everything above is linear and Gaussian, so the TMB Laplace objective equals
the exact marginal negative log-likelihood. With `u = vec(states × traits)`
and `Z` the row-to-`(state, trait)` incidence matrix (identity in the
unreplicated workflow), the marginal is

```
y ~ N( X beta,  Z (K_blockdiag ⊗ Sigma_T) Z' + sigma_eps^2 I )
```

where `K_blockdiag` is block diagonal over series with the AR1 or OU
correlation of that series' occasions. R's tests check the compiled objective
against an independently written dense version of this expression to
`2e-6` and the `theta_time` gradient to `2e-5` over all eight AR1/OU cells
(tests/testthat/test-temporal-sixth-source-oracles.R:214-248). The 2026-09-09
after-task note states the same covariance,
`V = J_unit ⊗ Sigma_B + K_temporal ⊗ Sigma_T + J_unit_obs ⊗ Sigma_W + R`
(docs/dev-log/after-task/2026-09-09-temporal-sixth-source.md:19-32).

Consequence for the port: no Laplace approximation, no mode finding, and no
`_laplace_mode` involvement. The port is a closed-form Gaussian marginal with
a parametric correlation kernel.

### 1.6 Identifiability and the sign convention

- `phi` is confined to `(-1+1e-6, 1-1e-6)` by the `tanh` map; `kappa > 0`
  by `exp`. Negative AR1 persistence is admitted and flips sign at odd lags
  (tests/testthat/test-temporal-sixth-source-oracles.R:342-357).
- `indep`, unreplicated: `psi_j` and `sigma_eps^2` are separated only by
  the temporal correlation; at `phi = 0` they are confounded. R states this
  as the reason the cross-source cells require replication
  (R/temporal.R:421-425). R does not guard the temporal-only case; the port
  should not either (R semantics), but the spec records it.
- `latent`, rank one: the likelihood is invariant to a joint sign flip of
  `lambda` and `z`. R does not constrain the TMB coordinates; it chooses a
  public orientation at report time: the first loading's sign unless it is
  below `1e-8` of the largest, then the first largest in trait order
  (R/temporal.R:466-483), applied in `extract_ordination`
  (R/extractors.R:514-537). `extract_temporal()` returns the raw
  `Lambda_temporal` without that flip (R/temporal.R:511-514).
- `latent(unique=TRUE)` needs `p >= 3` for the factor-plus-diagonal split;
  this is the reason behind the three-trait floor (R/temporal.R:367-370),
  which R applies to every mode.
- `dep`: `L` is a free lower-triangular factor, so `Sigma_T = L L'` is
  identified while `L` is identified only up to column signs. Receipts must
  compare `Sigma_T`, not `L`.
- Rank above one is refused (R/temporal.R:22-25, 329-332). Only Gaussian
  identity is admitted (section 1.2). So the admitted set is: family
  Gaussian; ranks 0 (`indep`), 1 (`latent`), `p` (`dep`); structures AR1, OU;
  workflows unreplicated and replicated.

### 1.7 What the lifecycle methods do on a temporal fit

- `predict(newdata=)` is refused (R/methods-gllvmTMB.R:2795-2800); in-sample
  `predict` returns `eta` with the series, time, replicate and trait columns
  (R/methods-gllvmTMB.R:2844-2848).
- `simulate(condition_on_RE = TRUE)` draws around the fitted `eta`;
  `condition_on_RE = FALSE` redraws the recursive states from their
  stationary prior plus every ordinary tier (R/methods-gllvmTMB.R:1535-1560,
  1653-1709).
- `update()` replays the saved public call with named overrides
  (R/methods-gllvmTMB.R:14-35).
- `getLV` / `extract_ordination` return scores only for the rank-one latent
  cell, with the sign anchor and a `temporal_index` attribute
  (R/extractors.R:514-537, R/output-methods.R:206-229); `indep` and `dep`
  return `NULL` there and expose their covariance through
  `extract_temporal()`.
- `confint`, `bootstrap_Sigma`, `ordination_uncertainty`, `loading_ci`,
  `loading_profile`, `profile_targets`, `extract_repeatability` and the other
  iid-score routes are refused with class
  `gllvmTMB_temporal_inference_unsupported` (R/temporal.R:455-464; callers
  listed at R/z-confint-gllvmTMB.R:1617, R/bootstrap-sigma.R:214,
  R/ordination-uncertainty.R:196 and twelve more).
- `select_lv` is refused on a temporal formula (R/select-lv.R:160-166).

### 1.8 What R claims about status

The validation-debt register row `TEMP-06-01` is `partial`
(docs/design/35-validation-debt-register.md:75): "Independently authored
dense Gaussian NLL and central-gradient checks cover all eight AR1/OU cells";
the helpers "each have a bounded temporal-only Gaussian `temporal_indep()`
contract"; and "Generic new-data prediction, generic intervals/profiles,
automatic selection, and broad bootstrap routes reject. The helper routes do
not establish calibration, coverage, source-pair support, or general
recovery." The four cross-source rows `TEMP-06-02..05` are `partial`; only the
kernel pair passed its recovery gate, the phylo, animal and spatial pairs are
recorded as failed engineering gates
(docs/design/35-validation-debt-register.md:76-79). No Wald interval is
claimed anywhere; `profile_temporal` is "not a calibrated interval"
(R/temporal-profile.R:4-5); `forecast_temporal`'s `se.fit` "is not a
calibrated prediction interval" (R/temporal-forecast.R:6-7);
`bootstrap_temporal` returns refit rows and no interval
(R/temporal-bootstrap.R:17-19); `compare_temporal` is an AIC table with "no
likelihood-ratio test" (R/temporal-selection.R:3). The port claims the same
and no more.

## 2. Public API contract (eight exports)

Julia twins keep R's names and argument names. Where R takes a bar formula or
a bare column, Julia takes a quoted `Expr` or a `Symbol`, following the
convention already used for `dep(0 + trait | g)` in `src/formula.jl:485-505`.
R's `cli` messages are rendered in Julia as `ArgumentError` text with the cli
markup removed; the words the R tests match on (section 4) are kept verbatim.

### 2.1 Constructors

R (R/temporal.R:56-59, 65-68, 93-98):

```
temporal_indep(formula, time, structure = "ar1", replicate = NULL)
temporal_dep(formula, time, structure = "ar1", replicate = NULL)
temporal_latent(formula, time, d = 1, structure = "ar1", replicate = NULL, unique = FALSE)
```

Julia:

```
temporal_indep(formula::Expr, time::Symbol; structure::Symbol = :ar1, replicate = nothing)
temporal_dep(formula::Expr, time::Symbol; structure::Symbol = :ar1, replicate = nothing)
temporal_latent(formula::Expr, time::Symbol; d::Integer = 1, structure::Symbol = :ar1,
                replicate = nothing, unique::Bool = false)
```

Return: `TemporalTerm` (fields `formula, time, mode, d, unique, structure,
replicate`), mirroring R's marker list and its classes
`gllvmTMB_temporal`, `gllvmTMB_temporal_<mode>` (R/temporal.R:34-42).
Constructor validation and messages (R/temporal.R:10-32):

| Condition | R message (cli stripped) |
| --- | --- |
| formula is not `lhs \| rhs` | "A temporal covariance term requires a formula of the form `0 + trait \| series`." |
| `time` not a bare name | "`time` must be a bare column name." |
| `replicate` not NULL/bare name | "`replicate` must be NULL or a bare column name." |
| `latent` with `d != 1` | "`temporal_latent()` currently supports rank one only (`d = 1`)." |
| `structure` not `ar1`/`ou` | "`structure` must be either \"ar1\" or \"ou\"." |
| `unique` not a single logical | "`unique` must be TRUE or FALSE." |

Every temporal error in R appends the action line "See `temporal_latent()`
for the admitted temporal workflow." (R/temporal.R:1-8). Julia appends the
same sentence.

### 2.2 Fit-time validation (the pre-pass)

Applied when the term meets the data (R/temporal.R:100-453). Messages, in
the order R checks them:

| Check | Message |
| --- | --- |
| more than one temporal term | "Only one temporal covariance term is supported in a model." |
| another covariance source (outside the four replicated AR1 `indep` cells) | "A temporal covariance term cannot be combined with another covariance source in this version." plus "Found source provider(s): ..." |
| response missing or NA | "`temporal_latent()` requires complete Gaussian response values." |
| bar not `0 + trait \| series` | "`temporal_latent()` requires one trait-intercept block `0 + trait \| series`." |
| columns absent | "Temporal data are missing column(s): ..." |
| time not finite numeric | "`time` must contain finite numeric occasions." |
| AR1 time not integer-valued | "AR1 `time` must contain finite integer-valued occasions." |
| NA in series/trait | "Temporal series and trait identifiers must be complete." |
| fewer than 3 traits | "`temporal_latent()` requires at least three traits." |
| fewer than 3 ordered occasions in a series | "Each temporal series needs at least three strictly ordered occasions." |
| replicate column absent / NA | "`replicate` must name a column in `data`." / "`replicate` must be complete." |
| duplicate replicated rows | "Temporal data contain duplicate series--occasion--replicate--trait rows." |
| incomplete replicate panel | "Each temporal replicate must contain one complete trait panel." |
| fewer than 2 measurements at a state | "Replicated temporal data require at least two measurements at every occasion." |
| duplicate unreplicated rows | "Repeated temporal observations require `replicate =` to distinguish measurements." |
| incomplete unreplicated panel | "Unreplicated temporal data require one complete trait panel at every occasion." |
| cross-source cell without replicated AR1 | "The temporal cross-source cell requires replicated AR1 observations." |

Fit-level gates (R/gllvmTMB.R:1144-1163): non-Gaussian family or non-identity
link gives "`temporal_latent()` currently requires the native Gaussian
identity-link ML route."; VA or AGHQ gives "`temporal_latent()` currently
requires native TMB Laplace integration." (Julia has no VA/AGHQ door for this
fit; the message is kept for the refused keyword). A stable-unit component
whose partition differs from `series` gives "The temporal `series` column must
have the same partition as `unit` when a stable unit covariance component is
included." (R/gllvmTMB.R:1257-1263).

### 2.3 `extract_temporal(fit)`

R (R/temporal.R:495-542) returns a list:

- `parameters`: one-row table `mode, structure, workflow, n_series, n_pairs`;
- `time`: one-row table `parameter ∈ {"phi", "ou_rate"}, value` on the
  persistence or rate scale (R/temporal.R:503-507);
- `pair_index`: the state table `pair_id, series, time`;
- `loadings`: `Lambda_temporal` (`p × rank`, trait row names) for `dep` and
  `latent`, `NULL` for `indep` (R/temporal.R:511-514), raw (no sign flip);
- `variance`: table `trait, value, component` with component
  `temporal_indep_variance` (`indep`) or `temporal_Psi_variance`
  (`latent(unique=TRUE)`), else an empty table (R/temporal.R:515-529).

Refusal on a non-temporal fit: "`extract_temporal()` requires a fit made with a
temporal covariance term." Julia returns a `NamedTuple` with the same five
names; each table is a `NamedTuple` of vectors (Tables.jl-compatible, no
DataFrames dependency); `loadings` is `Matrix{Float64}` or `nothing`.

### 2.4 `forecast_temporal(object, newdata, se.fit = FALSE)`

R (R/temporal-forecast.R:35-136). Contract: unreplicated Gaussian
`temporal_indep` fit with the temporal source alone; `newdata` is a complete
trait panel at future occasions of already fitted series. Computation:
residual `r = y - X beta_hat` on the fitted rows; `C_oo`, `C_ff`, `C_of`
built from the fitted `phi`/`kappa` and `psi` (R/temporal-forecast.R:138-164,
with `sigma_eps^2` on the diagonal of `C_oo` and `C_ff` only);
`est = X_new beta_hat + C_of' C_oo^{-1} r`;
`se.fit = sqrt(diag(C_ff - C_of' C_oo^{-1} C_of))` (R/temporal-forecast.R:125-134).
Return: `newdata` rows in input order plus `est` and, if requested, `se.fit`.

Refusals (message, R error class):

| Case | Message | Class |
| --- | --- | --- |
| not a temporal fit | "`forecast_temporal()` requires a native temporal `gllvmTMB()` fit." | (none) |
| `newdata` not a data frame | "`newdata` must be a data frame." | |
| `se.fit` not logical | "`se.fit` must be TRUE or FALSE." | |
| replicated fit | "`forecast_temporal()` does not yet support replicated temporal panels." | `gllvmTMB_temporal_forecast_replicated` |
| mode not `indep` | "`forecast_temporal()` currently supports `temporal_indep()` only." | `gllvmTMB_temporal_forecast_mode` |
| non-Gaussian rows | "`forecast_temporal()` currently requires a Gaussian identity-link temporal fit." | |
| another RE tier active | "`forecast_temporal()` currently supports the temporal source by itself." | `gllvmTMB_temporal_forecast_composed` |
| missing columns | "`newdata` is missing required temporal column(s): ..." | |
| NA keys | "`newdata` needs non-missing series, time, and trait values." | |
| non-numeric / non-finite time | "The temporal forecast time column must be numeric." / "... must contain finite values." | |
| AR1 non-integer time | "AR1 temporal forecasts require integer occasions." | |
| unknown series | "`forecast_temporal()` supports existing series only." | `gllvmTMB_temporal_forecast_new_series` |
| unknown trait | "`newdata` names unknown trait(s): ..." | |
| duplicate rows | "`newdata` has duplicate series--time--trait rows." | |
| incomplete panel | "`newdata` must contain a complete trait panel at every future series--occasion pair." | `gllvmTMB_temporal_forecast_panel` |
| not strictly future | "Temporal forecast occasions must be strictly after each series' fitted occasions." | `gllvmTMB_temporal_forecast_not_future` |
| `C_oo` not PD | "The fitted temporal response covariance was not positive definite for forecasting." | |
| negative conditional variance | "The temporal forecast produced a negative conditional variance." | |

Julia: `forecast_temporal(fit, newdata; se_fit::Bool = false)`; `newdata`
any Tables.jl table with the series, time and trait columns; returns a
`NamedTuple` of columns (`newdata` columns, `est`, optionally `se_fit`). The
R error classes are mirrored as a `kind::Symbol` field on a
`TemporalContractError <: Exception` so tests can match on class as R does.

### 2.5 `profile_temporal(object, level = 0.95, ...)`

R (R/temporal-profile.R:14-29): same fit contract as forecast (unreplicated
Gaussian `temporal_indep` alone); refusals "`profile_temporal()` requires a
native temporal fit.", "`profile_temporal()` currently supports unreplicated
Gaussian `temporal_indep()` fits only.", "`profile_temporal()` currently
requires the temporal source by itself.". It profiles `theta_temporal_time`
with all other coordinates re-optimised, through `tmbprofile_wrapper`
(R/profile-ci.R:372-456): threshold `crit = qchisq(level, 1) / 2` on the
negative log-likelihood scale (R/profile-ci.R:41-48), search budget
`ytol = crit + 1` (R/profile-ci.R:66-68), step `ystep = 0.5`, bounds found by
linear interpolation of the deviance trace with asymptotic (infinite) versus
truncated (`NA`) endpoints (R/profile-ci.R:188-235). Return: named vector
`estimate, lower, upper` on the `phi` or `kappa` scale; `NA` bounds mean no
endpoint was established (R/temporal-profile.R:10-12).

Julia: `profile_temporal(fit; level = 0.95, ystep = 0.5, ytol = nothing,
parm_range = (-Inf, Inf))` returning `(estimate, lower, upper)` with
`missing` where R returns `NA` and `-Inf`/`Inf` where R returns an
asymptotic bound (section 7, Q4). The profile is computed by a fixed-`theta`
re-optimisation of the remaining coordinates with the same LBFGS settings as
the fit, walking outward in `ystep` on the `theta` scale until the deviance
exceeds `ytol`, then interpolating the crossing of `crit` exactly as
R/profile-ci.R:188-235 does. Because the deviance trace itself is exact
(closed form), the only expected R-versus-Julia difference is the trace grid,
which is why receipts on `lower`/`upper` carry a looser tolerance than the
estimate (section 5).

### 2.6 `bootstrap_temporal(object, n_boot = 100L, seed = NULL)`

R (R/temporal-bootstrap.R:21-75): same fit contract, with refusals
"`bootstrap_temporal()` requires a native temporal fit.",
"`bootstrap_temporal()` currently requires the temporal source by itself."
(class `gllvmTMB_temporal_bootstrap_composed`), "`bootstrap_temporal()`
currently supports unreplicated Gaussian `temporal_indep()` fits only.",
"`n_boot` must be a positive integer.", "`seed` must be one non-negative
whole number or `NULL`.". Algorithm: draw `n_boot` seeds, for each draw an
unconditional response (`simulate(..., condition_on_RE = FALSE)`: redraw the
stationary states and the residual), refit with `update(object, data =)`,
and record `replicate, seed, convergence, objective, time_estimate, error`
where `time_estimate` is `phi` or `kappa` (R/temporal-bootstrap.R:1-8) and
failed or non-converged refits keep their message
(R/temporal-bootstrap.R:59-74). The caller's RNG state is restored
(R/temporal-bootstrap.R:45-53).

Julia: `bootstrap_temporal(fit; n_boot::Integer = 100, seed = nothing)`;
seeds drawn from a `StableRNG(seed)` so a run is reproducible across Julia
versions; returns a `NamedTuple` of the six columns; `convergence` is `0`
when the refit verdict is `:converged`, else the Julia stopping reason code,
and `error` holds the exception message. R's exact seeds and draws are not
reproducible in Julia (different generators); section 5 says what a receipt
compares instead.

### 2.7 `compare_temporal(...)`

R (R/temporal-selection.R:5-29): at least two named temporal fits, each
with the temporal source alone and identical response rows; returns
`model, logLik, df, AIC, convergence` where `logLik = -opt$objective`,
`df = length(opt$par)` and `AIC` from `stats::AIC`; no test statistic of
any kind. Refusals: "Supply at least two named temporal fits.", "Temporal
candidates must be named.", "Every candidate must be a native temporal fit.",
"`compare_temporal()` currently requires the temporal source by itself."
(class `gllvmTMB_temporal_selection_composed`), "Temporal candidates must
use identical response rows and ordering.".

Julia: `compare_temporal(; fits...)` with keyword names as model labels,
returning a `NamedTuple` of the five columns. `df` counts the free outer
coordinates (`beta`, `theta_time`, `theta_rr`, `theta_diag`, `log_sigma_eps`),
which is what R's `opt$par` holds (random effects are not in it).

### 2.8 Lifecycle methods twinned at R's scope

`update` (named overrides, replay), in-sample `predict`, `simulate` with
`condition_on_RE`, `getLV`/`extract_ordination` for the latent cell with the
sign anchor and `temporal_index`, and the refusal set of section 1.7. These
are not among the eight exports but the R tests exercise them on temporal
fits, so the twins need them to close the rows (section 4).

## 3. Julia design

### 3.1 Principle

The temporal likelihood is the closed-form Gaussian marginal of section 1.5.
GLLVModels.jl already has a dense closed-form fitter for
`V = sum_s P_s C_s P_s' ⊗ B_s + sigma_eps^2 I` with the same three trait
modes: `fit_gaussian_sources` (`src/source_fit.jl:290-392`) over
`SourceCovariance` objects (`src/source_fit.jl:18-58`), with trait blocks
built by `_source_trait_covariances` (`src/source_fit.jl:71-93`) and the
dense NLL in `_gaussian_sources_nll` (`src/source_fit.jl:108-135`). R's
temporal model is that model with one difference: the source correlation `C`
is not fixed, it is `K(theta_time)`. The port adds a parametric source
beside the fixed ones without editing the fixed-source code.

Nothing in the Laplace stack is involved. The spec does not touch
`_laplace_mode` (`src/families/laplace.jl:102`), `src/families/mixed.jl`,
`src/grouped_dispersion.jl`, `src/model_selection.jl`, `src/cv.jl`, or the
Normal branch of `gllvm()` in `src/formula.jl:234-250`.

### 3.2 New files

| File | Contents |
| --- | --- |
| `src/temporal.jl` | `TemporalTerm`; `temporal_indep`, `temporal_dep`, `temporal_latent`; `_parse_temporal_term(term, data; trait)` (the pre-pass of R/temporal.R:100-453, returning `TemporalSpec` with `pair_table`, `state_id`, `predecessor`, `gap`, `elapsed`, `mode`, `structure`, `rank`, `unique`, `workflow`, `series_col`, `time_col`, `replicate_col`); `TemporalContractError`. |
| `src/temporal_likelihood.jl` | `_temporal_correlation(spec, theta_time)` (block-diagonal `K` over series, AR1 integer power or OU exponential, on the marginal scale); `_temporal_trait_block(spec, theta_rr, theta_diag, p)` reusing `unpack_lambda` / `rr_theta_len` (`src/packing.jl`); `temporal_marginal_nll(theta, y, X, spec, sources)` building `V` per series (dense per-series Cholesky when no cross-series source is present) or as one dense matrix (composition with `unit`/`unit_obs` sources); `temporal_marginal_loglik` as the positive twin. |
| `src/temporal_fit.jl` | `TemporalGaussianFit <: StatsAPI.StatisticalModel`; `fit_temporal_gllvm(long_data; formula, temporal::TemporalTerm, trait = :trait, unit = nothing, unit_obs = nothing, structure = Expr[], start = nothing, g_tol = 1e-6, iterations = 500)`; Optim LBFGS with `autodiff = :forward`, the same verdict fields as `GaussianSourcesFit` (`converged, gradient_norm, hessian_min_eigenvalue, stopping_reason`); `loglikelihood`, `dof`, `nobs`, `aic`, `bic` methods for the new type (methods added from this file; `AnyGllvmFit` in `src/postfit.jl:9` is left alone). |
| `src/temporal_methods.jl` | `extract_temporal`, `forecast_temporal`, `profile_temporal`, `bootstrap_temporal`, `compare_temporal`, `update`, in-sample `predict`, `simulate`, `getLV`/`extract_ordination` for the latent cell, the sign anchor `_temporal_report_sign`, and the refusal methods (`confint`, `bootstrap_ci`, `ordination_uncertainty`, `select_lv` on a temporal fit throw the `gllvmTMB_temporal_inference_unsupported` message). |
| `test/test_temporal_*.jl` | one file per twinned R file (section 4). |
| `docs/src/temporal.md` | reference page; rows in `docs/design/capability-status.md` and the case map. |

Include order: after `source_fit.jl` and `packing.jl` (needed) and after
`postfit.jl` (so `aic`/`bic` generics exist), before `formula.jl`. Exports
added to the single export block of `src/GLLVModels.jl:205-369`: the eight
names plus `TemporalTerm`, `TemporalGaussianFit`, `fit_temporal_gllvm`.

### 3.3 Parameter vector and coordinates

`theta = [beta (q); theta_time (1); theta_rr (p*rank - rank(rank-1)/2); theta_diag (p, only when unique); log_sigma_eps (1)]`,
named `b_fix[i]`, `theta_temporal_time[1]`, `theta_temporal_rr[k]`,
`theta_temporal_diag[j]`, `log_sigma_eps[1]` so that an R `opt$par` vector
from the receipt can be placed on the Julia objective coordinate by
coordinate (this is how the existing B1 receipt test evaluates at R's
coordinates with `start = r_theta, iterations = 0`,
`test/test_destination_b_b1_cluster_paired_fit.jl`). The transforms are the
ones of section 1.3. Starting values follow R: `theta_time = 0`,
`theta_rr = init_rr_theta`, `theta_diag = 0`, `beta` from OLS
(R/fit-multi.R:5808-5819 and the source_fit start rule).

### 3.4 Likelihood evaluation

Unreplicated, temporal source alone: `V` is block diagonal over series with
blocks `K_g ⊗ Sigma_T + sigma_eps^2 I` of size `p T_g`. The NLL is the sum of
per-series dense Gaussian terms (`cholesky(Symmetric(V_g))`, `logdet`,
quadratic form), which is what `_gaussian_sources_nll` does for one dense
matrix (`src/source_fit.jl:124-135`); the per-series split is an exactness-
preserving cost reduction. Replicated: `V_g = Z_g (K_g ⊗ Sigma_T) Z_g' +
sigma_eps^2 I` with the row incidence `Z_g`. With ordinary `unit` /
`unit_obs` sources (slice 2), the cross-series identity blocks are added as
fixed `SourceCovariance` terms and `V` is assembled once, dense, exactly as
`_gaussian_sources_nll` does; the temporal kernel is the only parametric
piece. A Kronecker eigen-decomposition fast path
(`(Q_K ⊗ Q_T)(D_K ⊗ D_T + sigma^2 I)(...)'`) is an optional later
optimisation and stays a documented extra; R's tests use panels of 2 to 4
series and 3 to 4 occasions, where the dense path costs milliseconds.

The whole evaluation is ForwardDiff-generic (no `try`/`catch` on the AD
path beyond the `PosDefException` guard already used at
`src/source_fit.jl:128-132`), so gradients and the Hessian used for the
verdict come from ForwardDiff as in `fit_gaussian_sources`
(`src/source_fit.jl:361-372`). No hand-written gradient is needed.

### 3.5 Admitting the keyword without editing the Normal branch

R itself admits the keyword by a pre-pass that runs before the generic
formula parser and strips the marker (R/gllvmTMB.R:1138-1141,
R/temporal.R:428-439). Julia does the same, in two steps:

1. Slice 1: the door is `fit_temporal_gllvm(long_data; formula, temporal =
   temporal_indep(...), structure = Expr[])`. The `structure` vector keeps
   the existing raw-`Expr` convention (`src/formula.jl:387-505`,
   `fit_gaussian_structured` at `src/formula.jl:757`); `temporal.jl` calls
   the existing internal recognizer `_recognize_source_term` for the
   ordinary `indep`/`dep`/`latent` terms it finds there and refuses any
   source kind R refuses (section 1.2). No line of `formula.jl` changes.
2. Slice 2 (coordination needed): to make
   `gllvm(@formula(...), long_data; family = Normal(), structure = [...,
   :(temporal_indep(0 + trait | series, time = occasion))])` work, either
   (a) `gllvm`'s long-data method (`src/formula.jl:300`) calls a one-line
   pre-pass hook `_strip_temporal!(structure)` defined in `temporal.jl` and
   routes to `fit_temporal_gllvm` when a term was found, or (b)
   `_STRUCTURED_TERM_KINDS` (`src/formula.jl:387`) and
   `_build_source_term_spec` (`src/formula.jl:507-546`) gain the three
   temporal kinds. Both are edits inside `formula.jl` outside the Normal
   branch; (a) is the smaller diff and mirrors R. The lane holding
   `formula.jl` (grammar owner, the "Boole" review role) must accept the
   hook; the spec names this as coordination, not as a change this port
   makes on its own.

Julia has no `traits()` wide-format sugar, so R's wide route
(R/traits-keyword.R:187-194) is fenced (section 7, Q2).

### 3.6 Profile, bootstrap, compare, forecast

- `profile_temporal` re-optimises the remaining coordinates at fixed
  `theta_time` with the same objective; the trace and interpolation follow
  R/profile-ci.R:188-235. It does not call `profile_ci` from
  `src/confint_profile.jl` (typed on `GllvmFit`), so `confint_profile.jl` is
  untouched.
- `bootstrap_temporal` uses `simulate(fit; condition_on_RE = false, rng)`
  from `temporal_methods.jl` and `update` to refit; no use of
  `src/confint_bootstrap.jl`.
- `compare_temporal` computes `logLik`, `df`, `AIC = -2 logLik + 2 df` from
  the fit fields; `src/model_selection.jl` is not touched.
- `forecast_temporal` is the conditioning formula of section 2.4 with
  `LinearAlgebra.cholesky` on `C_oo`.

### 3.7 Existing docs that the implementing PR must reconcile

Three Julia-side statements contradict R at P1 and would mislead a reader:
`docs/src/gllvmtmb-parity.md:359-362` ("corAR1 ... not in gllvmTMB, so they
are out of scope"), `r/README_bridge.md:144`, and `ROADMAP.md:83-87` (ticked
as "subsumed"). The implementing PR should replace them with a pointer to
the temporal reference page (design rule 3 in `AGENTS.md`: a user-facing API
change updates docs in the same PR).

## 4. Block-by-block test map

Source: the 22 `tests/testthat/test-temporal-*.R` files plus the one
temporal-touching block of `test-anisotropy.R`, at P1. Counts were made by
reading every block: 91 `test_that` blocks and 396 `expect_*` calls (the raw
`grep -c 'test_that('` count of 92 includes two `test_that` strings inside
`writeLines()` in `test-temporal-ar1-verify-runner.R:23,29`, which are not
blocks). All fits are `gaussian()`.

Legend: T = twin (red-first Julia test written before the code), F = fenced
with reason. "tol" is the Julia test tolerance; R's own tolerance is quoted
where it differs. Slice numbers refer to section 8.

### 4.1 Twinned blocks (43 blocks, 150 expectations)

| R file:line | Title (short) | exp | Julia twin, tolerance, reason |
| --- | --- | --- | --- |
| ar1-methods.R:3 | legacy spelling resolves to native extractor | 4 | T `test_temporal_extract.jl`: mode/structure/component names, `n_pairs == 9`; exact. |
| ar1-methods.R:20 | latent lifecycle labels and guards | 10 | T `test_temporal_lifecycle.jl`: score row ids, ordination `row_index`, loading row names, in-sample `predict` keeps key columns, `simulate(nsim=2)` shape, `update` type; refusals `predict(newdata)`, `confint`, `bootstrap_ci`, `ordination_uncertainty` match "newdata.*temporal" / "not available.*temporal". Exact/regex. |
| ar1-methods.R:50 | stable sign anchor | 3 | T `test_temporal_sign.jl` on `_temporal_report_sign([0, -2, 0.5])`: `first_loading_negligible`, `anchor_trait == "largest"`, `multiplier == -1`. Exact. |
| ar1-methods.R:59 | missing-response, selection, integration guards | 3 | T `test_temporal_admission.jl`: NA response, `select_lv`, VA keyword each throw the R message. Regex. |
| ar1-oracles.R:3 | retired iid-Psi differs from current | 3 | T `test_temporal_oracles.jl`: closed-form entries `K λ1^2` vs `K(λ1^2 + ψ1)`; atol 1e-12. Pure arithmetic. |
| ar1-oracles.R:16 | AR1 sign, zero, gap, boundary identities | 5 (7 at run) | T: NLL symmetric in `rr` sign (atol 1e-10), `phi(0) == 0`, `0.65^6` gap identity, finite NLL and gradient at `theta = ±20`. Julia can go tighter than R's 1e-8 because the objective is exact. |
| ar1-parser.R:4 | migration fixture keeps gaps; OU distinct | 4 | T `test_temporal_parser.jl`: `pair_table.time == [1,3,7]`, structures, "integer-valued" refusal. Exact. |
| ar1-parser.R:25 | constructor and parser boundary checks | 8 | T: type, `d == 1`, structure; refusals "rank one", "bare column", "trait-intercept block", "complete trait panel", "replicate". Regex. |
| program-bootstrap.R:1 | bootstrap retains every attempt | 5 | T `test_temporal_bootstrap.jl`: `replicate == 1:2`, six column names, finite seeds, objective finite or error text; refusal "temporal_indep" on a `dep` fit. Structure only (RNG differs from R). |
| program-bootstrap.R:18 | reproducible seeds and OU scale | 6 | T: same seed reproduces seeds and `time_estimate` (atol 1e-10), `time_estimate > 0` on OU, caller RNG untouched (Julia: `Random.default_rng()` state unchanged), `time_estimate == exp(theta_time)` (atol 1e-12). |
| program-bootstrap.R:40 | refuses source pairs | 1 | T: a fit with an ordinary `unit` source refuses with "temporal source by itself" (slice 2; the kernel pair itself is fenced, the composed-refusal is what the block asserts). |
| program-composed-simulation.R:12 | unconditional sim redraws temporal and ordinary tiers | 1 | T slice 2 `test_temporal_simulate.jl`: with an `rng` that returns zeros (Julia `simulate` takes `rng`; the twin passes a zero-returning `AbstractRNG` stub), unconditional draw equals `X beta`; atol 1e-12. |
| program-composed-simulation.R:36 | conditional sim keeps the fitted predictor | 1 | T slice 2: zero-noise conditional draw equals fitted `eta`; atol 1e-12. |
| program-composed-simulation.R:84 | composed sim matches analytic moments | 1 (4 at run) | T slice 2: 10000 draws, four covariance entries within 4.2 Monte Carlo SE of the analytic `V`. |
| program-forecast.R:47 | forecasts match dense conditioning | 3 | T `test_temporal_forecast.jl`: `est` and `se_fit` against an independently built dense conditional; atol 1e-8 (R's tolerance). |
| program-forecast.R:85 | forecast rejects unsupported layouts | 3 | T: "existing series", "strictly after", "complete trait panel". Regex. |
| program-forecast.R:104 | negative AR1 and OU origin shift | 2 | T: with `theta_time` overwritten to `phi = -0.6`, `est` matches dense mean (atol 1e-8); OU `est`/`se_fit` invariant to `+100` time origin (atol 1e-10). |
| program-profile.R:1 | profile of the transformed time parameter | 6 | T `test_temporal_profile.jl`: names, `estimate == (1-1e-6) tanh(theta)` (atol 1e-10), trace value at the MLE equals the objective (atol 1e-8), trace maximum exceeds objective + 0.1, narrow `parm_range` gives `missing` bounds, refusal on `dep`. The R block reads `TMB::tmbprofile` directly; the twin reads the Julia trace. |
| program-profile.R:30 | OU profile invariant to time-origin shift | 2 | T: objectives (atol 1e-8) and profile outputs (atol 1e-8) equal after `+100`. |
| program-selection.R:1 | compare_temporal AIC semantics, no LRT | 6 | T `test_temporal_compare.jl`: column names, `logLik == -objective` (atol 1e-12), `df == length(theta)`, `AIC == -2 logLik + 2 df` (atol 1e-12), no `p_value` column. |
| program-selection.R:22 | compare refuses composed candidates | 1 | T slice 2: "source by itself" on a fit with an ordinary `unit` source. |
| program-simulation.R:32 | unconditional sim removes fitted recursive states | 1 | T `test_temporal_simulate.jl`: mock fit (`phi = 0.6`, 3 states, unique), zero-noise unconditional effect is `zeros(3)`; atol 1e-14. |
| program-simulation.R:41 | conditional sim centred on fitted predictor | 1 | T: zero-noise conditional equals `eta`; atol 1e-14. |
| sixth-source-api.R:1 | constructors preserve mode | 7 | T `test_temporal_api.jl`: types and modes; `unique == true` on latent. Exact. |
| sixth-source-api.R:16 | AR1 and OU distinct contracts | 2 | T: structure symbols. Exact. |
| sixth-source-api.R:26 | parser keeps ordered integer gaps | 3 | T: `active`, times `[1,3,7]`, `state_tier == :temporal`. Exact. |
| sixth-source-api.R:44 | series/unit partition required only with a stable unit component | 2 | T slice 2: temporal-only admitted; with `indep(0 + trait \| unit_group)` refuse "same partition as.*unit". |
| sixth-source-engine.R:12 | dedicated state tier, not B scores | 11 | T `test_temporal_engine.jl`: 9 states, `state_id` length, predecessors in `{-1, 0..8}`, gaps 2 and 4 present, no unit-level latent block, temporal diag block present, `extract_temporal().variance.component` all `temporal_indep_variance`, `p` rows. The two TMB-internal assertions (`use_rr_B == 0`, `q_temporal` in `parList`) map to "no `SourceCovariance` in `fit.sources`" and "`theta_diag` present in `fit.parameters`". |
| sixth-source-engine.R:34 | temporal plus ordinary unit intercept | 1 | T slice 2: admitted. |
| sixth-source-engine.R:43 | distinct series label, same partition | 1 | T slice 2: admitted. |
| sixth-source-engine.R:52 | unit_obs stays unit-nested | 2 | T slice 2: nested admitted; crossed refuses "unit_obs.*nested". |
| sixth-source-engine.R:69 | composes with every unit / unit_obs mode | 2 (6 at run) | T slice 2: six cells fit and report `temporal.active`. |
| sixth-source-engine.R:91 | ordinary ordination beside temporal | 3 | T slice 2: `extract_ordination` returns one row per series, no `pair_id`. |
| sixth-source-engine.R:104 | composed simulation redraws ordinary tiers | 2 | T slice 2: shapes of unconditional and conditional draws. |
| sixth-source-engine.R:119 | long and wide lifecycle (loop indep/dep/latent) | 14 | T (long form only): modes, `dep` loadings `3 × 3`, latent score ids and loading names, `getLV` is `nothing` otherwise, `predict` columns, `simulate` shape, `update` type, refusals for `predict(newdata)`, `confint`, `select_lv`. The wide-versus-long equality assertions are fenced inside this block (no `traits()` sugar in Julia; Q2). |
| sixth-source-engine.R:169 | refuses each deferred provider and duplicate terms | 3 (5 at run) | T: phylo/animal/spatial/kernel `indep` beside an unreplicated temporal term refuse "requires replicated AR1" (Julia refuses at the pre-pass with the same message; the cell itself is fenced); two temporal terms refuse "Only one temporal covariance term". |
| sixth-source-engine.R:196 | all modes and structures reach the tier | 3 (8 at run) | T: 8 cells (indep, dep, latent, latent-unique × AR1, OU) fit and report mode and structure. |
| sixth-source-oracles.R:214 | all cells match dense NLL and gradient | 2 (32 at run) | T `test_temporal_oracles.jl`: the Julia objective versus an independently written dense builder (a second, separate implementation in the test file, as R does at oracles.R:36-106) over 8 cells × 4 time values; NLL atol 1e-9, ForwardDiff `theta_time` gradient versus central difference atol 1e-6. Tighter than R's 2e-6 / 2e-5 because there is no Laplace or tape noise. |
| sixth-source-oracles.R:318 | dense oracle with unit and unit_obs | 2 | T slice 2: NLL atol 1e-9; gradient atol 1e-6. |
| sixth-source-oracles.R:342 | negative persistence flips sign at odd lag | 2 | T: lag 7 at occasions `(1,3,8)`, covariance entry `< 0`. Exact. |
| sixth-source-oracles.R:359 | OU extremes finite | 2 (4 at run) | T: NLL and gradient finite at `theta = ±45` on a `dep` fit. |
| sixth-source-oracles.R:375 | latent-unique Psi correlated across occasions | 4 | T: correct versus iid-Psi covariance and NLL differ by more than 1e-4. |
| sixth-source-regressions.R:1 | ordinary providers stay ordinary without temporal | 2 | T: `fit_gaussian_structured` with `dep` / `latent` and no temporal term has no temporal spec. |

Subtotal: 43 blocks, 150 expectations. Of these, 14 blocks (32 expectations)
need slice 2 (ordinary `unit` / `unit_obs` composition and simulate).

### 4.2 Fenced blocks (48 blocks, 246 expectations)

| Group | R file (blocks, exp) | Reason |
| --- | --- | --- |
| Cross-source cells (24 blocks, 119 exp) | program-animal-replicated.R (6, 40); program-kernel-replicated.R (6, 31); program-phylo-replicated.R (5, 24); program-spatial-replicated.R (6, 20); phylo-optimizer-qualification.R:208 (1, 4) | The replicated AR1 `temporal_indep` plus one `kernel_indep` / `phylo_indep` / `animal_indep` / `spatial_indep` cell. Spatial is outside P1 by D-295. The phylo, animal and spatial pairs are recorded by R as failed engineering gates (docs/design/35-validation-debt-register.md:77-79). The kernel pair is twinnable later because Julia has fixed kernel sources (section 7, Q1). Each pair file also carries R-only assertions (`optimizer_passes`, `engine = "julia"` refusal, `Lambda_phy` report). |
| Dev-only optimiser helpers and DRAC artefacts (19 blocks, 104 exp) | phylo-damped-newton.R (8, 34); phylo-third-pass.R (4, 17); phylo-optimizer-qualification.R:39, 50, 166, 244, 269 (5, 44); ar1-verify-runner.R (1, 6); sixth-source-regressions.R:19 (1, 3) | They test scripts under `dev/temporal-program/`, TMB internals (`spHess`, `EvalADFunObject`, `optimizer_pass_history`, a 44-column diagnostics table), a retained DRAC campaign CSV with an md5, and a seed plan for an R campaign. No public API and no Julia analogue. |
| Retired TMB branch (1 block, 5 exp) | sixth-source-oracles.R:250 | Rebuilds the legacy `use_temporal_B` TMB object with `MakeADFun` and compares; R-internal migration evidence. |
| Non-temporal regression guards (3 blocks, 11 exp) | ar1-regressions.R:5, 37, 71 | No temporal term; they guard R's ordinary `latent`, phylo-versus-kernel equality and spatial alias rewriting. Covered by existing Julia tests where the feature exists. |
| Spatial plotting (1 block, 7 exp) | anisotropy.R:76 | `plot_anisotropy` refusal on `use$spatiotemporal`; spatial and plotting are outside this port. |

Subtotal check: 24 + 19 + 1 + 3 + 1 = 48 blocks and 119 + 104 + 5 + 11 + 7 =
246 expectations, so twinned plus fenced is 43 + 48 = 91 blocks and 150 + 246
= 396 expectations. Two files split across groups:
`phylo-optimizer-qualification.R` (block 208 with 4 expectations is a
cross-source cell; the other five blocks carry 44) and
`sixth-source-regressions.R` (block 1 with 2 expectations is twinned; block
19 with 3 is fenced).

Inside the twinned set, two partial fences are recorded: the wide-format
(`traits()`) equalities in sixth-source-engine.R:119, and the exact R seed
values in program-bootstrap.R:18 (Julia reproduces its own seeds, not R's).

## 5. Receipts and case-map rows

Rows: `temporal/temporal_indep`, `temporal/temporal_dep`,
`temporal/temporal_latent`, `temporal/extract_temporal`,
`temporal/forecast_temporal`, `temporal/bootstrap_temporal`,
`temporal/profile_temporal`, `temporal/compare_temporal`, and the S3 row for
`update.gllvmTMB_multi` (all in
`docs/dev-log/core070/true-parity-latest/case-map.json`, PR #526, signed
D-297 as `required_core`; their `executable_case_ids`, `evidence`,
`measured_against` and `disposition` are empty today).

Which rows close per slice:

- Slice 1 closes `temporal_indep`, `temporal_dep`, `temporal_latent`,
  `extract_temporal`, `forecast_temporal`, `profile_temporal`,
  `compare_temporal`, `bootstrap_temporal` for the temporal-only cells
  (unreplicated and replicated, AR1 and OU).
- Slice 2 closes the `update` S3 row and the composition sub-cases inside
  the constructor rows (`temporal + indep/dep/latent(unit | unit_obs)`).
- Cross-source cells stay open with a `PARTIAL` disposition proposal (Q1).

Each receipt is one JSON file under
`docs/dev-log/core070/true-parity-latest/receipts/temporal/` produced by
the frozen-R runner used for the existing receipts
(`test/parity/core070_receipts.jl`, `docs/dev-log/core070/README.md`) and
records:

- pins: R commit `9539352f6`, `DESCRIPTION` version `0.7.1`, TMB and R
  versions, shared-library sha256; Julia commit, Julia version,
  `Manifest.toml` sha256;
- the public R call (formula string, `structure`, `time`, `replicate`,
  `unit`), the data md5 and the data itself when small (the R fixtures are
  9 to 96 rows);
- R's `opt$par` with names, `opt$objective`, `opt$convergence`,
  `report$phi_temporal` or `report$ou_rate`, `report$temporal_sd`,
  `report$Lambda_temporal`, and `extract_temporal()` output;
- Julia's estimate, `loglik`, verdict fields;
- comparisons: (i) Julia NLL at R's coordinates versus R's objective,
  target `|Δ| <= 1e-8` (both are the exact marginal; R's own oracle tolerance
  `2e-6` is the honest ceiling if the target is missed, and any gap above
  `1e-6` is a defect to explain rather than a tolerance to widen); (ii) R's
  objective at Julia's coordinates, computed by the R runner (cross-objective
  the other way); (iii) between optima: `|Δ logLik| <= 1e-6`, `phi`/`kappa`
  atol `1e-5`, `Sigma_T = L L'` entries atol `1e-5` (not `L`, section 1.6),
  `psi` and `sigma_eps` rtol `1e-5`, `beta` atol `1e-6`;
- helper receipts: `forecast_temporal` `est`/`se_fit` atol `1e-8` at R's
  coordinates; `profile_temporal` `estimate` atol `1e-8`, `lower`/`upper`
  atol `1e-3` (grid interpolation, section 2.5) with `NA`/`missing` agreement
  recorded as a boolean; `compare_temporal` `logLik`/`df`/`AIC` atol `1e-8`;
  `bootstrap_temporal` structural agreement (columns, `n_boot` rows, count of
  converged refits) and a distribution summary of `time_estimate` (mean and
  SD within Monte Carlo error at `n_boot = 200`), never per-row equality.

The receipt carry rule of D-295 applies: these are new P1 receipts, not
0.7.0 carries, so no `PARTIAL_STALE_AT_P1` question arises.

## 6. Symbolic alignment table

| Math term | R location (P1) | Planned Julia location |
| --- | --- | --- |
| state index `(series, occasion)`, ordered by series then time | R/temporal.R:378-388; R/fit-multi.R:5520-5540 | `src/temporal.jl` `_parse_temporal_term` → `TemporalSpec.pair_table`, `predecessor`, `gap`, `elapsed` |
| `phi = (1-1e-6) tanh(theta_time)` | src/gllvmTMB.cpp:1795; R/temporal.R:504 | `src/temporal_likelihood.jl` `_temporal_phi(theta)` |
| `kappa = exp(theta_time)` | src/gllvmTMB.cpp:1796; R/temporal.R:506 | `_temporal_kappa(theta)` |
| `a_s = phi^gap` (integer power) / `exp(-kappa elapsed)` | src/gllvmTMB.cpp:1804-1806, 62-75 | `_temporal_correlation` (marginal form `phi^{|t-u|}`, `exp(-kappa|t-u|)`; equal to the recursion's stationary marginal) |
| innovation SD `sqrt(1-a^2)` / `sqrt(1-exp(-2 kappa Δ))` | src/gllvmTMB.cpp:1811-1813, 77-90 | not needed in the marginal; used in `simulate` (`src/temporal_methods.jl`) to mirror R/methods-gllvmTMB.R:1685-1706 |
| `Lambda_temporal = unpack(theta_rr, p, rank)` lower-triangular | src/gllvmTMB.cpp:1792-1793 | `unpack_lambda` (`src/packing.jl`) inside `_temporal_trait_block` |
| `psi_j = exp(2 theta_diag_j)` | src/gllvmTMB.cpp:1797; R/temporal.R:518 | `_temporal_trait_block` |
| `Sigma_T` by mode (diag / `LL'` / `λλ'` / `λλ'+Ψ`) | R/fit-multi.R:5541-5544; src/gllvmTMB.cpp:1815-1834 | `_temporal_trait_block(mode, unique)` |
| `eta_o = x_o'β + Λ z_s + q_js` | src/gllvmTMB.cpp:3102-3112 | implicit in the marginal; explicit in `predict` / `simulate` |
| `y_o ~ N(eta_o, sigma_eps^2)`, `sigma_eps` free | src/gllvmTMB.cpp:3294; R/fit-multi.R:6959-6960 | `temporal_marginal_nll` residual term |
| marginal `V = Z(K ⊗ Sigma_T)Z' + sigma^2 I` (block over series) | tests/testthat/test-temporal-sixth-source-oracles.R:36-106 (R's dense oracle); docs/dev-log/after-task/2026-09-09-temporal-sixth-source.md:19-22 | `temporal_marginal_nll` |
| composition `+ J_unit ⊗ Sigma_B + J_unit_obs ⊗ Sigma_W` | same note, line 19; R/temporal.R:154-158 | fixed `SourceCovariance` terms via `_source_trait_covariances` (`src/source_fit.jl:71-93`), slice 2 |
| sign anchor for rank-one loadings | R/temporal.R:466-483; R/extractors.R:525-527 | `_temporal_report_sign` in `src/temporal_methods.jl` |
| forecast conditional mean and SD | R/temporal-forecast.R:122-134, 138-164 | `forecast_temporal` |
| profile threshold `qchisq(level,1)/2`, `ytol = crit + 1`, interpolation | R/profile-ci.R:41-48, 66-68, 188-235 | `profile_temporal` |
| `AIC = -2 logLik + 2 df`, `df = length(opt$par)` | R/temporal-selection.R:24-27 | `compare_temporal` |
| unconditional simulation: redraw states from the stationary recursion | R/methods-gllvmTMB.R:1684-1708 | `simulate(fit; condition_on_RE=false)` |

## 7. Open questions for the maintainer

Each with a recommendation. None of these blocks slice 1.

- Q1. Cross-source cells (replicated AR1 `temporal_indep` plus one
  `kernel_indep` / `phylo_indep` / `animal_indep` / `spatial_indep`; 24 R
  blocks). Options: twin all four; twin the kernel pair only; fence all.
  Recommendation: fence all four in this port, with a `PARTIAL` disposition
  proposal on the constructor rows naming the cells; revisit the kernel pair
  after slice 2 because Julia already has fixed kernel sources and R records
  that pair as the one with a passed recovery gate. The spatial pair is
  outside P1 by D-295; the phylo and animal pairs are failed gates on R's own
  ledger, so a twin would replicate an unvalidated route.
- Q2. Wide-format `traits(t1, t2, t3) ~ ...` route. Julia has no `traits()`
  sugar and its front door takes `Y` as a `p × n` matrix. Recommendation:
  fence the wide route; the long-form door is R's canonical temporal input
  (R/temporal.R:348-349 says the wide interface is expanded to the long form
  before temporal parsing), so nothing in the likelihood is lost.
- Q3. Formula admission mechanism (section 3.5). Recommendation: slice 1
  ships `fit_temporal_gllvm` with a raw-`Expr` `structure` vector and no
  edit to `formula.jl`; slice 2 asks the `formula.jl` lane for the one-line
  pre-pass hook (option a). The maintainer decides whether the hook lands in
  this port's PR or the grammar lane's.
- Q4. `NA` bounds in `profile_temporal`. Options: `missing` or `NaN`.
  Recommendation: `missing`, because R distinguishes a truncated search
  (`NA`) from an asymptotic bound (`±Inf`) and `NaN` would blur the first
  with a numeric failure.
- Q5. Bootstrap seeds. R draws `sample.int` seeds and reseeds base R per
  replicate; Julia cannot reproduce those draws. Recommendation: `StableRNG`
  with a `seed` keyword, receipts compare structure and a distribution
  summary (section 5), and the reference page says so.
- Q6. Ordinary `unit` / `unit_obs` composition. R admits `indep`, `dep`,
  `latent` at both levels beside the temporal term (14 twinned blocks need
  it). Recommendation: inside P1, as slice 2; it reuses `SourceCovariance`
  with identity blocks and costs little beyond assembling one dense `V`.
- Q7. The `sigma_eps` auto-suppression rule. R keeps `sigma_eps` free for an
  unreplicated temporal fit even when a per-row `indep` term is present
  (R/fit-multi.R:6959-6960), and suppresses it in the replicated case.
  Recommendation: port the rule as written (R semantics) in slice 2 and
  record it in the reference page; do not "improve" it.
- Q8. Three stale Julia docs (section 3.7). Recommendation: fix in the
  implementing PR under design rule 3, not in this spec PR.
- Q9. Bridge reachability. Should `bridge_capabilities` advertise temporal
  so R can call Julia for it? Recommendation: no in P1; R has its own
  engine for this row and the bridge would add a second inference surface to
  validate.
- Q10. Fit type integration. Join `AnyGllvmFit` in `src/postfit.jl:9` or
  define methods on the new type? Recommendation: define `loglikelihood`,
  `dof`, `nobs`, `aic`, `bic` for `TemporalGaussianFit` in the new file;
  no edit to `postfit.jl`.
- Q11. `dep` loadings in receipts. `L` is sign-ambiguous by column.
  Recommendation: receipts compare `Sigma_T = L L'`; `extract_temporal`
  still returns `L` as R does.
- Q12. Error type. R uses `cli::cli_abort` with classes. Recommendation: a
  single `TemporalContractError <: Exception` with `message::String` and
  `kind::Symbol` carrying the R class name, thrown from every temporal
  refusal; the tests match on both.

## 8. Estimate

Build: about 8 to 11 working days of one lane, split as slice 1 (constructors,
pre-pass, likelihood, fit, `extract_temporal`, `forecast_temporal`,
`profile_temporal`, `compare_temporal`, `bootstrap_temporal`, `update`,
in-sample `predict`, `simulate`, latent scores with the sign anchor, refusal
set) 5 to 7 days; slice 2 (ordinary `unit` / `unit_obs` composition, the
partition rule, the `sigma_eps` rule, composed simulate, the formula hook
after coordination) 3 to 4 days.

Tests and receipts: about 4 to 5 days: 43 red-first twins (29 in slice 1, 14
in slice 2), the independent dense builder for the oracle file, the R receipt
runs for the eight rows (an R-side lane task, since Codex holds the live R
toolchain per the operating contract), and the case-map evidence edits.

Total: about three working weeks of lane time, plus review.

What the estimate rests on:

- The likelihood is closed form and small. R's fixtures are 2 to 4 series
  by 3 to 4 occasions by 3 traits; every fit in the R suite runs in seconds,
  so no run approaches the 3-hour line (D-287) and no Totoro or DRAC time is
  needed. The R recovery campaigns (kernel, phylo, animal, spatial pairs) are
  not twinned.
- The fitting pattern already exists in `fit_gaussian_sources` (Optim LBFGS
  plus ForwardDiff plus the verdict fields); the port adds one parametric
  source and a per-series block loop.
- 34 of the 43 twins are contract, refusal and structural tests; 9 are
  numeric oracles with an exact target.
- Risk that stretches the estimate: `profile_temporal` bound agreement at
  `1e-3` depends on reproducing R's trace grid and interpolation; if the
  first receipt misses, the fix is in the walk, not in the likelihood, and
  the `estimate` receipt still closes. `bootstrap_temporal` receipts are by
  design distributional.
- What the estimate does not cover: the cross-source cells (Q1), the wide
  route (Q2), and any Kronecker fast path.
