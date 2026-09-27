# iSDM port spec: gllvmTMB's public integrated-SDM door at P1, in GLLVModels.jl

Status: DRAFT for maintainer review (workstream 1a of the true-parity plan; scope signed as
vault decision D-295, Packet 1 row 2, 2026-09-27).
Scope: twin R's public door `gllvmTMB(..., family = isdm_sources(...))` at pin
P1 = gllvmTMB commit `9539352f6`: non-spatial, Laplace, first order (point fit) plus `predict`.
No intervals, because R's own ledger claims none (`docs/design/capability-status.md:137-139` at
P1 rates ISDM-01 to ISDM-03 as `point-fit-recovery`). Spatial iSDM is outside the P1 boundary.
Julia's `SourceCovariance` (`src/source_fit.jl:18`) is unrelated to this model and stays a
documented Julia-only extra.

Reading rule used here: every R citation is `path:line` at `9539352f6`, read with
`git show 9539352f6:<path>`; every Julia citation is `path:line` at `origin/main` `1385b0490`.
The R repository was not checked out or edited.

## 0. What the port is, in one paragraph

R fits one ecological GLLVM linear predictor per (cell, species) and lets several named data
sources observe it, each under its own admitted observation law (Poisson-log counts or
Bernoulli-cloglog detections), with a known per-row support entering as an offset. The Julia
port keeps exactly that model, its parameterisation and its refusals, but needs one new axis
that the Julia engine does not have today: within one trait (species) the response family varies
by row, and one (cell, species) pair carries several rows. So the port cannot reuse the p x n
matrix substrate of `src/families/laplace.jl` or `src/families/mixed.jl`. It gets its own
long-table kernel in new files, sharing only the per-observation family pieces (`_glm_score`,
`_glm_weight`, `_glm_obs_weight`, `_glm_logpdf`, `linkinv`, `mu_eta`) that every kernel already
shares.

## 1. The exact model R fits

### 1.1 Sources, laws, and the per-row family axis

A source is a named observation stream. `isdm_sources(...)` builds a named list of family
objects with `attr(., "family_var") = "isdm_source"` (`R/isdm-sources.R:165-171`). Only two laws
are admitted (`R/isdm-sources.R:15-26`):

| law | R family object | `family_id` | `link_id` | why admitted |
|---|---|---|---|---|
| count stream | `poisson()` | 2 | 0 | `log E[Y] = eta + log a` |
| detection stream | `binomial("cloglog")` | 1 | 2 | `p = 1 - exp(-a e^eta) = P(N > 0)`, `N ~ Poisson(a e^eta)` |

The family id map is `R/fit-multi.R:1193-1213`. Each long row gets its own `(family_id,
link_id)` from its `isdm_source` value (`R/fit-multi.R:1432-1462`: the list is aligned to the
selector's levels by name at `R/fit-multi.R:1451-1452`), and the C++ template reads those two
per-row integer vectors (`src/gllvmTMB.cpp:838-848`).

The ordinary rule is one family/link per trait
(`.gllvmTMB_validate_family_scale_by_trait`, `R/fit-multi.R:293-318`, error class
`gllvmTMB_family_within_trait_unsupported`). The integrated contract is the single sanctioned
exception: it is admitted only when `.gllvmTMB_integrated_sources_contract()` returns `TRUE`
(`R/fit-multi.R:1471-1484`), and then `allow_isdm_mixed = TRUE` is passed to the validator
(`R/fit-multi.R:1497-1501`). This "family varies by row within a trait" is the new axis the
plan names, and it is the whole reason the Julia port needs new files.

### 1.2 Linear predictor and latent structure

For long row `o` with trait (species) `t(o)`, unit (cell) `s(o)` and source `d(o)`:

```
eta(o) = [X_fix b_fix](o) + offset(o) + sum_k Lambda_B[t(o), k] * z_B[k, s(o)]
```

- Fixed part and offset: `src/gllvmTMB.cpp:1676` (`eta = X_fix b_fix + offset_vec`).
- Latent part: `src/gllvmTMB.cpp:3083-3096` adds `Lambda_B * z_B[, s]` per row. One
  `latent(0 + trait | cell_id, d = 1)` term becomes the between-unit reduced-rank block `rr_B`
  when its grouping equals the `unit` column (`R/fit-multi.R:1843`, rank `d_B` at
  `R/fit-multi.R:3003-3005`, `site_id` at `R/fit-multi.R:3584`). In the fixture used by every
  public test, `unit = "cell_id"` and `d = 1`.
- Prior on scores: `z_B[, s] ~ N(0, I_d)` per cell (`src/gllvmTMB.cpp:1879-1882`); `z_B` is
  declared random and integrated by TMB's Laplace approximation (`R/fit-multi.R:7261`).
- Loadings: `Lambda_B` (`p x d`) is unpacked as a lower-triangular matrix with the diagonal
  first, then the strict lower triangle column by column (`src/gllvmTMB.cpp:34-59`,
  `gll_unpack_rr_loadings`). No positivity constraint on the diagonal; the identifiability
  convention is lower-triangular only. Julia's `packing.jl` uses the same lower-triangular
  convention (see `src/packing.jl`; `rr_theta_len(p, K)` in `src/families/mixed.jl:435`).
- The ecological predictor is shared across sources. Design 120 writes it as
  `eta_eco[i,j] = alpha[j] + x[i]' beta[j] + u[i]' lambda[j]` and the per-source part as
  `eta_obs[i,j,d] = gamma[d,j]` with reference coding `gamma[1,j] = 0`
  (`docs/design/120-multi-source-isdm-contract.md:13-25`). At P1 both live inside `X_fix b_fix`:
  the user writes the source indicator interaction (`trait:src_gbif` in
  `tests/testthat/test-isdm-predict.R:40`), or `isdm_source(family, observation = ~ ...)`
  supplies a source-masked block (section 1.4).

### 1.3 Observation likelihood per row

Given `eta(o)` and support `a(o)` with `offset(o) = log a(o)`:

```
count arm (fid 2):      Y(o) ~ Poisson(exp(eta(o)))
                        ll += dpois(y, exp(eta), log = TRUE)             src/gllvmTMB.cpp:3334-3336
detection arm (fid 1,   Y(o) ~ Bernoulli(1 - exp(-exp(eta(o))))
  lid 2):               ll += gll_dbinom_cloglog(y, n_trials = 1, eta)    src/gllvmTMB.cpp:3318-3320
```

`gll_dbinom_cloglog` (`src/gllvmTMB_cloglog.h:45-57`) evaluates both tails on the log scale:
`y * log(1 - exp(-exp(eta))) - (n - y) * exp(min(eta, 700))`, with `gll_log_cloglog_p` using a
series expansion below `eta = -20` (comment at `src/gllvmTMB.cpp:3306-3311`). The per-row
switch on `family_id_vec(o)` is `src/gllvmTMB.cpp:3296-3336`. Rows are conditionally
independent given `z_B`; there is no cross-arm term (Design 120 section 2,
`docs/design/120-multi-source-isdm-contract.md:36-49`).

The marginal is TMB's Laplace approximation over `z_B`, which is block-diagonal by cell, so it
equals a per-cell Laplace with the observed (AD) curvature of the joint negative log density.
R reports `eta` at the Laplace mode (`REPORT(eta)`, `src/gllvmTMB.cpp:4360`, surfaced as
`fit$report$eta`, `R/fit-multi.R:8982`).

### 1.4 Offsets and the cloglog exception

`offset(log_support)` is held out of the fixed formula and evaluated against the same `data`
(`R/fit-multi.R:3347-3360`; `R/offset.R:14-16`). `gll_prepare_offset()`
(`R/offset.R:108-195`) refuses a non-zero offset on any row whose family is not a count family
(`R/offset.R:22`, ids 2, 5, 10, 11, 15), except a Bernoulli-cloglog row inside the admitted
contract (`allow_isdm_cloglog`, `R/offset.R:157-159`). Zero offsets always pass
(`R/offset.R:29-33`). Non-finite offsets abort (`R/offset.R:140-146`).

### 1.5 Source-specific observation formulas (optional)

`isdm_source(family, observation = ~ ...)` attaches a one-sided formula
(`R/isdm-sources.R:48-65`). At fit time `.gll_isdm_observation_design()`
(`R/isdm-sources.R:178-277`) evaluates each formula only on that source's rows, names the
columns `isdm_source:<src>:<term>`, zero-fills them elsewhere, refuses a top-level
`isdm_source` fixed effect (`:181-187`), refuses missing variables or NA (`:207-225`), refuses
column-name collisions (`:240-249`, class `gllvmTMB_isdm_observation_name_collision`), and then
keeps a source column only if it raises the QR rank of `[X_fix, kept columns]`
(`:256-267`), reporting drops with `cli_inform` (`:268-273`). The frozen basis (terms,
xlevels, contrasts, kept column names) is stored on the fit as `isdm_observation_basis`
(`R/fit-multi.R:3335-3345`) for `predict(newdata=)`.

### 1.6 Identifiability and what is estimated

Design 120 section 3 (`docs/design/120-multi-source-isdm-contract.md:83-99`): with reference
coding the per-source terms are relative recording effects on the log scale; absolute
intensity, occupancy and detectability are not identified from presence-only arms (Fithian
et al. 2015, cited in vault note dr30). The once-per-session message at
`R/fit-multi.R:1487-1495` states this to the user. The port reports the same quantities and
makes the same claim, no more.

### 1.7 What `predict` returns and on which scale

`predict.gllvmTMB_multi` (`R/methods-gllvmTMB.R:2786-2793`; `type = c("link", "response")`,
`re_form = ~.`, `se.fit = FALSE`):

- In-sample, full random effects: `est = fit$report$eta` (`R/methods-gllvmTMB.R:2827-2833`).
- In-sample, `re_form` zero (`~0`, `NA`, numeric `0`; `.gllvmTMB_re_form_is_zero`):
  `est = X_fix b_fix + offset_vec` (`R/methods-gllvmTMB.R:2828-2830`, helpers at `:224-247`,
  `:2674-2685`, `R/offset.R:52-56`).
- Output columns in-sample: `unit`, `species`, `trait`, then the `family_var` column
  (`isdm_source`) on mixed-family fits, then `est` last (`R/methods-gllvmTMB.R:2839-2873`).
- `newdata`: the source column must be present, non-NA and declared
  (`R/methods-gllvmTMB.R:2876-2905`, classes `gllvmTMB_predict_isdm_source_missing`,
  `gllvmTMB_predict_isdm_source_unknown`); the design is rebuilt from the response-free RHS
  (`:2927-2930`), source blocks are reconstructed from the frozen basis
  (`R/isdm-sources.R:284-403`), the offset is re-evaluated against `newdata`
  (`R/offset.R:67-104`), and the `rr_B` contribution `Lambda_B[t, ] . z_B[, s]` is re-added for
  units seen at training (`R/methods-gllvmTMB.R:2949-2977`). An unseen unit level falls back to
  the fixed-only prediction for that row.
- `type = "response"` applies each row's own inverse link, dispatched on the per-row
  `(family_id, link_id)` (`R/methods-gllvmTMB.R:3155-3186` in-sample, `:3187-3220` newdata):
  `exp(eta)` on count rows, `-expm1(-exp(eta))` on detection rows. It includes the offset, so a
  count row returns an expected count at that support; effort-free relative intensity is
  obtained by setting the offset column to zero in `newdata`
  (`docs/design/127-isdm-prediction-map-implementation.md:78-93`).
- `se.fit = TRUE` is refused with `newdata` (`R/methods-gllvmTMB.R:483-491`, class
  `gllvmTMB_predict_se_newdata_unsupported`). In-sample it returns the fixed-effect-only
  delta-method standard error (see open question Q1).
- `fitted()` forwards to the in-sample path (`R/methods-gllvmTMB.R:3330-3333`).

## 2. The public API contract

### 2.1 `isdm_source()` and `isdm_sources()`

`isdm_source(family, observation)` (`R/isdm-sources.R:48-65`):

| check | error text (first line) |
|---|---|
| `family` not an R family | `family must be an R family object.` |
| `observation` not a one-sided formula | `observation must be a one-sided formula.` |

`isdm_sources(...)` (`R/isdm-sources.R:119-172`), in order:

| check | error text (first line) | line |
|---|---|---|
| fewer than two args, or any unnamed | `isdm_sources() needs at least two named sources.` | 128-133 |
| duplicate names | `Source names must be unique; <name> is declared twice.` | 134-139 |
| a law outside the admitted set | `Source(s) <names> declare(s) an observation law that is not admitted.` | 140-148 |
| no count arm at all | `An integrated declaration needs at least one count arm.` | 156-164 |

Return: the named law list with `family_var = "isdm_source"`, an informational
`isdm_source_laws` matrix, and `isdm_observation` when any wrapper carried a formula
(`:165-171`). An all-count declaration is accepted and fits through the ordinary mixed-family
route (`:149-155`).

### 2.2 The long-table input shape

One row per (unit, trait, source[, visit]). Required columns: the response (LHS), the `trait`
column, the `unit` column, `isdm_source` (values exactly the declared names), and the offset
variable named in `offset(...)`. The public fixture (`tests/testthat/test-isdm-predict.R:6-34`)
is 30 cells x 2 species x 2 sources = 120 rows with columns `cell_id, trait, isdm_source,
support, value, log_support, env, src_gbif`.

Admission (`.gllvmTMB_isdm_declared_core`, `R/isdm-sources.R:412-439`) returns `TRUE` only when:
the selector has one value per row, every value is a declared name and every declared name
occurs; at least one count arm and at least one detection arm are declared; each row's
`(family_id, link_id)` equals its declared law; and every trait carries every declared source
(row presence, not balance). `FALSE` means the ordinary rules apply, which then refuse a
within-trait family mix (`R/fit-multi.R:307-317`) and a non-zero offset on the cloglog rows
(`R/offset.R:187-191`).

Fit-time refusals after admission (`R/fit-multi.R:3462-3480`):

| input | class |
|---|---|
| `weights` supplied | `gllvmTMB_isdm_weights_unsupported` |
| any detection row with `n_trials != 1` (a `cbind(succ, fail)` LHS) | `gllvmTMB_isdm_multitrial_unsupported` |
| a declared source x trait arm with no observed response | `gllvmTMB_isdm_observed_source_incomplete` (`R/fit-multi.R:3502-3528`, `R/isdm-sources.R:445-477`) |

### 2.3 What the fit object exposes that the tests read

`class(fit) = c("gllvmTMB_multi", "gllvmTMB")` (`R/fit-multi.R:9291-9293`). Fields read by
`test-isdm-predict.R` and `test-isdm-developer-fit.R`: `fit$report$eta`, `fit$family_input`
(with `attr(, "family_var")`), `fit$tmb_data$family_id_vec`, `fit$tmb_data$link_id_vec`,
`fit$tmb_data$offset_vec`, `fit$X_fix_names` (`R/fit-multi.R:9275`), `fit$X_fix`,
`fit$isdm_observation_basis`, `fit$formula` (offset-free), `fit$data`, `fit$opt$convergence`,
and the predict output data frame.

## 3. The Julia design

### 3.1 Constraints

- New files only for the kernel and fitter. `src/families/mixed.jl` is under change in
  PR #514 (damped per-site mode search) and is not edited here; nothing in this port calls
  `_mixed_laplace_mode` or `_mixed_loglik_site`.
- Hard stop for this lane: the shared `_laplace_mode` (`src/families/laplace.jl:102`) is not
  edited and not called. Its observation vector is one entry per trait; the iSDM cell has one
  entry per row, several per trait, so the shapes do not match anyway.
- Reused as-is from the family layer: `_glm_score`, `_glm_weight`, `_glm_logpdf` for
  `Poisson` (`src/families/poisson.jl:12-13`) and `Binomial` (`src/families/binomial.jl:28-35`),
  `_glm_obs_weight` (`src/families/laplace.jl:260`), `_default_hessian(::Binomial,
  ::CLogLogLink) = :observed` (`src/families/binomial.jl:95`), the `CLogLogLink`
  (`src/families/links.jl:16, 29, 41, 52`), `_clamp_eta`, `_clamp_mu`.
- `src/spde_latent.jl` (joint Laplace over a spatial GMRF, `:1-45`) is the wrong substrate for
  P1: spatial is outside the boundary, and its sparse Hessian assembly has nothing to offer a
  per-cell independent-score model. Not touched, not reused.
- `gllvm(formula, long_data)` (`src/formula.jl:300-333`) requires a complete species x site
  grid with one row per cell, so the long iSDM table cannot pass through it unchanged.

### 3.2 New files and types

| file | contents |
|---|---|
| `src/families/isdm_sources.jl` | `IsdmSource` (law + optional observation formula), `IsdmSources` (ordered names, laws as `(family, link)` pairs, observation formulas), constructors `isdm_source(family; observation = nothing)` and `isdm_sources(; kwargs...)` with the four `isdm_sources()` checks of section 2.1 as `ArgumentError`s carrying R's first line verbatim; `_isdm_admitted_law_id` returning `(fid, lid)` or `nothing`. Admitted Julia spellings: `Poisson()` (implies `LogLink()`), `(Binomial(), CLogLogLink())` or `Binomial(CLogLogLink())` wrapper; `Binomial()` bare (logit) is refused exactly as R refuses `binomial()`. |
| `src/families/isdm_table.jl` | `IsdmTable`: the validated long table. Fields: `y`, `n_trials` (all 1 on detection rows), `trait_id`, `unit_id`, `source_id`, `fid`, `lid`, `offset`, `X_fix` (dense, with column names), `rows_by_unit` (ranges after a stable sort by unit), level vectors, and the frozen observation basis. Builder `isdm_table(formula, data; family::IsdmSources, trait, unit)` runs, in R's order: selector alignment by name (`R/fit-multi.R:1432-1452`), `_isdm_declared_core` (`R/isdm-sources.R:412-439`), the within-trait scale rule with the admitted exception, the observation design with QR rank retention (`R/isdm-sources.R:178-277`), the offset gate with the cloglog exception (`R/offset.R:108-195`), the weights and multitrial refusals, and the observed-arm check. |
| `src/families/isdm_formula.jl` | The small formula reader: pull `offset(expr)` and exactly one `latent(0 + trait \| unit, d = K)` out of the RHS (mirrors `R/parse-multi-formula.R:100, 287` and `R/offset.R:14-16`), refuse any other structured term (`indep`, `dep`, `spatial_*`, `phylo_*`, `(1 \| g)`) with a named error, and hand the remaining fixed RHS to StatsModels with `0 + trait` full dummy coding so `X_fix` column names can be aligned to R's `X_fix_names` by name. |
| `src/families/isdm_laplace.jl` | The per-cell kernel over long rows. `_isdm_cell_mode(cell, Λ, b, θ...)`: damped Fisher scoring on `z_s` (length `K`) where each row contributes score `s_o` and weight `W_o` to `Λ[t(o), :]`; the Newton matrix is `A = Σ_o W_o λ_{t(o)} λ_{t(o)}ᵀ + I_K`, step halving on a decrease of the cell log-posterior, a `converged` flag (the rule PR #514 introduces, copied, not shared). `_isdm_cell_loglik`: `Σ_o ℓ_o(ẑ) - ½ ẑᵀẑ - ½ logdet(A_obs)`, with `A_obs` built from `_glm_obs_weight` on cloglog rows and `_glm_weight` on Poisson-log rows (Fisher equals observed there), which is the observed curvature TMB obtains by AD. `isdm_marginal_loglik_laplace(table, Λ, b)`: the sum over cells. `_isdm_dbinom_cloglog(y, η)`: a Julia copy of `gll_dbinom_cloglog` including the `eta < -20` series and the `700` cap, used instead of `_glm_logpdf(::Binomial)` on detection rows so the two engines agree in the tails (risk R2). |
| `src/families/isdm_fit.jl` | `IsdmFit` (fields: `b_fix` with names, `Λ` (p x K), `zhat` (K x n_units), `eta` (per row, the twin of `fit$report$eta`), `loglik`, `converged`, `iterations`, `table`, `sources`, `formula`, `hessian_used`). `fit_isdm_gllvm(table::IsdmTable; K, b_init, Λ_init, optimizer controls)`: packed θ = `[b_fix; vech-lower(Λ)]` (same lower-triangular layout as `src/packing.jl`), Optim LBFGS on the negative marginal with a ForwardDiff or finite-difference gradient (decision R6), warm start from a per-row link-scale pseudodata regression. The name mirrors the `fit_<family>_gllvm` pattern of `src/families/fit_gllvm.jl`. |
| `src/families/isdm_predict.jl` | `predict(fit::IsdmFit; newdata = nothing, type = :link, re_form = :all)` and `fitted(fit::IsdmFit; ...)`. Returns a `NamedTuple` of columns `(unit, trait, isdm_source, est)` in R's order; `re_form = :zero` (also `nothing`, `0`) gives `X_fix b_fix + offset`; `newdata` rebuilds the fixed design from the frozen basis by column name, re-evaluates the offset, re-adds `Λ[t, :] . ẑ[:, s]` for seen units, falls back to fixed-only on unseen units, refuses unknown or missing source labels with the two R classes as `ArgumentError` subtypes, and applies the per-row inverse link for `type = :response`. No `se_fit` (Q1). |
| `src/families/isdm_public.jl` | The door: a two-line branch at the top of `gllvm(formula, long_data; family, ...)` in `src/formula.jl:300` (`family isa IsdmSources && return fit_isdm_gllvm(isdm_table(formula, long_data; family, trait, unit); K = d from the formula)`), plus the once-per-session experimental notice mirroring `R/fit-multi.R:1487-1495`. This is the only edit outside new files, besides the `include` lines in `src/GLLVModels.jl` and the export list. |

Calling shape, matching R's (`tests/testthat/test-isdm-predict.R:38-43`):

```julia
fam = isdm_sources(gbif = Poisson(), survey = Binomial(CLogLogLink()))
fit = gllvm(@formula(value ~ 0 + trait + trait & env + trait & src_gbif +
                     offset(log_support) + latent(0 + trait | cell_id, d = 1)),
            dat; family = fam, trait = :trait, unit = :cell_id)
predict(fit; type = :response)
```

### 3.3 Why a long-table kernel and not a p x n matrix with masks

One could stack the sources as extra pseudo-traits (`p x D` rows) and mask the absent cells.
That reproduces the likelihood but not the contract: it forces one family per pseudo-trait,
makes `Λ` `pD x K` instead of `p x K`, and breaks the shared-loading semantics that make the
arms consistent. The long-table kernel keeps `Λ` at `p x K` and lets a row carry any admitted
law, which is what R does (`src/gllvmTMB.cpp:3083-3096` indexes `Lambda_B` by the row's trait).

### 3.4 Mapping of R test assertions to planned Julia twin tests

Tolerance vocabulary: `exact` means the same floating-point expression on both sides of the
Julia identity (tested with `==` or `atol = 1e-12`); `paired` means an R-versus-Julia number
recorded in a receipt; `class` means a thrown `ArgumentError` whose message starts with R's
first line. Every twin is written red first. Twin file: `test/test_isdm.jl` (default suite,
no R) for identities and refusals; `test/parity/isdm_cases.jl` (opt-in, RCall) for paired
numbers.

`tests/testthat/test-isdm-public-door.R`:

| R lines | assertion | Julia twin | tolerance | reason |
|---|---|---|---|---|
| 5-40 | cloglog offset admitted inside the contract, refused outside | `isdm_table` on a two-row table with a non-zero offset: passes when `allow_isdm_cloglog`, throws "offsets are supported for count families" otherwise | class | pure contract; identical predicate |
| 42-69 | the flag never opens the offset for gaussian, logit, probit, Beta | four `@test_throws` on the same message | class | same |
| 71-110 | contract requires both arms within every trait; split traits and a dummy portal row are refused | `_isdm_declared_core` on the three selector/trait vectors: `true, false, false` | exact boolean | pure predicate |
| 145-161 | `weights` refused inside the contract | `gllvm(...; weights = ...)` on the door fixture throws `gllvmTMB_isdm_weights_unsupported` text | class | same rule; Julia has no `weights` keyword on this door, so the keyword exists only to be refused |
| 163-179 | multi-trial detection rows refused | table with `n_trials = 2` on a detection row throws the multitrial text | class | same |

`tests/testthat/test-isdm-predict.R` (non-spatial, on the fixture at `:6-45`, regenerated in
Julia from the same literal `x`, `u_cell`, `alpha`, `beta`, `lam_tr` and stored as a literal
CSV under `test/fixtures/`, hash-verified, never from a seed):

| R lines | assertion | Julia twin | tolerance | reason |
|---|---|---|---|---|
| 55-65 | in-sample `predict()` equals `report$eta`, 120 rows, `est` present | `predict(fit).est == fit.eta` | exact | same vector by construction |
| 67-82 | response scale applies each row's own inverse link; PA in [0,1], counts > 0 | `exp` on count rows, `-expm1(-exp(.))` on detection rows | exact | same expression |
| 84-92 | `predict(newdata = training)` equals in-sample | equality of the two vectors | `atol 1e-10` | design rebuilt from names; floating reassociation only |
| 94-109 | `re_form = ~0` on newdata equals `X_fix b_fix + offset` and differs from `~.` | same two checks with `re_form = :zero` | `atol 1e-10`, and `std(diff) > 0` | same |
| 111-126 | `se.fit` finite in-sample, refused with newdata | only the refusal is twinned: `predict(fit; newdata, se_fit = true)` throws the R first line | class | intervals are out of scope (Q1) |
| 128-140 | unseen unit level falls back to fixed-only | `predict` on one cell relabelled `cNEW` equals `re_form = :zero` | exact | same fallback rule |
| 150-174 | `#1132` defect 3: newdata response uses each row's arm | newdata response equals in-sample response; detection rows in [0,1]; equals the cloglog inverse of the link prediction | `atol 1e-10` / exact | same |
| 176-210 | `#1132` defect 2: `re_form` honoured in-sample for `~0`, `NA`, `0`; `fitted` forwards; `~1` warns | `re_form` in `(:zero, nothing, 0)` all equal `X b + offset`; `fitted(fit; re_form = :zero)` equal; an unsupported form throws (Julia has no formula-valued `re_form`, so `:something_else` is an `ArgumentError`, not a warning) | exact / class | same rule; the warning-versus-error difference is recorded as a fence |
| 467-488 | in-sample output carries the `isdm_source` column, `est` last | `keys(predict(fit)) == (:cell_id, :trait, :isdm_source, :est)` and the column equals the table's source labels | exact | same |
| 511-528 | zeroing the offset in newdata changes link predictions by exactly `log(support)` | same | `atol 1e-12` | same arithmetic |
| 530-555 | `predict(newdata)` without the response column | drop `value` from `newdata`; equal to with-response results on both scales | exact | same |

Not twinned, with the reason recorded in the ledger row: `:212-259, 293-360` (SPDE spatial,
outside P1), `:261-291` (augmented random-slope tier, column grammar, outside P1),
`:382-458, 557-608` (`diag_species`, `rr_W`, `diag_W`, `propto`, `re_int` tiers on non-iSDM fits;
they belong to the covariance rows, not to ISDM-01 to 03), `:490-509` (single-family output
shape, a `predict` row, not an iSDM row). The legacy two-source route
(`list(gbif = poisson(), survey_pa = binomial("cloglog"))` with `isdm_family`,
`R/fit-multi.R:364-390`) is not twinned; R itself calls it backward compatibility (Q4).

Paired numeric twins (`test/parity/isdm_cases.jl`), on the same fixture:

| case id | comparison | tolerance |
|---|---|---|
| `P1-ISDM-LOGLIK-XOBJ` | Julia `isdm_marginal_loglik_laplace` evaluated at R's `(b_fix, Lambda_B)` versus R's `-fit$opt$objective`; and R's objective (via `fit$tmb_obj$fn`) at Julia's estimate versus Julia's own | `abs 1e-6` | cross-objective identity in both directions, the gate-tier bar; the cloglog tail copy and observed-curvature logdet make this a like-for-like Laplace |
| `P1-ISDM-ESTIMATES` | `b_fix` by name; `Lambda_B Lambda_B'`; per-row `eta` at each engine's own optimum | `rel 1e-4` on `b_fix` and the loading crossproduct, `abs 1e-4` on `eta` | different optimisers, same objective; residual is optimiser noise, and any larger gap is a finding |
| `P1-ISDM-PREDICT` | `predict` on link and response scales, in-sample and on a newdata grid with the offset zeroed | `abs 1e-4` | follows from estimates |
| `P1-ISDM-ADMISSION-20` | the 20 `CORE070-ISDM-*-PAIRED-CONTROL` predicates (section 4), each now run natively in Julia and compared with the R replay | exact boolean or error substring | contract twins, first executed on a Julia surface |

## 4. Receipts and scoreboard rows

Rows this port closes (P1 scoreboard, rows to be created by workstream 0 per the plan's
"isdm 1FO plus predict" entry):

| row | R register | closed by |
|---|---|---|
| `isdm/ISDM-PUBLIC-DOOR-1FO` | ISDM-01, ISDM-02 (`docs/design/capability-status.md:137-138` at P1) | `P1-ISDM-LOGLIK-XOBJ` and `P1-ISDM-ESTIMATES` PASS, plus all section 3.4 contract twins green |
| `isdm/ISDM-PREDICT` | ISDM-03 (`:139`) | `P1-ISDM-PREDICT` PASS plus the predict identity twins green |

Rows this port reclassifies: the 20 `CORE070-ISDM-*-PAIRED-CONTROL` rows in
`docs/dev-log/core070/isdm-batch-contract.json` (case ids ALIASED, ALIGN, COUNT, EXTRA-SOURCE,
LEGACY, MASKED-ARM, MASKED-COLUMNS, MISSING-IN-TRAIT, MISSING-SOURCE, MIXED, NO-OFFSET,
NO-TRAITS, SUPPORT, THREE, UNBALANCED, WITHIN-TRAIT-ADMIT, WRAPPER-LAW, WRONG-ID, WRONG-LINK,
ZERO-ORDINARY). All 20 cite receipts at `.unlazy/core070-aghq/isdm-admission/attempt2/receipt.json`
and `.unlazy/core070-aghq/isdm-admission/totoro-readback/receipt/receipt.json`
(`docs/dev-log/core070/isdm-admission-evidence.json:15, 24`), a git-ignored path that no longer
exists, and their source pins are the 0.7.0 hashes of `R/isdm-sources.R`, `R/fit-multi.R`,
`R/offset.R` (`isdm-batch-contract.json:9-13`), which changed by P1, so under the carry rule
they read `PARTIAL_STALE_AT_P1` until re-measured. `P1-ISDM-ADMISSION-20` re-measures them
against a Julia surface for the first time; the LEGACY case is re-measured but stays an R-only
disposition (Q4).

Each receipt (`docs/dev-log/core070/true-parity-latest/receipts/isdm/<case>.json`, tracked
in git per D-295 rule 4) must record, not merely a case id:

- P1 commit `9539352f6` and sha256 of the four R files read (`R/isdm-sources.R`,
  `R/fit-multi.R`, `R/offset.R`, `R/methods-gllvmTMB.R`) and of `src/gllvmTMB.cpp`,
  `src/gllvmTMB_cloglog.h`; the Julia commit and the sha256 of the six new Julia files;
- the fixture file hash and its row count;
- R numbers and Julia numbers side by side: log-likelihood at each optimum, both
  cross-objective evaluations, `b_fix` by name, `Lambda_B Lambda_B'`, `eta` max absolute
  difference, predict max absolute difference on both scales;
- both convergence verdicts (`fit$opt$convergence`, `pdHess`; Julia `converged`, all cells'
  mode `converged`), iteration counts, and the Hessian condition number where available;
- the tolerance applied, the measured gap, and the verdict;
- host, R version, Julia version, wall time.

## 5. Symbolic alignment table

| math term | R code (P1) | planned Julia location |
|---|---|---|
| admitted laws: Poisson-log (2, 0), Bernoulli-cloglog (1, 2) | `R/isdm-sources.R:15-26`; ids `R/fit-multi.R:1193-1213` | `isdm_sources.jl::_isdm_admitted_law_id` |
| per-row `(family_id, link_id)` from the source selector | `R/fit-multi.R:1451-1452`; `src/gllvmTMB.cpp:838-848` | `IsdmTable.fid`, `.lid` |
| within-trait mixing admitted only under the contract | `R/fit-multi.R:293-318, 1471-1501` | `isdm_table.jl::_isdm_assert_trait_scale` |
| every trait carries every source | `R/isdm-sources.R:412-439` | `isdm_table.jl::_isdm_declared_core` |
| `eta = X_fix b_fix + offset` | `src/gllvmTMB.cpp:1676` | `isdm_laplace.jl` (`η0 = X b + offset` per row) |
| `+ Lambda_B[t, :] . z_B[:, s]` | `src/gllvmTMB.cpp:3083-3096` | `_isdm_cell_mode` (row-indexed `Λ[t(o), :]`) |
| `z_B[:, s] ~ N(0, I_d)` | `src/gllvmTMB.cpp:1879-1882` | `-½ ẑᵀẑ` in `_isdm_cell_loglik` |
| `Lambda_B` lower-triangular packing | `src/gllvmTMB.cpp:34-59` | `src/packing.jl` (existing convention) |
| Poisson row `dpois(y, exp(eta))` | `src/gllvmTMB.cpp:3334-3336` | `_glm_logpdf(::Poisson)` `src/families/poisson.jl:13` |
| cloglog row, log-scale tails | `src/gllvmTMB_cloglog.h:45-57` | `isdm_laplace.jl::_isdm_dbinom_cloglog` (copy) |
| offset gate and the cloglog exception | `R/offset.R:157-159, 187-191` | `isdm_table.jl::_isdm_prepare_offset` |
| source-masked observation columns, QR rank retention | `R/isdm-sources.R:178-277` | `isdm_table.jl::_isdm_observation_design` |
| Laplace marginal with observed curvature | TMB `MakeADFun(random = "z_B")`, `R/fit-multi.R:7261` | `_isdm_cell_loglik` (`_glm_obs_weight` on cloglog rows) |
| `report$eta` at the mode | `src/gllvmTMB.cpp:4360` | `IsdmFit.eta` |
| `predict`, link, in-sample | `R/methods-gllvmTMB.R:2827-2833` | `isdm_predict.jl` |
| `predict`, `re_form` zero | `R/methods-gllvmTMB.R:2828-2830, 224-247` | same, `re_form = :zero` |
| `predict`, newdata design and offset | `R/methods-gllvmTMB.R:2927-2944`; `R/isdm-sources.R:284-403`; `R/offset.R:67-104` | same, frozen basis by name |
| `predict`, `rr_B` re-add for seen units | `R/methods-gllvmTMB.R:2949-2977` | same |
| `predict`, per-row inverse link | `R/methods-gllvmTMB.R:3155-3220` | same, dispatch on `(fid, lid)` |
| weights and multitrial refusals | `R/fit-multi.R:3462-3480` | `isdm_table.jl` |
| observed-arm check | `R/isdm-sources.R:445-477`; `R/fit-multi.R:3502-3528` | `isdm_table.jl::_isdm_assert_observed_arms` |
| experimental once-per-session notice | `R/fit-multi.R:1487-1495` | `isdm_public.jl` |

## 6. Risks, open questions, and estimates

### 6.1 Open questions for the maintainer (each with a recommendation)

**Q1. In-sample `se.fit`.** `test-isdm-predict.R:111-126` asserts a finite positive `se.fit`
in-sample; it is the fixed-effect-only delta-method standard error, which Design 127 says must
not be read as a map interval (`docs/design/127-isdm-prediction-map-implementation.md:102-122`).
Recommendation: keep it out of 1b and 1c, twin only the `newdata` refusal, and record the
in-sample `se.fit` as a named fence on the `isdm/ISDM-PREDICT` row. It is second-order work
and R's ledger does not claim it.

**Q2. The cloglog tail kernel.** R evaluates the detection likelihood with a series below
`eta = -20` and a cap at `700`; Julia's generic `_glm_logpdf(::Binomial)` goes through
`Distributions.Binomial` with a clamped `mu`. The two differ only in the tails, but a paired
log-likelihood at `1e-6` can fail on a sparse detection arm. Recommendation: copy
`gll_dbinom_cloglog` into `isdm_laplace.jl` as a private function and pin it against R at 24
`eta` values from `-40` to `720` (a pure-logic test, no R at run time; values recorded once).

**Q3. Fixed-design column parity.** R's `model.matrix` with treatment contrasts and Julia's
StatsModels with `DummyCoding` produce the same columns for `0 + trait + trait:env +
trait:src_gbif` but may order or name them differently (`trait&env` versus `traitsp1:env`).
Recommendation: align by a name-normalising map, compare `b_fix` by name, and refuse to pair
when any column has no partner. Position-based pairing is not acceptable.

**Q4. The legacy two-source route.** `isdm_family` with hard-coded `gbif`/`survey_pa` names is
recognised as the `n = 2` instance of the same predicate (`R/fit-multi.R:364-390`).
Recommendation: do not twin it. Record it as an R-only backward-compatibility disposition on
the ISDM-01 row; the public door covers the same likelihood.

**Q5. Gradient route for the fitter.** TMB differentiates the joint by AD and applies the
implicit-function step through the Laplace mode. Options in Julia: ForwardDiff through the
damped mode search (simple, correct at convergence, slower), or the implicit-step analytic
gradient pattern of `src/laplace_grad.jl` (faster, more code). Recommendation: ForwardDiff
first in 1b, with a finite-difference check at three random θ; the analytic gradient is a
later speed slice, not part of the parity claim.

**Q6. All-count declarations.** `isdm_sources(a = Poisson(), b = Poisson())` is accepted by R and
fits through the ordinary mixed route with no relaxation (`R/isdm-sources.R:149-155`).
Recommendation: accept it at the Julia door and route it through the same long-table kernel
(all rows Poisson), but receipt it under the mixed-family rows, not under ISDM-01 to 03.

**Q7. Missing responses.** `miss_control(response = "include")` masks rows and the observed-arm
check exists for that path. Recommendation: refuse `missing` in the response at the Julia door
for P1 (one named `ArgumentError`), and port the observed-arm check as a plain contract test so
the row is not silently narrower than R's; masking is a `mi()`/missing-data row.

**Q8. The `species` column in `predict` output.** R emits `unit, species, trait` from
`object$species_col` (`R/methods-gllvmTMB.R:2838-2860`); the iSDM fixture has no species
column and R resolves that internally. Recommendation: Julia emits `(unit, trait,
isdm_source, est)` and the twin compares on those four; whether R's `species` column is a
copy of `trait` on this fixture is measured in 1c, not assumed.

**Q9. Multiple maxima.** Issue #477 recorded that small-data NB2 likelihoods have several
maxima that Laplace ranks differently. Sparse detection arms can do the same here.
Recommendation: the cross-objective identity (both directions) is the first-order pass
criterion; a same-objective, different-optimum case is a recorded finding, not a tolerance
change, and the fixture uses `n_cell = 30` as R does.

**Q10. Engine = "julia" route (1d).** `.gllvmTMB_julia_dispatch()` (`R/julia-bridge.R:3600`)
serialises a `p x n` matrix and refuses family lists and offsets (`GJL-GATE-FAMILY`,
`R/julia-bridge.R:728, 771`; the offset refusal text at `:3952-3955`). An iSDM route needs a
new long-table payload (rows, fid, lid, offset, `X_fix` with names, unit and trait ids) and
a new Julia entry `bridge_isdm_fit` beside `bridge_fit` (`src/bridge.jl:486`). Recommendation:
build it as an additive branch keyed on `family_var == "isdm_source"`, refuse everything the
Julia door refuses with the same gate ids, and return `eta`, `b_fix`, `Lambda_B`, `loglik`.
This is R-side work under D-292 and needs its own gllvmTMB PR.

### 6.2 Risks

- R1: PR #514 changes the mixed-family mode search; the iSDM kernel copies its damping rule
  rather than sharing it, which is a deliberate duplication. Mitigation: a comment in the new
  file naming #514 as the source, and the scope note in `src/families/laplace.jl:190-215` is
  extended in a later chore to list `isdm_laplace.jl` (not in 1b, to keep #514's file
  untouched).
- R2: tail behaviour of the cloglog kernel (Q2).
- R3: design column naming (Q3).
- R4: the QR rank-retention rule for observation formulas depends on `qr()$rank`'s default
  tolerance (`R/isdm-sources.R:258-267`). Julia's `rank` uses a different default. Mitigation:
  port the rule with an explicit relative tolerance and test only on the fixture where the
  alias is exact; record the tolerance in the receipt.
- R5: receipts for the 20 admission rows are lost and their pins are stale; nothing is
  carried, everything is re-measured (D-295 rule 1).
- R6: estimate risk on 1b if the ForwardDiff gradient through the damped mode search proves
  too slow on the 120-row fixture; fallback is a finite-difference gradient with a cost note.

### 6.3 Estimates

Stated before any run (D-287). No simulation or campaign is planned in 1a to 1d; every fit is
the 120-row fixture, under five seconds per fit in R and expected under one second in Julia,
so nothing approaches the three-hour line.

| slice | work | estimate |
|---|---|---|
| 1b build | `isdm_sources.jl`, `isdm_table.jl`, `isdm_formula.jl` (3 days); `isdm_laplace.jl` with the cloglog copy and the FD gradient check (3 days); `isdm_fit.jl`, `isdm_predict.jl`, `isdm_public.jl`, docstrings, tutorial page, README row, `capability-status.md` row (3 days); review and the Rose walk-around (1 day); buffer for Q3 and R6 (2 days) | 10 to 12 days |
| 1c tests and receipts | `test/test_isdm.jl` red-first twins (1 day); `test/parity/isdm_cases.jl` with the four paired cases and the 20 admission twins (1.5 days); receipts written and the two scoreboard rows plus the 20 reclassified rows updated (1 day); Totoro re-run for the second host (0.5 day) | 3 to 4 days |
| 1d bridge route | gllvmTMB additive branch and payload, `bridge_isdm_fit`, one paired receipt through `engine = "julia"`, gate ids and tests (Q10) | 2 to 3 days |

Dependencies: 1b starts after #514 merges (its file is not touched, but the damping rule is
copied from the merged text); 1c after 1b; 1d after 1c only.

## 7. Provenance and references

- gllvmTMB at `9539352f6` (read-only reference; no edits): `R/isdm-sources.R`,
  `R/isdm-contract.R` (developer route, not twinned), `R/fit-multi.R`, `R/offset.R`,
  `R/methods-gllvmTMB.R`, `R/julia-bridge.R`, `src/gllvmTMB.cpp`, `src/gllvmTMB_cloglog.h`,
  `tests/testthat/test-isdm-public-door.R`, `tests/testthat/test-isdm-predict.R`,
  `docs/design/120-multi-source-isdm-contract.md`, `docs/design/126-isdm-prediction-api.md`,
  `docs/design/127-isdm-prediction-map-implementation.md`,
  `docs/design/111-isdm-nonspatial-recovery-protocol.md`, `docs/design/capability-status.md`.
- gllvmTMB issue #1238 (iJSDM response-information forensic follow-up): parked as a Claude
  handover on 2026-09-24; its `EVIDENCE_INCOMPLETE` boundary concerns an internal
  response-information study (register row ISDM-RESP-INFO, R-only, no Julia twin expected),
  so it neither blocks nor widens this port.
- Vault: D-292, D-294, D-295; dr30 (Fithian et al. 2015 identifiability result).
- Julia at `1385b0490`: `src/families/laplace.jl`, `src/families/mixed.jl`,
  `src/families/binomial.jl`, `src/families/poisson.jl`, `src/families/links.jl`,
  `src/families/covariates.jl`, `src/spde_latent.jl`, `src/source_fit.jl`, `src/formula.jl`,
  `src/postfit.jl`, `src/bridge.jl`, `test/parity/README.md`, `test/parity/core070_receipts.jl`.
- Fithian, W., Elith, J., Hastie, T. and Keith, D. A. (2015). Bias correction in species
  distribution models: pooling survey and collection data for multiple species. Methods in
  Ecology and Evolution 6, 424-438.
- Kristensen, K. et al. (2016). TMB: automatic differentiation and Laplace approximation.
  Journal of Statistical Software 70(5).
