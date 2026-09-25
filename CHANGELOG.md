# Changelog

## Development

- **Package renamed to GLLVModels.jl.** Install and load it as
  `GLLVModels`; modelling functions such as `gllvm`, `fit_gllvm`, and `bf`
  retain their existing API. `GLLVModels.GLLVM` is a temporary source-level
  alias, but `using GLLVM` cannot remain available after a Julia package rename.
  The GitHub repository rename and Pages migration remain separate maintainer
  gates; historical development records retain their original spelling.

All notable changes to GLLVModels.jl are documented here.

## Unreleased

### Fixed
- **Gamma grouped fits no longer report convergence from a diverged inner search.**
  The per-site mode search inside `fit_gamma_gllvm_grouped` and its covariate and
  shared-shape routes could diverge at the fitter's own start and still return a
  finite value, so a fit could report `converged = true` far below the optimum (43
  log-likelihood units on the case in #479). The search now halves steps that lower
  the site objective and, under the log link, falls back to damped Newton; a site that
  still does not converge returns `-Inf`, so the fitter's failure sentinel fires.
  Values where the old search converged are unchanged (to 1e-8). Healthy fits run
  about 20 to 25 percent slower.
- **Parity tools now refuse a wrong `GLLVM_PARITY_R_LIBS`, dev-only.** Seven `tools/core070_*`
  parity scripts and `test/parity/parity_helpers.jl` called `library(gllvmTMB)` directly, so a
  library named by `GLLVM_PARITY_R_LIBS` with no gllvmTMB was silently ignored in favour of R's
  default library. They now route through the same guard as `tools/core070_second_order`
  (`second_order_r_lib()`) and refuse instead. No effect on `gllvm`/`fit_gllvm`.
- **Beta grouped fits no longer report convergence at a non-stationary point.**
  `fit_beta_gllvm_grouped` (the default route for `fit_gllvm(...; family = Beta())`)
  and `fit_beta_gllvm_grouped_cov` could report `converged = true` after Optim stopped
  on a zero-length line-search step with a large gradient, well below a better point
  (#480). They now report convergence only when the gradient meets a scale-aware test,
  `g_residual <= max(g_tol, g_tol * |nll|)` (the rule the Tweedie fitter uses), and
  otherwise restart once from a neutral start and once from the returned point,
  keeping the best only if it is better. Some fits that used to claim convergence now
  report `converged = false`; the cause there is the inner mode search (#482).
- **NB2 with per-trait dispersion: fewer fits stuck at the Poisson boundary.**
  `fit_nb_gllvm_grouped`, the default no-covariate route for
  `fit_gllvm(...; family = NegativeBinomial())`, could stop with a trait's `r`
  pushed toward the Poisson limit, well below a better point (#477). When a group
  ends at the boundary, the fit now restarts with the boundary groups at `r = 1`,
  together and each on its own, and keeps the best result only if it raises the
  log-likelihood. On 16 datasets of the NATIVE-06 design this lifted 7 fits by 0.46
  to 2.59 log-likelihood units and lowered none
  (`docs/dev-log/core070/nb2-boundary-screen-20260924/`). Fits that never reach the
  boundary are unchanged. On small data the NB2 likelihood can have several maxima,
  so a fit is not guaranteed to reach the highest one. The covariate route
  (`fit_nb_gllvm_grouped_cov`, used by `gllvm(@formula(...), ...)` with NB2 and site
  covariates) gets the same restart; it had the same stall on 3 of 10 screened datasets.
- **Gamma with fixed-effect covariates no longer throws `DomainError`.**
  `fit_gllvm_cov(Y; family = Gamma(...), X, K)` stopped with
  `DomainError("Gamma: alpha > 0")` on ordinary Gamma data, and so did
  `fit_gllvm_speciescov`, `fit_fourthcorner_gllvm`, `fit_roweffect_gllvm` and
  `fit_constrained_gllvm`, which share its per-site Laplace search. That search took
  undamped steps and could return a finite value from a diverged site (about -7.2e22
  at the fitter's own start), which sent the optimizer to a shape of 0. The search now
  halves steps that lower the site objective, falls back to damped Newton on the
  observed curvature, and returns `-Inf` for a site that still does not converge, so
  the fitter's failure sentinel fires. The dispersion is also built inside the
  objective's guard, so an out-of-range value returns the sentinel instead of an
  error. On the #479 data the Gamma fit now reaches -569.700833, the same optimum as
  `fit_gamma_gllvm`. Values where the old search converged are unchanged (to 1e-8).
  **Behaviour change:** `fit_gllvm_cov` with Binomial went from converged to NOT
  converged on 2 of 8 screened datasets, with loadings running away (largest 42.2 and
  41.2, was 3.5 and 3.1) at a higher log-likelihood (-211.64, was -236.13; -214.44,
  was -241.08). The old fits had stopped where unconverged sites fed the optimizer
  wrong values; the no-covariate `fit_binomial_gllvm` runs away on the same data too.
  Very dispersed Gamma data with responses near 1e-29 (true shape 0.1) now fail with
  log-likelihood `-Inf` and `converged = false` instead of throwing. Healthy fits run
  about 20 percent slower.

### Changed
- **Breaking (default change):** `fit_delta_lognormal_gllvm` / `fit_delta_gamma_gllvm`
  (and `fit_gllvm(...; family = DeltaLogNormal()/DeltaGamma())`) now default to
  `disp_group = :species` — **one dispersion per trait**, matching R gllvmTMB's
  per-trait dispersion, instead of one scalar shared across all traits.
  **Existing delta fits that omitted `disp_group` will see `fit.σ` / `fit.α`
  change from a scalar to a length-`p` vector.** Pass `disp_group = :shared`
  explicitly to keep the previous one-scalar-per-model behaviour. The per-trait
  default can also move an existing fit to a different optimum or change whether
  it converges: one existing offset test did, and now passes `disp_group = :shared`.
  Decision:
  `docs/dev-log/decisions/2026-09-15-delta-dispersion-alignment-pending.md`
  (accepted 2026-09-24).
- **Development (D3 Stage 1):** `fit_gaussian_gllvm(y; K, lambda_constraint = M)`
  fits a confirmatory ordinary J1 Gaussian model with the numeric entries of a
  `p × K` pin matrix `M` (`NaN` = free) held fixed, mirroring R gllvmTMB's
  `lambda_constraint`. `loading_profile(fit; y, ...)` now profiles the free
  entries of such a confirmatory fit on a grid — the confirmatory mirror of
  R's `loading_profile()`, distinct from the existing exploratory
  `loading_profile_exploratory`. Stage 1 scope only (`X = nothing`, no W/diag/
  phylo blocks, no `aghq`); this is a Stage 1 receipt, not full R grid parity.
  The deprecated generic `loading_profile` shim is unchanged and still serves
  the old three-positional-argument calling convention.
- **Development:** fixed Gaussian source fits accept a complete mean design `X`.
  Explicit-source wide/long formulas expose trait intercepts, shared slopes,
  categorical contrasts and interactions. Fits retain the copied design, response
  shape and coefficient labels. No R source-grammar, bridge, random-slope or
  calibrated-uncertainty claim is implied.
- Truncated NB2 exposes its existing per-trait fitter through explicit
  `disp_group=:species` in native and intercept-only wide/long formula calls.
  Shared dispersion remains the default; partial grouping is rejected.
- **Development:** per-trait Gaussian fitting accepts an explicit fixed residual
  SD and estimates unique variances within that constraint. Results retain unique
  and total diagonal variances; existing defaults and parameter counts are unchanged.
  The `pervar=true` Gaussian formula route preserves trait intercepts,
  zero-mean designs, shared slopes and categorical contrasts. Bridge and interval
  parity are not established. AGHQ requests on the fixed-residual unique model
  retain exact Gaussian/Laplace with warning and provenance; `k=1` is quiet.
  Plain heteroscedastic AGHQ remains unimplemented and is labeled separately.
- Fix default exact Gaussian fitting with a zero-column design or all fixed-zero
  coefficients, including ordinary and phylogenetic profiled likelihoods.
- Restore Wald/profile intervals for these models; failed profile refits no
  longer fabricate finite bounds without a finite likelihood crossing.
- Add explicit `sigma_eps_fixed` for Gaussian source fits, retaining the free
  residual default. Starts and parameter counts omit the fixed coordinate;
  diagnostics differentiate only free parameters and retain fixed-noise provenance.
- Development interface: typed fixed Gaussian source covariances, additive
  projected-source fitting, and explicit gradient/Hessian diagnostics. Independent,
  latent and dependent trait modes pass targeted tests. Six retained nonspatial
  models match public R 0.7.0 fits under the declared likelihood/health checks.
  The unique-variance case has nearly singular curvature; intervals are unverified.
  Existing fitters are unchanged. No source-kernel estimation, R source-grammar/bridge,
  non-Gaussian, inference, recovery or performance completion claim.
- Local binomial candidate: opt-in ordinary AGHQ with logit/probit/cloglog,
  retained trials/masks/offsets, estimator metadata and frozen-objective inference.
  Public prediction returns probabilities and simulation returns counts. The
  original k5 paired convergence/likelihood gate remains failed, not waived.
- Local candidate: ordinary log-link Poisson fits accept opt-in `aghq` controls
  and retain integration metadata, final adaptation and every start outcome.
  Predictions preserve offsets; Wald/profile intervals use the fitted frozen
  objective and bootstrap refits retain failures. Other-family/structured AGHQ
  and calibrated coverage are not established. Default Laplace is unchanged.
- Generic family Wald intervals now require a positive-definite Hessian;
  a positive inverse diagonal alone no longer yields apparent valid uncertainty.
- Ordinal fitters now reject unsupported links with an early `ArgumentError`;
  logit and probit models and their likelihoods are unchanged.
- Formula fits validate every supplied site-table column before response access,
  including intercept-only fits. Empty tables remain valid without covariates.
  The original NB2 wide/long formula model matches its native fit; broader
  interface qualification remains incomplete.
- **Local development candidate:** ordinary NB2 evaluates its density directly
  from the mean and uses an overflow-safe observed curvature. The original
  paired fit, scalar derivatives and required NB2/truncated-NB2 runner pass.
  Broader recovery, full package checks and independent review remain pending.
- Student-t fitting rejects infinite fixed degrees of freedom before reading
  responses, for scalar and per-trait inputs. Finite positive fixed values and
  the estimated-df route are unchanged.
- **Local development candidate:** grouped Tweedie distinguishes fixed common,
  shared estimated and per-species estimated power; `TweediePerTraitPowerFit`
  stores the latter. Student fits record whether degrees of freedom were estimated
  so information-criterion parameter counts are correct. These changes do not
  establish full Core070 parity: the original Student-t R health gate remains
  failed, and final candidate requalification is pending.
- Branch-RE uses an equivalent dense marginal fallback when the auxiliary sparse
  precision is numerically unsafe. This preserves valid marginal models without
  a ridge, at a possible O(p²) memory cost reported by a warning.
- **AGHQ Stage-1a site evaluator now threads the Fisher-vs-observed curvature
  selector (unpark Slice 0/1, 2026-08-28)**: `aghq_stage1a_loglik_site`
  (`src/families/aghq_grid.jl`, internal, no public `aghq=` surface) gains a
  `hessian::Symbol = _default_hessian(family, link)` keyword, mirroring the
  role-separation contract in `laplace_loglik_site`/`covariates.jl` exactly —
  the Newton mode search stays Fisher-scored; only the adaptation curvature
  (the log-det AND the per-site Cholesky reused across every quadrature node)
  is selectable. Previously the site evaluator computed an unconditional
  Fisher weight for both roles, which had silently diverged from the same
  family's own default Laplace fitter for every family whose
  `_default_hessian` is `:observed` (Beta, Gamma, NegativeBinomial, NB1,
  StudentT, Exponential, TruncatedNegBin2, and — since the 2026-08-28
  curvature-flip decision batch — TweedieED and Binomial-probit): the AGHQ
  `k=1` template no longer equaled that family's own Laplace golden once its
  default flipped. `hessian = :fisher` pinned reproduces the pre-fix value
  bit-for-bit (verified against an independent copy of the pre-change
  unconditional-Fisher formula); the family-default `k=1` template now
  matches that family's default dense Laplace marginal to 1e-10 for all 9
  affected families (new cross-check in `test/test_aghq_grid.jl`). A PD guard
  keyed on the weight's sign (matching `laplace.jl`) was added so a negative
  observed weight (Beta/Student-t) fails to `-Inf` rather than throwing out of
  `cholesky`. The module remains internal and FENCED — this closes only the
  AGHQ instance of the Fisher-vs-observed fault class, not the class
  generally (`docs/design/capability-status.md`). No ledger row promoted.
- **`confint`/bootstrap now rebuild the fit's own objective** (the
  curvature-consistency class from the 2026-08-27 adversarial audit): every
  one-part fit struct records the `hessian` its objective used, and the CI
  adapters thread `fit.hessian` into both the rebuilt marginal and the
  bootstrap refit. Previously an explicit `hessian = :fisher` fit received
  default-curvature CIs silently — the ":fisher restores previous behaviour"
  claim was true at fit time only; it is now true through `confint`.
  Positional fit-struct construction stays source-compatible (a compat
  constructor defaults the new field to the family default). **Closed
  2026-08-28**: the GROUPED fit structs (NBGroupedFit, NBGroupedCovFit,
  BetaGroupedFit, BetaGroupedCovFit, GammaGroupedFit, GammaGroupedCovFit,
  NB1GroupedFit, NB1GroupedCovFit) now also record `hessian`, and their
  `_family_ci` adapters thread it the same way; TweedieGroupedFit and
  BetaBinomialGroupedFit/BetaBinomialGroupedCovFit have no `hessian`
  selector on their underlying kernel at all (unconditional Fisher weight),
  so their field is fixed at `:fisher` by construction — there is nothing to
  thread. Residual: Student-t has no confint adapter at all (pre-existing
  gap, now recorded).
- **Beta, NB1 and Student-t Laplace log-determinants now use the observed
  conditional curvature, completing decision A (2026-08-27)** — with Gamma,
  NB2 and Exponential, every one-part family except Tweedie and GP-1 now
  matches TMB's log-det. The evidence (900-cell adjudication campaign): the
  observed curvature's *estimates* land closer to the exact-marginal optimum
  in 90–100% of medium/strong cells for all three; its *reported loglik* is
  measurably more biased for them (Beta |err| 0.50→1.22; NB1 0.54→1.09;
  Student-t 1.60→2.08) — that cost was accepted explicitly for TMB parity and
  estimator quality. Reported logliks, AIC/BIC and Wald SEs change for these
  families; `hessian = :fisher` restores the previous objective at fit time.
  Beta's analytic gradient log-det weight moved in the same commit; the
  grouped Beta/NB1 marginal evaluators align with their (already-observed)
  fitters. Beta's and Student-t's genuinely-negative observed curvature is
  handled by the assembly-level PD guard (measured, load-bearing).
- **Exponential's `_default_hessian` is now declared `:observed` at the
  registry level** (it was only a fitter-signature default since 2026-08-24),
  so the covariates/quadratic/mixed/SPDE/phylo-GLM/coevolution kernels agree
  with the shipped default. The `:observed` route also re-routed through the
  generic Laplace core, retiring a grouped-kernel detour whose undamped
  per-site Newton loop produced garbage marginals at moderate parameters
  (measured: −5.0e23 against an exact −1717.6) — the cause of runaway
  `‖Λ̂‖ ~ 10³` fits that still reported `converged = true`.
- **NB2's Laplace log-determinant now uses the observed conditional curvature
  `μ·r·(r+y)/(r+μ)²`, matching TMB / `gllvmTMB`** (previously the Fisher weight
  `μr/(r+μ)`). This **changes reported `loglik` values for
  `fit_nb_gllvm` / shared-route NB2 fits** and anything Hessian-derived (Wald
  SEs); point estimates move little. Evidence: the 900-cell curvature-
  adjudication campaign (2026-08-27) — observed's estimates land closer to the
  exact-marginal optimum in 100% of medium/strong cells *and* its objective
  value approximates the exact marginal better in 87% of cells; NB2 and Gamma
  are the two families where both metrics agree. The analytic gradient's
  log-det weight moved in the same commit (they must move together).
  `hessian = :fisher` restores the previous behaviour. The grouped
  (per-trait dispersion) NB2 route already used the observed weight.
- **Gamma's Laplace log-determinant now uses the observed conditional curvature
  `α·y/μ`, matching TMB / `gllvmTMB`** (previously the Fisher weight, the
  constant `α`). This **changes reported `loglik` values for Gamma fits**, and
  anything derived from the Hessian such as Wald standard errors; point
  estimates move very little, since the conditional mode is score-determined.
  Measured 12/12 closer to numerical quadrature with 20–60× smaller error.
  `hessian = :fisher` restores the previous behaviour. Other families are
  unchanged — this was a per-family decision on per-family evidence, not a
  global switch.
- **TweedieED (shared route) and Binomial/probit Laplace log-determinants now
  use the observed conditional curvature, matching TMB / `gllvmTMB`**
  (maintainer decision batch, 2026-08-28 — TMB structurally differentiates the
  joint negative log-likelihood, so its log-det is observed for every family
  it ships). This **changes reported `loglik`, AIC/BIC and Wald SEs** for
  `fit_tweedie_gllvm` and `fit_binomial_gllvm(...; link = ProbitLink())`;
  point estimates are expected to move little (the conditional mode stays
  Fisher-scored). `hessian = :fisher` restores the previous objective for
  both.
  - TweedieED/log: `μ^(1−p)·[(2−p)·μ + (p−1)·y] / φ`. The Tweedie density's
    normalising series is μ-free, so it contributes zero η-curvature and this
    closed form is exact (not an approximation); no analytic-gradient
    coupling was needed (`fit_tweedie_gllvm` is finite-difference only).
    Always non-negative.
  - Binomial/probit: `η·φ(η)·(y−nμ)/(μ(1−μ)) + φ(η)²·[y/μ²+(n−y)/(1−μ)²]`,
    μ = Φ(η). Provably non-negative for every cell (the probit binomial
    log-likelihood is globally concave in η, Pratt 1981 *JASA*) — unlike
    Beta/Student-t, the PD guard is not expected to fire here. No
    analytic-gradient coupling was needed either: `binomial_laplace_grad` is
    hardcoded to `LogitLink()` in its mode solve, so a probit fit was already
    routed to finite differences regardless of `hessian`, before and after
    this change.
  - Binomial/**cloglog** is explicitly excluded (the diagnosed Laplace
    saturation pathology) and stays `:fisher`.
  - **Recorded, not fixed:** the Tweedie **grouped** route
    (`fit_tweedie_gllvm_grouped`) has no `hessian` selector at all
    (unconditional Fisher) — with `G = 1` it no longer matches the shared
    route's new default, only `hessian = :fisher` on the shared route. Fenced
    in `docs/src/response-families.md` and `docs/src/gllvmtmb-parity.md`
    rather than aligned; out of scope for this change.

### Fixed
- **`compoisson_logz` no longer truncates silently past its term cap**: for
  large `λ^{1/ν}` the series' dominant terms outran the fixed summation cap and
  the partial sum understated log Z. Past 80% of the cap the function now
  switches to the Shmueli et al. (2005) asymptotic
  (`ν·λ^{1/ν} − ((ν−1)/(2ν))·log λ − ((ν−1)/2)·log 2π − ½·log ν`), exact at
  ν = 1 and validated against the direct series on the crossover band
  (rel. err. ≤ 3e-8); the switch is monotone across the branch boundary
  (tested at ν = 1 and ν = 2), and integer arguments still work (a
  pre-commit review caught the guard computing `T(0.8)` with integer `T`).
- **`fit_gaussian_gllvm` could report `converged = true` while stranded on the
  PosDef penalty plateau**: the catch's ad-hoc `1e10` penalty sat below the
  sentinel screen threshold and both return sites used raw `Optim.converged`.
  The catch now returns `_NLL_SENTINEL` and both return sites screen the
  verdict through `_fit_verdict` (the same sentinel-escape class fixed for
  Tweedie on 2026-08-26). Well-behaved fits are unchanged.
- **`fit_tweedie_gllvm` reported `converged = true` at points that were not
  maxima.** The log warm start `log(max(Y, 1e-6))` sent every structural zero to
  −13.8 regardless of the data scale, wrecking the intercepts and inflating the
  SVD loadings; from there the optimiser stalled, and two independent flaws in
  the convergence verdict advertised the stall as success. `Optim`'s *relative*
  f-change test fired on an objective of size ~1e11 while the gradient residual
  was still ~1e15, and the bare `1e12` failure value formed a perfectly flat
  plateau whose finite-difference gradient is exactly zero, so `Optim` reported
  gradient convergence and `−1e12` was returned as a maximised log-likelihood
  with `p̂ = 1.0`, outside the documented open interval `(1,2)`. On the family's
  own shipped test cell the default start landed ~9 orders of magnitude below a
  neighbouring start. The warm start now offsets by `0.1 · mean(Y[Y > 0])`, and
  `converged` additionally requires a successfully evaluated objective, a
  strictly interior power, and a gradient residual small *relative to the
  objective's scale*; failures return `converged = false` (with `loglik = -Inf`
  rather than the sentinel) and warn. A power-start sweep over
  `p_init ∈ {1.1 … 1.9}` now agrees to 8 significant figures where it previously
  spanned 9 orders of magnitude. Pinned by
  `test/test_tweedie_engine_health.jl`.
- **`fit_tweedie_gllvm_grouped` reported `converged = true` at the same class
  of non-maxima.** It carried the three defects #236 closed on the scalar
  fitter: `log(max(Y, 1e-6))` warm start, a bare `1e12` failure sentinel, and
  a naked `Optim.converged` verdict. It now uses `_tweedie_log_offset` and
  `_tweedie_verdict`. A one-group power-start sweep on the shipped cell agrees
  with `fit_tweedie_gllvm`, and a per-species sweep on the existing grouped
  smoke cell agrees with itself. Pinned by
  `test/test_tweedie_grouped_engine_health.jl`. The `fit_gllvm` bare-marker
  admit stays shut.
- **Every non-Gaussian Wald confidence-interval standard error was silently
  corrupted.** The observed-information finite-difference Hessian (`_fd_hessian`,
  backing `confint(fit, Y; method=:wald)` / `_family_wald` for Poisson / Binomial /
  NB / Beta / Gamma / Tweedie / two-part / SPDE-latent / structural CIs) wrote the
  Float32 literal `2f0` (== `2.0f0`) where `2 * f0` was intended, dropping the
  cached centre term so the diagonal exploded and `inv(H)` collapsed to SEs ~1e-6.
  Off-diagonals, the profile/bootstrap routes, and the Gaussian path were
  unaffected. Pinned by `test/test_fd_hessian.jl`.

### Added
- **`predictor::Symbol = :separate | :shared` on `fit_delta_lognormal_gllvm` /
  `fit_delta_gamma_gllvm`** (2026-08-28, maintainer decision "Twin identity
  MODE" — `docs/dev-log/decisions/2026-08-28-arc-decision-batch.md` gate 4):
  `:shared` reproduces gllvmTMB's delta-family design where ONE linear
  predictor drives both the occurrence and positive-value components
  (`gllvmTMB.cpp:2816-2844`) — `βz ≡ βc`, `Λz ≡ Λc`, optimised over the
  smaller `[β; vec(Λ); log dispersion]`. Offset threads to both parts
  symmetrically under `:shared`, matching the twin's single shared `eta`
  construction (`gllvmTMB.cpp:1401`). `:separate` (default) is unchanged and
  bit-identical to the pre-existing behaviour; `DeltaLogNormalFit` /
  `DeltaGammaFit` gain a `predictor::Symbol` field (positional-compat
  constructor keeps old 7-arg call sites defaulting to `:separate`). See
  `docs/dev-log/decisions/2026-08-28-delta-shared-predictor-identity.md` and
  `test/test_delta_shared_predictor.jl`.
- **All ten remaining two-part entry points expose the `hessian` curvature
  selector** (`fit_delta_lognormal_gllvm`, `fit_hurdle_poisson_gllvm`,
  `fit_hurdle_nb_gllvm`, `fit_zip_gllvm`, `fit_zinb_gllvm`, `fit_zib_gllvm`,
  `fit_beta_hurdle_gllvm`, and the covariate variants `fit_zip_gllvm_cov`,
  `fit_zinb_gllvm_cov`, `fit_zib_gllvm_cov`; DeltaGamma already had it) —
  kwarg validation and threading to the two-part Laplace kernel, default
  `:observed` (bit-identical to the previous behaviour; proven by test for
  an invalid selector and the `:fisher ≡ :observed` identity on all-ten
  coverage in `test_twopart_hessian_kwarg.jl`). Honest scope: the kernel's
  observed count-part weight is currently specialised only for DeltaGamma,
  so for the other families both selectors coincide until their observed
  weights land (the recorded two-part curvature gap); each docstring and the
  response-families page state this. The exposure is the measurement
  prerequisite for closing that gap.
- **`confint_lv_effects(fit, Y, X_lv; method = :wald | :bootstrap)`** — Wald
  (delta-method) and parametric-bootstrap confidence intervals for the
  predictor-informed latent-score trait-effect matrix `B_lv = Λ·α'`, for Gaussian
  and the five GLM families (Poisson / Binomial logit·probit·cloglog / NB2 / Gamma /
  Beta), `K ≥ 1` (`B_lv` is rotation-invariant). The bridge exposes them via
  `bridge_fit(...; X_lv, options = Dict("ci_method" => "wald"))`, which returns
  `lv_effects_lower` / `lv_effects_upper` / `lv_effects_se`. K = 1 interval
  coverage 0.915–0.955 across all eight routes; K = 2 recovery + coverage validated
  for Poisson. R-side reading of these CI fields is not yet wired.

## v0.3.0 — broad gllvmTMB-targeted capability build-out (2026-06-07)

Expanded toward R `gllvm` / `gllvmTMB` coverage with a broad response-family
surface, per-species dispersion across all five dispersion families plus Gaussian
per-species variance, ordinal logit + probit, fixed **and random** row effects, a
unified `fit_gllvm` dispatch, Wald/profile/bootstrap CI routes + aic/bic, plus
capabilities that are outside the current R bridge surface (Conway–Maxwell–Poisson,
phylogenetic-GLM and SPDE-spatial engines), a JuliaConnectoR R-bridge scaffold,
and strictly bit-exact performance work. Public parity is row-scoped through the
capability/bridge matrix; every promoted numerical addition is gated by
deterministic tests (machine-precision `Λ=0`/limit reductions,
gradient-vs-finite-difference checks), validated on Linux/macOS/Windows.

- **Phylogenetic GLM** (`fit_phylo_glm` / `PhyloGLMFit`) — a per-species
  phylogenetic random intercept for the non-Gaussian families (Poisson / NB /
  Binomial, with a dispersion parameter for the dispersion families) via an
  augmented-state joint Laplace over the sparse phylogenetic precision — the
  internal fast phylogenetic-Poisson path (issue #61).
- **Zero-inflated binomial (ZIB)** (`fit_zib_gllvm` / `ZIBFit`) — structural-zero
  × Binomial two-part family, with Wald / profile / parametric-bootstrap
  confidence intervals.
- **Negative-binomial type-1 (NB1)** (`fit_nb1_gllvm`) — linear variance
  `Var = μ(1+φ)`, alongside the existing NB2 (`Var = μ + μ²/r`).
- **Beta-binomial** (`fit_beta_binomial_gllvm` / `BetaBinomialFit`) — overdispersed
  binomial `BetaBinomial(N, μφ, (1−μ)φ)`, matching gllvm family 15 (`φ→∞ ⇒ Binomial`).
- **Conway–Maxwell–Poisson** (`fit_compoisson_gllvm` / `COMPoissonFit`) — counts with
  **under- or over-dispersion** (`ν>1`/`ν<1`; `ν=1 ⇒ Poisson`), a family beyond gllvmTMB.
- **Per-species / grouped dispersion** (`disp.group`) for all five dispersion
  families — `fit_{nb,beta,gamma,nb1,tweedie}_gllvm_grouped(Y; K, group)`; reduces
  exactly to the shared-dispersion fit at one group. Matches gllvm's per-species default.
- **Gaussian per-species (heteroscedastic) variance** (`fit_gaussian_pervar_gllvm`)
  — `Var_j = φ_j²`, gllvm's Gaussian default; reuses the low-rank Woodbury Cholesky.
- **Ordinal cumulative-probit link** (`fit_ordinal_gllvm(...; link=ProbitLink())`)
  alongside logit (gllvm's default ordinal); convention verified `P(y≤c)=F(τ_c−η)`.
- **Random row effects** (`fit_row_random_gllvm` / `RowRandomFit`) — per-site
  `ρ_s ~ N(0, σ_row²)` integrated out (gllvm `row.eff="random"`), alongside the
  existing fixed row effects.
- **Unified `fit_gllvm` dispatch** — `row_eff` / `disp_group` / `pervar` / `num_lv`
  keywords route to the right fitter (the call target for the JuliaConnectoR bridge).
- **Confidence intervals** extended to ZIB, beta-binomial, and random-row fits;
  **aic/bic** for all the new fit types.
- **JuliaConnectoR R-bridge scaffold** (`r/gllvmtmb_julia.R`, `r/parity_check.R`)
  mapping gllvmTMB-style calls to GLLVModels.jl with the documented parameterization
  conversions (NB `r=1/φ`, …) + an R-vs-Julia parity harness.
- **Performance** — strictly bit-exact allocation reductions in the Laplace and
  two-part mode-finders and the Poisson/NB fit objective (no result change; the
  suite's machine-precision anchors are the guard); a `bench/` speed harness and
  a literature-backed speed roadmap (`bench/SPEED_NOTES.md`).

## v0.2.0 — Full GLM family, VA, covariates, ordination

A large expansion from the v0.1.0 **Gaussian-only** pilot to a broad,
gllvmTMB-class GLLVM package. Every numerical addition is gated by deterministic
tests (exact `Λ=0`/`B=0`/`D=0` reductions, ELBO lower-bound / quadrature checks,
analytic-vs-finite-difference gradient checks), validated on Linux/macOS/Windows.

### Response families (Laplace-approximated marginal)
- Poisson, Negative binomial (NB2), Binomial/Bernoulli, Beta, Gamma, Exponential,
  Ordinal (cumulative logit).
- Two-part / zero-inflated: Delta-lognormal, Delta-Gamma, Hurdle-Poisson, Hurdle-NB,
  ZIP, ZINB.
- **Ordered-beta** (proportions/cover data with point masses at 0 and 1).
- **Beta-hurdle** (`fit_beta_hurdle_gllvm`; Bernoulli occurrence × positive Beta —
  closes the two-part/zero-inflated family set).
- **Tweedie** (compound Poisson–Gamma, 1<p<2; biomass/abundance with true zeros;
  `fit_tweedie_gllvm`, Dunn–Smyth density series).
- Links: identity, log, logit, probit, cloglog.

### Estimators
- **Variational approximation (VA)** as an opt-in alternative to Laplace
  (`fit_*_gllvm_va`): closed-form ELBO for Poisson/Gamma/Delta-Gamma, Gauss–Hermite
  for Binomial/NB/Beta. Steadier on dispersion/shape (e.g. the Delta-Gamma shape).
- **Analytic gradients throughout VA**: per-site inner gradients for every family,
  plus an **envelope-theorem analytic outer gradient** for the Gauss–Hermite families
  — the NB VA fit went from ~26× slower than Laplace to ~1.3×.

### Covariates, traits, structure
- Environmental covariates: shared-`γ` (`fit_gllvm_cov`) and **species-specific `B`**
  (`fit_gllvm_speciescov`).
- **Fourth-corner** trait–environment interactions (`fit_fourthcorner_gllvm`).
- **Community row effects** (`fit_roweffect_gllvm`).
- **Ordination trio** matching R `gllvm`: unconstrained (`num.lv`), **concurrent**
  (`num.lv.c`, covariate-informed LV mean + residual; `fit_concurrent_gllvm`, a.k.a.
  the as-built `fit_constrained_gllvm`), and **constrained / reduced-rank** (`num.RR`,
  deterministic `z_s = B'x_s`, a reduced-rank GLM with no latent integral;
  `fit_rrr_gllvm`).
- **Quadratic-response GLLVM** (species optima/tolerances; `fit_quadratic_gllvm`).

### Inference, ordination, workflow
- Confidence intervals: Wald, profile-likelihood, parametric bootstrap (incl. derived
  quantities), and a tidy `coef_table`.
- **Faster profile-likelihood CIs** — the LRT-crossing is now located by false position
  on `√D` (near-linear in the parameter, since `D ≈ (c−θ̂)²/SE²`) with a bisection
  safeguard, plus Wald-bound bracket seeding. This cuts the number of constrained refits
  (the dominant cost) per bound from ~15 bisections to a handful, at the same crossing —
  benefiting both the Gaussian and the GLM-family profile CIs.
- **VA-based standard errors** — `confint(fit, Y; method=:wald, objective=:va)` and
  `coef_table(...; objective=:va)` take the Hessian of the ELBO instead of the Laplace
  marginal (Poisson/NB/Binomial/Beta/Gamma), matching gllvm's approximate VA inference.
- **`ordination`** (site scores + species loadings + canonical rotation) and
  **`ordiplot`** (plot-ready biplot data: scores, arrows, labels, per-axis variance),
  paralleling R `gllvm`.
- **`select_lv`** — latent-dimension selection by AIC/BIC.
- `predict`, Dunn–Smyth `residuals`, `getLV`, `getLoadings`, `aic`, `bic`, `simulate` —
  with post-fit support (`getLV`/`predict`/`ordination`) extended to every new model
  type (species covariates, fourth-corner, row effects, concurrent, RRR, quadratic).
- `@formula` / `gllvm(...)` front end; an end-to-end docs tutorial.
- An R interface scaffold (`r/gllvmjl.R`, via `JuliaConnectoR`, mirroring DRM.jl) for
  calling the Julia fitters from R.

### Spatial — SPDE / Matérn-GMRF (Lindgren–Rue–Lindström 2011)
- A self-contained spatial module, shared-ready with DRM.jl: `spde_fem` (P1 finite-element
  mass/stiffness), `spde_precision` (sparse Matérn precision `Q(κ,τ)`, α=1 and α=2),
  `spde_projector` (barycentric `A` mapping mesh nodes → sites), `matern_correlation`
  (analytic reference), and `spde_mesh_grid` (auto-mesher over a bounding box).
- `fit_spde_gaussian` — INLA-style Gaussian spatial-field fit via the Woodbury identity
  and matrix-determinant lemma, gated by an exact dense-vs-sparse log-likelihood equality.
- **SPDE field as a latent variable inside the (non-Gaussian) GLLVM** —
  `spde_latent_marginal_loglik` / `fit_spde_latent_gllvm` make the `K` latent variables
  spatially-smooth Matérn-GMRF fields (`z_·k = A·u_k`, `u_k ~ N(0, Q⁻¹)`) via a **joint
  Laplace over the spatial GMRF** (sparse CHOLMOD Cholesky of the `K·N` field Hessian),
  gated by machine-precision anchors (the `Q=I, A=I` reduction to the independent-site
  Laplace, the conjugate-Gaussian reduction to `spde_gaussian_marginal_loglik`, and the
  `NB(r→∞)→Poisson` marginal reduction). `fit_spde_latent_gllvm` jointly estimates
  `β, Λ, κ, τ` for the no-dispersion families (Poisson/Binomial) and the dispersion
  families (Gaussian `σ²`, negative-binomial `r`).
- **Spatial prediction (kriging)** — `predict_spatial` interpolates the fitted Matérn
  field to new, unobserved locations (with `getLV`/`predict` post-fit for the
  SPDE-latent model); equals `predict` at the training locations (consistency anchor).
- **`spde_mesh_delaunay`** — Bowyer–Watson Delaunay triangulation, so a mesh can be built
  directly from observation points (gated by FEM-validity, convex-hull tiling, and the
  empty-circumcircle property).

### Missing data
- **NA handling in the Laplace core** — `marginal_loglik_laplace(...; mask)` drops
  unobserved cells from the score, Hessian weight, and log-density, so the marginal is
  over the observed entries and invariant to placeholder values in the masked cells.
  Wired into `fit_poisson_gllvm` / `fit_nb_gllvm` (pass a `mask`, or include `missing`
  in `Y`) with a mask-respecting warm start; `observed_mask(Y)` derives the mask.

### Beyond gllvm
- Phylogenetic GLLVM toolkit (sparse-precision, contrasts, edge-incidence, relaxed
  clock, branch random effects).

### Known limitations
- Laplace biases dispersion/shape parameters and the two-part/zero-inflated cells are
  multimodal — use VA there (see `ROADMAP.md`).
- Exponential LV recovery is weakly identified (CV = 1 swamps the SVD warm start); the
  test verifies machinery, not recovery.
- Wald Hessians are finite-difference (analytic per-family Hessians would speed CIs).
- Still open vs gllvm: full VA/EVA inference beyond Wald SEs. (Delaunay-of-points
  meshing, beta-hurdle, missing-data (NA) handling, and the internal fast
  phylogenetic-Poisson path of issue #61 have since landed — see the Unreleased
  section above.)

## v0.1.0

Gaussian + phylogenetic GLLVM pilot: closed-form Gaussian marginal, Woodbury/low-rank
Cholesky, PPCA & EM-FA initialisation, σ_eps profile-out, the phylogenetic
representations, and Wald/profile/bootstrap/derived confidence intervals.


### Local Core070 candidate — public Gaussian AGHQ

- Ordinary shared-SD Gaussian quadrature with explicit integration metadata;
  preserve default exact zero-mean fitting and legacy constructors.
- Carry fixed effects, fixed-zero coefficients, masks and offsets through
  prediction and same-objective inference; retain bootstrap failures.
- Original frozen-reference parity and full programme gates remain separate;
  no calibrated-coverage or performance claim follows from functional tests.
