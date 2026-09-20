# Changelog

Notable changes to GLLVModels.jl. Style mirrors `gllvmTMB`'s NEWS: status labels
**IN** (shipped), **PARTIAL** (limited), **PLANNED** (next), with issue/PR refs.

## GLLVModels.jl (development version)

### Changed
- **CHANGED:** the package and module are now named `GLLVModels`; the public
  fitting API remains unchanged. After the rename, `using GLLVM` cannot resolve
  a package. Once `GLLVModels` is loaded, `GLLVModels.GLLVM` is only a temporary
  qualified alias; it does not provide import compatibility and is deprecated.
- **Development:** fixed Gaussian source fits accept a complete mean design `X`.
  Explicit-source wide/long formulas expose trait intercepts, shared slopes,
  categorical contrasts and interactions. Fits retain the copied design, response
  shape and coefficient labels. No R source-grammar, bridge, random-slope or
  calibrated-uncertainty claim is implied.
- **Development:** per-trait Gaussian fitting accepts an explicit fixed residual
  SD and estimates unique variances within that constraint. Results retain unique
  and total diagonal variances; existing defaults and parameter counts are unchanged.
  The `pervar=true` Gaussian formula route preserves trait intercepts,
  zero-mean designs, shared slopes and categorical contrasts. Bridge and interval
  parity are not established. AGHQ requests on the fixed-residual unique model
  retain exact Gaussian/Laplace with warning and provenance; `k=1` is quiet.
  Plain heteroscedastic AGHQ remains unimplemented and is labeled separately.
- **PARTIAL:** `SourceCovariance` and `fit_gaussian_sources` fit fixed Gaussian
  covariance among source groups. Targeted tests and six retained public-R
  comparisons pass; bridge support and calibrated uncertainty remain
  unverified. The unique-variance comparison has nearly singular curvature.
- Exact Gaussian fits and Wald/profile intervals handle an empty fixed-effect
  design. Failed profile refits cannot supply finite bounds without a finite
  likelihood crossing.
- Formula fits validate every supplied site-table column before response access,
  including intercept-only fits. Empty tables remain valid without covariates.
  The original NB2 wide/long formula model matches its native fit; broader
  interface qualification remains incomplete.
- **EXPERIMENTAL:** ordinary NB2 evaluates its density directly
  from the mean and uses an overflow-safe observed curvature. The original
  paired fit, scalar derivatives and required NB2/truncated-NB2 runner pass.
  Broader recovery, full package checks and independent review remain pending.
- Student-t fitting rejects infinite fixed degrees of freedom before reading
  responses, for scalar and per-trait inputs. Finite positive fixed values and
  the estimated-df route are unchanged.
- **EXPERIMENTAL:** grouped Tweedie distinguishes fixed common,
  shared estimated and per-species estimated power; `TweediePerTraitPowerFit`
  stores the latter. Student fits record whether degrees of freedom were estimated
  so information-criterion parameter counts are correct. These changes do not
  establish full parity: the recorded Student-t R comparison still does not
  converge reliably, and broader qualification remains pending.
- Branch-RE uses an equivalent dense marginal fallback when the auxiliary sparse
  precision is numerically unsafe. This preserves valid marginal models without
  a ridge, at a possible O(p²) memory cost reported by a warning.
- **CHANGED:** Gamma's Laplace **log-determinant now uses the observed
  conditional curvature** `α·y/μ`, matching TMB and therefore `gllvmTMB`.
  It previously used the Fisher (expected) weight — the constant `α`, which is
  exactly the *expectation* of the correct one. **This moves reported `loglik`
  values for Gamma fits**, and anything derived from the Hessian such as Wald
  standard errors; point estimates move very little, because the conditional
  mode is determined by the score, which is unaffected. Measured against
  numerical quadrature the observed curvature is closer **12/12** seeds, with
  20–60× smaller error. `hessian = :fisher` restores the previous behaviour.

  This was **instance 8** of a Fisher-vs-observed fault class and the
  worst-placed one: `fit_gllvm(Y; family = Gamma())` reaches it, because Gamma
  is deliberately excluded from the `disp_group = :species` auto-coercion that
  shields the other dispersion families.

  **Per-family, not a global switch.** Other families keep the Fisher default:
  measured evidence says observed is *not* closer for Beta (2/12) and is
  measurably worse for GP-1's dispersion recovery. See
  `docs/design/capability-status.md` § *Laplace curvature* for the family-by-family
  table and which rows will not match `gllvmTMB` to machine precision.

### Fixed
- **FIX:** every non-Gaussian **Wald** standard error was wrong. The
  observed-information finite-difference Hessian (`_fd_hessian`, backing
  `confint(fit, Y; method=:wald)` and `_family_wald` for Poisson / Binomial / NB /
  Beta / Gamma / Tweedie / two-part / SPDE-latent / structural CIs) wrote `2f0` —
  which Julia lexes as the Float32 literal `2.0f0`, not `2 * f0` — so the diagonal
  second difference dropped the centre value and exploded with the objective's
  large constant, collapsing `inv(H)` to standard errors ~1e-6. Off-diagonals and
  the profile/bootstrap routes were unaffected. Added `test/test_fd_hessian.jl`
  pinning the Hessian to a known analytic value (the existing CI tests only
  checked structure / `pd_hessian`, never SE magnitude, so the bug was invisible).

### Engine
- **IN:** profile-likelihood hardening for admitted LV-effect routes:
  `profile_ci`, non-Gaussian `confint(...; method=:profile)`, and
  `confint_lv_effects(...; method=:profile)` now expose bounded refit/profile
  controls. `confint_lv_effects` also accepts `profile_indices` to profile
  selected entries of `vec(B_lv)` without running the whole trait-effect matrix.
- **IN:** phylogenetic GLM (`fit_phylo_glm` / `PhyloGLMFit`) — a per-species
  phylogenetic random intercept for the non-Gaussian families (Poisson / NB /
  Binomial) via an augmented-state joint Laplace over the sparse phylogenetic
  precision.
- **IN:** zero-inflated binomial (`fit_zib_gllvm` / `ZIBFit`), with Wald /
  profile / bootstrap confidence intervals.
- **IN:** negative-binomial type-1 (NB1, linear variance `Var = μ(1+φ)`;
  `fit_nb1_gllvm`).

### Documentation
- **IN:** pkgdown-style documentation site (DocumenterVitepress) — dropdown
  navbar, full-text search, light/dark mode; homepage mirrors `gllvmTMB`'s with
  a Julia flavour.

### Quality & infrastructure
- **IN:** `Pkg.test()` adopted as the full-suite command; Aqua (package
  hygiene) and JET (type-stability of the O(p) kernels) run in CI.
- **IN:** isolated RCall.jl parity scaffold (`test/parity/`, opt-in) for
  checking agreement against R `gllvmTMB`.

## GLLVModels.jl v0.2.0

A large expansion from the v0.1.0 Gaussian-only pilot to a broad, gllvmTMB-class
package; every numerical addition is gated by deterministic tests.

### Response families
- **IN:** broad one-part GLM family set via a Laplace marginal — Poisson, negative binomial
  (NB2), Binomial / Bernoulli, Beta, Gamma, Exponential, Ordinal, and Tweedie
  (compound Poisson–Gamma, `fit_tweedie_gllvm`).
- **IN:** two-part / mixture families — Delta-lognormal, Delta-Gamma,
  Hurdle-Poisson, Hurdle-NB, beta-hurdle (`fit_beta_hurdle_gllvm`), ordered-beta
  (`fit_ordered_beta_gllvm`), ZIP, and ZINB.
- **IN:** links — identity, log, logit, probit, cloglog.

### Estimators & inference
- **IN:** a variational (VA / ELBO) estimator alongside Laplace
  (`fit_*_gllvm_va`), with analytic inner gradients and envelope-theorem outer
  gradients for the Gauss–Hermite families, and **VA-based standard errors**
  (`confint(...; objective=:va)`).
- **IN:** confidence-interval routes — Wald, profile likelihood, and parametric
  bootstrap — for the GLM and two-part families via `confint(fit, Y; method=…)`,
  plus a tidy `coef_table`; public bridge promotion remains status-tracked by
  family/structure. Faster profile CIs use false-position on `√D`.

### Structure, covariates, ordination
- **PARTIAL:** predictor-informed latent-score means for the ordinary unit-tier
  path, with `fit_gaussian_gllvm(...; X_lv=...)`,
  `fit_poisson_gllvm(...; X_lv=...)`, `fit_nb_gllvm(...; X_lv=...)`,
  `fit_gamma_gllvm(...; X_lv=...)`, `fit_beta_gllvm(...; X_lv=...)`,
  `fit_binomial_gllvm(...; X_lv=...)`, and
  `fit_ordinal_gllvm(...; X_lv=...)` for complete-response Gaussian, Poisson
  (log link), shared-dispersion NB2, shared-shape Gamma, shared-precision Beta,
  binomial logit/probit/cloglog, and shared-cutpoint Ordinal logit point fits.
  `getLV(...;
  component=:mean/:innovation/:total)` reports score decompositions, while
  `extract_lv_effects()` reports the rotation-stable induced trait effect
  `B_lv = Λ * alpha_lv'` by default and the rotation-dependent CLV-style
  `alpha_lv` table with `type=:axis_effect`. `confint_lv_effects()` supplies
  uncertainty for `B_lv`, not for `alpha_lv`. Response masks, simultaneous
  fixed-effect `X`, raw axis-effect SEs, per-trait ordinal bridge parity,
  W-tier, phylogenetic/source-specific extensions, and R-package row promotion
  remain gated.
- **IN:** fixed-zero shared covariate coefficients for Gaussian (`β_fixed`) and
  non-Gaussian (`γ_fixed`) fixed-effect-X fits, plus bridge status fields for
  `gllvmTMB`'s `Xcoef_fixed` contract.
- **IN:** environmental covariates — shared-`γ` (`fit_gllvm_cov`) and
  species-specific `B` (`fit_gllvm_speciescov`); fourth-corner trait–environment
  interactions (`fit_fourthcorner_gllvm`); fixed community row effects
  (`fit_roweffect_gllvm`); quadratic response (`fit_quadratic_gllvm`).
- **IN:** the ordination trio — unconstrained, concurrent (`num.lv.c`,
  `fit_concurrent_gllvm`), and constrained / RRR (`num.RR`, `fit_rrr_gllvm`);
  `ordination` / `ordiplot` / `select_lv` post-fit helpers.

### Spatial & missing data
- **IN:** SPDE / Matérn-GMRF spatial field (Lindgren–Rue–Lindström 2011) —
  `fit_spde_gaussian` and the SPDE field as a latent variable inside the
  non-Gaussian GLLVM (`fit_spde_latent_gllvm`), with kriging prediction
  (`predict_spatial`) and auto-meshing (`spde_mesh_grid` / `spde_mesh_delaunay`).
- **IN:** NA handling in the Laplace core (`marginal_loglik_laplace(...; mask)`),
  with `observed_mask(Y)` and `missing` support in the GLM fitters.

### Interface
- **IN:** the `@formula` / `gllvm(...)` front-end (continuous fixed effects, wide
  + long input) and an R interface scaffold (`r/gllvmjl.R`, via JuliaConnectoR).

## GLLVModels.jl v0.1.0

- **IN:** Gaussian + phylogenetic GLLVM engine — closed-form marginal
  likelihood, PPCA / EM initialisation, multiple phylogenetic representations
  (sparse precision, Felsenstein contrasts, edge-incidence) agreeing to machine
  precision.
- **IN:** Wald / profile-likelihood / parametric-bootstrap confidence
  intervals, including derived quantities (Σ_y entries, communality,
  cross-trait correlation, phylogenetic signal).
- **IN:** ~340× median per-fit speedup over R `gllvmTMB` on the Gaussian
  benchmark grid, reproducing estimates and likelihoods to machine precision.
  *(Correction appended 2026-08-25 — the entry is left as originally published
  rather than rewritten. "Machine precision" overstates it: the measured
  worst case across that grid is `|Δ logLik| = 2.343e-07` and `Σ_y` relative
  Frobenius `4.424e-05`, i.e. agreement to at least six significant digits, not
  to ~2.2e-16. The `~340×` figure is also specific to the single-σ² Gaussian
  closed-form path and does NOT generalise: measured non-Gaussian speedups
  include truncated_poisson ≈2.2× and Gamma ≈1.6×.)*
