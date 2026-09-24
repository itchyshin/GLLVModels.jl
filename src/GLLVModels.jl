module GLLVModels

# Soft source-level migration aid: users who have already loaded GLLVModels
# can qualify the former module name while moving calls to GLLVModels. A
# renamed Julia package cannot keep `using GLLVM` alive because package
# resolution happens before this module is loaded.
const GLLVM = GLLVModels

using LinearAlgebra, Optim, ForwardDiff, Random, SparseArrays, Statistics, SHA
using SpecialFunctions: digamma, trigamma, besselk, gamma, loggamma
import StatsModels: coef, vcov, nobs, dof, loglikelihood, aic, bic, coeftable, stderror, confint, predict, residuals, fitted, StatsAPI, deviance
# Import Distributions without `Multinomial` so the Identity marker
# `GLLVModels.Multinomial` (unordered categorical, twin fid 16) can bind.
# `Distributions.Multinomial` is the count-vector law — still available qualified.
import Distributions
using Distributions: Distribution, Univariate, Discrete, Continuous,
    Poisson, Binomial, Normal, Gamma, Exponential, Beta,
    NegativeBinomial, LogNormal, TDist, Categorical, Bernoulli, Geometric,
    Chisq, cdf, pdf, logpdf, logcdf, quantile, ccdf

# Core
include("fit_verdict.jl")            # _fit_verdict: never report a failure sentinel as a loglik
include("packing.jl")
include("lowrank_cholesky.jl")          # used by likelihood
include("likelihood.jl")
include("source_covariance.jl")          # internal Gaussian additive source reference evaluator
include("source_fit.jl")                 # local candidate: fixed Gaussian source covariances
include("grouped_gaussian.jl")           # internal shared/crossed sparse Gaussian factor kernel
include("ppca_init.jl")                  # used by fit (warm-start)
include("em_fa.jl")                      # alternative EM solver
include("profile.jl")                    # σ_eps profile-out (used by fit)
include("fixed_coefficients.jl")         # zero masks for fixed-effect coefficients
include("fit.jl")
include("reml.jl")                       # REML for the Gaussian path (restricted ML)
include("random_effects.jl")             # RE foundation — grouping-factor coding
include("fit_random_effects.jl")         # Gaussian grouped random slopes (random regression)
include("twolevel.jl")                    # Gaussian two-level (between/within-individual) reduced-rank decomposition
include("simulate.jl")
include("families/gaussian_pervar.jl")   # Gaussian with per-species variance (gllvmTMB heteroscedastic default)
include("missing_predictor_fiml.jl")     # fit_gaussian_mi_fiml: closed-form FIML for a missing site-level predictor (mi() axis)
include("missing_predictor_phylo.jl")    # fit_gaussian_mi_phylo: phylo missing-predictor FIML (mi() axis, Phase 3)
include("structured_cov.jl")             # spatial_cov, relatedness_cov builders
include("cross_kernel.jl")               # make_cross_kernel: cross-lineage coevolution kernel K* (PGLLVM two-lineage, C0)
include("extract_gamma.jl")              # extract_Gamma: cross-lineage coevolution estimand Γ = Λ_phy Λ_phyᵀ block
include("coevolution_kronecker.jl")      # fit_coevolution_gaussian: faithful matrix-normal coevolution (Kronecker), recovers Γ
include("coevolution_blockna.jl")        # fit_coevolution_blockna: block-NA coevolution (host/partner each measure own traits)
include("spde.jl")                        # SPDE / Matérn-GMRF FEM spatial field (shared-ready with DRM.jl)
include("spde_mesh.jl")                   # SPDE grid auto-mesher
include("spde_delaunay.jl")               # SPDE Delaunay triangulation (Bowyer–Watson)
include("spde_fit.jl")                    # Gaussian SPDE spatial-field model + ML fit

# Sparse phylogenetic path (evaluation-only — see docstring for AD limitation)
include("sparse_phy.jl")
include("phylo_precision.jl")        # PrecisionPhy: R-convention sparse precision consumer (phylo transport S1)
include("precision_fit_admission.jl") # Validate and snapshot precision once at each fitting boundary.
include("precision_multivariate.jl") # internal Destination B multivariate precision kernel
include("likelihood_sparse_phy.jl")
include("sparse_phy_grad.jl")            # analytic gradient + SparsePhyState (self-includes takahashi_selinv.jl)
include("node_gradient.jl")              # O(p) node-frame gradient + per-species BLUPs (Phase 1.1)
include("fit_phylo.jl")                  # O(p) single-trait phylogenetic Gaussian fitter (Phase 1.4)
include("phylo_contrasts.jl")            # Felsenstein independent contrasts (AD-friendly)
include("likelihood_contrasts.jl")       # Gaussian marginal log-lik on contrast scale
include("edge_incidence.jl")             # Edge-node incidence sparse representation (Bolker phylog.rmd)
include("likelihood_edge_incidence.jl")  # Gaussian marginal log-lik via edge-node incidence
include("phylo_branch_re.jl")            # Single-variance branch random-effects model
include("em_phylo.jl")                   # Gradient-free EM for Gaussian phylogenetic GLLVM
include("em_squarem.jl")                 # SQUAREM acceleration for phylogenetic EM
include("relaxed_clock.jl")              # Relaxed-clock per-branch evolution rates

# Response families (Phase 3): Distributions types as markers + link functions
include("families/links.jl")
include("families/laplace.jl")           # generic family-dispatched Laplace marginal core
include("families/aghq_grid.jl")         # Stage-1a/1b live-pin AGHQ grid + Liu–Pierce site (not a public knob)
include("families/aghq_outer.jl")        # Internal unpenalized outer adaptation; public fit wiring pending
include("families/aghq_fit_info.jl")
include("families/binomial.jl")          # Binomial family pieces + fit (Phase 3)
include("families/aghq_gaussian.jl")
include("families/aghq_gaussian_fit.jl")
include("families/aghq_binomial.jl")      # Checked normalized three-link binomial adapter
include("families/poisson.jl")           # Poisson family pieces (Phase 3)
include("families/aghq_poisson.jl")      # Internal checked-mode Poisson AGHQ adapter
include("families/aghq_poisson_fit.jl")
include("families/aghq_binomial_fit.jl")
include("families/truncated_poisson.jl") # Zero-truncated Poisson (twin fid 10)
include("families/censored_poisson.jl") # Right-censored Poisson (Julia-forward; twin constructor-only)
include("families/truncated_nbinom2.jl") # Zero-truncated NB2 (twin fid 11; shared-r Arc1 + per-trait Arc1b)
include("families/negbin.jl")            # Negative-binomial (NB2) family pieces (Phase 3)
include("families/gp1.jl")               # Generalized-Poisson type-1 (GP-1, signed dispersion) — issue #104
include("families/negbin1.jl")           # Negative-binomial type-1 (NB1, linear variance)
include("families/beta.jl")              # Beta family pieces (Phase 3)
include("families/ordinal.jl")           # Ordinal (cumulative-logit) family pieces (Phase 3)
include("families/gamma.jl")             # Gamma (positive continuous) family pieces (Phase 3)
include("families/tweedie.jl")           # Tweedie (compound Poisson–Gamma, 1<p<2) — biomass/abundance with zeros
include("families/grouped_dispersion.jl") # Grouped / species-specific dispersion (disp.group)
include("families/exponential.jl")       # Exponential (positive continuous, no dispersion) — Gamma(α=1)
include("families/studentt.jl")          # Student-t (heavy-tailed continuous, fixed ν) family pieces
include("families/lognormal.jl")         # one-part lognormal (twin fid 3)
include("families/multinomial.jl")       # unordered categorical FE softmax (twin fid 16; v1 no LV)
include("families/twopart.jl")           # Two-part substrate + Delta-lognormal / Delta-Gamma / Hurdle (Phase 3)
include("families/beta_hurdle.jl")       # Beta-hurdle (Bernoulli × Beta) two-part family
include("families/beta_binomial.jl")     # Beta-binomial (overdispersed binomial) — twin fid 8
include("families/com_poisson.jl")        # Conway–Maxwell–Poisson (under/overdispersed counts) — beyond gllvmTMB
include("families/ordered_beta.jl")       # ordered-beta (must precede fit_gllvm)
include("families/fit_gllvm.jl")         # unified fit_gllvm(Y; family) dispatcher
include("none_dep.jl")                    # none × dep matrix fitter (K = p; no formula sugar)
include("phylo_dep.jl")                   # phylo × dep matrix fitter (K_phy = p; no formula sugar)
include("animal_dep.jl")                  # animal × dep matrix fitter (K_phy = p; no formula sugar)
include("animal_latent.jl")               # animal × latent matrix fitter (K_phy = d; no formula sugar)
include("spatial_dep.jl")                 # spatial × dep Arc 0 fail-loud admission (no mesh/SPDE yet)
include("kernel_indep.jl")                # kernel × indep Gaussian wrapper (SourceCovariance :indep)
include("kernel_dep.jl")                  # kernel × dep Gaussian wrapper (SourceCovariance :dep)
include("kernel_latent.jl")               # kernel × latent Gaussian wrapper (SourceCovariance :latent)
include("laplace_grad.jl")               # exact (AD + implicit-step) Poisson Laplace gradient (issue #65)
include("grouped_laplace.jl")            # internal global grouped non-Gaussian objective
include("missing_predictor_poisson.jl")  # non-Gaussian missing predictor (mi Phase 5a): Poisson augmented-Laplace FIML
include("missing_predictor_multi.jl")    # multiple missing predictors, jointly integrated (mi() vector axis, Track T3)
include("families/covariates.jl")        # fixed-effect covariates (Xβ) for the Laplace families
include("families/species_covariates.jl") # species-specific covariate coefficients (XB) for the Laplace families
include("families/constrained_ordination.jl") # constrained ordination (RRR of latent vars on env predictors)
include("families/rrr.jl")                # reduced-rank regression (num.RR) — deterministic constrained ordination
include("families/quadratic.jl")          # quadratic-response GLLVM (species optima/tolerances)
include("families/fourthcorner.jl")       # fourth-corner trait–environment interaction for the Laplace families
include("families/row_effects.jl")        # community row effects (per-site intercepts) for the Laplace families
include("families/row_random.jl")          # random row effects (ρ_s ~ N(0,σ_row²), gllvmTMB row.eff="random")
include("families/random_slopes.jl")       # non-Gaussian grouped random slopes (Poisson; per-group Laplace super-site)
include("families/variational.jl")       # Gaussian-variational (VA/ELBO) marginal — Poisson (increment 1) + GH helper
include("families/variational_binomial.jl") # VA/ELBO marginal — Binomial/Bernoulli (Gauss–Hermite)
include("families/variational_negbin.jl") # VA/ELBO marginal — Negative Binomial (Gauss–Hermite)
include("families/variational_gamma.jl") # VA/ELBO marginal — Gamma (closed form)
include("families/variational_beta.jl")  # VA/ELBO marginal — Beta (Gauss–Hermite)
include("families/variational_dgamma.jl") # VA/ELBO marginal — Delta-Gamma two-part (closed form)
include("families/variational_exponential.jl") # VA/ELBO marginal — Exponential (closed form, Gamma α=1)

# SPDE/Matérn-GMRF field as a latent variable inside a non-Gaussian GLLVM
# (joint Laplace over the spatial GMRF). Depends on the SPDE FEM machinery
# (src/spde.jl) and the family Laplace pieces above.
include("spde_latent.jl")
include("spde_latent_postfit.jl")
include("phylo_glm.jl")                   # phylogenetic GLLVM for non-Gaussian families (issue #61, working fit)
include("phylo_poisson_xlv.jl")           # internal S1 proof: phylo + Poisson + predictor-informed LV
include("phylo_binomial_xlv.jl")          # internal S1 proof: phylo + Binomial + predictor-informed LV
include("phylo_nb_xlv.jl")                # internal S1 proof: phylo + NB2 + predictor-informed LV
include("phylo_gamma_xlv.jl")             # internal S1 proof: phylo + Gamma + predictor-informed LV
include("phylo_beta_xlv.jl")              # internal S1 proof: phylo + Beta + predictor-informed LV
include("phylo_ordinal_xlv.jl")           # internal S1 proof: phylo + shared-cutpoint Ordinal + predictor-informed LV
include("coevolution_glm.jl")             # cross-family (non-Gaussian) cross-lineage coevolution (Track T4): K* through a dense Laplace

# Post-fit API (ordination, predict, residuals, summary)
include("postfit.jl")
include("lv_targets.jl")                # internal eta-scale realised LV targets
include("ordination.jl")                  # ordination output (site scores + species loadings, canonical rotation)
include("model_selection.jl")             # select_lv: latent-dimension selection by AIC/BIC
include("cv.jl")                          # cv_gllvm: K-fold cross-validation engine
include("simulate_fit.jl")               # simulate(fit, …) for the non-Gaussian families
include("ordination_uncertainty.jl")      # per-site latent-score uncertainty (conditional bootstrap of scores)

# Confidence intervals
include("confint.jl")                    # Wald
include("confint_profile.jl")            # profile likelihood
include("loading_profile_confirmatory_internal.jl")  # D3 confirmatory pins (not exported)
include("confint_bootstrap.jl")          # parametric bootstrap
include("confint_derived.jl")            # derived quantities (Σ_y, communality, ...)
include("confint_derived_wald.jl")       # transformed-Wald CIs for bounded derived quantities
# Cross-family latent-scale link-implicit residual table + non-Gaussian
# sigma_y_site/communality/correlation extractors. After postfit.jl (needs the
# family fit structs + predict) and confint_derived.jl (the Gaussian generics it
# adds methods to). Additive: the ::GllvmFit methods are unchanged.
include("link_residual.jl")
include("extractors.jl")                # extract_*/get* post-fit extractor family (core070 Cluster 1)
include("re_sd.jl")                      # latent_score_sd (renamed from getREsd): TMB-sdreport-style conditional-on-θ̂ random-effect SDs (core070 E-cluster)
include("families/mixed.jl")             # mixed-family GLLVM (cross-family VCV): fit_mixed_gllvm + MixedFamilyFit. AFTER link_residual + the family fitters so all dispatch targets exist.
include("boundary_inference.jl")         # χ̄² boundary LRT + boundary-aware profile CI for variance components
include("confint_family.jl")             # Wald / profile / bootstrap CIs for non-Gaussian families
include("marginal_target_intervals.jl")  # internal marginal intervals for grouped/precision candidates
include("grouped_fit.jl")               # internal joint grouped Gaussian fit and intervals
include("precision_multivariate_fit.jl") # internal multivariate phylogenetic fit and intervals
include("grouped_nongaussian_fit.jl")   # internal five-family joint grouping extension
include("destination_b_postfit.jl")     # explicit population-only Gaussian postfit routes
include("grouped_nongaussian_postfit.jl") # conditional means at zero random effects
include("destination_b_fixed_effects.jl") # full observed-marginal fixed-effect covariance
include("grouped_profile_foundation.jl") # exact-zero evaluator and numerical rank gates
include("grouped_profile_refit.jl") # retained, stationary constrained nuisance refits
include("grouped_profile_inversion.jl") # failure-safe callback-only LR bracket inversion
include("grouped_profile_interval.jl") # private actual-refit profile endpoints
include("joint_phylo_grouped_gaussian.jl") # one joint Gaussian marginal, evaluation only
include("joint_covariance_identification.jl") # actual covariance tangent aliases
include("joint_phylo_grouped_fit.jl") # private joint optimizer and marginal intervals
include("joint_phylo_grouped_postfit.jl") # explicit population mean and source-labelled extraction
include("diagnostics.jl")                # check_gllvmTMB / gllvmTMB_diagnose / predictive_check / sanity_multi / compare_* / confint_inspect (core070 diagnostics/compare cluster)
include("summary_table.jl")              # coef_table: tidy Wald inference table
include("postfit_tables.jl")             # final missing-surface cluster (core070 §1): deviance, cross-rho profiles,
                                          # predict_cross_covariance, predict_missing, rotate_loadings, tidy, summary, imputed
include("formula.jl")                    # @formula front-end (v1: fixed effects → engine)
include("bridge.jl")                      # R→Julia bridge_fit (JuliaCall flat contract); LAST
include("bridge_precision_multivariate.jl") # explicit multivariate precision bridge candidate

# Ordination naming: the implemented z_s ~ N(B'x_s, I) model (covariate-informed LV
# mean PLUS residual) is gllvm's *concurrent* ordination (num.lv.c). Expose the
# accurate name as an alias of the as-built `*_constrained` API.
const fit_concurrent_gllvm = fit_constrained_gllvm
const ConcurrentOrdinationFit = ConstrainedOrdinationFit

# Public API
export make_cross_kernel, extract_Gamma, fit_coevolution_gaussian, fit_coevolution_blockna,
       fit_gaussian_mi_fiml, fit_gaussian_mi_phylo, fit_gllvm_mi, fit_gllvm_mi_multi,
       spatial_cov, relatedness_cov,
       spde_fem, spde_precision, spde_projector, matern_correlation,
       spde_mesh_grid, spde_mesh_delaunay,
       spde_gaussian_marginal_loglik, fit_spde_gaussian, SPDEGaussianFit,
       spde_latent_marginal_loglik, fit_spde_latent_gllvm, SPDELatentFit,
       confint_spde_latent,
       confint_speciescov, confint_fourthcorner, confint_rrr, confint_constrained,
       confint_lv_effects,
       fit_gaussian_gllvm, GllvmModel, GllvmFit,
       SourceCovariance, fit_gaussian_sources, GaussianSourcesFit,
       GroupingTerm, GroupedGaussianFit, grouped_gaussian_intervals,
       grouped_gaussian_variance_profile,
       PrecisionMultivariateFit, precision_multivariate_intervals,
       JointPhyloGroupedGaussianFit, joint_phylo_grouped_intervals,
       GroupedNonGaussianFit, grouped_nongaussian_intervals,
       fit_gaussian_structured,
       gaussian_reml_loglik, fit_gaussian_reml, GaussianREMLFit,
       fit_gaussian_random_slope, GaussianRandomSlopeFit, gaussian_grouped_intercept_loglik,
       fit_twolevel_gaussian, TwoLevelFit, twolevel_marginal_loglik,
       repeatability, communality_B, communality_W, correlation_B, correlation_W,
       repeatability_wald_ci, repeatability_bootstrap_ci, repeatability_ci,
       TwoLevelRepeatabilityProfileWithdrawn,
       fit_poisson_random_slope, PoissonRandomSlopeFit, random_slope_marginal_loglik_laplace,
       fit_gaussian_pervar_gllvm, GaussianPerVarFit, gaussian_pervar_marginal_loglik,
       fit_compoisson_gllvm, COMPoisson, COMPoissonFit, compoisson_marginal_loglik_laplace,
       compoisson_logpdf, compoisson_logz,
       confint, profile_ci, bootstrap_ci,
       transformed_wald_ci_derived, correlation_wald_ci, communality_wald_ci,
       icc_wald_ci, phylo_signal_wald_ci,
       standardized_loading_wald_ci, raw_loading_wald_ci, loading_ci,
       loading_profile_exploratory, loading_profile,
       profile_ci_total_variance, profile_ci_phylo_signal,
       slope_sd_ci, standard_errors,
       latent_score_sd, getREsd, bootstrap_Sigma, tmbprofile_wrapper, profile_targets, profile_curve_targets,
       profile_phylo_signal,
       ppca_init, em_fa,
       sigma_y_site, communality, correlation, phylo_signal, link_residual,
       extract_Sigma, extract_Sigma_table, extract_loadings, extract_rotated_loadings,
       extract_communality, extract_correlations, extract_cross_correlations,
       extract_residual_cov, extract_residual_cor, getResidualCov, getResidualCor,
       extract_ordination, extract_cutpoints, extract_proportions, extract_phylo_signal,
       extract_repeatability, extract_ICC_site, extract_Omega,
       chibar2_pvalue, variance_lrt, profile_ci_variance,
       augmented_phy, AugmentedPhy, random_balanced_tree, sigma_phy_dense, make_phy,
       PrecisionPhy, precision_logdet_check,
       phylo_precision_payload, admit_phylo_precision_payload,
       gaussian_marginal_loglik_sparse_phy,
       node_grad, node_dσ_phy_only, NodePerSpecies, build_node_perspecies,
       grad_node_perspecies, node_blups,
       fit_phylo_gaussian, PhyloGaussianFit,
       phylo_glm_marginal_loglik, fit_phylo_glm, PhyloGLMFit,
       coevolution_glm_marginal_loglik, fit_coevolution_glm, CoevolutionGLMFit,
       coevolution_gamma,
       FelsensteinContrasts, felsenstein_contrast_matrix, felsenstein_contrasts,
       contrast_transform, gaussian_marginal_loglik_contrasts,
       EdgePhy, edge_phy, Q_times_x, sigma_phy_dense_edge, log_det_Q, solve_Q,
       gaussian_marginal_loglik_edge_phy,
       path_membership, simulate_branch_re, BranchRECache, branch_re_cache,
       branch_re_profile_negll, branch_blups, BranchREFit, fit_branch_re,
       fit_branch_re_dense, clade_edges, find_clade_root,
       welch_t, rank_sum_z, excess_kurtosis, qq_max_dev,
       AnBSparseSolver, build_AnB_sparse, solve_AnB, blup_phylo_sparse,
       EMPhyloFit, em_fit_phylo, fit_em_phylo, em_observed_information,
       em_fit_phylo_squarem, fit_phylo_squarem,
       edge_W_diag, Q_perbranch, simulate_relaxed_bm, estep_edge_moments,
       shrink_logrates, RelaxedClockFit, fit_relaxed_clock,
       spearman, shrinkage_factor, clade_detection,
       LogitLink, ProbitLink, CLogLogLink, IdentityLink, LogLink,
       fit_mixed_gllvm, MixedFamilyFit, mixed_marginal_loglik_laplace,
       fit_binomial_gllvm, BinomialFit, fit_poisson_gllvm, PoissonFit,
       TruncatedPoisson, fit_truncated_poisson_gllvm, TruncatedPoissonFit,
       truncated_poisson_marginal_loglik_laplace,
       CensoredPoisson, fit_censored_poisson_gllvm, CensoredPoissonFit,
       TruncatedNegBin2, fit_truncated_nbinom2_gllvm, TruncatedNegBin2Fit,
       truncated_nbinom2_marginal_loglik_laplace,
       fit_truncated_nbinom2_gllvm_pertrait, TruncatedNegBin2PerTraitFit,
       truncated_nbinom2_pertrait_marginal_loglik_laplace,
       poisson_laplace_grad, binomial_laplace_grad, nb_laplace_grad,
       gamma_laplace_grad, beta_laplace_grad,
       fit_nb_gllvm, NBFit, fit_beta_gllvm, BetaFit,
       GeneralizedPoisson1, fit_gp1_gllvm, GP1Fit, gp1_marginal_loglik_laplace,
       NB1, fit_nb1_gllvm, NB1Fit, nb1_marginal_loglik_laplace,
       fit_nb_gllvm_grouped, NBGroupedFit, nb_grouped_marginal_loglik_laplace,
       fit_nb_gllvm_grouped_cov, NBGroupedCovFit,
       fit_beta_gllvm_grouped, BetaGroupedFit, beta_grouped_marginal_loglik_laplace,
       fit_beta_gllvm_grouped_cov, BetaGroupedCovFit,
       fit_gamma_gllvm_grouped, GammaGroupedFit, gamma_grouped_marginal_loglik_laplace,
       fit_gamma_gllvm_grouped_cov, GammaGroupedCovFit,
       fit_nb1_gllvm_grouped, NB1GroupedFit, nb1_grouped_marginal_loglik_laplace,
       fit_nb1_gllvm_grouped_cov, NB1GroupedCovFit,
       fit_tweedie_gllvm_grouped, TweedieGroupedFit, TweediePerTraitPowerFit, tweedie_grouped_marginal_loglik_laplace,
       StudentTFamily, StudentT, fit_studentt_gllvm, StudentTFit, studentt_marginal_loglik_laplace,
       Lognormal, LognormalFit, fit_lognormal_gllvm,
       lognormal_marginal_loglik, lognormal_response_mean,
       Multinomial, MultinomialFit, fit_multinomial_gllvm,
       Ordinal, fit_ordinal_gllvm, OrdinalFit,
       fit_ordinal_gllvm_pertrait, OrdinalPerTraitFit,
       fit_ordinal_gllvm_pertrait_cov, OrdinalPerTraitCovFit,
       fit_gamma_gllvm, GammaFit,
       fit_exponential_gllvm, ExponentialFit, exponential_marginal_loglik_laplace,
       fit_tweedie_gllvm, TweedieFit, tweedie_marginal_loglik_laplace, tweedie_logpdf, tweedie_cdf,
       fit_delta_lognormal_gllvm, DeltaLogNormalFit, DeltaLogNormal,
       delta_lognormal_marginal_loglik_laplace,
       fit_hurdle_poisson_gllvm, HurdlePoissonFit, HurdlePoisson,
       hurdle_poisson_marginal_loglik_laplace,
       fit_hurdle_nb_gllvm, HurdleNBFit, HurdleNB,
       hurdle_nb_marginal_loglik_laplace,
       fit_delta_gamma_gllvm, DeltaGammaFit, DeltaGamma,
       delta_gamma_marginal_loglik_laplace,
       fit_beta_hurdle_gllvm, BetaHurdleFit, BetaHurdle, beta_hurdle_marginal_loglik_laplace,
       observed_mask,
       fit_zip_gllvm, ZIPFit, zip_marginal_loglik_laplace, ZIPoisson,
       fit_zip_gllvm_cov, ZIPCovFit,
       fit_zinb_gllvm, ZINBFit, zinb_marginal_loglik_laplace, ZINegBin,
       fit_zinb_gllvm_cov, ZINBCovFit,
       fit_zib_gllvm, ZIBFit, fit_zib_gllvm_cov, ZIBCovFit, zib_marginal_loglik_laplace, ZIB,
       fit_gllvm,
       fit_dep_gllvm,
       fit_phylo_dep_gllvm,
       fit_animal_dep_gllvm,
       fit_animal_latent_gllvm,
       fit_spatial_dep_gllvm,
       fit_kernel_indep_gllvm,
       fit_kernel_dep_gllvm,
       fit_kernel_latent_gllvm,
       fit_gllvm_cov, GllvmCovFit, gllvm, @formula,
       fit_gllvm_speciescov, GllvmSpeciesCovFit,
       fit_fourthcorner_gllvm, FourthCornerFit,
       fit_roweffect_gllvm, RowEffectFit,
       fit_row_random_gllvm, RowRandomFit, row_random_marginal_loglik_laplace, row_effects,
       fit_constrained_gllvm, ConstrainedOrdinationFit, constrained_marginal_loglik_laplace,
       fit_concurrent_gllvm, ConcurrentOrdinationFit,
       fit_rrr_gllvm, RRRFit, rrr_marginal_loglik,
       fit_quadratic_gllvm, QuadraticFit, quadratic_marginal_loglik_laplace,
       fit_ordered_beta_gllvm, OrderedBetaFit, OrderedBeta, ordered_beta_marginal_loglik_laplace,
       BetaBinom, fit_beta_binomial_gllvm, BetaBinomialFit, betabinomial_marginal_loglik_laplace,
       fit_beta_binomial_gllvm_grouped, BetaBinomialGroupedFit,
       betabinomial_grouped_marginal_loglik_laplace,
       fit_beta_binomial_gllvm_grouped_cov, BetaBinomialGroupedCovFit,
       beta_marginal_loglik_va, fit_beta_gllvm_va,
       delta_gamma_marginal_loglik_va, fit_delta_gamma_gllvm_va,
       poisson_marginal_loglik_va, fit_poisson_gllvm_va,
       binomial_marginal_loglik_va, nb_marginal_loglik_va,
       fit_binomial_gllvm_va, fit_nb_gllvm_va,
       gamma_marginal_loglik_va, fit_gamma_gllvm_va,
       exponential_marginal_loglik_va, fit_exponential_gllvm_va,
       getLV, getLoadings, rotation, ordination, ordiplot, ordination_uncertainty,
       extract_lv_effects, lv_effects, predict_spatial,
       coef_table, GllvmCoefTable, select_lv, LVSelection,
       cv_gllvm, CVResult,
       StatsAPI, coef, vcov, nobs, dof, loglikelihood, stderror, coeftable,
       predict, fitted, residuals, aic, bic, simulate,
       bridge_fit, bridge_capabilities,
       gllvmTMB_check_consistency, gllvmTMB_diagnose, check_gllvmTMB,
       check_auto_residual, sanity_multi, fit_diagnostic_table, diagnostic_table,
       diagnose_kernel_separability, compare_fits_Sigma_table, compare_Sigma_table,
       compare_fits_dep_vs_two_psi, compare_dep_vs_two_psi,
       compare_fits_indep_vs_two_psi, compare_indep_vs_two_psi, compare_loadings,
       predictive_check, confint_inspect,
       deviance, profile_cross_rho_ci, predict_cross_covariance, predict_missing,
       simulate_unit_trait, profile_cross_rho, rotate_loadings,
       extract_rotated_loadings_table, extract_coevolution_modules, imputed,
       tidy, GllvmSummary

end # module GLLVModels
