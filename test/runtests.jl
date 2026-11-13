using Test
using GLLVModels

# --- Deterministic Julia-suite sharding (GLLVM_TEST_SHARD="k/N") -----------
# Splits the `include("test_*.jl")` calls below across N shards by file
# position: shard k (1-based) gets file i when (i - 1) % N == k - 1. Unset
# (default) runs every file — unchanged behaviour. Pure-logic helpers live in
# shard_util.jl and are exercised standalone in test_shard_selection.jl.
include("shard_util.jl")

_shard_spec = get(ENV, "GLLVM_TEST_SHARD", "")
_SHARD = isempty(_shard_spec) ? nothing : _parse_shard_spec(_shard_spec)

_shard_pos = Ref(0)

# Files whose coverage depends on Julia threads (their testset skips legs when
# Threads.nthreads() is too small). CI sets GLLVM_TEST_REQUIRE_THREADS=4 on every
# sharded job and JULIA_NUM_THREADS=4 on every sharded job; if
# an include moves to a single-threaded shard this fails loudly rather than letting
# the legs skip silently. Unset (default, local runs) = no check.
_THREADED_FILES = ("test_bootstrap_sigma_r.jl",)
_REQUIRE_THREADS = something(tryparse(Int, get(ENV, "GLLVM_TEST_REQUIRE_THREADS", "")), 0)

# --- Duration-balanced assignment for the 4-way CI split --------------------
# Position sharding is fair by count, not by time. Measured per-file times
# (Totoro, 1 thread, 2026-11-09, Julia 1.10.12 and 1.13.3; see
# ci-speed-report-20261109) put the position shards at about 4103/5683/4187/3271 s
# on 1.10 (max/mean 1.32) and 4270/5073/4077/3414 s on 1.13. This table is greedy
# longest-first on the mean of the two legs' times, about 4.3 ks per shard on both.
# It applies ONLY when N == 4. Any other N, and any file not listed here (a newly
# added test), falls back to the position rule, so
# every file still runs in exactly one shard. A speed hint, not a contract:
# refresh it when the suite changes a lot.
const _SHARD4_TABLE = Dict{String,Int}()
for (_k, _files) in enumerate((
    (
        "test_package_rename_alias.jl", "test_none_dep.jl", "test_phylo_dep.jl",
        "test_kernel_indep.jl", "test_kernel_dep.jl", "test_kernel_latent.jl",
        "test_gaussian_K_bounds.jl", "test_unidentified_tier_warning.jl", "test_signed_sigma_phy.jl",
        "test_destination_b_grouping_invariance.jl", "test_destination_b_grouping_interval_matrix.jl", "test_grouped_profile_refit.jl",
        "test_grouped_profile_interval.jl", "test_joint_covariance_identification.jl", "test_joint_phylo_grouped_fit.jl",
        "test_extract_phylo_signal_joint.jl", "test_profile_derived_fix.jl", "test_aghq_grid.jl",
        "test_aghq_outer.jl", "test_aghq_ordinal.jl", "test_aghq_public_gaussian.jl",
        "test_gaussian_empty_design.jl", "test_aghq_gate.jl", "test_aghq_kd_bound.jl",
        "test_poisson_laplace.jl", "test_truncnb2_dispersion_boundary.jl", "test_truncnb2_pertrait_boundary_restart.jl",
        "test_nb2_precision.jl", "test_negbin_laplace.jl", "test_beta_laplace.jl",
        "test_binomial_fit.jl", "test_poisson_fit.jl", "test_nb_n_init.jl",
        "test_issue_783_curvature_docs.jl", "test_chibar2_variance_lrt_p1_twin.jl", "test_poisson_grad_perf.jl",
        "test_latte_kernel_identity.jl", "test_grouped_analytic_grad.jl", "test_destination_b_joint_nb2_replication.jl",
        "test_destination_b_quadrature.jl", "test_laplace_alloc_equiv.jl", "test_grouped_dispersion.jl",
        "test_nb_grouped_poisson_ridge.jl", "test_nb_cov_ridge_polish_grad.jl", "test_beta_grouped_convergence.jl",
        "test_gamma_x_identity.jl", "test_gamma_fit.jl", "test_tweedie_engine_health.jl",
        "test_tweedie_grouped_engine_health.jl", "test_tweedie_cdf.jl", "test_exponential.jl",
        "test_studentt.jl", "test_studentt_input_validation.jl", "test_studentt_boundary_honesty.jl",
        "test_mode_search_623.jl", "test_ordinal_probit.jl", "test_unified_api.jl",
        "test_gaussian_pervar_heywood_736.jl", "test_gaussian_wald_pd_hessian.jl", "test_ordination.jl",
        "test_destination_b_b1_unit_paired_fit.jl", "test_destination_b_b1_balanced_complete_crossed_design.jl", "test_precision_multivariate_fit.jl",
        "test_precision_fit_admission.jl", "test_phylo_latent_paired_p1.jl", "test_destination_b_phylo_independent_receipt.jl",
        "test_destination_b_a4_s4_tree_julia_own_optimum.jl", "test_destination_b_phylo_uncertainty_cli.jl", "test_destination_b_adapter_consumer.jl",
        "test_destination_b_pedigree_reference_checker.jl", "test_destination_b_postfit.jl", "test_gaussian_sources_fixed_residual.jl",
        "test_cross_objective_helpers.jl", "test_gaussian_source_design.jl", "test_bridge_sources.jl",
        "test_spde_latent.jl", "test_spde_latent_mode_flag_747.jl", "test_twopart_alloc_equiv.jl",
        "test_delta_shared_predictor.jl", "test_beta_binomial_mode_search.jl", "test_beta_binomial_phi_continuity_872.jl",
        "test_beta_binomial_grouped_verdict_515.jl", "test_zi_twin.jl", "test_zi_recovery.jl",
        "test_missing_predictor_phylo.jl", "test_confint_offset.jl", "test_predict_offset_shared.jl",
        "test_predict_modes_training.jl", "test_confint_bootstrap_verdict_nb1.jl", "test_confint_bootstrap_verdict_zi.jl",
        "test_confint_bootstrap_verdict_bhob.jl", "test_confint_bootstrap_verdict_nb.jl", "test_second_order_delta_followup.jl",
        "test_second_order_lognormal_ci.jl", "test_covariates.jl", "test_cov_mode_search_gamma.jl",
        "test_formula_input.jl", "test_row_random.jl", "test_nb1_row_effect_refusal.jl", "test_rrr.jl",
        "test_ordination_uncertainty.jl", "test_ordered_beta.jl", "test_ordered_beta_mode_search.jl",
        "test_variational.jl", "test_va_vs_laplace.jl", "test_bridge_ci.jl",
        "test_bridge_capabilities.jl", "test_phylo_eta_realized.jl", "test_bridge_missing_mask.jl",
        "test_known_sentinel_defects.jl", "test_edge_incidence_solveQ.jl", "test_branch_re_convergence.jl",
        "test_family11_p1_boundary.jl", "test_ordinal_link_input.jl", "test_em_phylo.jl",
        "test_em_squarem.jl", "test_em_squarem_safety.jl", "test_relaxed_clock_estep_clade.jl",
        "test_loading_profile_stage0.jl", "test_loading_profile_stage1_harness.jl", "test_loading_profile_confirmatory.jl",
        "test_derived_ci_surfaces.jl", "test_extractors.jl", "test_namespace_numeric_p1_twin_b.jl",
        "test_postfit_twins_p1.jl", "test_diagnostics_p1.jl", "test_off_predict_twin_p1.jl",
        "test_temporal_likelihood_internals.jl", "test_isdm_cloglog_internals.jl", "test_convergence_sweep.jl",
        "test_zinb_observed_laplace.jl", "test_input_validation.jl", "test_response_support_730.jl",
        "test_bootstrap_decisions.jl", "test_wtier_crosscov.jl", "test_branch_re_helpers.jl",
        "test_extractors_invariants.jl", "test_docstring_postfit_mask_offset_750.jl", "test_docstring_confint_lv_effects.jl",
        "test_mask_missing_763.jl", "test_bootstrap_sigma_r.jl",
        "test_sparse_phy_grad.jl",
        "test_nb_n_init_mite_1022.jl",
    ),
    (
        "test_shard_selection.jl", "test_animal_latent.jl", "test_precision_multivariate.jl",
        "test_destination_b_reference.jl", "test_destination_b_kernel_crosscheck.jl", "test_joint_phylo_grouped_alias_public.jl",
        "test_joint_phylo_grouped_postfit.jl", "test_confint_bootstrap.jl", "test_confint_derived.jl",
        "test_confint_derived_wald.jl", "test_takahashi_selinv.jl", "test_em_observed_information_pd.jl",
        "test_aghq_multistart.jl", "test_aghq_nb2.jl", "test_aghq_tweedie.jl",
        "test_aghq_delta_gamma.jl", "test_aghq_public_binomial.jl", "test_truncated_poisson.jl",
        "test_truncated_nbinom2.jl", "test_truncnb2_precision.jl", "test_truncnb2_mode_search.jl",
        "test_poisson_n_init.jl", "test_laplace_curvature_contract.jl", "test_laplace_curvature_oracle.jl",
        "test_laplace_dual_safety.jl", "test_phylo_nb_xlv.jl", "test_destination_b_joint_identification.jl",
        "test_destination_b_joint_poisson.jl", "test_gp1_laplace.jl", "test_gp1_verdict.jl",
        "test_nb_shared_r_boundary.jl", "test_grouped_dispersion_beta_gamma.jl", "test_nb_grouped_upper_warn.jl",
        "test_gamma_shape_upper_warn.jl", "test_nb_cov_two_starts.jl", "test_fit_verdict_gradient.jl",
        "test_nb_beta_x_identity.jl", "test_beta_grouped_mode_search.jl", "test_nb1_x_identity.jl",
        "test_zip_x_identity.jl", "test_issue_719_docs_deploy_guard.jl", "test_tweedie.jl",
        "test_tweedie_grouped_mode_search_503.jl", "test_studentt_core070.jl", "test_lognormal.jl",
        "test_zib_x_identity.jl", "test_ordinal_fit.jl", "test_ordinal_logit_twin.jl",
        "test_issue_749_fit_gllvm_doc.jl", "test_gaussian_pervar_design.jl", "test_gaussian_fixed_residual.jl",
        "test_formula_pervar.jl", "test_formula_default_x_guard.jl", "test_statsapi.jl",
        "test_postfit_zib_tweedie.jl", "test_extract_latent_scores.jl", "test_select_lv_759.jl",
        "test_select_lv_p1_twin.jl", "test_issue_108_anova_docs.jl", "test_cross_kernel.jl",
        "test_source_covariance.jl", "test_grouped_gaussian.jl", "test_destination_b_public.jl",
        "test_destination_b_b1_unit_obs_paired_fit.jl", "test_destination_b_b1_cluster2_paired_fit.jl", "test_b1_fixed_point_marginal_curvature_protocol.jl",
        "test_precision_shared_residual.jl", "test_phylo_latent_rounding_floor.jl", "test_destination_b_dense_uncertainty.jl",
        "test_bridge_precision_shared_residual.jl", "test_bridge_precision_multivariate.jl", "test_bridge_gradient_payload.jl",
        "test_bridge_phylo_precision.jl", "test_coevolution_glm.jl", "test_coevolution_glm_765.jl",
        "test_phylo_glm_families.jl", "test_phylo_beta_xlv_mode.jl", "test_phylo_ordinal_xlv_mode.jl",
        "test_xlv_refit_converged.jl", "test_twopart_mode_search.jl", "test_delta_disp_group.jl",
        "test_twopart_hessian_kwarg.jl", "test_zero_inflated.jl", "test_weights.jl",
        "test_weights_byte_identity.jl", "test_confint_bootstrap_verdict_tweedie.jl", "test_confint_bootstrap_verdict_ordinal.jl",
        "test_confint_bootstrap_verdict_misc.jl", "test_second_order_s2_cloglog_ratified.jl", "test_second_order_truncpois_ci.jl",
        "test_second_order_truncnb2_ci.jl", "test_second_order_multinomial_ci.jl", "test_second_order_betabinomial_shared_ci.jl",
        "test_nobs_pn_convention.jl", "test_cov_analytic_grad.jl", "test_optimizer_selection.jl",
        "test_formula_no_intercept_default.jl", "test_constrained_ordination.jl", "test_quadratic.jl",
        "test_variational_binomial.jl", "test_variational_beta.jl", "test_variational_exponential.jl",
        "test_random_slopes.jl", "test_twolevel.jl", "test_random_slopes_poisson.jl",
        "test_mixed_mode_search.jl", "test_mixed_contract.jl", "test_bridge_zib.jl",
        "test_bridge_zip_nox.jl", "test_bridge_truncated_poisson.jl", "test_bridge_lv_predictor.jl",
        "test_confint_hessian_twopart_truncnb2.jl", "test_grouped_hessian_consistency.jl", "test_saturation_health.jl",
        "test_phylo_branch_re.jl", "test_branch_re_marginal_fallback.jl", "test_core070_interface_registry.jl",
        "test_core070_shape_boundaries.jl", "test_cv_unsupported_773.jl", "test_gaussian_intercept.jl",
        "test_dunn_smyth_tail_767.jl", "test_predict_delta_shared_820.jl", "test_namespace_gaussian_w1_p1.jl",
        "test_confint_inspect_p1.jl", "test_data_twins_p1.jl", "test_predict_offset_twin_p1.jl",
        "test_off_mixed_twin_p1.jl", "test_destination_b_s4_public_phylo_dep_probe_harness.jl", "test_temporal_api.jl",
        "test_temporal_helpers.jl", "test_zip_betaz_wald_926.jl", "test_zerostep_verdict_sweep.jl",
        "test_derived_ci_sweep.jl", "test_nb_grouped_postfit.jl", "test_eta_clamp_rescue_874.jl",
        "test_derived_decisions.jl", "test_random_slope_internals.jl", "test_interrupt_rethrow.jl",
        "test_docstring_ref_linebreaks.jl",
        "test_nb_mite_multistart_1022.jl",
    ),
    (
        "test_animal_dep.jl", "test_animal_latent_slope_estimand_algebra.jl", "test_docstring_animal_latent_twin.jl",
        "test_W_and_diag.jl", "test_phy.jl", "test_phylo_precision.jl",
        "test_grouped_profile_foundation.jl", "test_joint_phylo_grouped_gaussian.jl", "test_joint_covariance_dense_gram.jl",
        "test_joint_phylo_grouped_interior.jl", "test_destination_b_precision_public.jl", "test_ppca_init.jl",
        "test_em_fa.jl", "test_confint.jl", "test_confint_profile.jl",
        "test_profile_rootfind.jl", "test_em_louis.jl", "test_em_sparse_estep_default.jl",
        "test_node_gradient.jl", "test_families.jl", "test_binomial_laplace.jl",
        "test_aghq_adapt.jl", "test_aghq_poisson.jl", "test_aghq_binomial.jl",
        "test_censored_poisson.jl", "test_truncnb2_laplace_breakdown.jl", "test_gamma_laplace.jl",
        "test_gamma_curvature_cross_kernel.jl", "test_phylo_beta_xlv.jl", "test_phylo_binomial_xlv.jl",
        "test_boundary_inference.jl", "test_grouped_laplace_identity.jl", "test_grouped_identification_diagnostics.jl",
        "test_destination_b_joint_other_families.jl", "test_nb_boundary_restart.jl", "test_nb1_grouped_mode_search.jl", "test_nb1_fisher_large_counts.jl",
        "test_nb2_grouped_mode_search.jl", "test_grouped_getlv_mode.jl", "test_gamma_grouped_mode_search.jl",
        "test_betabinomial_x_identity.jl", "test_zinb_x_identity.jl", "test_beta_fit.jl",
        "test_link_honoured.jl", "test_issue_725_extractor_docs.jl", "test_studentt_disp_group.jl",
        "test_studentt_normalizer_precision.jl", "test_studentt_retained_precision.jl", "test_studentt_grouped_mode_search.jl",
        "test_ordinal_pertrait.jl", "test_ordinal_level_check.jl", "test_ordinal_rowcheck.jl", "test_getlv_rowcheck.jl", "test_fit_gllvm.jl",
        "test_truncated_formula.jl", "test_gaussian_pervar.jl", "test_ledermann_pervar_k.jl",
        "test_postfit.jl", "test_issue_737_omega_doc.jl", "test_model_selection.jl",
        "test_select_lv_print.jl", "test_binomial_ridge.jl", "test_c1_behaviour_p1.jl",
        "test_structured_cov.jl", "test_symmetric_admission.jl", "test_destination_b_b1_joint_gaussian_paired_fit.jl",
        "test_phylo_latent_twin.jl", "test_destination_b_phylo_reference_checker.jl", "test_destination_b_pedigree_precision.jl",
        "test_destination_b_tree_precision.jl", "test_destination_b_tree_reference_checker.jl", "test_destination_b_a4_s4_public_r_formula_receipt.jl",
        "test_precision_objective_failures.jl", "test_grouped_nongaussian_postfit.jl", "test_formula_sources.jl",
        "test_formula_structured_terms.jl", "test_gaussian_source_bindings.jl", "test_phylo_precision_animal.jl",
        "test_coevolution_kronecker.jl", "test_spde_delaunay.jl", "test_spde_fit.jl",
        "test_spde_latent_postfit.jl", "test_phylo_binomial_xlv_mode.jl", "test_delta_fit.jl",
        "test_delta_postfit.jl", "test_hurdle_poisson.jl", "test_hurdle_nb.jl",
        "test_delta_gamma.jl", "test_missing_data.jl", "test_missing_response_extra.jl",
        "test_missing_predictor_poisson.jl", "test_missing_predictor_multi.jl", "test_nb_grouped_wald_speed_552.jl",
        "test_confint_family.jl", "test_predict_offset_grouped.jl", "test_family_profile_open_lower.jl",
        "test_confint_bootstrap_verdict_poisson.jl", "test_confint_bootstrap_verdict_beta.jl", "test_confint_bootstrap_verdict_hurdle_delta.jl",
        "test_truncnb2_ci_boundary.jl", "test_confint_bootstrap_ordered_beta.jl", "test_confint_bootstrap_verdict_rest.jl",
        "test_confint_bootstrap_boundary_per_param.jl", "test_second_order_tweedie_grouped_ci.jl", "test_diagnostics.jl",
        "test_summary_table.jl", "test_formula.jl", "test_simulate.jl",
        "test_species_covariates.jl", "test_fourthcorner.jl", "test_structural_confint.jl",
        "test_variational_negbin.jl", "test_variational_dgamma.jl", "test_bridge_grouped_dispersion.jl",
        "test_bridge_x.jl", "test_bridge_lognormal.jl", "test_bridge_truncated_input.jl",
        "test_bridge_integer_counts.jl", "test_lv_ci.jl", "test_lv_profile_pd_hessian.jl",
        "test_bridge_gaussian_offset.jl", "test_confint_hessian_consistency.jl", "test_curvature_census.jl",
        "test_phylo_contrasts_struct.jl", "test_parity_trial_inputs.jl", "test_em_phylo_sigma_eps_floor.jl",
        "test_nparams_em_branch_re.jl", "test_spatial_cov.jl", "test_cv.jl",
        "test_namespace_numeric_p1_twin.jl", "test_mc_simulate_p1.jl", "test_variance_decomp_p1.jl",
        "test_off_all_count_twin_p1.jl", "test_weights_twins_p1.jl", "test_temporal_lowrank.jl",
        "test_temporal_composed.jl", "test_isdm.jl", "test_zip_observed_laplace.jl",
        "test_tweedie_speed.jl", "test_clamp_and_replicate_retention_docs.jl", "test_lambda_constraint_wald_794.jl",
        "test_comparator_targets.jl", "test_destination_b_a4_s4_fixed_coordinate_evaluator.jl",
    ),
    (
        "test_likelihood.jl", "test_packing.jl", "test_docstring_animal_dep_twin.jl",
        "test_spatial_dep.jl", "test_fit.jl", "test_laplace_K_bounds.jl",
        "test_fixed_effects.jl", "test_reml.jl", "test_lv_predictor.jl",
        "test_sparse_phy.jl", "test_grouped_profile_inversion.jl", "test_grouped_profile_dense_oracle.jl",
        "test_grouped_profile_public.jl", "test_marginal_target_intervals.jl", "test_lowrank_cholesky.jl",
        "test_profile_sigma_phy_identity.jl", "test_confint_method_routes.jl", "test_fit_phylo.jl",
        "test_cloglog_likelihood.jl", "test_aghq_frozen.jl", "test_aghq_public_poisson.jl",
        "test_aghq_gaussian.jl", "test_profile_failure_bounds.jl", "test_aghq_p1_twin.jl",
        "test_nb2_ad_large_r.jl", "test_ordinal_laplace.jl", "test_phylo_xlv.jl",
        "test_xlv_nb_binomial_wired.jl", "test_laplace_grad.jl", "test_grouped_laplace.jl",
        "test_laplace_grad_identity.jl", "test_masked_dispersion_grad.jl", "test_nb_fit.jl",
        "test_nb1.jl", "test_gp1_mode_backtrack.jl", "test_gamma_beta_upper_boundary.jl",
        "test_grouped_dispersion_tweedie_nb1.jl", "test_nb1_poisson_ridge.jl", "test_nb1_scalar_gradient_gate.jl",
        "test_grouped_init_kwargs.jl", "test_grouped_getlv_offset.jl", "test_ordinal_x_identity.jl",
        "test_zinb_cov_iterations.jl", "test_issue_721_link_docs.jl", "test_tweedie_power_contract.jl",
        "test_studentt_boundary.jl", "test_multinomial.jl", "test_multinomial_separation.jl",
        "test_com_poisson.jl", "test_com_poisson_mode_search.jl", "test_gaussian_pervar_fallback.jl",
        "test_aicbic_newfits.jl", "test_vcov_full_covariance.jl", "test_model_comparison.jl",
        "test_extract_gamma.jl", "test_cross_kernel_fit.jl", "test_gaussian_sources.jl",
        "test_grouped_fit.jl", "test_destination_b_b1_cluster_paired_fit.jl", "test_b1_fixed_coordinate_curvature_protocol.jl",
        "test_destination_b_a4_s4_public_dense_bridge.jl", "test_destination_b_phylo_uncertainty.jl", "test_grouped_nongaussian_fit.jl",
        "test_destination_b_nongaussian_public.jl", "test_destination_b_fixed_effects.jl", "test_source_fit_optimizer_health.jl",
        "test_cross_objective_known_answer.jl", "test_coevolution_blockna.jl", "test_spde.jl",
        "test_spde_mesh.jl", "test_spde_delaunay_scale.jl", "test_spde_latent_binomial_trials.jl",
        "test_phylo_glm.jl", "test_phylo_poisson_xlv.jl", "test_phylo_xlv_verdict.jl",
        "test_twopart_substrate.jl", "test_twopart_input_check.jl", "test_beta_hurdle.jl",
        "test_beta_binomial.jl", "test_beta_binomial_verdict_515.jl", "test_missing_response.jl",
        "test_masked_gaussian_posdef.jl", "test_masked_gaussian_converged_735.jl", "test_missing_predictor_fiml.jl",
        "test_missing_predictor_z.jl", "test_missing_predictor_dispersion.jl", "test_mi_fitter.jl",
        "test_offset.jl", "test_fd_hessian.jl", "test_confint_mask_refused.jl",
        "test_predict_offset.jl", "test_predict_offset_remaining_788.jl", "test_confint_bootstrap_verdict_binomial.jl",
        "test_confint_bootstrap_verdict_gamma.jl", "test_confint_bootstrap_verdict_truncated.jl", "test_confint_bootstrap_verdict_betabinomial.jl",
        "test_second_order_ordinal_pertrait_ci.jl", "test_second_order_studentt_ci.jl", "test_second_order_r_libs.jl",
        "test_se_machinery.jl", "test_row_effects.jl", "test_structural_confint_edge.jl",
        "test_variational_gamma.jl", "test_bridge_mixed.jl", "test_hessian_kwarg.jl",
        "test_phylo_contrasts.jl", "test_edge_incidence.jl", "test_core070_receipts.jl",
        "test_core070_link_boundaries.jl", "test_sparse_phy_identities.jl", "test_relaxed_clock.jl",
        "test_relaxed_clock_boundary_876.jl", "test_gaussian_formula_covariates.jl", "test_postfit_tables.jl",
        "test_docs_snippets_734_738.jl", "test_postfit_communality_confint_default_parity.jl", "test_animal_scalar_p1.jl",
        "test_cross_lineage_p1.jl", "test_cross_lineage_extract_internals.jl", "test_bridge_readback_p1.jl",
        "test_predict_missing_p1.jl", "test_parity_binomial_gaussian_fixtures.jl", "test_data_twins_2_p1.jl",
        "test_cov_ord_latent_p1.jl", "test_cov_phylo_twins_p1.jl", "test_totoro_323_track_a_harness.jl",
        "test_temporal_oracles.jl", "test_temporal_fit_receipts.jl", "test_temporal_composed_receipts.jl",
        "test_zip_zinb_stall.jl", "test_robustness_sweep.jl", "test_convergence_553_554.jl",
        "test_eta_clamp_869.jl", "test_response_support_730_va_ordinal.jl", "test_truncpois_boot_782.jl",
        "test_n_lt_p.jl", "test_spde_projector_validation.jl", "test_response_support_730_siblings.jl",
        "test_phylo_sparse_solvers.jl", "test_changelog_structure.jl",
        "test_gaussian_nonfinite_y.jl",
        "test_predictive_check_binomial.jl", "test_kernel_latent_unique_bridge.jl", "test_destination_b_a4_s4_receipts.jl",
        "test_core070_covariance_registry.jl", "test_core070_covariance_formula_registry.jl",
    ),
))
    for _f in _files
        _SHARD4_TABLE[_f] = _k
    end
end

# Does the include at 1-based position `pos` (file `path`) belong to this run?
function _in_shard(path::AbstractString, pos::Integer)
    _SHARD === nothing && return true
    k, N = _SHARD
    N == 4 && haskey(_SHARD4_TABLE, path) && return _SHARD4_TABLE[path] == k
    return (pos - 1) % N == k - 1
end

# Gated files (opt-in or version-limited). `_shard_include_if(cond, path)` selects
# exactly like `_shard_include` (it ALWAYS advances the position counter, so a
# gate never shifts the position of any later file, and uses `_in_shard` with the
# 4-way table) and then runs the file only when `cond` is true. Keep these calls at
# the END of the include list so earlier positions are unchanged. Their lines do
# not match the `_shard_include("` header count below, so the header counts only
# ungated files.
#   GLLVM_SPARSE_PHY_GRAD_TESTS=1 : test_sparse_phy_grad.jl (about 900-1100 s alone; opt-in,
#                        a dedicated variable so it does not also switch on the other
#                        slow-gated tests that read GLLVM_SLOW_TESTS).
#   Julia 1.10.x only  : test_destination_b_a4_s4_fixed_coordinate_evaluator.jl
#                        (asserts the Julia runner matches ^1\.10\.; fails on 1.13).
_RUN_SPARSE_PHY_GRAD = get(ENV, "GLLVM_SPARSE_PHY_GRAD_TESTS", "") == "1"
_IS_JULIA_110 = VERSION.major == 1 && VERSION.minor == 10

function _shard_include(path::AbstractString)
    _shard_pos[] += 1
    if _in_shard(path, _shard_pos[])
        if _REQUIRE_THREADS > 0 && path in _THREADED_FILES
            @testset "threaded file $path needs Julia threads" begin
                @test Threads.nthreads() >= _REQUIRE_THREADS
            end
        end
        include(path)
    end
end

function _shard_include_if(cond::Bool, path::AbstractString)
    _shard_pos[] += 1
    if cond && _in_shard(path, _shard_pos[])
        include(path)
    end
end

# test_quality.jl (Aqua + JET) measured standalone at ~35s combined (2026-09)
# — far cheaper than the suite split across the files below, so it is not
# worth spreading across shards: it runs once, in shard 1 only (or always
# when unsharded), and is deliberately NOT behind `_shard_include` above.
_RUN_QUALITY = _SHARD === nothing || _SHARD[1] == 1

# File count for the header below: every `_shard_include("` call literally
# present in this file's own source — the same ordered list `_shard_include`
# walks at runtime — so the count can never drift from the include list as
# files are added or removed. Count only the INDENTED call sites: the bare
# literal also matches this very expression, a silent off-by-one in the header
# (harmless to selection, which uses the runtime `_shard_pos` counter, but
# wrong in the log). Carried over from PR #278.
_n_test_files = count("\n    _shard_include(\"", read(@__FILE__, String))
_n_selected = count(enumerate(m.captures[1] for m in eachmatch(
    r"\n    _shard_include\(\"([^\"]+)\"\)", read(@__FILE__, String)))) do (i, f)
    _in_shard(f, i)
end
_n_reported = _n_selected + (_RUN_QUALITY ? 1 : 0)

println(_SHARD === nothing ?
    "GLLVModels tests: all files ($_n_reported files)" :
    "GLLVModels tests: shard $(_SHARD[1])/$(_SHARD[2]) — $_n_reported files")

# All suites run under one outer testset so that a failure in any included file
# does not abort the run before later files execute: included `@testset`s nest
# (Test uses dynamic scoping), accumulate, and only this root throws at the end —
# so one CI run reports every failure across all files, not just the first.
@testset "GLLVModels.jl" begin
    @testset "smoke" begin
        @test 1 + 1 == 2
        @test GLLVModels.GLLVM === GLLVModels
    end

    _shard_include("test_package_rename_alias.jl")
    _shard_include("test_shard_selection.jl")
    _shard_include("test_likelihood.jl")
    _shard_include("test_packing.jl")
    _shard_include("test_none_dep.jl")
    _shard_include("test_phylo_dep.jl")
    _shard_include("test_animal_dep.jl")
    _shard_include("test_docstring_animal_dep_twin.jl")
    _shard_include("test_animal_latent.jl")
    _shard_include("test_animal_latent_slope_estimand_algebra.jl")
    _shard_include("test_docstring_animal_latent_twin.jl")
    _shard_include("test_spatial_dep.jl")
    _shard_include("test_kernel_indep.jl")
    _shard_include("test_kernel_dep.jl")
    _shard_include("test_kernel_latent.jl")
    _shard_include("test_fit.jl")
    _shard_include("test_gaussian_K_bounds.jl")
    _shard_include("test_laplace_K_bounds.jl")
    _shard_include("test_fixed_effects.jl")
    _shard_include("test_reml.jl")
    _shard_include("test_lv_predictor.jl")
    _shard_include("test_W_and_diag.jl")
    _shard_include("test_unidentified_tier_warning.jl")
    _shard_include("test_phy.jl")
    _shard_include("test_signed_sigma_phy.jl")
    _shard_include("test_sparse_phy.jl")
    _shard_include("test_phylo_precision.jl")
    _shard_include("test_precision_multivariate.jl")
    _shard_include("test_destination_b_reference.jl")
    _shard_include("test_destination_b_kernel_crosscheck.jl")
    _shard_include("test_destination_b_grouping_invariance.jl")
    _shard_include("test_destination_b_grouping_interval_matrix.jl")
    _shard_include("test_grouped_profile_foundation.jl")
    _shard_include("test_grouped_profile_refit.jl")
    _shard_include("test_grouped_profile_inversion.jl")
    _shard_include("test_grouped_profile_interval.jl")
    _shard_include("test_grouped_profile_dense_oracle.jl")
    _shard_include("test_grouped_profile_public.jl")
    _shard_include("test_joint_phylo_grouped_gaussian.jl")
    _shard_include("test_joint_covariance_identification.jl")
    _shard_include("test_joint_covariance_dense_gram.jl")
    _shard_include("test_joint_phylo_grouped_fit.jl")
    _shard_include("test_joint_phylo_grouped_alias_public.jl")
    _shard_include("test_joint_phylo_grouped_interior.jl")
    _shard_include("test_joint_phylo_grouped_postfit.jl")
    _shard_include("test_extract_phylo_signal_joint.jl")
    _shard_include("test_destination_b_precision_public.jl")
    _shard_include("test_marginal_target_intervals.jl")
    _shard_include("test_ppca_init.jl")
    _shard_include("test_em_fa.jl")
    _shard_include("test_lowrank_cholesky.jl")
    _shard_include("test_confint.jl")
    _shard_include("test_confint_profile.jl")
    _shard_include("test_profile_sigma_phy_identity.jl")
    _shard_include("test_profile_rootfind.jl")
    _shard_include("test_confint_bootstrap.jl")
    _shard_include("test_confint_derived.jl")
    _shard_include("test_confint_derived_wald.jl")
    _shard_include("test_confint_method_routes.jl")
    _shard_include("test_profile_derived_fix.jl")
    _shard_include("test_takahashi_selinv.jl")
    _shard_include("test_em_louis.jl")
    _shard_include("test_em_observed_information_pd.jl")
    _shard_include("test_em_sparse_estep_default.jl")
    _shard_include("test_node_gradient.jl")
    _shard_include("test_fit_phylo.jl")
    _shard_include("test_families.jl")
    _shard_include("test_binomial_laplace.jl")
    _shard_include("test_cloglog_likelihood.jl")
    _shard_include("test_aghq_grid.jl")
    _shard_include("test_aghq_adapt.jl")
    _shard_include("test_aghq_frozen.jl")
    _shard_include("test_aghq_outer.jl")
    _shard_include("test_aghq_multistart.jl")
    _shard_include("test_aghq_poisson.jl")
    _shard_include("test_aghq_public_poisson.jl")
    _shard_include("test_aghq_nb2.jl")
    _shard_include("test_aghq_ordinal.jl")
    _shard_include("test_aghq_tweedie.jl")
    _shard_include("test_aghq_delta_gamma.jl")
    _shard_include("test_aghq_gaussian.jl")
    _shard_include("test_aghq_public_gaussian.jl")
    _shard_include("test_gaussian_empty_design.jl")
    _shard_include("test_profile_failure_bounds.jl")
    _shard_include("test_aghq_binomial.jl")
    # P1 numeric twins of the aghq policy rows vs gllvmTMB (Poisson / binomial / Gaussian fits).
    _shard_include("test_aghq_p1_twin.jl")
    _shard_include("test_aghq_public_binomial.jl")
    _shard_include("test_aghq_gate.jl")
    _shard_include("test_aghq_kd_bound.jl")
    _shard_include("test_poisson_laplace.jl")
    _shard_include("test_truncated_poisson.jl")
    _shard_include("test_censored_poisson.jl")
    _shard_include("test_truncated_nbinom2.jl")
    _shard_include("test_truncnb2_precision.jl")
    _shard_include("test_truncnb2_laplace_breakdown.jl")
    _shard_include("test_truncnb2_mode_search.jl")
    _shard_include("test_truncnb2_dispersion_boundary.jl")
    _shard_include("test_truncnb2_pertrait_boundary_restart.jl")
    _shard_include("test_nb2_precision.jl")
    _shard_include("test_nb2_ad_large_r.jl")
    _shard_include("test_negbin_laplace.jl")
    _shard_include("test_beta_laplace.jl")
    _shard_include("test_ordinal_laplace.jl")
    _shard_include("test_gamma_laplace.jl")
    _shard_include("test_binomial_fit.jl")
    _shard_include("test_poisson_fit.jl")
    _shard_include("test_poisson_n_init.jl")
    _shard_include("test_nb_n_init.jl")
    _shard_include("test_laplace_curvature_contract.jl")
    _shard_include("test_issue_783_curvature_docs.jl")
    _shard_include("test_laplace_curvature_oracle.jl")
    _shard_include("test_laplace_dual_safety.jl")
    _shard_include("test_gamma_curvature_cross_kernel.jl")

    # Wired in 2026-08-25. These five were ORPHANED — present in test/, absent
    # from this file, and therefore never run in CI — while their sources
    # (src/phylo_*_xlv.jl) ARE shipped, included at src/GLLVModels.jl:103-107. That
    # is untested shipped code. All five pass; each was run individually first.
    #
    # test_phylo_gamma_xlv.jl is deliberately NOT wired in yet: its :123
    # assertion compares against a reference implementation inside the test file
    # that still computes the Fisher log-det, and updating that oracle should
    # not be done by whoever changed the code it judges.
    _shard_include("test_phylo_xlv.jl")
    _shard_include("test_phylo_beta_xlv.jl")

    # #948: wire NB and binomial xlv. Profile-CI canaries stay behind
    # GLLVM_XLV_CANARY=1 (pd_hessian / endpoint_status / NaN bounds are
    # optimiser-path and BLAS sensitive; they failed on this 1.10.12
    # snapshot too). Likelihood identities, likelihood ArgumentError
    # guards, and the cheap profile ArgumentError guards run in CI.
    # The NB sparse-vs-dense-leaf identity is `@test_broken` (1.3e-2
    # gap at equal parameters; do not widen atol). test_phylo_ordinal_xlv
    # stays unwired (same flake class; not in #948).
    _shard_include("test_phylo_binomial_xlv.jl")
    _shard_include("test_phylo_nb_xlv.jl")
    _shard_include("test_xlv_nb_binomial_wired.jl")

    #   include("test_phylo_ordinal_xlv.jl")
    #   test_sparse_phy_grad.jl is now opt-in: GLLVM_SPARSE_PHY_GRAD_TESTS=1 (end of the list).


    # chibar2_pvalue / variance_lrt / profile_ci_variance are EXPORTED and had
    # zero tests. They return p-values and confidence intervals — the outputs
    # most likely to end up in a paper — so untested was the least acceptable
    # place for it. Every assertion is against an independently derived value.
    _shard_include("test_boundary_inference.jl")
    # P1 twin vs gllvmTMB's R/chibar.R (D-297 case map: a name match alone never
    # counts as parity; this proves equal outputs on a literal fixture computed by
    # the real installed R package at the pin).
    _shard_include("test_chibar2_variance_lrt_p1_twin.jl")
    _shard_include("test_laplace_grad.jl")
    _shard_include("test_poisson_grad_perf.jl")
    _shard_include("test_grouped_laplace.jl")
    _shard_include("test_grouped_laplace_identity.jl")
    _shard_include("test_latte_kernel_identity.jl")
    _shard_include("test_grouped_analytic_grad.jl")
    _shard_include("test_laplace_grad_identity.jl")
    _shard_include("test_grouped_identification_diagnostics.jl")
    _shard_include("test_destination_b_joint_identification.jl")
    _shard_include("test_destination_b_joint_poisson.jl")
    _shard_include("test_destination_b_joint_other_families.jl")
    _shard_include("test_destination_b_joint_nb2_replication.jl")
    _shard_include("test_destination_b_quadrature.jl")
    _shard_include("test_masked_dispersion_grad.jl")
    _shard_include("test_laplace_alloc_equiv.jl")
    _shard_include("test_nb_fit.jl")
    _shard_include("test_nb1.jl")
    _shard_include("test_gp1_laplace.jl")
    _shard_include("test_gp1_verdict.jl")
    _shard_include("test_gp1_mode_backtrack.jl")
    _shard_include("test_grouped_dispersion.jl")
    _shard_include("test_nb_boundary_restart.jl")
    _shard_include("test_nb_shared_r_boundary.jl")
    _shard_include("test_grouped_dispersion_beta_gamma.jl")
    _shard_include("test_gamma_beta_upper_boundary.jl")
    _shard_include("test_nb_grouped_upper_warn.jl")
    _shard_include("test_gamma_shape_upper_warn.jl")
    _shard_include("test_nb_grouped_poisson_ridge.jl")
    _shard_include("test_nb_cov_ridge_polish_grad.jl")
    _shard_include("test_nb_cov_two_starts.jl")
    _shard_include("test_beta_grouped_convergence.jl")
    _shard_include("test_fit_verdict_gradient.jl")
    _shard_include("test_grouped_dispersion_tweedie_nb1.jl")
    _shard_include("test_nb1_poisson_ridge.jl")
    _shard_include("test_nb1_scalar_gradient_gate.jl")
    _shard_include("test_nb1_grouped_mode_search.jl")
    _shard_include("test_nb1_fisher_large_counts.jl")
    _shard_include("test_nb2_grouped_mode_search.jl")
    _shard_include("test_grouped_init_kwargs.jl")
    _shard_include("test_grouped_getlv_mode.jl")
    _shard_include("test_nb_beta_x_identity.jl")
    _shard_include("test_gamma_x_identity.jl")
    _shard_include("test_gamma_grouped_mode_search.jl")
    _shard_include("test_beta_grouped_mode_search.jl")
    _shard_include("test_grouped_getlv_offset.jl")
    _shard_include("test_nb1_x_identity.jl")
    _shard_include("test_betabinomial_x_identity.jl")
    _shard_include("test_ordinal_x_identity.jl")
    _shard_include("test_zip_x_identity.jl")
    _shard_include("test_zinb_x_identity.jl")
    _shard_include("test_zinb_cov_iterations.jl")
    _shard_include("test_beta_fit.jl")
    _shard_include("test_gamma_fit.jl")
    _shard_include("test_link_honoured.jl")
    _shard_include("test_issue_721_link_docs.jl")
    _shard_include("test_issue_725_extractor_docs.jl")
    _shard_include("test_issue_719_docs_deploy_guard.jl")
    _shard_include("test_tweedie.jl")
    _shard_include("test_tweedie_engine_health.jl")
    _shard_include("test_tweedie_grouped_engine_health.jl")
    _shard_include("test_tweedie_grouped_mode_search_503.jl")
    _shard_include("test_tweedie_power_contract.jl")
    _shard_include("test_tweedie_cdf.jl")
    _shard_include("test_exponential.jl")
    _shard_include("test_studentt.jl")
    _shard_include("test_studentt_input_validation.jl")
    _shard_include("test_studentt_disp_group.jl")
    _shard_include("test_studentt_core070.jl")
    _shard_include("test_studentt_boundary_honesty.jl")
    _shard_include("test_studentt_boundary.jl")
    _shard_include("test_studentt_normalizer_precision.jl")
    _shard_include("test_studentt_retained_precision.jl")
    _shard_include("test_studentt_grouped_mode_search.jl")
    _shard_include("test_mode_search_623.jl")
    _shard_include("test_lognormal.jl")
    _shard_include("test_multinomial.jl")
    _shard_include("test_multinomial_separation.jl")
    _shard_include("test_zib_x_identity.jl")
    _shard_include("test_ordinal_fit.jl")
    _shard_include("test_ordinal_pertrait.jl")
    _shard_include("test_ordinal_probit.jl")
    _shard_include("test_ordinal_logit_twin.jl")
    _shard_include("test_ordinal_level_check.jl")
    _shard_include("test_ordinal_rowcheck.jl")
    _shard_include("test_getlv_rowcheck.jl")
    _shard_include("test_fit_gllvm.jl")
    _shard_include("test_issue_749_fit_gllvm_doc.jl")
    _shard_include("test_truncated_formula.jl")
    _shard_include("test_unified_api.jl")
    _shard_include("test_com_poisson.jl")
    _shard_include("test_com_poisson_mode_search.jl")
    _shard_include("test_gaussian_pervar.jl")
    _shard_include("test_ledermann_pervar_k.jl")
    _shard_include("test_gaussian_pervar_design.jl")
    _shard_include("test_gaussian_fixed_residual.jl")
    _shard_include("test_gaussian_pervar_fallback.jl")
    _shard_include("test_gaussian_pervar_heywood_736.jl")   # #736
    _shard_include("test_formula_pervar.jl")
    _shard_include("test_formula_default_x_guard.jl")
    _shard_include("test_aicbic_newfits.jl")
    _shard_include("test_postfit.jl")
    _shard_include("test_issue_737_omega_doc.jl")
    _shard_include("test_statsapi.jl")
    _shard_include("test_vcov_full_covariance.jl")
    _shard_include("test_gaussian_wald_pd_hessian.jl")
    _shard_include("test_postfit_zib_tweedie.jl")
    _shard_include("test_ordination.jl")
    _shard_include("test_extract_latent_scores.jl")
    _shard_include("test_model_selection.jl")
    _shard_include("test_select_lv_759.jl")
    # P1 numeric twin of select_lv() vs gllvmTMB R/select-lv.R (Gaussian rank sweep).
    _shard_include("test_select_lv_p1_twin.jl")
    # select_lv result: the nine fields R prints (AICc, conv, pdHess added).
    _shard_include("test_select_lv_print.jl")
    _shard_include("test_binomial_ridge.jl")
    _shard_include("test_model_comparison.jl")
    # #108: tutorial gllvm_anova section (status vocabulary, claim boundary) and the behaviour it states.
    _shard_include("test_issue_108_anova_docs.jl")
    # P1 behavioural twins (itchyshin/GLLVModels.jl#684 item 2): print.anova table fields and the
    # extract_latent_scores refusal vs gllvmTMB R/aghq-report.R and R/extract-latent-scores.R.
    _shard_include("test_c1_behaviour_p1.jl")
    _shard_include("test_structured_cov.jl")
    _shard_include("test_cross_kernel.jl")
    _shard_include("test_extract_gamma.jl")
    _shard_include("test_cross_kernel_fit.jl")
    _shard_include("test_source_covariance.jl")
    _shard_include("test_symmetric_admission.jl")
    _shard_include("test_gaussian_sources.jl")
    _shard_include("test_grouped_gaussian.jl")
    _shard_include("test_grouped_fit.jl")
    _shard_include("test_destination_b_public.jl")
    _shard_include("test_destination_b_b1_unit_paired_fit.jl")
    _shard_include("test_destination_b_b1_unit_obs_paired_fit.jl")
    _shard_include("test_destination_b_b1_cluster_paired_fit.jl")
    _shard_include("test_destination_b_b1_cluster2_paired_fit.jl")
    _shard_include("test_destination_b_b1_joint_gaussian_paired_fit.jl")
    _shard_include("test_destination_b_b1_balanced_complete_crossed_design.jl")
    _shard_include("test_b1_fixed_coordinate_curvature_protocol.jl")
    _shard_include("test_b1_fixed_point_marginal_curvature_protocol.jl")
    _shard_include("test_precision_multivariate_fit.jl")
    _shard_include("test_precision_fit_admission.jl")
    _shard_include("test_precision_shared_residual.jl")
    _shard_include("test_phylo_latent_twin.jl")
    _shard_include("test_phylo_latent_paired_p1.jl")
    _shard_include("test_phylo_latent_rounding_floor.jl")
    _shard_include("test_destination_b_phylo_reference_checker.jl")
    _shard_include("test_destination_b_phylo_independent_receipt.jl")
    _shard_include("test_destination_b_pedigree_precision.jl")
    _shard_include("test_destination_b_tree_precision.jl")
    _shard_include("test_destination_b_tree_reference_checker.jl")
    _shard_include("test_destination_b_a4_s4_tree_julia_own_optimum.jl")
    _shard_include("test_destination_b_a4_s4_public_dense_bridge.jl")
    _shard_include("test_destination_b_a4_s4_public_r_formula_receipt.jl")
    _shard_include("test_destination_b_phylo_uncertainty.jl")
    _shard_include("test_destination_b_phylo_uncertainty_cli.jl")
    _shard_include("test_destination_b_dense_uncertainty.jl")
    _shard_include("test_destination_b_adapter_consumer.jl")
    _shard_include("test_destination_b_pedigree_reference_checker.jl")
    _shard_include("test_bridge_precision_shared_residual.jl")
    _shard_include("test_precision_objective_failures.jl")
    _shard_include("test_bridge_precision_multivariate.jl")
    _shard_include("test_grouped_nongaussian_fit.jl")
    _shard_include("test_destination_b_nongaussian_public.jl")
    _shard_include("test_destination_b_postfit.jl")
    _shard_include("test_grouped_nongaussian_postfit.jl")
    _shard_include("test_destination_b_fixed_effects.jl")
    _shard_include("test_gaussian_sources_fixed_residual.jl")
    _shard_include("test_source_fit_optimizer_health.jl")
    _shard_include("test_cross_objective_known_answer.jl")
    _shard_include("test_cross_objective_helpers.jl")
    _shard_include("test_gaussian_source_design.jl")
    _shard_include("test_formula_sources.jl")
    _shard_include("test_formula_structured_terms.jl")
    _shard_include("test_gaussian_source_bindings.jl")
    _shard_include("test_bridge_gradient_payload.jl")
    _shard_include("test_bridge_phylo_precision.jl")
    _shard_include("test_phylo_precision_animal.jl")
    _shard_include("test_bridge_sources.jl")
    _shard_include("test_coevolution_kronecker.jl")
    _shard_include("test_coevolution_blockna.jl")
    _shard_include("test_coevolution_glm.jl")
    _shard_include("test_coevolution_glm_765.jl")
    _shard_include("test_spde.jl")
    _shard_include("test_spde_mesh.jl")
    _shard_include("test_spde_delaunay.jl")
    _shard_include("test_spde_delaunay_scale.jl")
    _shard_include("test_spde_fit.jl")
    _shard_include("test_spde_latent.jl")
    _shard_include("test_spde_latent_postfit.jl")
    _shard_include("test_spde_latent_binomial_trials.jl")
    _shard_include("test_spde_latent_mode_flag_747.jl")
    _shard_include("test_phylo_glm.jl")
    _shard_include("test_phylo_glm_families.jl")
    _shard_include("test_phylo_poisson_xlv.jl")
    _shard_include("test_phylo_binomial_xlv_mode.jl")
    _shard_include("test_phylo_beta_xlv_mode.jl")
    _shard_include("test_phylo_ordinal_xlv_mode.jl")
    _shard_include("test_xlv_refit_converged.jl")
    _shard_include("test_phylo_xlv_verdict.jl")
    _shard_include("test_twopart_substrate.jl")
    _shard_include("test_twopart_alloc_equiv.jl")
    _shard_include("test_twopart_mode_search.jl")
    _shard_include("test_delta_fit.jl")
    _shard_include("test_delta_shared_predictor.jl")
    _shard_include("test_delta_disp_group.jl")
    _shard_include("test_delta_postfit.jl")
    _shard_include("test_hurdle_poisson.jl")
    _shard_include("test_hurdle_nb.jl")
    _shard_include("test_delta_gamma.jl")
    _shard_include("test_twopart_hessian_kwarg.jl")
    _shard_include("test_twopart_input_check.jl")
    _shard_include("test_beta_hurdle.jl")
    _shard_include("test_beta_binomial.jl")
    _shard_include("test_beta_binomial_mode_search.jl")
    _shard_include("test_beta_binomial_verdict_515.jl")
    _shard_include("test_beta_binomial_phi_continuity_872.jl")
    _shard_include("test_beta_binomial_grouped_verdict_515.jl")
    _shard_include("test_zero_inflated.jl")
    _shard_include("test_zi_twin.jl")
    _shard_include("test_zi_recovery.jl")
    _shard_include("test_missing_data.jl")
    _shard_include("test_missing_response.jl")
    _shard_include("test_missing_response_extra.jl")
    _shard_include("test_masked_gaussian_posdef.jl")   # #716
    _shard_include("test_masked_gaussian_converged_735.jl")   # #735
    _shard_include("test_fix_1107.jl")   # #1107 derive Gaussian mask from missing
    _shard_include("test_missing_predictor_fiml.jl")
    _shard_include("test_missing_predictor_phylo.jl")
    _shard_include("test_missing_predictor_z.jl")
    _shard_include("test_missing_predictor_poisson.jl")
    _shard_include("test_missing_predictor_dispersion.jl")
    _shard_include("test_missing_predictor_multi.jl")
    _shard_include("test_mi_fitter.jl")
    _shard_include("test_offset.jl")
    # Observation weights (Poisson Laplace) and the guard that weights = nothing is byte-identical to main.
    _shard_include("test_weights.jl")
    _shard_include("test_weights_byte_identity.jl")
    _shard_include("test_fd_hessian.jl")
    _shard_include("test_nb_grouped_wald_speed_552.jl")
    _shard_include("test_confint_family.jl")
    _shard_include("test_confint_offset.jl")
    _shard_include("test_confint_mask_refused.jl")
    _shard_include("test_predict_offset.jl")
    _shard_include("test_predict_offset_grouped.jl")
    _shard_include("test_predict_offset_remaining_788.jl")
    _shard_include("test_predict_offset_shared.jl")
    _shard_include("test_predict_modes_training.jl")
    _shard_include("test_family_profile_open_lower.jl")
    _shard_include("test_confint_bootstrap_verdict_poisson.jl")
    _shard_include("test_confint_bootstrap_verdict_binomial.jl")
    _shard_include("test_confint_bootstrap_verdict_gamma.jl")
    _shard_include("test_confint_bootstrap_verdict_beta.jl")
    _shard_include("test_confint_bootstrap_verdict_nb1.jl")
    _shard_include("test_confint_bootstrap_verdict_tweedie.jl")
    _shard_include("test_confint_bootstrap_verdict_ordinal.jl")
    _shard_include("test_confint_bootstrap_verdict_zi.jl")
    _shard_include("test_confint_bootstrap_verdict_hurdle_delta.jl")
    _shard_include("test_confint_bootstrap_verdict_truncated.jl")
    _shard_include("test_truncnb2_ci_boundary.jl")
    _shard_include("test_confint_bootstrap_verdict_misc.jl")
    _shard_include("test_confint_bootstrap_verdict_bhob.jl")
    _shard_include("test_confint_bootstrap_ordered_beta.jl")
    _shard_include("test_confint_bootstrap_verdict_rest.jl")
    _shard_include("test_confint_bootstrap_verdict_betabinomial.jl")
    _shard_include("test_confint_bootstrap_verdict_nb.jl")
    _shard_include("test_confint_bootstrap_boundary_per_param.jl")
    _shard_include("test_second_order_ordinal_pertrait_ci.jl")
    _shard_include("test_second_order_delta_followup.jl")
    _shard_include("test_second_order_s2_cloglog_ratified.jl")
    _shard_include("test_second_order_lognormal_ci.jl")
    _shard_include("test_second_order_truncpois_ci.jl")
    _shard_include("test_second_order_truncnb2_ci.jl")
    _shard_include("test_second_order_tweedie_grouped_ci.jl")
    _shard_include("test_second_order_multinomial_ci.jl")
    _shard_include("test_second_order_betabinomial_shared_ci.jl")
    _shard_include("test_second_order_studentt_ci.jl")
    _shard_include("test_second_order_r_libs.jl")
    _shard_include("test_diagnostics.jl")
    _shard_include("test_se_machinery.jl")
    _shard_include("test_nobs_pn_convention.jl")
    _shard_include("test_summary_table.jl")
    _shard_include("test_covariates.jl")
    _shard_include("test_cov_mode_search_gamma.jl")
    _shard_include("test_cov_analytic_grad.jl")
    _shard_include("test_formula_input.jl")
    _shard_include("test_formula.jl")
    _shard_include("test_optimizer_selection.jl")
    _shard_include("test_formula_no_intercept_default.jl")
    _shard_include("test_simulate.jl")
    _shard_include("test_species_covariates.jl")
    _shard_include("test_fourthcorner.jl")
    _shard_include("test_row_effects.jl")
    _shard_include("test_row_random.jl")
    _shard_include("test_nb1_row_effect_refusal.jl")
    _shard_include("test_constrained_ordination.jl")
    _shard_include("test_rrr.jl")
    _shard_include("test_quadratic.jl")
    _shard_include("test_ordination_uncertainty.jl")
    _shard_include("test_structural_confint.jl")
    _shard_include("test_structural_confint_edge.jl")
    _shard_include("test_ordered_beta.jl")
    _shard_include("test_ordered_beta_mode_search.jl")
    _shard_include("test_variational.jl")
    _shard_include("test_variational_binomial.jl")
    _shard_include("test_variational_negbin.jl")
    _shard_include("test_variational_gamma.jl")
    _shard_include("test_variational_beta.jl")
    _shard_include("test_variational_dgamma.jl")
    _shard_include("test_variational_exponential.jl")
    _shard_include("test_fix_1108.jl")
    _shard_include("test_va_vs_laplace.jl")
    _shard_include("test_random_slopes.jl")
    _shard_include("test_twolevel.jl")
    _shard_include("test_random_slopes_poisson.jl")
    _shard_include("test_bridge_ci.jl")
    _shard_include("test_bridge_grouped_dispersion.jl")
    _shard_include("test_bridge_capabilities.jl")
    _shard_include("test_mixed_mode_search.jl")
    _shard_include("test_mixed_contract.jl")
    _shard_include("test_bridge_mixed.jl")
    _shard_include("test_bridge_x.jl")
    _shard_include("test_bridge_zib.jl")
    _shard_include("test_bridge_zip_nox.jl")
    _shard_include("test_bridge_lognormal.jl")
    _shard_include("test_bridge_truncated_poisson.jl")
    _shard_include("test_bridge_truncated_input.jl")
    _shard_include("test_bridge_integer_counts.jl")
    _shard_include("test_bridge_lv_predictor.jl")
    _shard_include("test_lv_ci.jl")
    _shard_include("test_lv_profile_pd_hessian.jl")
    _shard_include("test_phylo_eta_realized.jl")
    _shard_include("test_bridge_missing_mask.jl")
    _shard_include("test_bridge_gaussian_offset.jl")
    _shard_include("test_hessian_kwarg.jl")
    _shard_include("test_confint_hessian_consistency.jl")
    _shard_include("test_confint_hessian_twopart_truncnb2.jl")
    _shard_include("test_grouped_hessian_consistency.jl")
    _shard_include("test_saturation_health.jl")
    _shard_include("test_known_sentinel_defects.jl")
    _shard_include("test_curvature_census.jl")
    _shard_include("test_phylo_contrasts.jl")
    _shard_include("test_phylo_contrasts_struct.jl")
    _shard_include("test_edge_incidence.jl")
    _shard_include("test_edge_incidence_solveQ.jl")
    _shard_include("test_phylo_branch_re.jl")
    _shard_include("test_branch_re_convergence.jl")
    _shard_include("test_branch_re_marginal_fallback.jl")
    _shard_include("test_core070_receipts.jl")
    _shard_include("test_family11_p1_boundary.jl")
    _shard_include("test_core070_interface_registry.jl")
    _shard_include("test_core070_shape_boundaries.jl")
    _shard_include("test_ordinal_link_input.jl")
    _shard_include("test_core070_link_boundaries.jl")
    _shard_include("test_parity_trial_inputs.jl")
    _shard_include("test_em_phylo.jl")
    _shard_include("test_em_phylo_sigma_eps_floor.jl")
    _shard_include("test_nparams_em_branch_re.jl")
    _shard_include("test_sparse_phy_identities.jl")
    _shard_include("test_em_squarem.jl")
    _shard_include("test_em_squarem_safety.jl")
    _shard_include("test_relaxed_clock.jl")
    _shard_include("test_relaxed_clock_estep_clade.jl")
    _shard_include("test_relaxed_clock_boundary_876.jl")
    _shard_include("test_spatial_cov.jl")
    _shard_include("test_cv.jl")
    _shard_include("test_cv_unsupported_773.jl")
    _shard_include("test_gaussian_intercept.jl")
    _shard_include("test_gaussian_formula_covariates.jl")
    _RUN_QUALITY && include("test_quality.jl")
    _shard_include("test_loading_profile_stage0.jl")
    _shard_include("test_loading_profile_stage1_harness.jl")
    _shard_include("test_loading_profile_confirmatory.jl")
    _shard_include("test_derived_ci_surfaces.jl")
    _shard_include("test_extractors.jl")
    _shard_include("test_postfit_tables.jl")
    _shard_include("test_docs_snippets_734_738.jl")
    _shard_include("test_dunn_smyth_tail_767.jl")
    _shard_include("test_predict_delta_shared_820.jl")
    # P1 numeric twins of six gllvmTMB namespace rows (loadings, rotated table, lv effects, communality, Sigma_B/W).
    _shard_include("test_namespace_numeric_p1_twin.jl")
    # P1 numeric twins of three more namespace rows (tidy fixed effects, Beta, nbinom2).
    _shard_include("test_namespace_numeric_p1_twin_b.jl")
    # P1 numeric twins of three gllvmTMB postfit rows (tidy fixed effects, coef, deviance) on an uncentred fit.
    _shard_include("test_postfit_twins_p1.jl")
    # P1 Monte-Carlo moment twins (D-319 item N4) of simulate_unit_trait and the simulate() default.
    _shard_include("test_mc_simulate_p1.jl")
    # P1 numeric twins of gllvmTMB extract_proportions (one-tier and two-level) and extract_residual_split (two-level Gaussian).
    _shard_include("test_variance_decomp_p1.jl")
    # Post-fit parity vs gllvmTMB P1 fixtures: tier-local communality (refs #131) and confint default method (@test_broken, refs #1020).
    _shard_include("test_postfit_communality_confint_default_parity.jl")
    # P1 numeric twin of the gllvmTMB namespace row animal_scalar (one shared animal variance on a pedigree A, Gaussian).
    _shard_include("test_animal_scalar_p1.jl")
    # P1 numeric twins of the gllvmTMB namespace rows extract_Gamma and extract_coevolution_modules (cross-lineage kernel tier, Gaussian).
    _shard_include("test_cross_lineage_p1.jl")
    _shard_include("test_cross_lineage_extract_internals.jl")
    # P1 bridge readback: gllvmTMB_julia coef/fitted/logLik/predict/residuals/summary and gllvm_julia_fit, recorded live through JuliaCall.
    _shard_include("test_bridge_readback_p1.jl")
    # P1 numeric twins of the gllvmTMB namespace rows gllvmTMB_wide, ordiplot.gllvmTMB_multi and flag_unreliable_loadings (one shared Gaussian fixture).
    _shard_include("test_namespace_gaussian_w1_p1.jl")
    # P1 numeric twin of the gllvmTMB namespace row confint_inspect (estimate, Wald and profile bounds on four direct targets, Gaussian rank 2).
    _shard_include("test_confint_inspect_p1.jl")
    # P1 numeric twin of the gllvmTMB postfit row predict_missing (masked-cell predictions of a masked Gaussian fit).
    _shard_include("test_predict_missing_p1.jl")
    # P1 numeric twins of gllvmTMB sanity_multi (max fixed-effect SE, min |diag Lambda_B|) and compare_loadings (Procrustes output).
    _shard_include("test_diagnostics_p1.jl")
    # P1 fit-level twins of core070 data rows (six bind; see tools/core070_data_p1_receipts.py) (offset and missing-response handling).
    _shard_include("test_data_twins_p1.jl")
    _shard_include("test_parity_binomial_gaussian_fixtures.jl")
    _shard_include("test_predict_offset_twin_p1.jl")
    # P1 fit-level twin of data/DATA-OFF-PREDICT (training units, new offset, training modes kept; ruling D-319).
    _shard_include("test_off_predict_twin_p1.jl")
    # P1 fit-level twins of data/DATA-MISS-MODEL, data/DATA-MISS-BOTH and namespace imputed (modelled missing predictor, response mask).
    _shard_include("test_data_twins_2_p1.jl")
    # P1 fit-level twin of data/DATA-OFF-ALL-COUNT (exposure offset on nbinom2, truncated Poisson, truncated nbinom2).
    _shard_include("test_off_all_count_twin_p1.jl")
    # P1 fit-level twin of data/DATA-OFF-MIXED (one poisson/gaussian/nbinom2 fit, offset zero on the gaussian trait).
    _shard_include("test_off_mixed_twin_p1.jl")
    # P1 fit-level twins of the 13 data/DATA-W-* rows (observation weights on a Poisson fit: long, wide-matrix and traits() shapes, masks).
    _shard_include("test_weights_twins_p1.jl")
    # P1 fit-level twins of covariance/COV-ORD-LATENT-BARE, -DEFAULT and -COMMON (ordinary latent(), Gaussian rank 1).
    _shard_include("test_cov_ord_latent_p1.jl")
    # P1 fit-level twins of covariance/COV-PHYLO-DEP, COV-PHYLO-A-ALIAS and COV-PHYLO-FOLDED-UNIQUE (Gaussian phylo_latent fits).
    _shard_include("test_cov_phylo_twins_p1.jl")
    _shard_include("test_destination_b_s4_public_phylo_dep_probe_harness.jl")
    _shard_include("test_totoro_323_track_a_harness.jl")
    _shard_include("test_temporal_api.jl")
    _shard_include("test_temporal_oracles.jl")
    _shard_include("test_temporal_lowrank.jl")
    _shard_include("test_temporal_likelihood_internals.jl")
    _shard_include("test_temporal_fit_receipts.jl")
    _shard_include("test_temporal_helpers.jl")
    _shard_include("test_temporal_composed.jl")
    _shard_include("test_temporal_composed_receipts.jl")
    _shard_include("test_isdm.jl")
    _shard_include("test_isdm_cloglog_internals.jl")
    _shard_include("test_convergence_sweep.jl")
    # #573: ZIP/ZINB fits never end below the nested Poisson / shared-r NB2 fit.
    _shard_include("test_zip_zinb_stall.jl")
    _shard_include("test_zip_observed_laplace.jl")
    # #926: ZIP betaz-hat matches the exact-likelihood betaz (Wald centre), not the Fisher-Laplace one.
    _shard_include("test_zip_betaz_wald_926.jl")
    _shard_include("test_zinb_observed_laplace.jl")
    _shard_include("test_zerostep_verdict_sweep.jl")
    _shard_include("test_input_validation.jl")
    _shard_include("test_robustness_sweep.jl")
    _shard_include("test_derived_ci_sweep.jl")
    _shard_include("test_tweedie_speed.jl")
    _shard_include("test_nb_grouped_postfit.jl")
    _shard_include("test_convergence_553_554.jl")
    _shard_include("test_eta_clamp_869.jl")
    _shard_include("test_eta_clamp_rescue_874.jl")
    _shard_include("test_response_support_730.jl")
    _shard_include("test_response_support_730_va_ordinal.jl")
    _shard_include("test_bootstrap_decisions.jl")
    _shard_include("test_clamp_and_replicate_retention_docs.jl")
    _shard_include("test_truncpois_boot_782.jl")
    _shard_include("test_derived_decisions.jl")
    _shard_include("test_n_lt_p.jl")
    _shard_include("test_wtier_crosscov.jl")
    _shard_include("test_spde_projector_validation.jl")
    # #794 / #764: Wald routes on lambda_constraint fits (gllvmTMB's pdHess gate; lognormal refuses pins).
    _shard_include("test_lambda_constraint_wald_794.jl")
    _shard_include("test_random_slope_internals.jl")
    # Unit tests for the branch-RE statistics / tree helpers (welch_t, rank_sum_z, clade_edges, ...).
    _shard_include("test_branch_re_helpers.jl")
    _shard_include("test_response_support_730_siblings.jl")
    _shard_include("test_interrupt_rethrow.jl")
    # NaN or Inf at an observed Gaussian response cell is an ArgumentError (K == p gave loglik = NaN).
    _shard_include("test_gaussian_nonfinite_y.jl")
    # Algebraic invariants of the residual cov/cor, rotation, cutpoint and repeatability extractors.
    _shard_include("test_extractors_invariants.jl")
    _shard_include("test_phylo_sparse_solvers.jl")
    # #750: getLV / predict / residuals / simulate docstring signatures list mask and offset.
    _shard_include("test_docstring_postfit_mask_offset_750.jl")
    _shard_include("test_docstring_confint_lv_effects.jl")
    # #1021: no `](@ref)` survives Markdown parsing in a docstring or docs/src page (dead `./@ref` link).
    _shard_include("test_docstring_ref_linebreaks.jl")
    # #763: an explicit mask that marks a `missing` cell observed throws an ArgumentError.
    _shard_include("test_mask_missing_763.jl")
    _shard_include("test_changelog_structure.jl")
    # #276: fixture index (comparator_targets.toml) parses, every row has its keys, every path exists. Index only, no comparator claim.
    _shard_include("test_comparator_targets.jl")

    # Previously unwired, green on Julia 1.10.12 and 1.13.3 (ci-speed-report-20261109):
    # core070 covariance registries (self-contained modules), A4/S4 receipts,
    # kernel-latent unique bridge, #770 predictive_check binomial trials, #1022 NB mite regressions.
    # Gated at the end of this list (see _shard_include_if): test_sparse_phy_grad
    # (912 s, opt-in) and test_destination_b_a4_s4_fixed_coordinate_evaluator (Julia 1.10.x only).
    # Still unwired on purpose: test_phylo_ordinal_xlv and
    # test_phylo_gamma_xlv (optimiser-sensitive canaries), test_binomial_case_contract
    # (stale SHA pins).
    _shard_include("test_core070_covariance_registry.jl")
    _shard_include("test_core070_covariance_formula_registry.jl")
    _shard_include("test_destination_b_a4_s4_receipts.jl")
    _shard_include("test_kernel_latent_unique_bridge.jl")
    _shard_include("test_predictive_check_binomial.jl")
    _shard_include("test_nb_mite_multistart_1022.jl")
    _shard_include("test_nb_n_init_mite_1022.jl")
    # bootstrap_Sigma_r driver (pinned-R behaviour target; no R-output comparison).
    _shard_include("test_bootstrap_sigma_r.jl")
    # Gated (see `_shard_include_if`); keep last so no other position shifts.
    _shard_include_if(_RUN_SPARSE_PHY_GRAD, "test_sparse_phy_grad.jl")
    _shard_include_if(_IS_JULIA_110, "test_destination_b_a4_s4_fixed_coordinate_evaluator.jl")
end
# README examples: shard 1 only. The blocks include Poisson fits and a Gaussian
# CI trio; they must not run on every CI shard.
if _SHARD === nothing || _SHARD[1] == 1
    include("test_readme_quickstart.jl")
end
