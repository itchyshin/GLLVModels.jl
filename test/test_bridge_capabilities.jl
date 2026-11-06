using Test
using GLLVModels

@testset "bridge capabilities ledger" begin
    caps = bridge_capabilities()

    @test propertynames(caps) == (
        :family,
        :fit_no_x,
        :fixed_effect_X,
        :predictor_informed_lv,
        :missing_response,
        :cbind_binomial,
        :ci_no_x_wald,
        :ci_no_x_profile,
        :ci_no_x_bootstrap,
        :ci_mask_wald,
        :ci_mask_profile,
        :ci_mask_bootstrap,
        :ci_x_wald,
        :ci_x_profile,
        :ci_x_bootstrap,
        :postfit_coef,
        :postfit_fit_stats,
        :postfit_summary,
        :postfit_predict,
        :postfit_residuals,
        :postfit_simulate,
        :postfit_ordination,
        :status,
        :notes,
    )

    @test caps.family == [
        "gaussian",
        "poisson",
        "lognormal",
        "binomial",
        "binomial_probit",
        "binomial_cloglog",
        "negbinomial",
        "nb1",
        "beta",
        "gamma",
        "betabinomial",
        "ordinal",
        "ordinal_probit",
        "zip",
        "zinb",
        "zib",
        "truncated_poisson",
        "mixed-family vector",
    ]
    @test caps.family[caps.fit_no_x] == caps.family
    @test caps.family[caps.fixed_effect_X] == [
        "gaussian",
        "poisson",
        "binomial",
        "negbinomial",
        "nb1",
        "beta",
        "gamma",
        "betabinomial",
        "ordinal",
        "ordinal_probit",
        "zip",
        "zinb",
    ]
    @test caps.family[caps.predictor_informed_lv] == [
        "gaussian",
        "poisson",
        "binomial",
        "binomial_probit",
        "binomial_cloglog",
        "negbinomial",
        "beta",
        "gamma",
    ]
    for fam in caps.family[caps.predictor_informed_lv]
        idx = findfirst(==(fam), caps.family)
        @test idx !== nothing
        @test occursin("predictor-informed latent-score X_lv", caps.notes[idx])
        @test occursin("X_lv Wald B_lv CI payloads are routed", caps.notes[idx])
        @test occursin("profile/bootstrap X_lv CIs", caps.notes[idx])
        @test occursin("remain follow-ups", caps.notes[idx])
    end
    @test all(!occursin("non-Gaussian non-binomial X_lv remain follow-ups", note)
              for note in caps.notes)
    @test all(!occursin("broader non-Gaussian X_lv routes remain separate", note)
              for note in caps.notes)
    @test caps.family[caps.missing_response] == [
        "gaussian",
        "poisson",
        "binomial",
        "binomial_probit",
        "binomial_cloglog",
        "negbinomial",
        "nb1",
        "beta",
        "gamma",
        "betabinomial",
        "ordinal",
        "ordinal_probit",
    ]
    @test caps.family[caps.cbind_binomial] == [
        "binomial",
        "binomial_probit",
        "binomial_cloglog",
        "betabinomial",
    ]
    ci_routed = [
        "gaussian",
        "poisson",
        "binomial",
        "binomial_probit",
        "binomial_cloglog",
        "negbinomial",
        "nb1",
        "beta",
        "gamma",
        "betabinomial",
        "zip",
        "zinb",
        "zib",
    ]
    @test caps.family[caps.ci_no_x_wald] == ci_routed
    @test caps.family[caps.ci_no_x_profile] == ci_routed
    @test caps.family[caps.ci_no_x_bootstrap] == ci_routed
    mask_ci_routed = [
        "poisson",
        "binomial",
        "binomial_probit",
        "binomial_cloglog",
        "negbinomial",
        "nb1",
        "beta",
        "gamma",
        "betabinomial",
    ]
    @test caps.family[caps.ci_mask_wald] == mask_ci_routed
    @test caps.family[caps.ci_mask_profile] == mask_ci_routed
    @test caps.family[caps.ci_mask_bootstrap] == mask_ci_routed
    x_ci_routed = [
        "gaussian",
        "poisson",
        "binomial",
        "negbinomial",
        "nb1",
        "beta",
        "gamma",
        "betabinomial",
        "zip",
        "zinb",
    ]
    @test caps.family[caps.ci_x_wald] == x_ci_routed
    @test caps.family[caps.ci_x_profile] == x_ci_routed
    @test caps.family[caps.ci_x_bootstrap] == x_ci_routed
    @test caps.family[caps.postfit_coef] == caps.family
    @test caps.family[caps.postfit_fit_stats] == caps.family
    @test caps.family[caps.postfit_summary] == caps.family
    # predict() now covers EVERY family: ordinal/ordinal_probit predict via the
    # cutpoints payload (per-category probabilities / modal class), so postfit_predict
    # is the full family list.
    @test caps.family[caps.postfit_predict] == caps.family
    # Scalar-mean post-fit (residuals = y - mu) still EXCLUDES the ordinal families,
    # which have no scalar response mean on the payload.
    scalar_mean_postfit = [
        "gaussian",
        "poisson",
        "binomial",
        "binomial_probit",
        "binomial_cloglog",
        "negbinomial",
        "nb1",
        "beta",
        "gamma",
        "zip",
        "zinb",
        "zib",
        "mixed-family vector",
    ]
    @test caps.family[caps.postfit_residuals] == scalar_mean_postfit
    # `simulate` is NARROWER than `residuals`: no simulate method exists for any of
    # the three zero-inflated fit types (ZIPFit / ZINBFit / ZIBFit), so those rows
    # report false rather than inheriting the residuals column.
    @test caps.family[caps.postfit_simulate] == [
        "gaussian",
        "poisson",
        "binomial",
        "binomial_probit",
        "binomial_cloglog",
        "negbinomial",
        "nb1",
        "beta",
        "gamma",
        "mixed-family vector",
    ]
    @test caps.family[caps.postfit_ordination] == caps.family
    @test all(==("partial"), caps.status)
    mixed_idx = findfirst(==("mixed-family vector"), caps.family)
    @test mixed_idx !== nothing
    @test caps.fit_no_x[mixed_idx]
    @test !caps.fixed_effect_X[mixed_idx]
    @test !caps.predictor_informed_lv[mixed_idx]
    @test !caps.missing_response[mixed_idx]
    @test !caps.ci_no_x_wald[mixed_idx]
    @test !caps.ci_no_x_profile[mixed_idx]
    @test !caps.ci_no_x_bootstrap[mixed_idx]
    @test !caps.ci_mask_wald[mixed_idx]
    @test !caps.ci_mask_profile[mixed_idx]
    @test !caps.ci_mask_bootstrap[mixed_idx]
    @test !caps.ci_x_wald[mixed_idx]
    @test !caps.ci_x_profile[mixed_idx]
    @test !caps.ci_x_bootstrap[mixed_idx]
    @test caps.postfit_predict[mixed_idx]
    @test caps.postfit_residuals[mixed_idx]
    @test caps.postfit_simulate[mixed_idx]
    @test occursin("no X", caps.notes[mixed_idx])
    @test occursin("CI", caps.notes[mixed_idx])

    # betabinomial: one-part + X (grouped_cov, twin API B), missing-response
    # masks, and Wald/profile/bootstrap CI (FD Hessian) are wired; still NO
    # scalar-mean postfit (residuals/simulate) extractor — pin that honestly.
    bb_idx = findfirst(==("betabinomial"), caps.family)
    @test bb_idx !== nothing
    @test caps.fit_no_x[bb_idx]
    @test caps.fixed_effect_X[bb_idx]
    @test !caps.predictor_informed_lv[bb_idx]
    @test caps.missing_response[bb_idx]
    @test caps.cbind_binomial[bb_idx]
    @test caps.ci_no_x_wald[bb_idx]
    @test caps.ci_no_x_profile[bb_idx]
    @test caps.ci_no_x_bootstrap[bb_idx]
    @test caps.ci_mask_wald[bb_idx]
    @test caps.ci_mask_profile[bb_idx]
    @test caps.ci_mask_bootstrap[bb_idx]
    @test caps.ci_x_wald[bb_idx]
    @test caps.ci_x_profile[bb_idx]
    @test caps.ci_x_bootstrap[bb_idx]
    @test caps.postfit_predict[bb_idx]
    @test !caps.postfit_residuals[bb_idx]
    @test !caps.postfit_simulate[bb_idx]
    @test caps.postfit_ordination[bb_idx]
    @test occursin("Wald/profile/bootstrap CI payloads are routed", caps.notes[bb_idx])
    @test occursin("finite-difference Hessian", caps.notes[bb_idx])
    @test occursin("residuals/simulate are not wired", caps.notes[bb_idx])

    # lognormal: no-X only (twin fid 3). CI / X / X_lv / masks / scalar-mean
    # postfit remain follow-ups — LognormalFit is not in `_CIFit` and has no
    # residuals/simulate extractor. Light no-X logLik Δ is wired via parity cell.
    ln_idx = findfirst(==("lognormal"), caps.family)
    @test ln_idx !== nothing
    @test ln_idx == findfirst(==("poisson"), caps.family) + 1
    @test caps.fit_no_x[ln_idx]
    @test !caps.fixed_effect_X[ln_idx]
    @test !caps.predictor_informed_lv[ln_idx]
    @test !caps.missing_response[ln_idx]
    @test !caps.cbind_binomial[ln_idx]
    @test !caps.ci_no_x_wald[ln_idx]
    @test !caps.ci_no_x_profile[ln_idx]
    @test !caps.ci_no_x_bootstrap[ln_idx]
    @test !caps.ci_mask_wald[ln_idx]
    @test !caps.ci_x_wald[ln_idx]
    @test caps.postfit_predict[ln_idx]
    @test !caps.postfit_residuals[ln_idx]
    @test !caps.postfit_simulate[ln_idx]
    @test caps.postfit_ordination[ln_idx]
    @test occursin("twin fid 3", caps.notes[ln_idx])
    @test occursin("fit_lognormal_gllvm", caps.notes[ln_idx])
    @test occursin("test/parity/test_lognormal_parity.jl", caps.notes[ln_idx])
    @test occursin("light no-X RCall logLik Δ PAID", caps.notes[ln_idx])
    @test !occursin("still OWED", caps.notes[ln_idx])
    @test occursin("narrower than full R-user parity", caps.notes[ln_idx])

    # truncated_poisson: no-X only (twin fid 10). CI / X / X_lv / masks /
    # scalar-mean postfit remain follow-ups — TruncatedPoissonFit is not in
    # `_CIFit` and has no residuals/simulate extractor. Light no-X logLik Δ wired.
    tp_idx = findfirst(==("truncated_poisson"), caps.family)
    @test tp_idx !== nothing
    @test tp_idx == findfirst(==("zib"), caps.family) + 1
    @test caps.fit_no_x[tp_idx]
    @test !caps.fixed_effect_X[tp_idx]
    @test !caps.predictor_informed_lv[tp_idx]
    @test !caps.missing_response[tp_idx]
    @test !caps.cbind_binomial[tp_idx]
    @test !caps.ci_no_x_wald[tp_idx]
    @test !caps.ci_no_x_profile[tp_idx]
    @test !caps.ci_no_x_bootstrap[tp_idx]
    @test !caps.ci_mask_wald[tp_idx]
    @test !caps.ci_x_wald[tp_idx]
    @test caps.postfit_predict[tp_idx]
    @test !caps.postfit_residuals[tp_idx]
    @test !caps.postfit_simulate[tp_idx]
    @test caps.postfit_ordination[tp_idx]
    @test occursin("twin fid 10", caps.notes[tp_idx])
    @test occursin("fit_truncated_poisson_gllvm", caps.notes[tp_idx])
    @test occursin("test/parity/test_truncated_poisson_parity.jl", caps.notes[tp_idx])
    @test occursin("light no-X RCall logLik Δ PAID", caps.notes[tp_idx])
    @test !occursin("still OWED", caps.notes[tp_idx])
    @test occursin("narrower than full R-user parity", caps.notes[tp_idx])

    # zib: no-X only. `cbind_binomial` stays FALSE (ZIB's N is one shared scalar,
    # not the per-observation cbind contract), masks and X are unwired, and no-X CI
    # routes all three methods through _family_ci(::ZIBFit).
    zib_idx = findfirst(==("zib"), caps.family)
    @test zib_idx !== nothing
    @test caps.fit_no_x[zib_idx]
    @test !caps.fixed_effect_X[zib_idx]
    @test !caps.predictor_informed_lv[zib_idx]
    @test !caps.missing_response[zib_idx]
    @test !caps.cbind_binomial[zib_idx]
    @test caps.ci_no_x_wald[zib_idx]
    @test caps.ci_no_x_profile[zib_idx]
    @test caps.ci_no_x_bootstrap[zib_idx]
    @test !caps.ci_mask_wald[zib_idx]
    @test !caps.ci_x_wald[zib_idx]
    @test !caps.ci_x_profile[zib_idx]
    @test !caps.ci_x_bootstrap[zib_idx]
    @test caps.postfit_predict[zib_idx]
    @test caps.postfit_residuals[zib_idx]
    @test !caps.postfit_simulate[zib_idx]
    @test caps.postfit_ordination[zib_idx]

    # The postfit_simulate narrowing covers all three zero-inflated rows at once,
    # not just the new one — none of them has a simulate method.
    for fam in ("zip", "zinb", "zib")
        idx = findfirst(==(fam), caps.family)
        @test caps.postfit_residuals[idx]
        @test !caps.postfit_simulate[idx]
    end

    grouped = Set(["negbinomial", "nb1", "beta", "gamma"])
    pertrait_ordinal = Set(["ordinal", "ordinal_probit"])
    for (fam, note) in zip(caps.family[1:(end - 1)], caps.notes[1:(end - 1)])
        if fam in grouped
            @test occursin("grouped dispersion", note)
            @test occursin("Wald/profile/bootstrap CI payloads are routed", note)
        elseif fam in pertrait_ordinal
            @test occursin("per-trait ordinal cutpoints", note)
            @test occursin("fixed-effect-X", note)
            @test occursin("CI routing is a follow-up", note)
        elseif fam == "zip"
            @test occursin("Julia-forward", note)
            @test occursin("twin-asymmetric", note)
            @test occursin("Wald/profile/bootstrap CI under X", note)
            @test occursin("finite-difference Hessian", note)
            @test occursin("no twin light RCall Δ", note)
            @test occursin("narrower than full R-user parity", note)
            zip_idx = findfirst(==("zip"), caps.family)
            @test caps.ci_x_wald[zip_idx]
            @test caps.ci_x_profile[zip_idx]
            @test caps.ci_x_bootstrap[zip_idx]
            @test caps.ci_no_x_wald[zip_idx]
        elseif fam == "zinb"
            @test occursin("Julia-forward", note)
            @test occursin("twin-asymmetric", note)
            @test occursin("shared scalar r", note)
            @test occursin("Wald/profile/bootstrap CI under X", note)
            @test occursin("finite-difference Hessian", note)
            @test occursin("no twin light RCall Δ", note)
            @test occursin("narrower than full R-user parity", note)
            zinb_idx = findfirst(==("zinb"), caps.family)
            @test caps.ci_x_wald[zinb_idx]
            @test caps.ci_x_profile[zinb_idx]
            @test caps.ci_x_bootstrap[zinb_idx]
            @test caps.ci_no_x_wald[zinb_idx]
            @test caps.fixed_effect_X[zinb_idx]
            @test caps.fit_no_x[zinb_idx]
        elseif fam == "zib"
            @test occursin("Julia-forward", note)
            @test occursin("twin-asymmetric", note)
            @test occursin("shared scalar trials count N", note)
            @test occursin("not per-observation cbind", note)
            @test occursin("no twin light RCall Δ", note)
            @test occursin("the twin gllvmTMB has no ZIB", note)
            @test occursin("narrower than full R-user parity", note)
            # The X arm is a separate arc: no fixed-effect-X, no CI under X, no masks.
            @test !occursin("Wald/profile/bootstrap CI under X", note)
        elseif fam == "lognormal"
            @test occursin("twin fid 3", note)
            @test occursin("fit_lognormal_gllvm", note)
            @test occursin("test/parity/test_lognormal_parity.jl", note)
            @test occursin("light no-X RCall logLik Δ PAID", note)
            @test !occursin("still OWED", note)
            @test occursin("narrower than full R-user parity", note)
            @test !occursin("Wald/profile/bootstrap CI payloads are routed", note)
        elseif fam == "truncated_poisson"
            @test occursin("twin fid 10", note)
            @test occursin("fit_truncated_poisson_gllvm", note)
            @test occursin("test/parity/test_truncated_poisson_parity.jl", note)
            @test occursin("light no-X RCall logLik Δ PAID", note)
            @test !occursin("still OWED", note)
            @test occursin("narrower than full R-user parity", note)
            @test !occursin("Wald/profile/bootstrap CI payloads are routed", note)
        else
            @test occursin("narrower than full R-user parity", note)
        end
        fam == "gaussian" && @test occursin("predictor-informed latent-score", note)
    end
    @test occursin("mixed-family", caps.notes[end])
    @test occursin("no X", caps.notes[end])
end
