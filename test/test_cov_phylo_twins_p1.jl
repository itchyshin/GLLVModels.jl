# gllvm-parity-tag: P1
#
# Fit-level twins for three core070 covariance rows against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9). Each row's P1 batch case is an R-only formula-grammar
# check (parse_multi_formula(desugar_brms_sugar(f))$covstructs, docs/dev-log/core070/true-parity-latest/
# covariance-batch-contract-p1.json) with no fit number. Here each is one real Gaussian fit in R and in
# Julia on one shared fixture, compared on the maximised logLik, the fitted parameters, and Julia's
# objective evaluated at R's own optimum (a fixed-parameter likelihood check that does not depend on
# either optimiser):
#
#   covariance/COV-PHYLO-DEP            R: phylo_dep(0 + trait | species, tree = tree), which gllvmTMB
#                                       rewrites to phylo_rr(d = n_traits, .dep = TRUE), the
#                                       phylo_latent(d = T) engine path (full unstructured
#                                       Sigma_phy (x) A). Julia: fit_phylo_latent_gllvm(...; d = T).
#   covariance/COV-PHYLO-A-ALIAS        R: phylo_latent(species, A = A), rewritten to phylo_rr(vcv = A);
#                                       R's own vcv = A spelling gives the identical objective (asserted
#                                       in the generator). Julia: fit_phylo_latent_gllvm(...; A = A), and
#                                       Julia's vcv = A spelling must give the identical fit.
#   covariance/COV-PHYLO-FOLDED-UNIQUE  R: phylo_latent(species, unique = TRUE, tree = tree): phylo_rr
#                                       (d = 1) plus the folded per-trait unique companion, the phylo_diag
#                                       block of src/gllvmTMB.cpp (g_phy_diag[, t] ~ N(0, A) with the same
#                                       Ainv_phy_rr, scaled by exp(log_sd_phy_diag[t])), so the model is
#                                       (Lambda Lambda' + diag(sd^2)) (x) A + sigma_eps^2 I.
#                                       Julia: fit_phylo_latent_gllvm(...; unique = true), mode
#                                       :explicitunique, the same covariance with unique variance
#                                       exp(2 * log_sd). Same parameterisation: the fixed-parameter check
#                                       maps R's (b_fix, log_sigma_eps, theta_rr_phy, log_sd_phy_diag)
#                                       onto Julia's coordinates one to one.
#
# No R at test time: R's values are read from test/fixtures/cov_phylo_twins_p1.toml (generated once
# by test/fixtures/gen_cov_phylo_twins_p1.R against a lane-local gllvmTMB install at the pin; the
# file records R version and commit). Every R fit converged (nlminb code 0) with a positive-definite
# Hessian (asserted there). 120-tip coalescent tree, 4 traits, 2 observations per species.
#
# Tolerances, not loosened to pass: logLik 1e-6 absolute (as every P1 twin); Julia's objective at
# R's parameters vs R's objective 1e-8 absolute (the A14/A15 cross-objective bar,
# test_phylo_latent_paired_p1.jl); intercepts, sigma_eps and sd_phy_diag 1e-4; Sigma_phy 1e-3 per
# element (sibling twins' Lambda Lambda' bar). The sign of a loading column is not identified, so
# loadings are compared through Sigma_phy.
#
# Stationarity of the DEP fit (recorded, not hidden): from its default start Julia's LBFGS stops at
# max |FD gradient| 4.0e-5, above its absolute default g_tol = 1e-5, so that fit reports
# converged = false. The ForwardDiff gradient of the same likelihood (a dense AD-capable copy that
# matches the sparse objective to 8e-12) agrees, 4.0e-5 on the log residual-SD coordinate, so the
# flag is not finite-difference noise; the stop is at the objective's rounding floor (curvature
# about 1.6e3 there, Newton decrement 2.6e-12, so no descent step is resolvable). The DEP twin is
# therefore the default fit followed by one warm restart from its parameters shifted by -0.01
# (Julia-only information, no change to the fitter): the restart reports converged = true at
# max |FD gradient| 8.6e-7 (max |AD gradient| 8.9e-7) with an objective within 3e-12 of the
# default stop. R's own nlminb optimum has max |AD gradient| 6.1e-4. The A-ALIAS and
# FOLDED-UNIQUE fits report converged = true from their default starts.
using Test
using GLLVModels
using LinearAlgebra
using TOML

const _CPT_TOML = joinpath(@__DIR__, "fixtures", "cov_phylo_twins_p1.toml")

_cpt_mat(v, p) = permutedims(reshape(Float64.(v), p, p))

# Response (traits x observations), observation species, tip labels, and A rebuilt from the Newick
# string in tip-label order (the fixture records A's sum, norm and first row, not A itself).
function _cpt_data(fx)
    T = Int(fx["n_traits"])
    tips = String.(fx["tip_labels"])
    sp = String.(fx["observation_species"])
    Y = permutedims(reshape(Float64.(fx["Y"]), length(sp), T))
    phy = augmented_phy(fx["newick"]; correlation = true)
    C = GLLVModels.sigma_phy_dense(phy)
    o = [findfirst(==(t), phy.leaf_names) for t in tips]
    return (T = T, tips = tips, sp = sp, Y = Y, A = Matrix{Float64}(C[o, o]))
end

# The three Julia twins, one call each.
# DEP: default fit, then one warm restart from its parameters shifted by -0.01 (see the header).
_cpt_fit_dep(d) = fit_phylo_latent_gllvm(d.Y, d.sp; species_levels = d.tips, d = d.T, tree = d.newick,
    start = fit_phylo_latent_gllvm(d.Y, d.sp; species_levels = d.tips, d = d.T, tree = d.newick).parameters .- 0.01)
_cpt_fit_alias(d) = fit_phylo_latent_gllvm(d.Y, d.sp; species_levels = d.tips, d = 1, A = d.A, tip_labels = d.tips)
_cpt_fit_vcv(d) = fit_phylo_latent_gllvm(d.Y, d.sp; species_levels = d.tips, d = 1, vcv = d.A, tip_labels = d.tips)
_cpt_fit_unique(d) = fit_phylo_latent_gllvm(d.Y, d.sp; species_levels = d.tips, d = 1, unique = true, tree = d.newick)

# Julia's negative log-likelihood objective of `fit`'s model at a packed parameter vector.
_cpt_objective(fit) = θ -> GLLVModels._precision_multivariate_nll(fit.response, fit.phy, θ;
    rank = fit.rank, mode = fit.mode, residual_mode = fit.residual_mode,
    species_id = fit.species_id, mean_design = fit.mean_design)

# R's optimum (named opt$par) mapped onto Julia's [beta; theta_rr; log_sd_unique?; log_sd_eps].
# Both engines pack the loadings with the gllvmTMB.cpp layout (src/packing.jl), so the map is a
# reordering by name.
function _cpt_r_theta(blk)
    names = String.(blk["par_names"]); par = Float64.(blk["par"])
    pick(n) = par[names .== n]
    return vcat(pick("b_fix"), pick("theta_rr_phy"), pick("log_sd_phy_diag"), pick("log_sigma_eps"))
end

# Julia objective at R's optimum minus R's own objective there.
_cpt_cross(fit, blk) = _cpt_objective(fit)(_cpt_r_theta(blk)) - Float64(blk["objective"])

_cpt_sigma_phy(fit) = fit.phylo_unique_variance === nothing ? fit.loading * fit.loading' :
    fit.loading * fit.loading' + Diagonal(fit.phylo_unique_variance)

@testset "phylo_dep / phylo_latent(A =) / phylo_latent(unique = TRUE) fits: gllvmTMB P1 (9539352f6)" begin
    if !isfile(_CPT_TOML)
        @warn "cov-phylo P1 fixture absent; twin gate NOT RUN" _CPT_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_CPT_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        @test fx["converged"] && fx["pd_hessian"]
        d0 = _cpt_data(fx)
        d = merge(d0, (newick = fx["newick"],))
        T = d.T
        @test size(d.Y) == (T, Int(fx["n_species"]) * Int(fx["replicates"]))
        # A rebuilt from the tree is the matrix R fitted (sum, Frobenius norm, first row).
        @test isapprox(sum(d.A), Float64(fx["A_sum"]); rtol = 1e-10)
        @test isapprox(norm(d.A), Float64(fx["A_frobenius"]); rtol = 1e-10)
        @test maximum(abs.(d.A[1, :] .- Float64.(fx["A_first_row"]))) <= 1e-12

        @testset "COV-PHYLO-DEP: phylo_dep(0 + trait | species) = phylo_latent(d = T)" begin
            r = fx["dep"]
            fit = _cpt_fit_dep(d)
            @test fit.rank == T && fit.mode === :barelowrank
            @test fit.converged && fit.hessian_positive_definite   # after the warm restart; see the header
            @test fit.gradient_norm <= 1e-5                         # measured 8.6e-7
            @test abs(fit.loglik - Float64(r["loglik"])) <= 1e-6
            @test abs(_cpt_cross(fit, r)) <= 1e-8
            @test maximum(abs.(fit.beta .- Float64.(r["beta"]))) <= 1e-4
            @test abs(sqrt(only(unique(fit.residual_variance))) - Float64(r["sigma_eps"])) <= 1e-4
            @test maximum(abs.(_cpt_sigma_phy(fit) .- _cpt_mat(r["Sigma_rr"], T))) <= 1e-3
        end

        @testset "COV-PHYLO-A-ALIAS: phylo_latent(species, A = A) = the vcv = A model" begin
            r = fx["alias"]
            @test Float64(r["objective_vcv_spelling"]) == Float64(r["objective"])   # R: alias, same objective
            fit = _cpt_fit_alias(d)
            fitv = _cpt_fit_vcv(d)
            @test fit.converged
            @test fit.hessian_positive_definite
            @test fit.parameters == fitv.parameters && fit.loglik == fitv.loglik   # Julia: alias, same fit
            @test abs(fit.loglik - Float64(r["loglik"])) <= 1e-6
            @test abs(_cpt_cross(fit, r)) <= 1e-8
            @test maximum(abs.(fit.beta .- Float64.(r["beta"]))) <= 1e-4
            @test abs(sqrt(only(unique(fit.residual_variance))) - Float64(r["sigma_eps"])) <= 1e-4
            @test maximum(abs.(_cpt_sigma_phy(fit) .- _cpt_mat(r["Sigma_rr"], T))) <= 1e-3
            # Both A spellings are refused together, as in R.
            @test_throws ArgumentError fit_phylo_latent_gllvm(d.Y, d.sp; species_levels = d.tips,
                A = d.A, vcv = d.A, tip_labels = d.tips)
        end

        @testset "COV-PHYLO-FOLDED-UNIQUE: phylo_latent(species, unique = TRUE)" begin
            r = fx["unique"]
            fit = _cpt_fit_unique(d)
            @test fit.mode === :explicitunique && fit.rank == 1
            @test fit.converged
            @test fit.hessian_positive_definite
            @test abs(fit.loglik - Float64(r["loglik"])) <= 1e-6
            @test abs(_cpt_cross(fit, r)) <= 1e-8
            @test maximum(abs.(fit.beta .- Float64.(r["beta"]))) <= 1e-4
            @test abs(sqrt(only(unique(fit.residual_variance))) - Float64(r["sigma_eps"])) <= 1e-4
            @test maximum(abs.(sqrt.(fit.phylo_unique_variance) .- Float64.(r["sd_phy_diag"]))) <= 1e-4
            @test maximum(abs.(_cpt_sigma_phy(fit) .- _cpt_mat(r["Sigma_total"], T))) <= 1e-3
            # The unique companion is not inert: the fit beats phylo_latent(d = 1) alone on the same tree.
            bare = fit_phylo_latent_gllvm(d.Y, d.sp; species_levels = d.tips, d = 1, tree = d.newick)
            @test fit.loglik - bare.loglik > 1
        end
    end
end
