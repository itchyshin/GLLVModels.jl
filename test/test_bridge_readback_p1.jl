# gllvm-parity-tag: P1
#
# Bridge readback at P1 (gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9): namespace rows
# S3method(coef|fitted|logLik|predict|residuals|summary, gllvmTMB_julia) and export(gllvm_julia_fit).
# No R and no JuliaCall at test time. test/fixtures/bridge_readback_p1.toml was recorded once by
# test/fixtures/gen_bridge_readback_p1.R in a live run: R (gllvmTMB at the pin) -> JuliaCall ->
# this package, on the namespace twin data ns_gauss_p1_data.csv (sha256 checked), model
#     value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE),  Gaussian, p = 6, n = 200.
# The TOML holds, from that one R session:
#   bridge        the outputs of the R methods on the object gllvmTMB(..., engine = "julia") returned;
#   julia_direct  the same quantities from the native GLLVModels accessors, computed in the same
#                 JuliaCall session on a fit made with the call bridge_fit makes for this row
#                 (alpha = row means of Y, fit_gaussian_gllvm(Y .- alpha; K = 2); src/bridge.jl);
#   gllvm_julia_fit, tmb   the export called directly, and the native R fit (engine = "tmb").
#
# What this checks. (1) The R methods return what Julia computes: each bridge output against its
# julia_direct twin. These pass through R-side arithmetic (fitted = alpha + rotated loadings x rotated
# scores', residuals = y - fitted, Pearson / sigma_eps, AIC / BIC from the payload) and marshalling,
# so a transposition, rotation or scale slip would show at O(1); the tolerance 1e-12 is a roundoff
# floor (observed <= 1.4e-15). (2) The export against the native R engine: log-likelihood, df,
# intercepts, the latent covariance Lambda Lambda' (rotation invariant) and sigma_eps, each side at its
# own optimum (log-likelihood and loadings-scale tolerances as the sibling namespace twins,
# test_namespace_numeric_p1_twin.jl; intercepts and sigma_eps about 3-10x the observed difference).
# (3) Drift guard: refitting here reproduces the recorded julia_direct values (loose, since the
# optimiser path may differ across Julia versions and platforms).
#
# Limits, disclosed: one Gaussian fixture, no-X, complete response; predict has no newdata route on
# the bridge (gated in R), so predict is in-sample; julia_direct is a refit with the identical call,
# because the bridge returns a payload, not the Julia fit object; StatsAPI.coef on the centred fit is
# empty, so coef is compared to the intercepts and getLoadings(fit; rotate = true) the bridge returns.
using Test
using GLLVModels
using LinearAlgebra
using Statistics
using TOML
using SHA

const _BR_DIR = joinpath(@__DIR__, "fixtures")
const _BR_TOML = joinpath(_BR_DIR, "bridge_readback_p1.toml")

_br_vec(x) = x isa AbstractDict ? Float64.(collect(x["colmajor"])) :
             x isa AbstractVector ? Float64.(collect(x)) : [Float64(x)]
_br_mat(x) = reshape(_br_vec(x), Int(x["nrow"]), Int(x["ncol"]))
_br_maxdiff(a, b) = (va = _br_vec(a); vb = _br_vec(b); @assert length(va) == length(vb); maximum(abs.(va .- vb)))

@testset "bridge readback P1 (gllvmTMB_julia methods, gllvm_julia_fit)" begin
    fx = TOML.parsefile(_BR_TOML)
    @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
    @test bytes2hex(sha256(read(joinpath(_BR_DIR, fx["data_file"])))) == fx["data_sha256"]
    b, j, g, t = fx["bridge"], fx["julia_direct"], fx["gllvm_julia_fit"], fx["tmb"]
    @test b["converged"] && j["converged"] && g["converged"]
    @test t["convergence"] == 0 && t["pd_hessian"]
    Y = _br_mat(fx["y"])
    @test size(Y) == (6, 200)

    @testset "R method output == direct Julia accessor (same session)" begin
        tol = 1e-12   # roundoff floor; observed <= 1.4e-15
        pairs = [
            ("coef_alpha", "alpha"), ("coef_loadings", "loadings"),                 # coef
            ("fitted_response", "fitted_response"), ("fitted_link", "fitted_link"), # fitted
            ("logLik", "loglik"), ("logLik_df", "df"), ("logLik_nobs", "nobs"),     # logLik
            ("predict_link_est", "fitted_link"), ("predict_response_est", "fitted_response"), # predict
            ("residuals_response", "resid_response"), ("residuals_pearson", "resid_pearson"), # residuals
            ("summary_logLik", "loglik"), ("summary_AIC", "aic"), ("summary_BIC", "bic"),       # summary
            ("summary_df", "df"), ("summary_nobs", "nobs"),
            ("summary_coef_alpha", "alpha"), ("summary_coef_loadings", "loadings"),
            ("summary_Sigma", "Sigma"), ("summary_correlation", "correlation"),
            ("summary_communality", "communality"),
        ]
        for (rb, jd) in pairs
            @test _br_maxdiff(b[rb], j[jd]) <= tol   # BRIDGE-READBACK-TOL
        end
    end

    @testset "gllvm_julia_fit export vs the formula route and vs engine = \"tmb\"" begin
        @test _br_maxdiff(g["loglik"], b["logLik"]) <= 1e-12        # GJF-FORMULA-ROUTE-TOL
        @test _br_maxdiff(g["loglik"], t["logLik"]) <= 1e-6         # GJF-LOGLIK-TOL (observed 9.7e-9; as test_namespace_numeric_p1_twin.jl:94)
        @test Int(g["df"]) == Int(t["df"])                          # GJF-DF
        @test _br_maxdiff(g["alpha"], t["beta"]) <= 3e-12           # GJF-INTERCEPT-TOL (observed 3.0e-13; both at the sample mean)
        L = _br_mat(g["loadings"])
        @test maximum(abs.(vec(L * L') .- _br_vec(t["Sigma_latent"]))) <= 5e-5  # GJF-LATENT-SIGMA-TOL (observed 9.2e-6; as the loadings, test_namespace_numeric_p1_twin.jl:97)
        @test abs(Float64(g["sigma_eps"]) - Float64(t["sigma_eps"])) <= 1e-6    # GJF-SIGMA-EPS-TOL (observed 3.0e-7)
    end

    @testset "drift guard: a refit here reproduces julia_direct" begin
        alpha = vec(mean(Y; dims = 2))
        Yc = Y .- alpha
        f = fit_gaussian_gllvm(Yc; K = 2)
        @test f.converged
        @test isapprox(loglikelihood(f), Float64(j["loglik"]); atol = 1e-6, rtol = 0)
        mu = alpha .+ predict(f, Yc; type = :response)
        @test isapprox(vec(mu), _br_vec(j["fitted_response"]); atol = 1e-5, rtol = 0)
        @test isapprox(vec(Matrix(sigma_y_site(f))), _br_vec(j["Sigma"]); atol = 1e-5, rtol = 0)
    end
end
