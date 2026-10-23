# gllvm-parity-tag: P1
#
# Numeric twins of gllvmTMB's ordinary latent() covariance term at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9): covariance rows COV-ORD-LATENT-BARE, -DEFAULT and
# -COMMON. Their R cases at P1 (CORE070-COV-ORD-LATENT-*-FORMULA, tools/core070_covariance_batch.R)
# parse these three formulas only; here each one is fitted. No R at test time: R's recorded values
# are read from test/fixtures/cov_ord_latent_p1.toml (generated once by
# test/fixtures/gen_cov_ord_latent_p1.R against a lane-local gllvmTMB install at the pin; it records
# R version, commit and the sha256 of the data CSV). Every R fit converged (nlminb code 0) with a
# positive-definite Hessian (asserted in the generator).
#
# Data: n = 60 sites, p = 4 traits, one Gaussian observation per trait and site, unit = "site".
#   bare     latent(0 + trait | site, unique = FALSE)  Sigma_B = L L',            sigma_eps free
#   default  latent(0 + trait | site)                  Sigma_B = L L' + diag(psi), sigma_eps fixed
#   common   latent(0 + trait | site, common = TRUE)   Sigma_B = L L' + psi I_p,   sigma_eps fixed
# with L a rank-1 loading column and vec(Y) ~ N(trait means, I_n (x) Sigma_B + sigma_eps^2 I).
# With a site-level unique diagonal gllvmTMB fixes sigma_eps at max(0.001 sd(y), 1e-6) (recorded).
# <-> fit_gaussian_sources(Y; sources = [SourceCovariance(I_n; groups = 1:n, mode = :latent,
#     rank = 1, unique, common)], sigma_eps_fixed = R's fixed value for default and common).
# Julia's `common = true` on a latent-unique block ties the unique diagonal to one log SD, as R's
# common = TRUE ties theta_diag_B; the model identity is checked directly: Julia's objective at R's
# optimum and at a perturbed off-optimum point (R's -obj$fn there) both reproduce R's value.
#
# Compared per fit: log-likelihood, trait intercepts, the site-level trait covariance Sigma_B
# (invariant to the loading sign), sigma_eps (bare only; fixed elsewhere), the free-parameter
# count, and the Julia objective at R's optimum and at R's probe point.
#
# Tolerances: about 7-10x the largest observed difference over the three fits (R stops at max
# |gradient| 3e-5 to 5e-5, Julia at g_tol 1e-8, so Sigma_B and sigma_eps carry R's optimizer
# stopping error); the objective-at-a-fixed-point checks use a roundoff floor of 2e-11 (about
# 300 eps |logLik|; observed <= 5.2e-13).
#
# Limits, disclosed: Gaussian only; rank 1 only; one observation per site and trait; the Julia side
# is the native source fitter with an identity site covariance, not formula sugar for latent().
using Test
using GLLVModels
using LinearAlgebra
using TOML
using SHA

const _COL_DIR = joinpath(@__DIR__, "fixtures")

# R's write.csv long data (site, trait, value) -> p x n_site response matrix.
function _col_load_data(path::AbstractString, trait_names::Vector{String}, n_site::Integer)
    Y = fill(NaN, length(trait_names), n_site)
    open(path) do io
        readline(io) == "\"site\",\"trait\",\"value\"" || error("unexpected header in $path")
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, parse(Int, parts[1])] = parse(Float64, parts[3])
        end
    end
    return Y
end

# R's row-by-row TOML matrix -> Matrix{Float64}
_col_matrix(rows) = reduce(vcat, [permutedims(Float64.(r)) for r in rows])

# R parameter vector (par_names order) -> Julia's (means, rr loadings, unique log SDs, residual
# log SD when free). Only the bare fit reorders: R puts log_sigma_eps before theta_rr_B.
function _col_julia_point(r_par::Vector{Float64}, names::Vector{String})
    pick(nm) = r_par[names .== nm]
    return vcat(pick("b_fix"), pick("theta_rr_B"), pick("theta_diag_B"), pick("log_sigma_eps"))
end

const _COL_CASES = (("bare", false, false), ("default", true, false), ("common", true, true))

@testset "ordinary latent() twins: gllvmTMB P1 (9539352f6)" begin
    fxp = joinpath(_COL_DIR, "cov_ord_latent_p1.toml")
    if !isfile(fxp)
        @warn "cov_ord_latent P1 fixture absent; twin gate NOT RUN" fxp
        @test_skip false
    else
        fx = TOML.parsefile(fxp)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        dp = joinpath(_COL_DIR, fx["data_file"])
        @test bytes2hex(sha256(read(dp))) == fx["data_sha256"]
        p, n = Int(fx["p"]), Int(fx["n_site"])
        Y = _col_load_data(dp, String.(fx["trait_names"]), n)
        @test all(isfinite, Y)

        for (key, uniq, common) in _COL_CASES
            @testset "$key" begin
                s = fx[key]
                @test s["converged"] && s["pd_hessian"]
                fixed = s["sigma_eps_fixed"]
                @test fixed == uniq                                # R fixes sigma_eps exactly when a site diag is present
                fixed && @test isapprox(s["sigma_eps"], fx["sigma_eps_rule"]; rtol = 1e-12)   # exp(log(rule)) in TMB
                sigma_fixed = fixed ? Float64(s["sigma_eps"]) : nothing
                source = SourceCovariance(Matrix(1.0I, n, n); groups = 1:n, name = :site, mode = :latent, rank = 1, unique = uniq, common = common)
                fit = fit_gaussian_sources(Y; sources = [source], sigma_eps_fixed = sigma_fixed, g_tol = 1e-8, iterations = 2000)
                @test fit.converged && fit.hessian_positive_definite
                @test GLLVModels.dof(fit) == s["r_df"]               # same free-parameter count as R's logLik df

                r_beta = Float64.(s["beta"])
                r_Sigma = _col_matrix(s["trait_covariance"])
                U = only(fit.trait_covariances)

                @test isapprox(fit.loglik, Float64(s["loglik"]); atol = 3e-10, rtol = 0)       # logLik (observed <= 2.6e-11)
                @test isapprox(fit.beta, r_beta; atol = 3e-10, rtol = 0)                       # trait intercepts (observed <= 3.8e-11)
                @test isapprox(U, r_Sigma; atol = 5e-6, rtol = 0)                              # Sigma_B (observed <= 6.4e-7)
                @test isapprox(fit.sigma_eps, Float64(s["sigma_eps"]); atol = 5e-7, rtol = 0)  # sigma_eps (observed 6.5e-8 bare; fixed elsewhere)

                # Julia's objective at R's optimum and at R's perturbed probe point
                names = String.(s["par_names"])
                r_point = vcat(r_beta, Float64.(s["lambda"]), Float64.(s["diag_logsd"]), fixed ? Float64[] : [log(Float64(s["sigma_eps"]))])
                nll_at_r = GLLVModels._gaussian_sources_nll(Y, [source], r_point; sigma_eps_fixed = sigma_fixed)
                @test isapprox(-nll_at_r, Float64(s["loglik"]); atol = 2e-11, rtol = 0)        # objective at R's optimum (observed <= 5.2e-13)
                probe = _col_julia_point(Float64.(s["probe_par"]), names)
                nll_at_probe = GLLVModels._gaussian_sources_nll(Y, [source], probe; sigma_eps_fixed = sigma_fixed)
                @test isapprox(-nll_at_probe, Float64(s["probe_loglik"]); atol = 2e-11, rtol = 0)  # objective at R's probe point
            end
        end
    end
end
