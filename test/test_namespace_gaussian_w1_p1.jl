# gllvm-parity-tag: P1
#
# Numeric twins of three gllvmTMB namespace rows at P1 (9539352f66f2db2cc26b1c393e67212a359b60c9),
# all on ONE shared Gaussian dataset, test/fixtures/ns_gauss_p1_data.csv (p = 6 traits, n = 200
# units; sha256 checked): export/gllvmTMB_wide, S3method/ordiplot,gllvmTMB_multi and
# export/flag_unreliable_loadings. No R at test time: R's recorded values are read from
# test/fixtures/ns_gauss_w1_p1.toml (generated once by test/fixtures/gen_namespace_gaussian_w1_p1.R
# against a lane-local gllvmTMB install at the pin). Every R fit converged (nlminb code 0, tight
# tolerances) with a positive-definite Hessian (asserted).
#
#   wide    : gllvmTMB_wide(Y, d = 2). The wrapper's latent() keeps its default unique = TRUE, so R
#             fits Sigma = Lambda Lambda' + diag(sd_B^2) + sigma_eps^2 I with sigma_eps mapped off at
#             its data-derived start (recorded). <-> fit_gaussian_pervar_gllvm(Y; K = 2,
#             fixed_residual_sd = R's sigma_eps), the same covariance with the same fixed residual.
#             Compared: log-likelihood, trait intercepts, Lambda Lambda' (rotation-invariant), sd_B,
#             and the Julia log-likelihood at R's estimates.
#   ordiplot: ordiplot(fit) on latent(d = 2, unique = FALSE) <-> ordiplot(fit_gllvm(Y; family =
#             Normal(), K = 2), Y). R returns list(scores, loadings) unrotated; Julia returns
#             principal-rotated sites/species by default. Both are compared through the
#             rotation-invariant products scores * loadings' (n x p) and loadings * loadings'.
#   flag    : flag_unreliable_loadings(fit) on a confirmatory fit (lambda_constraint pins
#             Lambda[1,2] = 0 and Lambda[2,1] = 0) <-> flag_unreliable_loadings(fit_gaussian_gllvm(Yc;
#             K = 2, lambda_constraint = M), Yc). Julia's confirmatory fit is zero-mean, so it is
#             fitted to Y centred by trait means. With complete balanced Gaussian data the trait-mean
#             MLE is the sample mean and the observed information is block diagonal between the means
#             and the covariance parameters at the optimum, so the log-likelihood and the loading
#             standard errors are those of R's fit with intercepts. Compared: the loading intervals
#             (estimate, se, lower, upper), the flags at the default null region (-0.1, 0.1) and at
#             (-0.5, 0.5) as exact integer codes (1 TRUE, 0 FALSE, -1 NA for pinned), and the
#             log-likelihood.
#
# Tolerances: about 10x the observed differences. The flag row is looser because Julia's
# confirmatory refit stops at g_tol 1e-4 (loadings within ~7e-5 of R); the closest non-pinned
# interval bound sits 0.012 from a null-region edge, 25x the bound tolerance, so the flags are
# not near a tie.
#
# Limits, disclosed: Gaussian only; the wide wrapper is twinned on its default call (no X, weights,
# phylo_vcv or formula_extra); ordiplot is compared on its returned data, not its drawing;
# flag_unreliable_loadings is compared on the raw Wald route only (R's default), not "wald_asym",
# "profile" or the standardized scale.
using Test
using GLLVModels
using Distributions: Normal
using LinearAlgebra
using TOML
using SHA

const _W1_DIR = joinpath(@__DIR__, "fixtures")

# R's write.csv long data (unit, trait, value) -> p x n response matrix.
function _w1_load(path::AbstractString, trait_names::Vector{String}, n::Integer)
    Y = fill(NaN, length(trait_names), n)
    open(path) do io
        readline(io) == "\"unit\",\"trait\",\"value\"" || error("unexpected header in $path")
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, parse(Int, strip(parts[1], '"'))] = parse(Float64, parts[3])
        end
    end
    return Y
end

_w1_mat(v, nrow, ncol) = permutedims(reshape(Float64.(v), ncol, nrow))   # row-major vector -> matrix
_w1_code(x) = x === missing ? -1 : Int(x)                                  # flag -> integer code
_w1_rcode(s) = s == "NA" ? -1 : s == "TRUE" ? 1 : s == "FALSE" ? 0 : error("bad flag $s")

@testset "namespace Gaussian twins (wide, ordiplot, flag): gllvmTMB P1 (9539352f6)" begin
    fxp = joinpath(_W1_DIR, "ns_gauss_w1_p1.toml")
    if !isfile(fxp)
        @warn "namespace Gaussian W1 P1 fixture absent; twin gate NOT RUN" fxp
        @test_skip false
    else
        fx = TOML.parsefile(fxp)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        dp = joinpath(_W1_DIR, fx["data_file"])
        @test bytes2hex(sha256(read(dp))) == fx["data_sha256"]
        p, n = Int(fx["p"]), Int(fx["n"])
        Y = _w1_load(dp, String.(fx["trait_names"]), n)
        @test all(isfinite, Y)

        @testset "gllvmTMB_wide" begin
            w = fx["wide"]
            @test w["converged"] && w["pd_hessian"]
            r_L = _w1_mat(w["Lambda"], p, 2)
            r_beta, r_sdB, c = Float64.(w["beta"]), Float64.(w["sd_B"]), Float64(w["sigma_eps_fixed"])
            fit = fit_gaussian_pervar_gllvm(Y; K = 2, fixed_residual_sd = c)
            @test fit.converged
            @test fit.fixed_residual_sd == c
            @test isapprox(fit.loglik, Float64(w["loglik"]); atol = 5e-10, rtol = 0)        # logLik (observed 5.1e-11)
            @test isapprox(fit.β, r_beta; atol = 5e-7, rtol = 0)                            # trait intercepts (observed 5.5e-8)
            @test isapprox(fit.Λ * fit.Λ', r_L * r_L'; atol = 5e-6, rtol = 0)              # Lambda Lambda' (observed 5.1e-7)
            @test isapprox(sqrt.(fit.ψ²), r_sdB; atol = 3e-6, rtol = 0)                     # unique SDs sd_B (observed 3.1e-7)
            X = zeros(p, n, p)
            for t in 1:p, s in 1:n
                X[t, s, t] = 1.0
            end
            ll_at_r = gaussian_pervar_marginal_loglik(Y, r_L, r_sdB .^ 2 .+ c^2; X = X, β = r_beta)
            @test isapprox(ll_at_r, Float64(w["loglik"]); atol = 6e-11, rtol = 0)          # objective at R's point (observed 3.0e-12; floor ~190 eps |logLik|)
        end

        @testset "ordiplot" begin
            o = fx["ordiplot"]
            @test o["converged"] && o["pd_hessian"]
            r_S = _w1_mat(o["scores"], n, 2)
            r_L = _w1_mat(o["loadings"], p, 2)
            fit = fit_gllvm(Y; family = Normal(), K = 2)
            @test fit.converged
            od = ordiplot(fit, Y)
            @test size(od.sites) == (n, 2) && size(od.species) == (p, 2)
            @test isapprox(fit.logLik, Float64(o["loglik"]); atol = 1.5e-9, rtol = 0)                # logLik (observed 1.6e-10)
            @test isapprox(od.sites * od.species', r_S * r_L'; atol = 1e-5, rtol = 0)              # scores * loadings' (observed 9.9e-7)
            @test isapprox(od.species * od.species', r_L * r_L'; atol = 1e-5, rtol = 0)            # loadings * loadings' (observed 1.2e-6)
        end

        @testset "flag_unreliable_loadings" begin
            f = fx["flag"]
            @test f["converged"] && f["pd_hessian"]
            Yc = Y .- sum(Y; dims = 2) ./ n
            M = fill(NaN, p, 2)
            for pin in f["pins"]
                M[pin[1], pin[2]] = 0.0
            end
            fit = fit_gaussian_gllvm(Yc; K = 2, lambda_constraint = M)
            @test fit.converged
            rows = flag_unreliable_loadings(fit, Yc)
            rows_w = flag_unreliable_loadings(fit, Yc; null_region = (-0.5, 0.5))
            @test [(r.trait, r.axis) for r in rows] == [(t, k) for k in 1:2 for t in 1:p]
            @test all(r -> r.pd_hessian, rows)
            @test [r.pinned for r in rows] == Bool.(f["pinned"])
            est, se = [r.estimate for r in rows], [r.se for r in rows]
            lo, hi = [r.lower for r in rows], [r.upper for r in rows]
            @test isapprox(fit.logLik, Float64(f["loglik"]); atol = 1e-5, rtol = 0)               # logLik (observed 1.5e-6)
            @test isapprox(est, Float64.(f["estimate"]); atol = 5e-4, rtol = 0)                   # loading estimates (observed 7.0e-5)
            @test isapprox(se, Float64.(f["se"]); atol = 5e-5, rtol = 0)                          # raw Wald SEs (observed 7.6e-6)
            @test isapprox(lo, Float64.(f["lower"]); atol = 5e-4, rtol = 0)                       # lower bounds (observed 5.7e-5)
            @test isapprox(hi, Float64.(f["upper"]); atol = 5e-4, rtol = 0)                       # upper bounds (observed 8.5e-5)
            code_d = _w1_code.([r.unreliable for r in rows])
            code_w = _w1_code.([r.unreliable for r in rows_w])
            r_code_d = _w1_rcode.(f["unreliable_default"])
            r_code_w = _w1_rcode.(f["unreliable_wide"])
            @test maximum(abs.(code_d .- r_code_d)) < 0.5   # default-region flags, exact integer codes
            @test maximum(abs.(code_w .- r_code_w)) < 0.5   # (-0.5, 0.5)-region flags, exact integer codes
            @test length(unique(r_code_w)) == 3             # the wide region gives TRUE, FALSE and NA rows
            free = .!Bool.(f["pinned"])
            for (a, b) in ((-0.1, 0.1), (-0.5, 0.5))         # no bound within 10x its tolerance of an edge
                @test minimum(abs.(vcat(lo[free] .- a, lo[free] .- b, hi[free] .- a, hi[free] .- b))) > 5e-3
            end
            # The row-table method on the same intervals gives the same flags.
            @test _w1_code.([r.unreliable for r in flag_unreliable_loadings(rows_w; null_region = (-0.1, 0.1))]) == code_d
        end
    end
end

@testset "flag_unreliable_loadings: refusals" begin
    # Deterministic one-factor data (no RNG): loadings times a score, plus a non-collinear term.
    y = [0.9, 0.6, -0.5, 0.7] * [sin(1.3s) for s in 1:60]' .+ 0.5 .* [cos(0.7t * s + t) for t in 1:4, s in 1:60]
    y = y .- sum(y; dims = 2) ./ 60
    exploratory = fit_gaussian_gllvm(y; K = 1)
    @test_throws ArgumentError flag_unreliable_loadings(exploratory, y)        # no pins: rotation only
    M = fill(NaN, 4, 1); M[2, 1] = 0.0
    fit = fit_gaussian_gllvm(y; K = 1, lambda_constraint = M)
    @test_throws ArgumentError flag_unreliable_loadings(fit, y; null_region = (0.1, -0.1))
    @test_throws ArgumentError flag_unreliable_loadings(fit, y; null_region = (0.0,))
    @test_throws ArgumentError flag_unreliable_loadings(fit, y; level = :unit_obs)
    @test_throws ArgumentError flag_unreliable_loadings([(estimate = 0.2, lower = 0.1)])
    rows = flag_unreliable_loadings(fit, y)
    @test rows[2].pinned && rows[2].se == 0 && rows[2].unreliable === missing
    @test all(r -> r.null_region_lo == -0.1 && r.null_region_hi == 0.1, rows)
end
