# gllvm-parity-tag: P1
#
# Numeric twins of gllvmTMB's extract_Gamma() and extract_coevolution_modules() at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9): namespace rows export/extract_Gamma and
# export/extract_coevolution_modules. No R at test time: R's recorded values are read from
# test/fixtures/cross_lineage_p1.toml (generated once by test/fixtures/gen_cross_lineage_p1.R
# against a lane-local gllvmTMB install at the pin; it records R version, commit and the sha256 of
# both CSVs). The R fit converged (nlminb code 0) with a positive-definite Hessian (asserted).
#
# Model: value ~ 0 + trait + kernel_latent(species, K = K, d = 3, name = "cross", unique = FALSE),
# Gaussian, unit = "site"; K = make_cross_kernel(A_H, A_P, W, rho = 0.6) over 8 host and 8 partner
# species, 4 sites per species, traits h_size, h_defence, p_size, p_attack (complete data):
#     vec(Y) ~ N(trait means, (P K_eff P') (x) Lambda Lambda' + sigma_eps^2 I),
# P the site -> species incidence and K_eff = K + 1e-8 I (gllvmTMB's jitter, measured in the
# generator). <-> fit_kernel_latent_gllvm(Y, K_eff, groups, 3; name = :cross), the native source
# fitter with one latent rank-3 source on K_eff (same model; log-likelihoods compared).
#
# Estimands, in R's orientation: both R functions slice extract_Sigma(fit, "cross", part =
# "shared"), the 4 x 4 TRAIT covariance Lambda Lambda', by trait name. The Julia methods on a
# GaussianSourcesFit slice the source's fitted trait covariance the same way (rows = row_traits,
# columns = col_traits, in the order given). The older extract_Gamma(::GllvmFit) method, which is
# positional over the stacked species of the Hadamard fit, is a different estimand and is not used.
#
# Compared: extract_Gamma for the host x partner block and for a reordered call that mixes the
# lineages (2 x 3); extract_coevolution_modules R matrix, singular values, squared shares, module and
# trait labels, and both axis tables (each pair of axes up to one joint sign, which neither engine
# identifies; the sign agrees here but is not relied on), plus the n_modules = 1 call. With d = 3 and
# two traits per lineage the first singular value is 1 by construction and the second (0.297) is the
# informative one.
#
# Tolerances: about 7-12x the observed difference (R stops at max |gradient| 4.4e-5, Julia at
# g_tol 1e-8); the log-likelihood uses about 10x the observed 1.3e-11.
#
# Limits, disclosed: Gaussian only; scale = "effect" is not twinned (R multiplies by the rho it
# records on the fitted kernel; a GaussianSourcesFit does not record rho, and the Julia method
# refuses :effect); complete data (R's block-NA layout, where each lineage measures only its own
# traits, needs missing responses, which fit_gaussian_sources does not accept); the Julia side is the
# native source fitter, not formula sugar for kernel_latent().
using Test
using GLLVModels
using LinearAlgebra
using TOML
using SHA

const _CL_DIR = joinpath(@__DIR__, "fixtures")

# R's write.csv long data (site, trait, species, value) -> p x n_site response matrix and the
# species index (into `species`) of each site.
function _cl_load_data(path::AbstractString, trait_names::Vector{String}, species::Vector{String},
                       n_site::Integer)
    Y = fill(NaN, length(trait_names), n_site)
    grp = zeros(Int, n_site)
    open(path) do io
        readline(io) == "\"site\",\"trait\",\"species\",\"value\"" || error("unexpected header in $path")
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            site = parse(Int, parts[1])
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            s = findfirst(==(strip(parts[3], '"')), species)
            (t === nothing || s === nothing) && error("unrecognised trait or species in $path")
            Y[t, site] = parse(Float64, parts[4])
            grp[site] = s
        end
    end
    return Y, grp
end

# R's write.csv of the named n x n kernel (header = species names, in order).
function _cl_load_K(path::AbstractString, species::Vector{String})
    lines = filter(!isempty, readlines(path))
    String.(strip.(split(lines[1], ","), '"')) == species || error("unexpected header in $path")
    length(lines) == length(species) + 1 || error("unexpected row count in $path")
    return reduce(vcat, [permutedims(parse.(Float64, split(l, ","))) for l in lines[2:end]])
end

@testset "cross-lineage kernel twins: gllvmTMB P1 (9539352f6)" begin
    clf = joinpath(_CL_DIR, "cross_lineage_p1.toml")
    if !isfile(clf)
        @warn "cross-lineage P1 fixture absent; twin gate NOT RUN" clf
        @test_skip false
    else
        fx = TOML.parsefile(clf)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        f = fx["fit"]
        dp = joinpath(_CL_DIR, f["data_file"])
        kp = joinpath(_CL_DIR, f["K_file"])
        @test bytes2hex(sha256(read(dp))) == f["data_sha256"]
        @test bytes2hex(sha256(read(kp))) == f["K_sha256"]
        @test f["converged"] && f["pd_hessian"]

        tr = String.(f["trait_names"])
        sp = String.(f["species"])
        p = length(tr)
        Y, grp = _cl_load_data(dp, tr, sp, Int(f["n_site"]))
        K = _cl_load_K(kp, sp)
        @test all(isfinite, Y) && sort(unique(grp)) == collect(1:length(sp))
        @test issymmetric(K) && all(==(1.0), diag(K))
        @test maximum(abs, K[1:Int(f["n_host"]), (Int(f["n_host"]) + 1):end]) > 0.1   # a real host-partner bridge
        K_eff = K + Float64(f["K_jitter"]) * I

        fit = fit_kernel_latent_gllvm(Y, K_eff, grp, Int(f["d"]); name = :cross, g_tol = 1e-8,
                                      iterations = 2000)
        @test fit.converged && fit.hessian_positive_definite
        @test isapprox(fit.loglik, Float64(f["loglik"]); atol = 2e-10, rtol = 0)   # same optimum (observed 1.3e-11)
        @test isapprox(fit.beta, Float64.(f["beta"]); atol = 2e-5, rtol = 0)       # trait intercepts (observed 2.1e-6)
        @test isapprox(fit.sigma_eps, Float64(f["sigma_eps"]); atol = 4e-7, rtol = 0)   # residual SD (observed 4.4e-8)
        r_sig = reshape(Float64.(f["Sigma_shared"]), p, p)
        @test isapprox(fit.trait_covariances[1], r_sig; atol = 3e-6, rtol = 0)     # shared Lambda Lambda' (observed 3.0e-7)

        host, partner = tr[1:2], tr[3:4]
        @testset "extract_Gamma" begin
            g = fx["gamma"]
            @test String.(g["row_traits"]) == host && String.(g["col_traits"]) == partner
            r_gam = reshape(Float64.(g["Gamma"]), 2, 2)
            @test length(unique(round.(abs.(r_gam); digits = 3))) == 4                 # four distinct R entries
            Gam = extract_Gamma(fit; level = :cross, row_traits = host, col_traits = partner,
                                trait_names = tr)
            @test size(Gam) == (2, 2)
            @test isapprox(Gam, r_gam; atol = 2e-6, rtol = 0)   # host x partner Gamma (observed 2.2e-7)
            # positional selection and Symbol names give the same block
            @test extract_Gamma(fit; level = "cross", row_traits = [1, 2], col_traits = [3, 4]) == Gam
            @test extract_Gamma(fit; level = :cross, row_traits = Symbol.(host),
                                col_traits = Symbol.(partner), trait_names = tr) == Gam
            # orientation: swapping the lineages transposes the block
            @test extract_Gamma(fit; level = :cross, row_traits = partner, col_traits = host,
                                trait_names = tr) == permutedims(Gam)
            rp, cp = String.(g["row_perm"]), String.(g["col_perm"])
            r_perm = reshape(Float64.(g["Gamma_perm"]), length(rp), length(cp))
            Gp = extract_Gamma(fit; level = :cross, row_traits = rp, col_traits = cp, trait_names = tr)
            @test size(Gp) == (2, 3)
            @test isapprox(Gp, r_perm; atol = 2e-6, rtol = 0)   # reordered, lineage-mixing Gamma (observed 2.4e-7)
            @test_throws ArgumentError extract_Gamma(fit; level = :cross, row_traits = host,
                                                     col_traits = partner, trait_names = tr, scale = :effect)
            @test_throws ArgumentError extract_Gamma(fit; level = :cross, row_traits = ["h_size", "h_size"],
                                                     col_traits = partner, trait_names = tr)
            @test_throws ArgumentError extract_Gamma(fit; level = :cross, row_traits = ["nope"],
                                                     col_traits = partner, trait_names = tr)
            @test_throws ArgumentError extract_Gamma(fit; level = :cross, row_traits = host,
                                                     col_traits = partner)   # names need trait_names
            @test_throws ArgumentError extract_Gamma(fit; level = :other, row_traits = [1], col_traits = [3])
        end

        @testset "extract_coevolution_modules" begin
            m = fx["modules"]
            mo = extract_coevolution_modules(fit; level = :cross, row_traits = host,
                                             col_traits = partner, trait_names = tr)
            r_R = reshape(Float64.(m["R"]), 2, 2)
            r_sv = Float64.(m["singular_value"])
            @test 0.05 < r_sv[2] < 0.95                                                 # informative second axis
            @test isapprox(mo.R, r_R; atol = 1e-6, rtol = 0)                            # standardised R (observed 9.5e-8)
            @test mo.modules.module == String.(m["module"])
            @test mo.modules.component == fill("cross", 2)
            @test isapprox(mo.modules.singular_value, r_sv; atol = 3e-7, rtol = 0)      # singular values (observed 3.5e-8)
            @test isapprox(mo.modules.squared_share, Float64.(m["squared_share"]); atol = 2e-7, rtol = 0)   # squared shares (observed 1.7e-8)
            @test mo.row_axes.trait == String.(m["row_axes_trait"])
            @test mo.row_axes.module == String.(m["row_axes_module"])
            @test mo.col_axes.trait == String.(m["col_axes_trait"])
            @test mo.col_axes.module == String.(m["col_axes_module"])
            @test mo.row_axes.side == fill("row", 4) && mo.col_axes.side == fill("column", 4)
            # each pair of axes is identified up to one joint sign
            U_r = reshape(Float64.(m["row_axes_loading"]), 2, 2)
            V_r = reshape(Float64.(m["col_axes_loading"]), 2, 2)
            U_j = reshape(mo.row_axes.loading, 2, 2)
            V_j = reshape(mo.col_axes.loading, 2, 2)
            sgn = [sign(dot(U_j[:, k], U_r[:, k])) for k in 1:2]
            @test isapprox(U_j .* sgn', U_r; atol = 1e-6, rtol = 0)                     # row axes (observed 9.5e-8)
            @test isapprox(V_j .* sgn', V_r; atol = 1.5e-7, rtol = 0)                   # column axes, same signs (observed 1.4e-8)
            # R = U diag(d) V'
            @test U_j * Diagonal(mo.modules.singular_value) * V_j' ≈ mo.R atol = 1e-12

            mo1 = extract_coevolution_modules(fit; level = :cross, row_traits = host,
                                              col_traits = partner, trait_names = tr, n_modules = 1)
            @test mo1.R == mo.R && length(mo1.modules.module) == 1 && length(mo1.row_axes.loading) == 2
            @test isapprox(mo1.modules.singular_value, Float64.(m["n1_singular_value"]); atol = 3e-7, rtol = 0)   # n_modules = 1: the structural 1 (observed 1e-16; tolerance of the singular values)
            @test isapprox(mo1.modules.squared_share, Float64.(m["n1_squared_share"]); atol = 2e-7, rtol = 0)   # (observed 1.7e-8)
            @test isapprox(mo1.row_axes.loading .* sgn[1], Float64.(m["n1_row_axes_loading"]); atol = 1e-6, rtol = 0)     # (observed 9.5e-8)
            @test isapprox(mo1.col_axes.loading .* sgn[1], Float64.(m["n1_col_axes_loading"]); atol = 1.5e-7, rtol = 0)   # (observed 1.4e-8)
            @test_throws ArgumentError extract_coevolution_modules(fit; level = :cross, row_traits = host,
                                                                   col_traits = partner, trait_names = tr,
                                                                   n_modules = 0)
            @test_throws ArgumentError extract_coevolution_modules(fit; level = :cross, row_traits = host,
                                                                   col_traits = partner, trait_names = tr,
                                                                   scale = "effect")
        end
    end
end
