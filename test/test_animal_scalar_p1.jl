# gllvm-parity-tag: P1
#
# Numeric twin of gllvmTMB's animal_scalar() at P1 (9539352f66f2db2cc26b1c393e67212a359b60c9):
# namespace row export/animal_scalar. No R at test time: R's recorded values are read from
# test/fixtures/animal_scalar_p1.toml (generated once by test/fixtures/gen_animal_scalar_p1.R
# against a lane-local gllvmTMB install at the pin; it records R version, commit and the sha256 of
# both CSVs). The R fit converged (nlminb code 0) with a positive-definite Hessian (asserted).
#
# Model: value ~ 0 + trait + animal_scalar(id, A = A), Gaussian, unit = "site", cluster = "id";
# a 60-individual pedigree (A = pedigree_to_A(ped)), 2 sites per individual, p = 3 traits:
#     vec(Y) ~ N(trait means, sigma2_a (P A_eff P') (x) I_p + sigma_eps^2 I),
# P the site -> individual incidence and A_eff = A + 1e-8 I (gllvmTMB's jitter, measured in the
# generator). Each trait carries an independent draw with one shared variance sigma2_a; R routes the
# keyword to phylo(id, vcv = A) and fits it as loglambda_phy (log variance) + log_sigma_eps.
# <-> fit_gaussian_sources(Y; sources = [SourceCovariance(A_eff; groups, mode = :indep,
#     common = true)]): the same covariance, one common log SD for the source and one residual SD.
# Compared: log-likelihood, trait intercepts, sigma2_a, sigma_eps, and the Julia objective
# evaluated at R's estimates against R's log-likelihood. There are no loadings in this model.
#
# Tolerances: about 7-8x the observed difference for the intercepts, sigma2_a and sigma_eps (R stops
# at max |gradient| 2.1e-6, Julia at g_tol 1e-8); the two log-likelihood checks use a roundoff floor
# of about 190 eps |logLik| (2e-11), since 10x the observed 3e-13 would sit at a few ulps.
#
# Limits, disclosed: Gaussian only; the dense A = route only (R's pedigree = route is checked in the
# generator to reach the same log-likelihood up to the jitter, 2e-7, and is not compared here);
# the Julia side is the native source fitter, not formula sugar for animal_scalar().
using Test
using GLLVModels
using LinearAlgebra
using TOML
using SHA

const _AS_DIR = joinpath(@__DIR__, "fixtures")

# R's write.csv long data (site, trait, id, value) -> p x n_site response matrix and the
# individual index (1-based) of each site.
function _as_load_data(path::AbstractString, trait_names::Vector{String}, n_site::Integer)
    p = length(trait_names)
    Y = fill(NaN, p, n_site)
    ind = zeros(Int, n_site)
    open(path) do io
        readline(io) == "\"site\",\"trait\",\"id\",\"value\"" || error("unexpected header in $path")
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            site = parse(Int, parts[1])
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, site] = parse(Float64, parts[4])
            ind[site] = parse(Int, parts[3])
        end
    end
    return Y, ind
end

# R's write.csv of the unnamed n x n relatedness matrix (header V1..Vn).
function _as_load_A(path::AbstractString, n::Integer)
    lines = filter(!isempty, readlines(path))
    length(lines) == n + 1 || error("unexpected row count in $path")
    return reduce(vcat, [permutedims(parse.(Float64, split(l, ","))) for l in lines[2:end]])
end

@testset "animal_scalar twin: gllvmTMB P1 (9539352f6)" begin
    asf = joinpath(_AS_DIR, "animal_scalar_p1.toml")
    if !isfile(asf)
        @warn "animal_scalar P1 fixture absent; twin gate NOT RUN" asf
        @test_skip false
    else
        fx = TOML.parsefile(asf)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        s = fx["scalar"]
        dp = joinpath(_AS_DIR, s["data_file"])
        ap = joinpath(_AS_DIR, s["A_file"])
        @test bytes2hex(sha256(read(dp))) == s["data_sha256"]
        @test bytes2hex(sha256(read(ap))) == s["A_sha256"]
        @test s["converged"] && s["pd_hessian"]

        p, n_id, n_site = Int(s["p"]), Int(s["n_id"]), Int(s["n_site"])
        Y, ind = _as_load_data(dp, String.(s["trait_names"]), n_site)
        A = _as_load_A(ap, n_id)
        @test all(isfinite, Y) && sort(unique(ind)) == collect(1:n_id)
        @test issymmetric(A) && all(>=(1.0), diag(A)) && count(>(0.0), A - Diagonal(A)) > 0   # related individuals
        A_eff = A + Float64(s["A_jitter"]) * I

        source = SourceCovariance(A_eff; groups = ind, name = :animal, mode = :indep, common = true)
        fit = fit_gaussian_sources(Y; sources = [source], g_tol = 1e-8, iterations = 2000)
        @test fit.converged && fit.hessian_positive_definite
        @test GLLVModels.dof(fit) == p + 2   # p intercepts, one shared variance, one residual SD

        r_beta = Float64.(s["beta"])
        r_s2a = Float64(s["sigma2_a"])
        r_se = Float64(s["sigma_eps"])
        @test length(unique(round.(r_beta; digits = 3))) == p                  # R intercepts are distinct
        U = only(fit.trait_covariances)
        @test U == Diagonal(fill(U[1, 1], p))                                  # sigma2_a I_p: one shared variance

        @test isapprox(fit.loglik, Float64(s["loglik"]); atol = 2e-11, rtol = 0)   # logLik (observed 2.8e-13; floor ~190 eps |logLik|)
        @test isapprox(fit.beta, r_beta; atol = 2e-7, rtol = 0)                    # trait intercepts (observed 2.9e-8)
        @test isapprox(U[1, 1], r_s2a; atol = 1e-8, rtol = 0)                      # sigma2_a (observed 1.2e-9)
        @test isapprox(fit.sigma_eps, r_se; atol = 3e-8, rtol = 0)                 # sigma_eps (observed 3.6e-9)
        # Julia's objective at R's estimates (R loglambda_phy is a log variance; Julia uses a log SD)
        r_point = vcat(r_beta, log(sqrt(r_s2a)), log(r_se))
        nll_at_r = GLLVModels._gaussian_sources_nll(Y, [source], r_point)
        @test isapprox(-nll_at_r, Float64(s["loglik"]); atol = 2e-11, rtol = 0)    # objective at R's point (observed 4.0e-13; floor ~190 eps |logLik|)
    end
end
