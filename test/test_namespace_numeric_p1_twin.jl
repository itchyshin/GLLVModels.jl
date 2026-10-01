# gllvm-parity-tag: P1
#
# Numeric twins of six gllvmTMB namespace rows against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9): extract_loadings,
# extract_rotated_loadings_table, extract_lv_effects, extract_communality,
# extract_Sigma_B and extract_Sigma_W. No R at test time: R's recorded values are
# read from test/fixtures/ns_numeric_p1.toml (generated once by
# test/fixtures/gen_namespace_numeric_p1.R against a lane-local gllvmTMB install at
# the pin; the file records R version, commit, seeds and the data sha256s) and the
# same three datasets are fitted in Julia.
#
# Three fits, grouped so rows that share a model share a fixture:
#   main: Gaussian, p = 6, n = 200, rank 2 (R: latent(d = 2, unique = FALSE))
#         <-> fit_gllvm(Y; family = Normal(), K = 2).
#   lv  : the same plus predictor-informed latent scores
#         (R: latent(..., lv = ~ x1 + x2)) <-> fit_gllvm(Y; K = 2, X_lv = X).
#   two : two-level Gaussian, p = 5, 120 units x 4 observations, rank 1 per tier plus
#         diagonal per tier (R: latent + unique at unit and unit_obs)
#         <-> fit_twolevel_gaussian(y, individual; K_B = 1, K_W = 1).
# Each R fit converged with a positive-definite Hessian (asserted below). Julia fits
# are compared at each side's own optimum; the log-likelihood guard is atol 1e-6.
#
# Tolerances are set above the differences observed when the twin was built (shown in
# the comments) and are not loosened to pass: loadings, effects, communality and
# Sigma differ only by the two optimisers' convergence noise (~1e-5).
#
# vcov.gllvmTMB_multi is bound on the main fit: R's fixed-effect (trait-mean) block of
# the inverse joint Hessian against the beta[1:p] block of Julia's vcov(fit, Y). Both
# are on the natural scale of the trait means, in trait order t1..t6; the block is
# invariant to how each engine parameterises the variance terms (both are evaluated at
# the joint optimum), so no reparameterisation map is needed.
#
# NOT bound here (reproducer: test/fixtures/repro_namespace_twin_gaps_p1.jl): the
# single-level communality / proportions split under unique (the decomposition is not
# identified from one observation per cell, so the engines split it differently).
using Test
using GLLVModels
using Distributions: Normal
using TOML
using LinearAlgebra
using SHA

const _NS_DIR = joinpath(@__DIR__, "fixtures")
const _NS_TOML = joinpath(_NS_DIR, "ns_numeric_p1.toml")

# R's write.csv long data -> (p x n response matrix, n x q unit-level predictor matrix,
# unit index per column). `obs_col` = column holding the observation id (two-level data).
function _ns_load_csv(path::AbstractString, trait_names::Vector{String}, n_cols::Integer;
                      x_cols::Vector{Int} = Int[], obs_col::Union{Nothing,Int} = nothing)
    p = length(trait_names)
    Y = zeros(Float64, p, n_cols)
    X = zeros(Float64, n_cols, length(x_cols))
    ind = zeros(Int, n_cols)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            unit = parse(Int, strip(parts[1], '"'))
            col = obs_col === nothing ? unit : parse(Int, strip(parts[obs_col], '"'))
            t = findfirst(==(strip(parts[obs_col === nothing ? 2 : 3], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, col] = parse(Float64, parts[obs_col === nothing ? 3 : 4])
            ind[col] = unit
            for (j, c) in enumerate(x_cols)
                X[col, j] = parse(Float64, parts[c])
            end
        end
    end
    return Y, X, ind
end

_ns_mat(v, nrow, ncol) = permutedims(reshape(Float64.(v), ncol, nrow))   # row-major vector -> matrix

@testset "namespace numeric twins: gllvmTMB P1 (9539352f6)" begin
    if !isfile(_NS_TOML)
        @warn "namespace numeric P1 fixture absent; twin gate NOT RUN" _NS_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_NS_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        tn = String.(fx["trait_names"])
        p, n = Int(fx["p"]), Int(fx["n_unit"])

        @testset "main: extract_loadings, extract_rotated_loadings_table" begin
            m = fx["main"]
            dp = joinpath(_NS_DIR, m["data_file"])
            @test bytes2hex(sha256(read(dp))) == m["data_sha256"]
            @test m["converged"] && m["pd_hessian"]
            Y, _, _ = _ns_load_csv(dp, tn, n)
            fit = fit_gllvm(Y; family = Normal(), K = 2)
            @test fit.converged
            # guard: same optimum (observed 1e-8)
            @test isapprox(fit.logLik, Float64(m["loglik"]); atol = 1e-6, rtol = 0)
            # raw lower-triangular loadings, both engines' native convention (observed 9.8e-6)
            L = extract_loadings(fit; rotate = false)
            @test isapprox(L, _ns_mat(m["loadings"], p, 2); atol = 5e-5, rtol = 0)
            # varimax table (observed: loadings 6.5e-5 raw / 7.2e-5 standardized, axis variance 8e-5, share 1.4e-5)
            tr = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :raw)
            @test tr.axis == repeat([1, 2]; inner = p)
            @test isapprox(tr.loading, Float64.(m["rot_raw_loading"]); atol = 2e-4, rtol = 0)
            @test isapprox(tr.axis_variance[[1, p + 1]], Float64.(m["rot_raw_axis_variance"]); atol = 3e-4, rtol = 0)
            @test isapprox(tr.axis_share[[1, p + 1]], Float64.(m["rot_raw_axis_share"]); atol = 1e-4, rtol = 0)
            ts = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :standardized)
            @test isapprox(ts.loading, Float64.(m["rot_std_loading"]); atol = 2e-4, rtol = 0)
            # vcov.gllvmTMB_multi: full trait-mean covariance, off-diagonals included
            # (itchyshin/GLLVModels.jl#652 found Julia returning Diagonal(se^2)).
            Vr = _ns_mat(m["vcov"], p, p)
            V = vcov(fit, Y)
            @test V isa Matrix{Float64}
            @test confint(fit, Y).term[1:p] == ["beta[$j]" for j in 1:p]
            Vb = V[1:p, 1:p]
            offd(A) = A - Diagonal(diag(A))
            @test maximum(abs, offd(Vr)) > 1e-3                       # R's block is not diagonal
            @test isapprox(diag(Vb), diag(Vr); atol = 1e-6, rtol = 0)   # observed 4.4e-8
            @test isapprox(offd(Vb), offd(Vr); atol = 1e-6, rtol = 0)   # observed 4.6e-8 (1.2e-5 relative)
            @test issymmetric(V)
        end

        @testset "lv: extract_lv_effects" begin
            l = fx["lv"]
            dp = joinpath(_NS_DIR, l["data_file"])
            @test bytes2hex(sha256(read(dp))) == l["data_sha256"]
            @test l["converged"] && l["pd_hessian"]
            @test String.(l["predictors"]) == ["x1", "x2"]
            Y, X, _ = _ns_load_csv(dp, tn, n; x_cols = [4, 5])
            fit = fit_gllvm(Y; family = Normal(), K = 2, X_lv = X)
            @test fit.converged
            @test isapprox(fit.logLik, Float64(l["loglik"]); atol = 1e-6, rtol = 0)   # observed 1.8e-8
            # trait-scale B_lv = Lambda alpha' (rotation-stable; observed 3.0e-6)
            Bj = extract_lv_effects(fit; type = :trait_effect)
            @test isapprox(Bj, _ns_mat(l["trait_effect"], p, 2); atol = 2e-5, rtol = 0)
            # latent-axis alpha (both engines use the same lower-triangular Lambda convention; observed 9e-6)
            Aj = extract_lv_effects(fit; type = :axis_effect)
            @test isapprox(Aj, _ns_mat(l["axis_effect"], 2, 2); atol = 5e-5, rtol = 0)
        end

        @testset "two: extract_communality, extract_Sigma_B, extract_Sigma_W" begin
            t = fx["two"]
            dp = joinpath(_NS_DIR, t["data_file"])
            @test bytes2hex(sha256(read(dp))) == t["data_sha256"]
            @test t["converged"] && t["pd_hessian"]
            p2 = Int(t["p"])
            Y, _, ind = _ns_load_csv(dp, String.(t["trait_names"]), Int(t["n_obs"]); obs_col = 2)
            fit = fit_twolevel_gaussian(Y, ind; K_B = 1, K_W = 1)
            @test fit.converged
            @test isapprox(fit.loglik, Float64(t["loglik"]); atol = 1e-6, rtol = 0)   # observed 8.9e-8
            # per-trait communality at each tier (observed unit 5.0e-5, unit_obs 2.2e-6)
            @test isapprox(extract_communality(fit; level = :unit), Float64.(t["communality_unit"]); atol = 2e-4, rtol = 0)
            @test isapprox(extract_communality(fit; level = :unit_obs), Float64.(t["communality_unit_obs"]); atol = 2e-4, rtol = 0)
            # tier total covariance (observed unit 1.7e-5, unit_obs 5.2e-6)
            @test isapprox(extract_Sigma(fit; level = :unit).Sigma, _ns_mat(t["sigma_unit"], p2, p2); atol = 1e-4, rtol = 0)
            @test isapprox(extract_Sigma(fit; level = :unit_obs).Sigma, _ns_mat(t["sigma_unit_obs"], p2, p2); atol = 1e-4, rtol = 0)
        end
    end
end
