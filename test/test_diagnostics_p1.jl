# gllvm-parity-tag: P1
#
# Numeric twins of gllvmTMB's sanity_multi() and compare_loadings() at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9): postfit rows POSTFIT-SURFACE-sanity_multi and
# POSTFIT-SURFACE-compare_loadings, namespace row export/sanity_multi. No R at test time: R's
# recorded values are read from test/fixtures/diagnostics_p1.toml (generated once by
# test/fixtures/gen_diagnostics_p1.R against a lane-local gllvmTMB install at the pin; it records
# R version, commit and the data sha256).
#
# sanity_multi     : value ~ 0 + trait + latent(0 + trait | unit, d = D, unique = FALSE), D = 1, 2,
#                    on ns_gauss_p1_data.csv (p = 6, 200 units, one observation per cell)
#                    <-> fit_gllvm(Y; family = Normal(), K = D) and sanity_multi(fit; y = Y). Both R
#                    fits converged with a positive-definite Hessian (asserted). Compared numerically:
#                    max_se (largest fixed-effect SE) and rr_B_min_loading (min |diag Lambda_B|); the
#                    flags converged, sdreport_ok and pd_hessian are checked equal (booleans, not
#                    receipt cases); the report lines are checked line by line except for the
#                    gradient value.
# compare_loadings : two recorded input pairs (6 x 2: a fitted Lambda_B against the simulating
#                    loadings; 8 x 3 with a reflection as the optimal transform), R's full output
#                    (R, Lambda_a_rot, frobenius, cor_per_factor) against the matrix method.
#
# Disclosed: max_gradient is not compared. It is the largest gradient component at each optimiser's
# stopping point (R's nlminb stops at ~1e-3, the Julia fit at ~1e-12), so its value says how far
# each optimiser went, not what the model is; both are below R's 1e-2 threshold (asserted).
using Test
using GLLVModels
using Distributions: Normal
using TOML
using SHA
using LinearAlgebra: I

const _DG_DIR = joinpath(@__DIR__, "fixtures")

_dg_mat(rows) = permutedims(reduce(hcat, [Float64.(r) for r in rows]))   # TOML array of rows -> Matrix

function _dg_load_Y(path::AbstractString, trait_names::Vector{String}, n::Integer)
    Y = zeros(Float64, length(trait_names), n)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            a = split(line, ",")
            Y[findfirst(==(strip(a[2], '"')), trait_names), parse(Int, strip(a[1], '"'))] = parse(Float64, a[3])
        end
    end
    return Y
end

@testset "diagnostics twins: gllvmTMB P1 (9539352f6)" begin
    dgf = joinpath(_DG_DIR, "diagnostics_p1.toml")
    if !isfile(dgf)
        @warn "diagnostics P1 fixture absent; twin gate NOT RUN" dgf
        @test_skip false
    else
        fx = TOML.parsefile(dgf)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"

        @testset "sanity_multi on Gaussian rank-D fits" begin
            dp = joinpath(_DG_DIR, fx["data_file"])
            @test bytes2hex(sha256(read(dp))) == fx["data_sha256"]
            Y = _dg_load_Y(dp, String.(fx["trait_names"]), Int(fx["n_unit"]))
            r_names = Symbol[]
            r_se = Float64[]; j_se = Float64[]
            r_ld = Float64[]; j_ld = Float64[]
            for key in ("d1", "d2")
                s = fx["sanity"][key]
                @test s["converged"] && s["pd_hessian"]
                fit = fit_gllvm(Y; family = Normal(), K = Int(s["d"]))
                @test fit.converged
                @test isapprox(fit.logLik, Float64(s["loglik"]); atol = 1e-6, rtol = 0)   # same optimum as R
                io = IOBuffer()
                js = sanity_multi(fit; y = Y, io = io)
                # R's flags, under R's names and in R's order, lead the result
                r_names = Symbol.(String.(s["flag_names"]))
                @test collect(keys(js))[1:length(r_names)] == r_names
                @test js.converged === s["flag_converged"]
                @test js.sdreport_ok === s["flag_sdreport_ok"]
                @test js.pd_hessian === s["flag_pd_hessian"]
                @test Float64(s["max_gradient"]) < 1e-2 && js.max_gradient < 1e-2   # both PASS R's threshold; values not compared
                push!(r_se, Float64(s["max_se"])); push!(j_se, js.max_se)
                push!(r_ld, Float64(s["rr_B_min_loading"])); push!(j_ld, js.rr_B_min_loading)
                # report lines: R's layout and labels; every line but the gradient value matches R's
                jl = split(String(take!(io)), '\n'; keepempty = false)
                rl = String.(s["printed"])
                @test length(jl) == length(rl)
                for (a, b) in zip(jl, rl)
                    if startswith(b, "Max |gradient|")
                        @test replace(a, r"= [^)]*\)" => "") == replace(b, r"= [^)]*\)" => "")
                    else
                        @test a == b
                    end
                end
            end
            # the R values are not degenerate: well away from zero and from each other
            @test all(>(0.05), r_se) && abs(r_se[1] - r_se[2]) > 1e-3
            @test all(>(0.5), r_ld) && abs(r_ld[1] - r_ld[2]) > 0.05
            @test isapprox(j_se, r_se; atol = 2e-6, rtol = 0)   # max_se, d = 1 and 2 (observed 2.2e-7)
            @test isapprox(j_ld, r_ld; atol = 2e-5, rtol = 0)   # rr_B_min_loading, d = 1 and 2 (observed 2.8e-6)
        end

        @testset "compare_loadings on recorded loading pairs" begin
            for key in ("fit_vs_truth", "reflected")
                c = fx["compare_loadings"][key]
                A = _dg_mat(c["Lambda_a"]); B = _dg_mat(c["Lambda_b"])
                r = compare_loadings(A, B)
                @test collect(keys(r)) == [:R, :Lambda_a_rot, :frobenius, :cor_per_factor]
                rR = _dg_mat(c["R"])
                @test size(r.R) == size(rR)
                # the R values are not degenerate: a non-trivial transform and a non-zero residual
                @test maximum(abs, rR - I) > 1e-3 && Float64(c["frobenius"]) > 0.05
                # 1e-12: identical inputs, so only BLAS/LAPACK summation order differs (x86 vs ARM); a real port error is 1e-2 to 1
                @test maximum(abs, r.R .- rR) <= 1e-12   # R (observed 4.4e-16)
                @test maximum(abs, r.Lambda_a_rot .- _dg_mat(c["Lambda_a_rot"])) <= 1e-12   # Lambda_a_rot (observed 8.9e-16)
                @test abs(r.frobenius - Float64(c["frobenius"])) <= 1e-12   # frobenius (observed 2.8e-16)
                @test maximum(abs, r.cor_per_factor .- Float64.(c["cor_per_factor"])) <= 1e-12   # cor_per_factor (observed 3.3e-16)
            end
        end
    end
end
