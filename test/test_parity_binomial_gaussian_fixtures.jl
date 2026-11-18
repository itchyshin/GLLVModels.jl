# gllvm-parity-tag: P1
#
# Rotation-invariant parity of the Binomial and Gaussian GLLVM fits against stored gllvmTMB
# fixtures (gllvmTMB 0.7.1 at 9539352f66f2db2cc26b1c393e67212a359b60c9). No R at test time: R's
# fits are read from test/fixtures/aghq_p1/aghq_p1.toml (binomial_p5, gaussian_p5; one latent
# axis) and test/fixtures/data_twins_p1.toml ([gauss_zero], Gaussian rank 2 on ns_gauss_p1_data).
#
# Compared per fit, all invariant to the rotation / sign of the loadings:
#   * logLik;
#   * Lambda Lambda' (p x p) from R's recorded loadings (K = 1) or R's recorded Lambda Lambda'
#     (K = 2) against Julia's fitted loadings;
#   * Sigma: the latent-scale covariance Lambda Lambda' (Binomial, no residual term) and
#     Lambda Lambda' + sigma_eps^2 I (Gaussian, where R recorded sigma_eps).
# Tests only: no tolerance was widened, and no row currently mismatches. A mismatch found later
# should be listed in _PBG_BROKEN so it is reported as @test_broken, not silently loosened.
using Test
using GLLVModels
using Distributions: Normal
using LinearAlgebra
using TOML
using SHA

include(joinpath(@__DIR__, "fixtures", "aghq_p1", "aghq_p1_helpers.jl"))

const _PBG_ATOL_LL  = 1e-6   # logLik, absolute (matches the AGHQ twin)
const _PBG_ATOL_SIG = 1e-4   # Lambda Lambda' / Sigma entries (observed ~1e-5 or smaller)
# "<case>:<quantity>" pairs known to mismatch; reported with @test_broken. Empty on purpose.
const _PBG_BROKEN = Set{String}()

function _pbg_check(key::AbstractString, got::Real, want::Real, atol::Real)
    if key in _PBG_BROKEN
        @test_broken isapprox(got, want; atol = atol, rtol = 0)
    else
        @test isapprox(got, want; atol = atol, rtol = 0)
    end
end

_pbg_check_mat(key, A, B, atol) = _pbg_check(key, maximum(abs.(A .- B)), 0.0, atol)

@testset "Binomial / Gaussian rotation-invariant parity vs gllvmTMB fixtures" begin
    fx = TOML.parsefile(AGHQ_P1_TOML)
    @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"

    @testset "Binomial (K = 1): $row" for row in
            ("AGHQ-POLICY-OFF", "AGHQ-AUTO-K-BINOMIAL", "AGHQ-POLICY-EXPLICIT")
        id = aghq_p1_fit_id(row)
        @test fx["dataset"][fx["case"][id]["dataset"]]["family"] == "binomial"
        r = fx["case"][id]["r"]
        @test r["converged"] === true
        j = aghq_p1_fit(fx, id)
        @test j.converged
        lr = Float64.(r["lambda"])
        ΛΛ_r = lr * lr'
        ΛΛ_j = j.lambda * j.lambda'
        _pbg_check("$row:logLik", j.loglik, Float64(r["loglik"]), _PBG_ATOL_LL)
        _pbg_check_mat("$row:LambdaLambda", ΛΛ_j, ΛΛ_r, _PBG_ATOL_SIG)
        # binomial has no residual variance: Sigma is Lambda Lambda' on the link scale
        _pbg_check_mat("$row:Sigma", ΛΛ_j, ΛΛ_r, _PBG_ATOL_SIG)
        @test all(isfinite, ΛΛ_j)
        @test issymmetric(ΛΛ_j) && minimum(eigvals(Symmetric(ΛΛ_j))) > -1e-10
    end

    @testset "Gaussian (K = 1): AGHQ-AUTO-K-GAUSSIAN" begin
        row = "AGHQ-AUTO-K-GAUSSIAN"
        @test fx["dataset"][fx["case"][row]["dataset"]]["family"] == "gaussian"
        r = fx["case"][row]["r"]
        @test r["converged"] === true
        j = aghq_p1_fit(fx, row)
        @test j.converged
        lr = Float64.(r["lambda"])
        p = length(lr)
        ΛΛ_r = lr * lr'
        ΛΛ_j = j.lambda * j.lambda'
        Σ_r = ΛΛ_r + Float64(r["sigma_eps"])^2 * I(p)
        Σ_j = ΛΛ_j + j.sigma_eps^2 * I(p)
        _pbg_check("$row:logLik", j.loglik, Float64(r["loglik"]), _PBG_ATOL_LL)
        _pbg_check_mat("$row:LambdaLambda", ΛΛ_j, ΛΛ_r, _PBG_ATOL_SIG)
        _pbg_check("$row:sigma_eps", j.sigma_eps, Float64(r["sigma_eps"]), _PBG_ATOL_SIG)
        _pbg_check_mat("$row:Sigma", Σ_j, Σ_r, _PBG_ATOL_SIG)
    end

    @testset "Gaussian (K = 2): data_twins_p1 [gauss_zero]" begin
        dtoml = joinpath(@__DIR__, "fixtures", "data_twins_p1.toml")
        dfx = TOML.parsefile(dtoml)
        g = dfx["gauss_zero"]
        csv = joinpath(@__DIR__, "fixtures", g["data_file"])
        @test bytes2hex(sha256(read(csv))) == g["data_sha256"]
        @test g["converged"] && g["pd_hessian"]
        p, n = Int(dfx["p"]), Int(dfx["n_unit_gauss"])
        # R's write.csv long data (unit, trait "t#", value, ...) -> p x n
        hdr = split(readline(csv), ",")
        ci = findfirst(==("\"value\""), hdr)
        @test ci !== nothing
        Y = zeros(p, n)
        open(csv) do io
            readline(io)
            for line in eachline(io)
                isempty(line) && continue
                a = split(line, ",")
                Y[parse(Int, strip(a[2], ['"', 't'])), parse(Int, strip(a[1], '"'))] = parse(Float64, a[ci])
            end
        end
        fit = fit_gllvm(Y; family = Normal(), K = 2)
        @test fit.converged
        Λ = fit.pars.Λ
        ΛΛ_j = Λ * Λ'
        ΛΛ_r = permutedims(reshape(Float64.(g["lambda_lambdat"]), p, p))
        _pbg_check("gauss_zero:logLik", fit.logLik, Float64(g["loglik"]), _PBG_ATOL_LL)
        _pbg_check_mat("gauss_zero:LambdaLambda", ΛΛ_j, ΛΛ_r, _PBG_ATOL_SIG)
        # R recorded no residual SD for this fit, so Sigma is not compared here.
    end
end
