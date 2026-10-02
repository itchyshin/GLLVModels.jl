# gllvm-parity-tag: P1
#
# Numeric twins of three more gllvmTMB namespace rows against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9): tidy.gllvmTMB_multi (fixed effects), and the
# Beta and nbinom2 family exports. Follows test_namespace_numeric_p1_twin.jl (itchyshin/GLLVModels.jl#652).
# No R at test time: R's recorded values are read from test/fixtures/ns_numeric_p1_b.toml
# (generated once by test/fixtures/gen_namespace_numeric_p1_b.R against a lane-local gllvmTMB
# install at the pin; the file records R version, commit, seeds and the data sha256s) and the
# same datasets are fitted in Julia. Each R fit converged with a positive-definite Hessian
# (asserted below); both sides are compared at their own optimum, with a log-likelihood guard.
#
#   tidy : the rank-2 Gaussian fit of the #652 twin (fit_gllvm(Y; K = 2) <->
#          latent(d = 2, unique = FALSE)); tidy(fit, Y) fixed rows vs R tidy(fit, "fixed").
#   beta : Beta family, p = 6, n = 200, K = 1 (R: family = Beta(), latent(d = 1, unique = FALSE)).
#   nb2  : NB2 family, same design (R: family = nbinom2()); per-trait dispersion on both sides.
#
# Tolerances are set above the differences observed when the twin was built (shown in the
# comments) and are not loosened to pass. The sign of a K = 1 loading axis is not identified, so
# loadings are compared through Lambda Lambda' (rotation- and sign-stable).
using Test
using GLLVModels
using Distributions: Normal, NegativeBinomial
using TOML
using LinearAlgebra
using SHA

const _NSB_DIR = joinpath(@__DIR__, "fixtures")
const _NSB_TOML = joinpath(_NSB_DIR, "ns_numeric_p1_b.toml")

# R's write.csv long data (unit, trait "t#", value) -> p x n response matrix.
function _nsb_load_csv(path::AbstractString, p::Integer, n::Integer)
    Y = zeros(Float64, p, n)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            a = split(line, ",")
            Y[parse(Int, strip(a[2], ['"', 't'])), parse(Int, strip(a[1], '"'))] = parse(Float64, a[3])
        end
    end
    return Y
end
_nsb_mat(v, nrow, ncol) = permutedims(reshape(Float64.(v), ncol, nrow))

@testset "namespace numeric twins (b): gllvmTMB P1 (9539352f6)" begin
    if !isfile(_NSB_TOML)
        @warn "namespace numeric P1 (b) fixture absent; twin gate NOT RUN" _NSB_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_NSB_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        p, n = Int(fx["p"]), Int(fx["n_unit"])

        @testset "tidy.gllvmTMB_multi (fixed)" begin
            t = fx["tidy"]
            dp = joinpath(_NSB_DIR, t["data_file"])
            @test bytes2hex(sha256(read(dp))) == t["data_sha256"]
            @test t["converged"] && t["pd_hessian"]
            Y = _nsb_load_csv(dp, p, n)
            fit = fit_gllvm(Y; family = Normal(), K = 2)
            @test fit.converged
            @test isapprox(fit.logLik, Float64(t["loglik"]); atol = 1e-6, rtol = 0)
            tb = tidy(fit, Y)
            @test length(tb) == p == length(t["terms"])
            @test all(r -> r.effect === :fixed && r.link === :identity, tb)
            @test all(==("identity"), t["link"])
            # estimates: observed 3.0e-13
            @test isapprox([r.estimate for r in tb], Float64.(t["estimate"]); atol = 1e-6, rtol = 0)
            # Wald standard errors from the inverse joint Hessian: observed 3.0e-7
            @test isapprox([r.std_error for r in tb], Float64.(t["std_error"]); atol = 1e-5, rtol = 0)
        end

        @testset "beta" begin
            b = fx["beta"]
            dp = joinpath(_NSB_DIR, b["data_file"])
            @test bytes2hex(sha256(read(dp))) == b["data_sha256"]
            @test b["converged"] && b["pd_hessian"]
            Y = _nsb_load_csv(dp, p, n)
            fit = fit_gllvm(Y; family = GLLVModels.Beta(), K = 1)
            @test fit.converged
            @test fit.group == collect(1:p)                 # one dispersion per trait, as in R
            @test isapprox(fit.loglik, Float64(b["loglik"]); atol = 1e-6, rtol = 0)
            # observed: logLik 1.5e-10, intercepts 3.9e-7, Lambda Lambda' 7.3e-7, phi 6.4e-6
            bj = fit.β
            @test isapprox(bj, Float64.(b["beta"]); atol = 1e-5, rtol = 0)
            LLt = fit.Λ * fit.Λ'
            @test isapprox(LLt, _nsb_mat(b["lambda_lambdat"], p, p); atol = 1e-5, rtol = 0)
            phij = fit.φ
            @test isapprox(phij, Float64.(b["phi"]); atol = 1e-4, rtol = 0)
        end

        @testset "nb2" begin
            b = fx["nb2"]
            dp = joinpath(_NSB_DIR, b["data_file"])
            @test bytes2hex(sha256(read(dp))) == b["data_sha256"]
            @test b["converged"] && b["pd_hessian"]
            Y = _nsb_load_csv(dp, p, n)
            fit = fit_gllvm(Y; family = NegativeBinomial(1.0, 0.5), disp_group = :species, K = 1)
            @test fit.converged
            @test fit.group == collect(1:p)                 # one dispersion per trait, as in R
            @test isapprox(fit.loglik, Float64(b["loglik"]); atol = 1e-6, rtol = 0)
            # observed: logLik 4.5e-8, intercepts 6.2e-6, Lambda Lambda' 1.2e-5, phi 3.7e-4
            bj = fit.β
            @test isapprox(bj, Float64.(b["beta"]); atol = 5e-5, rtol = 0)
            LLt = fit.Λ * fit.Λ'
            @test isapprox(LLt, _nsb_mat(b["lambda_lambdat"], p, p); atol = 1e-4, rtol = 0)
            phij = fit.r_group
            @test isapprox(phij, Float64.(b["phi"]); atol = 1e-3, rtol = 0)
        end
    end
end
