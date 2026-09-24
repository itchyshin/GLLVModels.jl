# test_nb2_finite_dispersion_parity.jl — NB2 GLLVModels vs gllvmTMB where every trait keeps
# finite overdispersion, plus a boundary-agreement check on the NATIVE-06 data.
#
# Developer check (#476, option "B-lite"): included by runparity.jl's optional developer
# cohort only, NOT by the frozen required contract. NATIVE-06 itself stays frozen; on its
# data traits 1 and 3 sit at the Poisson boundary on both engines, so its "both optimizers
# converge" rule cannot hold there. This file carries the parity evidence NATIVE-06 cannot.
#
# R's gradient at its own optimum is recorded, not gated: on this design gllvmTMB's nlminb
# stops with max |gradient| between 2e-4 and 2e-3 even where both engines agree to 1e-11 in
# logLik, and the value depends on the machine.

using GLLVModels, RCall, Test, SHA, TOML
isdefined(@__MODULE__, :parity_nb2_original_Y) || include(joinpath(@__DIR__, "nb2_health.jl"))

function _nb2_r_side(Y, K)
    r = fit_gllvmtmb_parity_loglik(Y, K; family = :negbinomial)
    rr = rcopy(Vector{Float64}, R"exp(as.numeric(fit_r$tmb_obj$env$parList(fit_r$opt$par)$log_phi_nbinom2))")
    grad = rcopy(Float64, R"max(abs(as.numeric(fit_r$tmb_obj$gr(fit_r$opt$par))))")
    code = rcopy(Int, R"as.integer(fit_r$opt$convergence)")
    return (; logLik = r.logLik, r = rr, grad, code)
end

function _nb2_report(label, jl, rs)
    println("── ", label, " ──")
    println("  Julia logLik = ", jl.loglik, "   gllvmTMB logLik = ", rs.logLik,
            "   Δ (jl − r) = ", jl.loglik - rs.logLik)
    println("  Julia r = ", round.(jl.r_group; sigdigits = 5), "   boundary = ", jl.dispersion_boundary)
    println("  gllvmTMB r = ", round.(rs.r; sigdigits = 5))
    println("  gllvmTMB optimizer code = ", rs.code, "   r_gradient_max = ", rs.grad, " (recorded, not a gate)")
end

@testset "NB2 parity with finite dispersion on every trait (developer check, #476)" begin
    d = TOML.parsefile(joinpath(@__DIR__, "..", "fixtures", "nb2_finite_dispersion_data.toml"))
    Y = reshape(Int.(d["Y_column_major"]), d["p"], d["n"])
    K = d["K"]
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y))))) == d["data_sha256"]

    jl = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = K, g_tol = 1e-7, iterations = 800)
    rs = _nb2_r_side(Y, K)
    _nb2_report("NB2 finite dispersion (seed $(d["seed"]), r_true $(d["r_true"]), n $(d["n"]))", jl, rs)

    @test jl.converged
    @test !any(jl.dispersion_boundary)
    @test rs.code == 0
    @test all(<(1e3), rs.r)
    @test abs(jl.loglik - rs.logLik) <= 1e-6 * abs(rs.logLik)
    @test maximum(abs.(jl.r_group .- rs.r) ./ rs.r) <= 1e-3
end

@testset "NATIVE-06 data: both engines put traits 1 and 3 at the Poisson boundary (developer check, #476)" begin
    Y = parity_nb2_original_Y()
    K = 2
    jl = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = K, g_tol = 1e-7, iterations = 800)
    rs = _nb2_r_side(Y, K)
    _nb2_report("NATIVE-06 data, boundary agreement", jl, rs)

    identified = [2, 4, 5]
    @test jl.dispersion_boundary == [true, false, true, false, false]
    @test all(>(1e5), rs.r[[1, 3]])
    @test maximum(abs.(jl.r_group[identified] .- rs.r[identified]) ./ rs.r[identified]) <= 1e-3
    @test abs(jl.loglik - rs.logLik) <= 1e-6 * abs(rs.logLik)
end
