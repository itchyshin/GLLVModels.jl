# Two-part fitters reject an observed value outside the family's support (2026-09-28).
#
# WHY THIS FILE EXISTS: every two-part log-density branches on `if y > 0`, so an
# observed NaN or negative value took the zero branch and was scored as an observed
# zero; the fit then reported `converged = true`. Beta-hurdle additionally clamped
# values >= 1 into (0,1). NaN is not a missing-value marker in this package (only
# `missing` is, via `observed_mask`), so such a value is invalid input and must be
# refused up front with an ArgumentError naming the value, its index and the support.
#
# What is deliberately NOT rejected here (bootstrap draws in the #504 verdict
# slices rely on these failing softly, as a non-converged fit, not as a throw):
# a half count 0.5 (count families), 1e300 (all), and N + 1 for ZIB. Those already
# end on the fitter's failure verdict, so they are not silently mis-scored.

using GLLVModels, Test, Random

const GM = GLLVModels

function _tp_argerr_msg(f)
    try
        f()
    catch e
        return e isa ArgumentError ? e.msg : "not an ArgumentError: $(typeof(e))"
    end
    return "no throw"
end

@testset "two-part fitters reject observed out-of-support values" begin
    Random.seed!(11)
    p, n, K = 4, 30, 1
    Z = randn(K, n); L = 0.5 .* randn(p, K); H = 0.3 .* randn(p) .+ L * Z
    occ = rand(p, n) .< 0.7
    Ycount = Float64[occ[t, s] ? rand(1:6) : 0 for t in 1:p, s in 1:n]
    Ypos = [occ[t, s] ? exp(0.2 * randn() + H[t, s]) : 0.0 for t in 1:p, s in 1:n]
    Yprop = [occ[t, s] ? clamp(0.3 + 0.1 * randn(), 0.05, 0.95) : 0.0 for t in 1:p, s in 1:n]
    N = 8
    Ybin = Float64[occ[t, s] ? rand(0:N) : 0 for t in 1:p, s in 1:n]
    X = 0.2 .* randn(p, n, 2)

    # (label, fitter, valid Y, invalid observed values)
    counts_bad = (NaN, -1.0)
    positive_bad = (NaN, -1.0, Inf)
    entries = [
        ("fit_zip_gllvm", Y -> GM.fit_zip_gllvm(Y; K), Ycount, counts_bad),
        ("fit_zip_gllvm_cov", Y -> GM.fit_zip_gllvm_cov(Y; X, K), Ycount, counts_bad),
        ("fit_zinb_gllvm", Y -> GM.fit_zinb_gllvm(Y; K), Ycount, counts_bad),
        ("fit_zinb_gllvm_cov", Y -> GM.fit_zinb_gllvm_cov(Y; X, K), Ycount, counts_bad),
        ("fit_hurdle_poisson_gllvm", Y -> GM.fit_hurdle_poisson_gllvm(Y; K), Ycount, counts_bad),
        ("fit_hurdle_nb_gllvm", Y -> GM.fit_hurdle_nb_gllvm(Y; K), Ycount, counts_bad),
        ("fit_zib_gllvm", Y -> GM.fit_zib_gllvm(Y; K, N), Ybin, counts_bad),
        ("fit_zib_gllvm_cov", Y -> GM.fit_zib_gllvm_cov(Y; X, K, N), Ybin, counts_bad),
        ("fit_delta_lognormal_gllvm", Y -> GM.fit_delta_lognormal_gllvm(Y; K), Ypos, positive_bad),
        ("fit_delta_gamma_gllvm", Y -> GM.fit_delta_gamma_gllvm(Y; K), Ypos, positive_bad),
        ("fit_delta_gamma_gllvm_va", Y -> GM.fit_delta_gamma_gllvm_va(Y; K), Ypos, positive_bad),
        ("fit_beta_hurdle_gllvm", Y -> GM.fit_beta_hurdle_gllvm(Y; K), Yprop,
         (NaN, -0.2, -Inf, 1.0, 1.5)),
    ]

    @testset "$label rejects $v" for (label, fit, Y, bad) in entries, v in bad
        B = copy(Y)
        B[2, 3] = v
        msg = _tp_argerr_msg(() -> fit(B))
        @test occursin(label, msg)
        @test occursin(string(v), msg)
        @test occursin("[2, 3]", msg)
    end

    @testset "$label: valid data still fits" for (label, fit, Y, bad) in entries
        f = fit(Y)
        @test f.converged == true
        @test isfinite(f.loglik)
    end

    # The dispatcher routes through the same fitters, so the check reaches it too.
    @testset "fit_gllvm dispatcher propagates the check" begin
        B = copy(Ycount); B[1, 1] = NaN
        @test_throws ArgumentError GM.fit_gllvm(B; family = ZIPoisson(), K)
        B = copy(Yprop); B[1, 1] = 1.0
        @test_throws ArgumentError GM.fit_gllvm(B; family = BetaHurdle(), K)
    end

    # Soft-failure draws used by the #504 bootstrap-verdict tests stay soft:
    # no ArgumentError, the fitter returns its failure verdict instead.
    @testset "soft-failure draws are not newly rejected" begin
        B = copy(Ycount); B[1, 1] = 0.5
        f = GM.fit_zip_gllvm(B; K)
        @test !f.converged
        B = copy(Ybin); B[1, 1] = N + 1
        f = GM.fit_zib_gllvm(B; K, N)
        @test !f.converged
        B = copy(Ypos); B[1, 1] = 1e300
        f = GM.fit_delta_lognormal_gllvm(B; K)
        @test !f.converged
    end
end
