using GLLVModels, Test, TOML

# Dispersion-boundary verdict for the zero-truncated NB2 fitters. r below 1e-6 is a
# degenerate fit (extreme overdispersion), yet Optim can stop there with converged = true
# (r = 3.1e-46 from one count of 10^13 on macOS; the same data end at r = 0.98 on a Linux
# CI runner, so that trigger is not used here). Such fits are reported as not converged,
# with a warning. Above 1e6 (the Poisson limit) r is not identified but the fit is usually
# sound, so the fitters only warn and keep Optim's verdict.

const _BD = GLLVModels._truncnb2_dispersion_verdict

@testset "truncated NB2 dispersion verdict: the rule" begin
    @test (@test_logs _BD(true, [0.5, 3.0, 2e5], "fit")) === true
    @test (@test_logs _BD(false, [0.5, 3.0], "fit")) === false
    @test (@test_logs (:warn, r"below 1e-6") _BD(true, [3.0, 1e-7], "fit")) === false
    @test (@test_logs (:warn, r"Poisson limit") _BD(true, [3.0, 2e6], "fit")) === true
    @test (@test_logs (:warn, r"Poisson limit") _BD(false, [3.0, 2e6], "fit")) === false
    @test (@test_logs (:warn, r"Poisson limit") (:warn, r"below 1e-6") _BD(true, [1e-7, 2e6], "fit")) === false
end

# Wiring: start r at the boundary and run zero optimiser iterations, so the fit returns
# that r on every machine and the fitter's own verdict must act on it. For the shared-r
# fitter, `eigmin_floor = -Inf` keeps the Laplace breakdown retry (#581) from replacing
# the start.
const _TNB2_BD = TOML.parsefile(joinpath(@__DIR__, "fixtures",
                                         "truncnb2_dispersion_boundary.toml"))
const _TNB2_BD_Y = reshape(Int.(_TNB2_BD["upper_Y_column_major"]),
                           _TNB2_BD["upper_p"], _TNB2_BD["n"])

@testset "truncated NB2 (shared r): the fitter applies the verdict" begin
    f = @test_logs (:warn, r"fit_truncated_nbinom2_gllvm: .*below 1e-6") match_mode = :any fit_truncated_nbinom2_gllvm(
        _TNB2_BD_Y; K = 1, r_init = 1e-9, iterations = 0, eigmin_floor = -Inf)
    @test f.r ≈ 1e-9
    @test !f.converged
end

@testset "truncated NB2 (per-trait r): the fitter applies the verdict" begin
    # The per-trait fitter restarts any trait whose r ends outside [1e-6, 1e6] from r = 1
    # before the verdict (`_nb_boundary_restart`), and keeps the restart when it fits
    # better. A start at r = 1e-9 is replaced that way (it ends at r = 1), so the lower
    # end cannot be pinned here; the start at r = 1e9 is kept, which shows the verdict
    # runs on the returned r. The lower-end rule itself is covered by the unit tests.
    p = _TNB2_BD["upper_p"]
    g = @test_logs (:warn, r"fit_truncated_nbinom2_gllvm_pertrait: .*Poisson limit") match_mode = :any fit_truncated_nbinom2_gllvm_pertrait(
        _TNB2_BD_Y; K = 1, r_init = [fill(3.0, p - 1); 1e9], iterations = 0)
    @test g.r[end] ≈ 1e9
end
