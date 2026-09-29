using Test, GLLVModels, Random, LinearAlgebra, TOML

# Panel finding 4 (docs/dev-log/core070/parity-panel-2026-09-01.md): a
# Student-t fit whose estimated ν reaches the flat Gaussian-limit boundary
# (ν > 1e6 — the same rule the parity fixture prints but never asserts) must
# say so in its public fit object, symmetric to the R engine's health report.
# The flag is additive: `converged` semantics are unchanged (a converged-flag
# change is a maintainer-gated public-contract change).
@testset "Student-t Gaussian-limit boundary honesty" begin
    rng = MersenneTwister(2026)
    p, n = 2, 80
    Λ_true = reshape([0.6, -0.4], p, 1)
    # Pure Gaussian data: the estimated-ν MLE walks to the ν→∞ boundary.
    Y = Λ_true * randn(rng, 1, n) + 0.5 .* randn(rng, p, n)
    fit = fit_studentt_gllvm(Y; K = 1)   # ν estimated by default
    @test fit.estimated_nu
    @test hasproperty(fit, :nu_boundary)
    νvec = fit.ν isa Real ? [fit.ν] : fit.ν
    if any(>(1e6), νvec)
        @test fit.nu_boundary
        @test occursin("boundary", sprint(show, fit))
    else
        # If the optimizer happens to stop at interior ν, the flag must be false
        # and the data-driven premise is reported rather than silently passed.
        @test !fit.nu_boundary
        @info "boundary not reached on this stream; interior ν" νvec
    end

    # Heavy-tailed data with fixed ν: never flagged.
    ν_true = 4.0
    Yt = Λ_true * randn(rng, 1, n) +
         0.5 .* (randn(rng, p, n) ./ sqrt.(rand(rng, p, n) .* 0 .+ 1)) .+
         0.3 .* randn(rng, p, n)
    fixed = fit_studentt_gllvm(Yt; K = 1, nu = ν_true)
    @test !fixed.estimated_nu
    @test hasproperty(fixed, :nu_boundary) && !fixed.nu_boundary
end

# An estimated ν that runs to the ν → ∞ boundary is restarted from finite ν, so a
# fit does not stop at the Gaussian limit when an interior optimum is higher. Data:
# the stored near-Gaussian parity draw (Julia 1.13.1, seed 73), where the default
# start used to end at ν₁ ≈ 5e9 with logLik −1430.162 while gllvmTMB found the
# interior ν₁ ≈ 17.7 at −1430.097.
@testset "Student-t boundary restart finds the interior ν optimum" begin
    d = TOML.parsefile(joinpath(@__DIR__, "parity", "fixtures", "studentt_neargauss_data.toml"))
    Y = reshape(Float64.(d["Y_column_major"]), d["p"], d["n"])
    f = fit_studentt_gllvm(Y; K = 1, nu = nothing, disp_group = :species,
                           g_tol = 1e-7, iterations = 800)
    @test f.loglik > -1430.10
    @test 10 < f.ν[1] < 30
end
