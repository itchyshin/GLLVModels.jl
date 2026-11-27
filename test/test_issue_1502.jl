# refs #1502: fit_gllvm(Normal()) on masked data with trait intercepts must not fail
# the AGHQ conditional-mode check that already accepts fit_gaussian_gllvm.
using Test, GLLVModels, Distributions

@testset "issue 1502: fit_gllvm Normal with missing and trait intercepts" begin
    Y = Matrix{Union{Missing, Float64}}(undef, 2, 3)
    Y[1, :] = [1.0, 2.0, missing]
    Y[2, :] = [3.0, 4.0, 5.0]
    f1 = fit_gaussian_gllvm(Y; K = 1)
    f2 = fit_gllvm(Y; family = Normal(), K = 1)
    @test isfinite(f1.logLik) && f1.converged
  # Route must not abort at the Laplace mode check; finite fit on this tiny panel
    @test isfinite(f2.logLik)
    @test length(f2.pars.β) == 2
end