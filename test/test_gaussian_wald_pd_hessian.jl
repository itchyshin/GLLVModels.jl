# Gaussian Wald pd_hessian must be a real positive-definite test (#727).
# An indefinite Hessian can invert with an all-positive diagonal, so
# `diag(inv(H)) > 0` reports a saddle as positive definite. The family
# Wald path uses Cholesky; this file pins the same test on the Gaussian path.
using Test, LinearAlgebra
using GLLVModels

@testset "Gaussian Wald indefinite Hessian is not PD" begin
    # Eigenvalues 1 and -100. diag(inv(H)) is positive, so the old check passes.
    Q = [1.0 1.0; 1.0 -1.0] / sqrt(2)
    H = Q * Diagonal([1.0, -100.0]) * Q'
    Hsym = Symmetric((H .+ H') ./ 2)
    Σ = inv(Hsym)
    @test all(diag(Σ) .> 0)
    @test !isposdef(Hsym)

    V, se, pd = GLLVModels._gaussian_wald_covariance_from_hessian(H, 2)
    @test pd == false
    @test all(isnan, se)
    @test all(isnan, V)

    # A positive-definite Hessian still reports finite SEs.
    Hpd = [2.0 0.3; 0.3 1.5]
    Vpd, sepd, pdpd = GLLVModels._gaussian_wald_covariance_from_hessian(Hpd, 2)
    @test pdpd
    @test isposdef(Symmetric(Hpd))
    @test sepd ≈ sqrt.(diag(inv(Symmetric(Hpd))))
    @test diag(Vpd) == sepd .^ 2
    @test issymmetric(Vpd)
end
