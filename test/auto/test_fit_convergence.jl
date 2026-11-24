using Test
using LinearAlgebra
using GLLVModels

@testset "Boundary-aware fit convergence" begin
    H = Matrix{Float64}(I, 2, 2)
    # Minimization KKT: positive score at an active lower bound and negative
    # score at an active upper bound project to zero.
    lower = GLLVModels._fit_projected_gradient([0.2, 0.0]; lower=(1,))
    upper = GLLVModels._fit_projected_gradient([-0.2, 0.0]; upper=(1,))
    @test lower == [0.0, 0.0]
    @test upper == [0.0, 0.0]

    wrong_lower = GLLVModels._fit_projected_gradient([-0.2, 0.0]; lower=(1,))
    wrong_upper = GLLVModels._fit_projected_gradient([0.2, 0.0]; upper=(1,))
    @test wrong_lower == [-0.2, 0.0]
    @test wrong_upper == [0.2, 0.0]
    @test_throws ArgumentError GLLVModels._fit_projected_gradient([0.0]; lower=(2,))
    @test_throws ArgumentError GLLVModels._fit_projected_gradient([0.0]; lower=(1,), upper=(1,))
    @test GLLVModels._fit_natural_positive_score(-0.2, 0.1) == -2.0
    @test isnan(GLLVModels._fit_natural_positive_score(0.0, 0.0))
    unknown = GLLVModels._fit_projected_gradient([0.0]; lower=(1,),
        lower_scores=Dict(1 => NaN))
    @test isnan(only(unknown))

    good = GLLVModels._fit_convergence(optimizer_finished=true,
        objective=0.0, logLik=0.0, gradient=[0.2, 0.0],
        projected_gradient=lower, hessian=H,
        boundary_params=["variance"], boundary_check_available=true)
    bad = GLLVModels._fit_convergence(optimizer_finished=true,
        objective=0.0, logLik=0.0, gradient=[-0.2, 0.0],
        projected_gradient=wrong_lower, hessian=H,
        boundary_params=["variance"], boundary_check_available=true)
    @test good.status === :boundary
    @test good.relative_gradient == 0.0
    @test bad.status === :not_converged
    @test bad.relative_gradient > 0.0
    unknown_boundary = GLLVModels._fit_convergence(optimizer_finished=true,
        objective=0.0, logLik=0.0, gradient=[0.0],
        projected_gradient=unknown, hessian=ones(1, 1),
        boundary_params=["variance"], boundary_check_available=true)
    @test unknown_boundary.status === :degenerate
end
